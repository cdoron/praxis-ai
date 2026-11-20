// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! A minimal server-owned tool for the Anthropic Messages agentic loop,
//! modeled on [`super::AnthropicWebSearchFilter`] but with no outbound
//! callout: it answers a managed tool call with a fixed, configured
//! string instead of querying an external provider.
//!
//! Every round is forced to the buffered subrequest transport regardless
//! of the client's requested `stream` value, which keeps this filter free
//! of incremental SSE parsing. When the original client asked for
//! streaming, the terminal (non-looping) round is re-encoded as a single
//! Anthropic Messages SSE lifecycle so the wire contract still matches
//! what the client asked for.

use async_trait::async_trait;
use bytes::Bytes;
use http::header::{ACCEPT_ENCODING, CONTENT_TYPE, HeaderValue};
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, IterationState, NextIterationBody,
    Rejection, SubRequestResponseMode, parse_filter_config,
};
use serde::Deserialize;
use serde_json::{Value, json};

use super::wire;

/// Registry name and filter-results namespace.
const FILTER_NAME: &str = "where_is_treasure";
/// IRR action that re-enters the inference step.
const ACTION_LOOP: &str = "loop";
/// IRR action that returns the current response to the client.
const ACTION_DONE: &str = "done";
/// IRR accumulator entry holding the latest serialized Messages request.
const REQUEST_ACCUMULATOR_KEY: &str = "where_is_treasure.request";
/// Default tool name advertised to and matched against the client.
const DEFAULT_TOOL_NAME: &str = "WhereIsTreasure";

/// Default maximum body size (1 MiB).
fn default_max_body_bytes() -> usize {
    1_048_576
}

/// YAML configuration for [`WhereIsTreasureFilter`].
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WhereIsTreasureConfig {
    /// Tool name advertised to and matched against the client. Defaults to
    /// [`DEFAULT_TOOL_NAME`].
    tool_name: Option<String>,
    /// The fixed answer returned as the tool result.
    location: String,
    /// Maximum request/response body size in bytes for `StreamBuffer` mode.
    #[serde(default = "default_max_body_bytes")]
    max_body_bytes: usize,
}

/// Executes a server-owned managed tool call in an Anthropic Messages loop,
/// answering with a fixed configured string.
pub struct WhereIsTreasureFilter {
    /// Tool name advertised to and matched against the client.
    tool_name: String,
    /// The fixed answer returned as the tool result.
    location: String,
    /// Maximum request/response body size in bytes for `StreamBuffer` mode.
    max_body_bytes: usize,
}

impl WhereIsTreasureFilter {
    /// Build the filter from its YAML configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when `config` does not match
    /// [`WhereIsTreasureConfig`] (missing `location`, or an unknown field).
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: WhereIsTreasureConfig = parse_filter_config(FILTER_NAME, config)?;
        Ok(Box::new(Self {
            tool_name: cfg.tool_name.unwrap_or_else(|| DEFAULT_TOOL_NAME.to_owned()),
            location: cfg.location,
            max_body_bytes: cfg.max_body_bytes,
        }))
    }

    /// Whether this request body is an IRR re-entry (a later agentic
    /// round), distinguished by the router-owned [`IterationState`]
    /// carrying the previous round's response. A fresh, first-round
    /// request has none.
    fn is_reentry(ctx: &HttpFilterContext<'_>) -> bool {
        ctx.extensions
            .get::<IterationState>()
            .and_then(|state| state.previous_response.as_ref())
            .is_some()
    }

    /// Force the upcoming backend round to the buffered subrequest
    /// transport, regardless of what the client asked for.
    fn force_buffered_transport(ctx: &mut HttpFilterContext<'_>) {
        ctx.set_subrequest_response_mode(SubRequestResponseMode::Buffered);
    }

    /// Whether the original client request asked for a streamed response.
    /// Read fresh from the IRR-retained first-round body rather than
    /// cached, since that body is never mutated once retained.
    fn original_wanted_streaming(ctx: &HttpFilterContext<'_>) -> bool {
        ctx.extensions
            .get::<IterationState>()
            .and_then(|state| serde_json::from_slice::<Value>(&state.original_request.body).ok())
            .and_then(|value| value.get("stream").and_then(Value::as_bool))
            .unwrap_or(false)
    }

    /// Find a managed tool-use block anywhere in an assistant message's
    /// content and return its `id` plus the complete content array (so
    /// any preceding text blocks are preserved verbatim in the next
    /// round's conversation history).
    fn managed_call_from_response(&self, bytes: &[u8]) -> Option<(String, Vec<Value>)> {
        let response: Value = serde_json::from_slice(bytes).ok()?;
        if response.get("type").and_then(Value::as_str) != Some("message") {
            return None;
        }
        if response.get("role").and_then(Value::as_str) != Some("assistant") {
            return None;
        }
        let content = response.get("content")?.as_array()?;
        let tool_use = content.iter().find(|block| {
            block.get("type").and_then(Value::as_str) == Some("tool_use")
                && block.get("name").and_then(Value::as_str) == Some(self.tool_name.as_str())
        })?;
        let id = tool_use.get("id").and_then(Value::as_str)?;
        if id.is_empty() {
            return None;
        }
        Some((id.to_owned(), content.clone()))
    }

    /// Force `"stream": false` on a request body, forwarding every round
    /// to the backend as plain buffered JSON.
    fn force_non_streaming_body(mut request: Value) -> Value {
        if let Some(object) = request.as_object_mut() {
            object.insert("stream".to_owned(), Value::Bool(false));
        }
        request
    }

    /// Handle round 0 (or any round that is not an IRR re-entry): force
    /// non-streaming transport and continue unmodified otherwise.
    fn handle_fresh_request(
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
    ) -> Result<FilterAction, FilterError> {
        let Some(bytes) = body.as_deref() else {
            return Ok(FilterAction::Continue);
        };
        let Ok(request) = serde_json::from_slice::<Value>(bytes) else {
            return Ok(FilterAction::Continue);
        };
        let request = Self::force_non_streaming_body(request);
        let rebuilt = serde_json::to_vec(&request)
            .map_err(|error| FilterError::from(format!("{FILTER_NAME}: request serialization failed: {error}")))?;
        *body = Some(Bytes::from(rebuilt));
        Self::force_buffered_transport(ctx);
        Ok(FilterAction::Continue)
    }

    /// Answer the pending managed call recovered from the previous round's
    /// response and rebuild the request for the next round.
    fn handle_reentry(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
    ) -> Result<FilterAction, FilterError> {
        let Some(iteration_state) = ctx.extensions.get::<IterationState>() else {
            return Err(FilterError::from(format!(
                "{FILTER_NAME}: IRR iteration state unavailable during re-entry"
            )));
        };
        let Some(previous_response) = iteration_state.previous_response.as_ref() else {
            return Err(FilterError::from(format!(
                "{FILTER_NAME}: previous IRR response unavailable during re-entry"
            )));
        };
        let Some((id, assistant_content)) = self.managed_call_from_response(&previous_response.body) else {
            return Err(FilterError::from(format!(
                "{FILTER_NAME}: previous response no longer contains a managed {} call",
                self.tool_name
            )));
        };

        let request_bytes = iteration_state
            .accumulator
            .get(REQUEST_ACCUMULATOR_KEY)
            .unwrap_or(&iteration_state.original_request.body);
        let mut request: Value = serde_json::from_slice(request_bytes)
            .map_err(|error| FilterError::from(format!("{FILTER_NAME}: retained request parsing failed: {error}")))?;
        if let Err(rejection) = append_treasure_turn(&mut request, &id, assistant_content, &self.location) {
            return Ok(FilterAction::Reject(rejection));
        }
        Self::commit_reentry_request(ctx, body, request)
    }

    /// Force non-streaming transport, serialize the rebuilt request, and
    /// retain it for the IRR's next round.
    fn commit_reentry_request(
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        request: Value,
    ) -> Result<FilterAction, FilterError> {
        let request = Self::force_non_streaming_body(request);
        let rebuilt = serde_json::to_vec(&request)
            .map_err(|error| FilterError::from(format!("{FILTER_NAME}: request serialization failed: {error}")))?;
        let rebuilt = Bytes::from(rebuilt);
        Self::force_buffered_transport(ctx);
        let iteration_state = ctx.extensions.get_mut::<IterationState>().ok_or_else(|| {
            FilterError::from(format!(
                "{FILTER_NAME}: IRR iteration state unavailable while retaining request"
            ))
        })?;
        iteration_state
            .accumulator
            .insert(REQUEST_ACCUMULATOR_KEY.to_owned(), rebuilt.clone());
        ctx.extensions.insert(NextIterationBody(rebuilt.clone()));
        ctx.request_headers_to_set
            .push((CONTENT_TYPE, HeaderValue::from_static("application/json")));
        *body = Some(rebuilt);
        Ok(FilterAction::Continue)
    }

    /// Re-encode a buffered Anthropic Messages JSON response as a single
    /// Anthropic Messages SSE lifecycle, for a client that asked to stream
    /// but whose backend round was forced to buffered transport.
    fn buffered_to_sse(bytes: &[u8]) -> Option<Vec<u8>> {
        let response: Value = serde_json::from_slice(bytes).ok()?;
        Some(render_sse_lifecycle(&TerminalAnswer::from_response(&response)))
    }
}

/// Fields of a buffered Messages response needed to re-encode it as SSE.
struct TerminalAnswer {
    /// Message ID, echoed into `message_start`.
    id: String,
    /// Model name, echoed into `message_start`.
    model: String,
    /// Stop reason, echoed into `message_delta`.
    stop_reason: String,
    /// Usage object, echoed into `message_delta` verbatim.
    usage: Value,
    /// Concatenated text of every `text` content block.
    text: String,
}

impl TerminalAnswer {
    /// Extract the fields needed to re-encode `response` as SSE.
    fn from_response(response: &Value) -> Self {
        let text = response
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|block| block.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|block| block.get("text").and_then(Value::as_str))
            .collect();
        Self {
            id: response
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or("msg_where_is_treasure")
                .to_owned(),
            model: response.get("model").and_then(Value::as_str).unwrap_or("").to_owned(),
            stop_reason: response
                .get("stop_reason")
                .and_then(Value::as_str)
                .unwrap_or("end_turn")
                .to_owned(),
            usage: response.get("usage").cloned().unwrap_or_else(|| json!({})),
            text,
        }
    }
}

/// Render one coherent Anthropic Messages SSE lifecycle for a terminal,
/// non-looping answer.
fn render_sse_lifecycle(answer: &TerminalAnswer) -> Vec<u8> {
    let mut out = Vec::new();
    push_opening_events(&mut out, answer);
    push_closing_events(&mut out, answer);
    out
}

/// Push `message_start` and `content_block_start`.
fn push_opening_events(out: &mut Vec<u8>, answer: &TerminalAnswer) {
    push_event(
        out,
        "message_start",
        &json!({"type": "message_start", "message": {
            "id": answer.id, "type": "message", "role": "assistant", "model": answer.model,
            "content": [], "stop_reason": null, "stop_sequence": null,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }}),
    );
    push_event(
        out,
        "content_block_start",
        &json!({"type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "citations": null, "text": ""}}),
    );
}

/// Push `content_block_delta`, `content_block_stop`, `message_delta`, and
/// `message_stop`.
fn push_closing_events(out: &mut Vec<u8>, answer: &TerminalAnswer) {
    push_event(
        out,
        "content_block_delta",
        &json!({"type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": answer.text}}),
    );
    push_event(
        out,
        "content_block_stop",
        &json!({"type": "content_block_stop", "index": 0}),
    );
    push_event(
        out,
        "message_delta",
        &json!({"type": "message_delta",
            "delta": {"stop_reason": answer.stop_reason, "stop_sequence": null}, "usage": answer.usage}),
    );
    push_event(out, "message_stop", &json!({"type": "message_stop"}));
}

/// Append the assistant tool call and matching user result turn.
fn append_treasure_turn(
    request: &mut Value,
    id: &str,
    assistant_content: Vec<Value>,
    location: &str,
) -> Result<(), Rejection> {
    let Some(messages) = request.get_mut("messages").and_then(Value::as_array_mut) else {
        return Err(wire::error_rejection(
            400,
            wire::ErrorType::InvalidRequest,
            "messages must be an array for treasure re-entry",
        ));
    };
    let mut assistant_turn = serde_json::Map::new();
    assistant_turn.insert("role".to_owned(), Value::String("assistant".to_owned()));
    assistant_turn.insert("content".to_owned(), Value::Array(assistant_content));
    messages.push(Value::Object(assistant_turn));
    messages.push(json!({
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": id, "content": location}],
    }));
    if request.get("tool_choice").is_some()
        && let Some(object) = request.as_object_mut()
    {
        object.insert("tool_choice".to_owned(), json!({"type": "auto"}));
    }
    Ok(())
}

/// Serialize one Anthropic Messages SSE frame: `event: <name>\ndata:
/// <json>\n\n`.
fn push_event(out: &mut Vec<u8>, event: &str, data: &Value) {
    out.extend_from_slice(b"event: ");
    out.extend_from_slice(event.as_bytes());
    out.extend_from_slice(b"\ndata: ");
    out.extend_from_slice(&serde_json::to_vec(data).unwrap_or_default());
    out.extend_from_slice(b"\n\n");
}

/// Publish the loop decision for the IRR transition table.
fn set_action(ctx: &mut HttpFilterContext<'_>, action: &'static str) -> Result<(), FilterError> {
    ctx.filter_results
        .entry(FILTER_NAME)
        .or_default()
        .set("action", action)?;
    Ok(())
}

/// Whether the upstream response succeeded, treating an absent header
/// (not yet populated) as success so classification proceeds.
fn is_success_response(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.response_header
        .as_ref()
        .is_none_or(|response| response.status.is_success())
}

#[async_trait]
impl HttpFilter for WhereIsTreasureFilter {
    fn name(&self) -> &'static str {
        FILTER_NAME
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(self.max_body_bytes),
        }
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    fn response_body_mode(&self) -> BodyMode {
        // Every round is forced to the buffered subrequest transport (see
        // `force_buffered_transport`), so there is no live stream to
        // compose with; StreamBuffer hands this filter the complete body
        // in one call.
        BodyMode::StreamBuffer {
            max_bytes: Some(self.max_body_bytes),
        }
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        // A compressed body cannot be parsed as JSON.
        ctx.request_headers_to_remove.push(ACCEPT_ENCODING);
        Ok(FilterAction::Continue)
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }
        if Self::is_reentry(ctx) {
            return self.handle_reentry(ctx, body);
        }
        Self::handle_fresh_request(ctx, body)
    }

    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        // Every round talks to the backend as plain JSON, so the real
        // Content-Type is always "application/json". Only the round that
        // turns out to be terminal is ever externally visible (the IRR
        // discards every looping round's response before it reaches the
        // client), so unconditionally overriding it here on every round is
        // safe and correctly applies to whichever round ends up terminal.
        if Self::original_wanted_streaming(ctx)
            && let Some(response) = ctx.response_header.as_mut()
        {
            response
                .headers
                .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        }
        Ok(FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }
        if !is_success_response(ctx) {
            set_action(ctx, ACTION_DONE)?;
            return Ok(FilterAction::Continue);
        }
        let Some(bytes) = body.as_deref() else {
            set_action(ctx, ACTION_DONE)?;
            return Ok(FilterAction::Continue);
        };
        if self.managed_call_from_response(bytes).is_some() {
            set_action(ctx, ACTION_LOOP)?;
            return Ok(FilterAction::Continue);
        }
        set_action(ctx, ACTION_DONE)?;
        if Self::original_wanted_streaming(ctx)
            && let Some(sse) = Self::buffered_to_sse(bytes)
        {
            *body = Some(Bytes::from(sse));
        }
        Ok(FilterAction::Continue)
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests {
    use super::*;

    fn filter() -> WhereIsTreasureFilter {
        WhereIsTreasureFilter {
            tool_name: DEFAULT_TOOL_NAME.to_owned(),
            location: "Buried beneath the old lighthouse on Craggy Isle.".to_owned(),
            max_body_bytes: default_max_body_bytes(),
        }
    }

    #[test]
    fn finds_managed_call_among_other_content() {
        let body = json!({
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_1", "name": "WhereIsTreasure", "input": {}},
            ],
        });
        let bytes = serde_json::to_vec(&body).unwrap();
        let (id, content) = filter().managed_call_from_response(&bytes).expect("managed call");
        assert_eq!(id, "toolu_1");
        assert_eq!(content.len(), 2);
    }

    #[test]
    fn ignores_unmanaged_tool_name() {
        let body = json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_1", "name": "SomeOtherTool", "input": {}}],
        });
        let bytes = serde_json::to_vec(&body).unwrap();
        assert!(filter().managed_call_from_response(&bytes).is_none());
    }

    #[test]
    fn ignores_plain_text_response() {
        let body = json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello"}],
        });
        let bytes = serde_json::to_vec(&body).unwrap();
        assert!(filter().managed_call_from_response(&bytes).is_none());
    }

    #[test]
    fn force_non_streaming_body_overwrites_stream_field() {
        let request = json!({"stream": true, "messages": []});
        let rewritten = WhereIsTreasureFilter::force_non_streaming_body(request);
        assert_eq!(rewritten.get("stream"), Some(&Value::Bool(false)));
    }

    #[test]
    fn buffered_to_sse_carries_full_text_and_stop_reason() {
        let body = json!({
            "id": "msg_1", "model": "test-model", "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 7},
            "content": [{"type": "text", "text": "Buried beneath the old lighthouse."}],
        });
        let bytes = serde_json::to_vec(&body).unwrap();
        let sse = WhereIsTreasureFilter::buffered_to_sse(&bytes).expect("sse bytes");
        let text = String::from_utf8(sse).unwrap();
        assert!(text.starts_with("event: message_start\n"));
        assert!(text.contains("Buried beneath the old lighthouse."));
        assert!(text.contains("\"stop_reason\":\"end_turn\""));
        assert!(text.ends_with("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"));
    }
}
