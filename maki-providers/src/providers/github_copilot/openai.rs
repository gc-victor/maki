use serde_json::{Value, json};

use crate::model::Model;
use crate::providers::openai::responses::{convert_input, convert_tools_for_responses};
use crate::{ThinkingConfig, TokenUsage};

/// Thinking mode for GitHub Copilot's `/responses` endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponsesThinkingMode {
    Off,
    Enabled,
}

impl ResponsesThinkingMode {
    /// Map shared thinking config to the Responses wire format.
    pub(crate) fn from_thinking(thinking: ThinkingConfig) -> Self {
        match thinking {
            ThinkingConfig::Off => Self::Off,
            ThinkingConfig::Adaptive | ThinkingConfig::Budget(_) => Self::Enabled,
        }
    }

    pub(crate) fn apply_to_body(self, body: &mut Value) {
        match self {
            Self::Off => {}
            Self::Enabled => {
                body["reasoning"] = json!({"effort": "high", "summary": "detailed"});
                body["include"] = json!(["reasoning.encrypted_content"]);
            }
        }
    }
}

/// Copilot uses prompt_tokens/completion_tokens instead of OpenAI's input_tokens/output_tokens.
pub(crate) fn parse_usage(u: &Value) -> TokenUsage {
    let input = u["prompt_tokens"]
        .as_u64()
        .or_else(|| u["input_tokens"].as_u64())
        .unwrap_or(0) as u32;
    let output = u["completion_tokens"]
        .as_u64()
        .or_else(|| u["output_tokens"].as_u64())
        .unwrap_or(0) as u32;
    let cached = u["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .or_else(|| u["input_tokens_details"]["cached_tokens"].as_u64())
        .unwrap_or(0) as u32;
    TokenUsage {
        input: input.saturating_sub(cached),
        output,
        cache_read: cached,
        cache_creation: 0,
    }
}

fn derive_cache_key(session_id: &str) -> String {
    format!("maki-{}", session_id)
}

/// Copilot rejects cache_control.scope on /responses but accepts other cache_control fields.
pub(crate) fn apply_responses_extensions(body: &mut Value, session_id: Option<&str>) {
    if let Some(id) = session_id {
        body["prompt_cache_key"] = json!(derive_cache_key(id));
    }
    strip_unsupported_cache_scope(body);
}

/// Copilot rejects `cache_control.scope` on `/responses`, but accepts the rest.
fn strip_unsupported_cache_scope(value: &mut Value) {
    match value {
        Value::Object(map) => {
            if let Some(Value::Object(cache_map)) = map.get_mut("cache_control") {
                cache_map.remove("scope");
                if cache_map.is_empty() {
                    map.remove("cache_control");
                }
            }
            for v in map.values_mut() {
                strip_unsupported_cache_scope(v);
            }
        }
        Value::Array(arr) => {
            for v in arr.iter_mut() {
                strip_unsupported_cache_scope(v);
            }
        }
        _ => {}
    }
}

/// Build a `/responses` request body.
pub fn build_responses_body(
    model: &Model,
    messages: &[crate::Message],
    system: &str,
    tools: &Value,
    strict_tools: bool,
    thinking: ThinkingConfig,
) -> Value {
    let input = convert_input(messages);
    let wire_tools = convert_tools_for_responses(tools, strict_tools);

    let mut body = json!({
        "model": model.id,
        "instructions": system,
        "input": input,
        "stream": true,
        "store": false,
    });

    if wire_tools.as_array().is_some_and(|a| !a.is_empty()) {
        body["tools"] = wire_tools;
    }

    let mode = ResponsesThinkingMode::from_thinking(thinking);
    mode.apply_to_body(&mut body);

    body
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::github_copilot::test_constants::{GPT5_MODEL, GPT5_SPEC};
    use crate::{Message, ThinkingConfig};

    fn user_messages() -> Vec<Message> {
        vec![Message::user("hello".into())]
    }

    fn sample_tools() -> Value {
        serde_json::json!([{
            "name": "bash",
            "description": "Run a shell command",
            "input_schema": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            }
        }])
    }

    #[test]
    fn gpt_5_family_models_use_responses_body_format() {
        let model = Model::from_spec(GPT5_SPEC).unwrap();

        let body = build_responses_body(
            &model,
            &user_messages(),
            "system",
            &serde_json::json!([]),
            false,
            ThinkingConfig::Off,
        );

        assert!(body.get("input").is_some(), "/responses uses input array");
        assert!(
            body.get("messages").is_none(),
            "/responses does not use messages array"
        );
        assert_eq!(body["model"], GPT5_MODEL);
        assert_eq!(body["instructions"], "system");
        assert_eq!(body["stream"], true);
        assert_eq!(body["store"], false);
        assert!(body.get("tools").is_none());

        let input = body["input"].as_array().expect("input should be an array");
        assert!(!input.is_empty());
        assert_eq!(input[0]["type"], "message");
        assert_eq!(input[0]["role"], "user");
    }

    #[test]
    fn responses_body_includes_tools_when_provided() {
        let model = Model::from_spec(GPT5_SPEC).unwrap();
        let tools = sample_tools();

        let body = build_responses_body(
            &model,
            &user_messages(),
            "system",
            &tools,
            false,
            ThinkingConfig::Off,
        );

        let tools_array = body["tools"].as_array().expect("tools should be an array");
        assert_eq!(tools_array.len(), 1);
        assert_eq!(tools_array[0]["type"], "function");
        assert_eq!(tools_array[0]["name"], "bash");
        assert_eq!(tools_array[0]["description"], "Run a shell command");
        assert!(
            tools_array[0].get("function").is_none(),
            "responses endpoint tools must not have nested 'function' object"
        );
    }

    #[test]
    fn responses_body_with_thinking_adaptive_includes_reasoning() {
        let model = Model::from_spec(GPT5_SPEC).unwrap();

        let body = build_responses_body(
            &model,
            &user_messages(),
            "system",
            &serde_json::json!([]),
            false,
            ThinkingConfig::Adaptive,
        );

        assert_eq!(body["reasoning"]["effort"], "high");
        assert_eq!(body["reasoning"]["summary"], "detailed");
        assert_eq!(
            body["include"],
            serde_json::json!(["reasoning.encrypted_content"])
        );
    }

    #[test]
    fn responses_body_with_thinking_off_omits_reasoning() {
        let model = Model::from_spec(GPT5_SPEC).unwrap();

        let body = build_responses_body(
            &model,
            &user_messages(),
            "system",
            &serde_json::json!([]),
            false,
            ThinkingConfig::Off,
        );

        assert!(body.get("reasoning").is_none());
        assert!(body.get("include").is_none());
    }

    #[test]
    fn parse_usage_cached_tokens_accounting() {
        let u = serde_json::json!({
            "prompt_tokens": 500,
            "completion_tokens": 100,
            "prompt_tokens_details": { "cached_tokens": 300 },
        });
        let usage = parse_usage(&u);
        assert_eq!(usage.input, 200);
        assert_eq!(usage.output, 100);
        assert_eq!(usage.cache_read, 300);
    }

    #[test]
    fn parse_usage_cached_tokens_saturating() {
        let u = serde_json::json!({
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "prompt_tokens_details": { "cached_tokens": 150 },
        });
        let usage = parse_usage(&u);
        assert_eq!(usage.input, 0);
        assert_eq!(usage.cache_read, 150);
    }

    #[test]
    fn parse_usage_input_tokens_details_cached_tokens() {
        let u = serde_json::json!({
            "input_tokens": 400,
            "output_tokens": 80,
            "input_tokens_details": { "cached_tokens": 250 },
        });
        let usage = parse_usage(&u);
        assert_eq!(usage.input, 150);
        assert_eq!(usage.output, 80);
        assert_eq!(usage.cache_read, 250);
    }

    #[test]
    fn apply_responses_extensions_injects_prompt_cache_key() {
        let mut body = serde_json::json!({"model": "gpt-5"});
        apply_responses_extensions(&mut body, Some("session-xyz"));
        assert_eq!(body["prompt_cache_key"], "maki-session-xyz");
    }

    #[test]
    fn apply_responses_extensions_strips_nested_cache_control_scope() {
        let mut body = serde_json::json!({
            "model": "gpt-5",
            "input": [
                {
                    "type": "message",
                    "content": [{ "type": "output_text", "text": "hello" }],
                    "cache_control": { "scope": "some_scope", "type": "some_type" }
                }
            ]
        });
        apply_responses_extensions(&mut body, None);

        let inner_cache = &body["input"][0]["cache_control"];
        assert!(inner_cache.get("scope").is_none());
        assert_eq!(inner_cache.get("type").unwrap(), "some_type");
    }

    #[test]
    fn apply_responses_extensions_removes_empty_cache_control_objects() {
        let mut body = serde_json::json!({
            "model": "gpt-5",
            "cache_control": { "scope": "only_scope" }
        });
        apply_responses_extensions(&mut body, None);

        assert!(body.get("cache_control").is_none());
    }

    #[test]
    fn derive_cache_key_prefixes_session_id() {
        assert_eq!(derive_cache_key("session-abc"), "maki-session-abc");
        assert_eq!(derive_cache_key("uuid-123e-4567"), "maki-uuid-123e-4567");
        assert_ne!(derive_cache_key("session-a"), derive_cache_key("session-b"));
    }
}
