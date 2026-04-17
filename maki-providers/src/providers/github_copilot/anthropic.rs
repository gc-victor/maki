//! Anthropic Messages support for GitHub Copilot `/v1/messages`.

use serde_json::{Value, json};

use crate::model::Model;
use crate::providers::anthropic::build_body;
use crate::{ContentBlock, Message, ThinkingConfig};

#[cfg(test)]
use crate::Role;

const HIGH_EFFORT: &str = "high";
const MEDIUM_EFFORT: &str = "medium";
const DEFAULT_THINKING_BUDGET: u32 = 1600;
const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";
const ADVANCED_TOOL_USE_BETA: &str = "advanced-tool-use-2025-11-20";
const PLACEHOLDER_THINKING: &str = "Thinking...";

/// Thinking mode for the `/v1/messages` endpoint (Anthropic Messages API).
///
/// Controls how reasoning/thinking is configured for Claude-family models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessagesThinkingMode {
    Off,
    Adaptive,
    Manual(u32),
}

impl MessagesThinkingMode {
    pub(crate) fn from_thinking_config(thinking: ThinkingConfig, model_id: &str) -> Self {
        match thinking {
            ThinkingConfig::Off => Self::Off,
            ThinkingConfig::Adaptive => {
                // Only 4.6+ models support adaptive. Older models fall back to
                // the Anthropic SDK's default manual budget.
                // See: https://github.com/anthropics/anthropic-sdk-python/blob/78de297e71bacbe6acf4d3b420edcaad90ce1045/examples/thinking_stream.py#L8
                if model_supports_adaptive(model_id) {
                    Self::Adaptive
                } else {
                    Self::Manual(DEFAULT_THINKING_BUDGET)
                }
            }
            ThinkingConfig::Budget(budget) => Self::Manual(budget),
        }
    }

    pub(crate) fn apply_to_body(self, body: &mut Value, model_id: &str) {
        match self {
            Self::Off => {}
            Self::Adaptive => {
                body["thinking"] = json!({"type": "adaptive"});
                let effort = if model_supports_high_effort(model_id) {
                    HIGH_EFFORT
                } else {
                    MEDIUM_EFFORT
                };
                body["output_config"] = json!({"effort": effort});
            }
            Self::Manual(budget_tokens) => {
                body["thinking"] = json!({"type": "enabled", "budget_tokens": budget_tokens});
            }
        }
    }
}

/// Parse a model version like `claude-4.6` into `(major, minor)`.
fn parse_model_version(model_id: &str) -> Option<(u32, u32)> {
    let version = &model_id[model_id.find(|c: char| c.is_ascii_digit())?..];
    let (major, rest) = parse_leading_u32(version)?;

    let minor = rest
        .strip_prefix('.')
        .and_then(|rest| parse_leading_u32(rest).map(|(minor, _)| minor))
        .unwrap_or(0);

    Some((major, minor))
}

fn parse_leading_u32(input: &str) -> Option<(u32, &str)> {
    let end = input
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(input.len());

    input[..end]
        .parse()
        .ok()
        .map(|value| (value, &input[end..]))
}

/// Check if a model supports adaptive thinking (Claude 4.6 or higher).
fn model_supports_adaptive(model_id: &str) -> bool {
    parse_model_version(model_id)
        .is_some_and(|(major, minor)| major > 4 || (major == 4 && minor >= 6))
}

/// Check if a model requires the advanced-tool-use beta (Claude 4.5 or higher).
fn model_requires_advanced_tool_use(model_id: &str) -> bool {
    parse_model_version(model_id)
        .is_some_and(|(major, minor)| major > 4 || (major == 4 && minor >= 5))
}

/// Check if a model supports high effort (Claude Opus 4.7+ does not support high).
fn model_supports_high_effort(model_id: &str) -> bool {
    parse_model_version(model_id).is_some_and(|(major, minor)| !(major == 4 && minor == 7))
}

// Managed beta tokens. Keep this in sync with the logic below.
const MANAGED_BETAS: &[&str] = &[INTERLEAVED_THINKING_BETA, ADVANCED_TOOL_USE_BETA];

/// Rebuild `anthropic-beta` with the managed betas this request needs.
pub(crate) fn adjust_anthropic_beta_header(
    headers: &mut Vec<(String, String)>,
    mode: MessagesThinkingMode,
    model_id: &str,
) {
    // Preserve caller-managed betas and replace the ones we own.
    let existing: Vec<String> = headers
        .iter()
        .position(|(k, _)| k.eq_ignore_ascii_case("anthropic-beta"))
        .map(|idx| {
            let (_, value) = headers.remove(idx);
            value
                .split(',')
                .map(|s| s.trim())
                .filter(|s| {
                    !MANAGED_BETAS
                        .iter()
                        .any(|managed| managed.eq_ignore_ascii_case(s))
                })
                .map(String::from)
                .collect()
        })
        .unwrap_or_default();

    let mut betas = existing;

    if matches!(mode, MessagesThinkingMode::Manual(_)) {
        betas.push(INTERLEAVED_THINKING_BETA.into());
    }
    if model_requires_advanced_tool_use(model_id) {
        betas.push(ADVANCED_TOOL_USE_BETA.into());
    }

    if !betas.is_empty() {
        headers.push(("anthropic-beta".into(), betas.join(", ")));
    }
}

/// Signatures containing `@` are Copilot placeholders.
fn is_invalid_thinking(thinking: &str, signature: Option<&str>) -> bool {
    thinking.is_empty()
        || thinking == PLACEHOLDER_THINKING
        || signature.is_some_and(|s| s.contains('@'))
}

fn normalize_for_claude(blocks: &[ContentBlock]) -> Vec<ContentBlock> {
    blocks
        .iter()
        .filter(|block| {
            let ContentBlock::Thinking {
                thinking,
                signature,
            } = block
            else {
                return true;
            };
            !is_invalid_thinking(thinking, signature.as_deref())
        })
        .map(|block| match block {
            ContentBlock::Thinking {
                thinking,
                signature: None,
            } => ContentBlock::Text {
                text: thinking.clone(),
            },
            _ => block.clone(),
        })
        .collect()
}

fn sanitize_tools(tools: &Value) -> Value {
    let Some(arr) = tools.as_array() else {
        return tools.clone();
    };
    Value::Array(
        arr.iter()
            .map(|t| {
                let mut tool = t.clone();
                tool.as_object_mut().map(|o| o.remove("input_examples"));
                tool
            })
            .collect(),
    )
}

pub fn build_anthropic_messages_body(
    model: &Model,
    messages: &[Message],
    system: &str,
    tools: &Value,
    thinking: ThinkingConfig,
) -> (Value, MessagesThinkingMode) {
    let normalized_messages: Vec<Message> = messages
        .iter()
        .map(|m| Message {
            role: m.role,
            content: normalize_for_claude(&m.content),
            display_text: None,
        })
        .collect();

    let sanitized_tools = sanitize_tools(tools);

    // Apply thinking after the shared builder. Copilot rejects explicit effort
    // when the shared body already set one.
    let mut body = build_body(model, &normalized_messages, &sanitized_tools, system, None);
    let mode = MessagesThinkingMode::from_thinking_config(thinking, &model.id);
    mode.apply_to_body(&mut body, &model.id);

    (body, mode)
}

#[cfg(test)]
mod tests {
    use super::*;

    use test_case::test_case;

    const TEST_PLACEHOLDER_THINKING: &str = PLACEHOLDER_THINKING;

    fn user_messages() -> Vec<Message> {
        vec![Message::user("hello".into())]
    }

    fn sample_tools() -> Value {
        serde_json::json!([{
            "name": "bash",
            "description": "Run a command",
            "input_schema": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]},
            "input_examples": [{"cmd": "pwd"}]
        }])
    }

    fn find_beta_header(headers: &[(String, String)]) -> Option<&str> {
        headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("anthropic-beta"))
            .map(|(_, v)| v.as_str())
    }

    #[test_case(MessagesThinkingMode::Manual(1024), "claude-sonnet-4",
        Some(INTERLEAVED_THINKING_BETA); "manual adds interleaved")]
    #[test_case(MessagesThinkingMode::Adaptive, "claude-sonnet-4",
        None; "adaptive no betas")]
    #[test_case(MessagesThinkingMode::Off, "claude-sonnet-4",
        None; "off no betas")]
    #[test_case(MessagesThinkingMode::Off, "claude-sonnet-4.5",
        Some(ADVANCED_TOOL_USE_BETA); "45 gets advanced-tool")]
    #[test_case(MessagesThinkingMode::Off, "claude-opus-4.6",
        Some(ADVANCED_TOOL_USE_BETA); "46 gets advanced-tool")]
    fn beta_header_added_when_missing(
        mode: MessagesThinkingMode,
        model_id: &str,
        expected: Option<&str>,
    ) {
        let mut headers = vec![];
        adjust_anthropic_beta_header(&mut headers, mode, model_id);
        assert_eq!(find_beta_header(&headers), expected);
    }

    #[test]
    fn manual_appends_to_existing() {
        let mut headers = vec![(
            "anthropic-beta".into(),
            "context-management-2025-06-27".into(),
        )];
        adjust_anthropic_beta_header(&mut headers, MessagesThinkingMode::Manual(1024), "claude-4");

        let beta = find_beta_header(&headers).unwrap();
        assert!(
            beta.contains("context-management-2025-06-27"),
            "preserves unmanaged"
        );
        assert!(beta.contains(INTERLEAVED_THINKING_BETA), "adds managed");
    }

    #[test]
    fn adaptive_removes_interleaved() {
        let mut headers = vec![(
            "anthropic-beta".into(),
            format!("{INTERLEAVED_THINKING_BETA},other-beta"),
        )];
        adjust_anthropic_beta_header(&mut headers, MessagesThinkingMode::Adaptive, "claude-4");

        let beta = find_beta_header(&headers).unwrap();
        assert!(!beta.contains(INTERLEAVED_THINKING_BETA), "removes managed");
        assert!(beta.contains("other-beta"), "preserves unmanaged");
    }

    #[test]
    fn managed_betas_deduplicated() {
        let mut headers = vec![(
            "anthropic-beta".into(),
            format!("{ADVANCED_TOOL_USE_BETA},custom-beta"),
        )];
        adjust_anthropic_beta_header(&mut headers, MessagesThinkingMode::Off, "claude-4.5");

        let beta = find_beta_header(&headers).unwrap();
        assert_eq!(beta.matches(ADVANCED_TOOL_USE_BETA).count(), 1);
    }

    #[test]
    fn case_insensitive_header_handling() {
        let mut headers = vec![("Anthropic-Beta".into(), "custom".into())];
        adjust_anthropic_beta_header(&mut headers, MessagesThinkingMode::Manual(1024), "claude-4");

        assert!(find_beta_header(&headers).unwrap().contains("custom"));
    }

    #[test]
    fn manual_thinking_on_45_plus_has_both_betas() {
        let mut headers = vec![];
        adjust_anthropic_beta_header(
            &mut headers,
            MessagesThinkingMode::Manual(1024),
            "claude-4.5",
        );

        let beta = find_beta_header(&headers).unwrap();
        assert!(beta.contains(INTERLEAVED_THINKING_BETA));
        assert!(beta.contains(ADVANCED_TOOL_USE_BETA));
    }

    #[test]
    fn messages_thinking_mode_off_does_not_modify_body() {
        let mut body = serde_json::json!({"model": "claude-sonnet-4"});
        MessagesThinkingMode::Off.apply_to_body(&mut body, "claude-sonnet-4");

        assert!(body.get("thinking").is_none());
        assert!(body.get("output_config").is_none());
    }

    #[test]
    fn messages_thinking_mode_adaptive_adds_thinking_and_output_config() {
        let mut body = serde_json::json!({"model": "claude-sonnet-4.6"});
        MessagesThinkingMode::Adaptive.apply_to_body(&mut body, "claude-sonnet-4.6");

        assert_eq!(body["thinking"]["type"], "adaptive");
        assert_eq!(body["output_config"]["effort"], "high");
    }

    #[test]
    fn messages_thinking_mode_adaptive_uses_medium_for_opus_47() {
        let mut body = serde_json::json!({"model": "claude-opus-4.7"});
        MessagesThinkingMode::Adaptive.apply_to_body(&mut body, "claude-opus-4.7");

        assert_eq!(body["thinking"]["type"], "adaptive");
        assert_eq!(body["output_config"]["effort"], "medium");
    }

    #[test]
    fn messages_thinking_mode_manual_adds_enabled_thinking_with_budget() {
        let mut body = serde_json::json!({"model": "claude-sonnet-4"});
        MessagesThinkingMode::Manual(4096).apply_to_body(&mut body, "claude-sonnet-4");

        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["budget_tokens"], 4096);
        assert!(body.get("output_config").is_none());
    }

    #[test]
    fn claude_body_format() {
        let model = Model::from_spec("github-copilot/claude-sonnet-4").unwrap();
        let (body, _) = build_anthropic_messages_body(
            &model,
            &user_messages(),
            "system",
            &serde_json::json!([]),
            ThinkingConfig::Off,
        );
        assert!(body.get("messages").is_some());
        assert!(body.get("input").is_none());
        assert!(body.get("system").is_some());
        assert!(body.get("max_tokens").is_some());
        assert!(body.get("max_completion_tokens").is_none());
    }

    #[test]
    fn top_level_system_field() {
        let model = Model::from_spec("github-copilot/claude-sonnet-4").unwrap();
        let (body, _) = build_anthropic_messages_body(
            &model,
            &user_messages(),
            "You are helpful",
            &serde_json::json!([]),
            ThinkingConfig::Off,
        );
        let system = body["system"].as_array().unwrap();
        assert_eq!(system[0]["text"], "You are helpful");
        assert!(
            !body["messages"]
                .as_array()
                .unwrap()
                .iter()
                .any(|m| m["role"] == "system")
        );
    }

    #[test]
    fn thinking_adaptive_for_46_plus() {
        let model = Model::from_spec("github-copilot/claude-sonnet-4.6").unwrap();
        let (body, _) = build_anthropic_messages_body(
            &model,
            &user_messages(),
            "system",
            &serde_json::json!([]),
            ThinkingConfig::Adaptive,
        );
        assert_eq!(body["thinking"]["type"], "adaptive");
        assert_eq!(body["output_config"]["effort"], "high");
    }

    #[test]
    fn thinking_fallback_for_pre_46() {
        let model = Model::from_spec("github-copilot/claude-sonnet-4").unwrap();
        let (body, _) = build_anthropic_messages_body(
            &model,
            &user_messages(),
            "system",
            &serde_json::json!([]),
            ThinkingConfig::Adaptive,
        );
        assert_eq!(body["thinking"]["type"], "enabled");
        assert!(body.get("output_config").is_none());
    }

    #[test]
    fn thinking_manual_budget() {
        let model = Model::from_spec("github-copilot/claude-opus-4.6").unwrap();
        let (body, _) = build_anthropic_messages_body(
            &model,
            &user_messages(),
            "system",
            &serde_json::json!([]),
            ThinkingConfig::Budget(8192),
        );
        assert_eq!(body["thinking"]["budget_tokens"], 8192);
    }

    #[test]
    fn unsigned_thinking_normalized() {
        let model = Model::from_spec("github-copilot/claude-sonnet-4").unwrap();
        let messages = vec![Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Thinking {
                    thinking: "reasoning".into(),
                    signature: None,
                },
                ContentBlock::Text {
                    text: "answer".into(),
                },
            ],
            ..Default::default()
        }];
        let (body, _) = build_anthropic_messages_body(
            &model,
            &messages,
            "system",
            &serde_json::json!([]),
            ThinkingConfig::Off,
        );
        let content = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "reasoning");
    }

    #[test]
    fn signed_thinking_preserved() {
        let model = Model::from_spec("github-copilot/claude-sonnet-4").unwrap();
        let messages = vec![Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Thinking {
                thinking: "reasoning".into(),
                signature: Some("sig".into()),
            }],
            ..Default::default()
        }];
        let (body, _) = build_anthropic_messages_body(
            &model,
            &messages,
            "system",
            &serde_json::json!([]),
            ThinkingConfig::Off,
        );
        let content = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "thinking");
        assert_eq!(content[0]["signature"], "sig");
    }

    #[test]
    fn tools_included() {
        let model = Model::from_spec("github-copilot/claude-sonnet-4").unwrap();
        let (body, _) = build_anthropic_messages_body(
            &model,
            &user_messages(),
            "system",
            &sample_tools(),
            ThinkingConfig::Off,
        );
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools[0]["name"], "bash");
        assert!(tools[0].get("function").is_none());
    }

    #[test]
    fn tools_sanitized() {
        let sanitized = sanitize_tools(&sample_tools());
        let tools = sanitized.as_array().unwrap();
        assert!(tools[0].get("input_examples").is_none());
    }

    #[test]
    fn tools_non_array_passthrough() {
        let tools = serde_json::json!({"not": "an array"});
        let sanitized = sanitize_tools(&tools);
        assert_eq!(sanitized, tools);
    }

    #[test]
    fn cache_breakpoint_single_user_message() {
        let model = Model::from_spec("github-copilot/claude-sonnet-4").unwrap();
        let messages = vec![Message::user("hello".into())];
        let (body, _) = build_anthropic_messages_body(
            &model,
            &messages,
            "system",
            &serde_json::json!([]),
            ThinkingConfig::Off,
        );

        let wire_messages = body["messages"].as_array().unwrap();
        assert_eq!(wire_messages.len(), 1);
        let content = wire_messages[0]["content"].as_array().unwrap();
        assert_eq!(content[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn cache_breakpoints_last_two_messages_only() {
        let model = Model::from_spec("github-copilot/claude-sonnet-4").unwrap();
        let messages = vec![
            Message::user("first".into()),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text {
                    text: "reply".into(),
                }],
                ..Default::default()
            },
            Message {
                role: Role::User,
                content: vec![
                    ContentBlock::ToolResult {
                        tool_use_id: "t1".into(),
                        content: "ok".into(),
                        is_error: false,
                    },
                    ContentBlock::Text {
                        text: "second".into(),
                    },
                ],
                ..Default::default()
            },
        ];
        let (body, _) = build_anthropic_messages_body(
            &model,
            &messages,
            "system",
            &serde_json::json!([]),
            ThinkingConfig::Off,
        );

        let wire_messages = body["messages"].as_array().unwrap();
        assert_eq!(wire_messages.len(), 3);

        assert!(
            wire_messages[0]["content"][0]
                .get("cache_control")
                .is_none()
        );

        assert_eq!(
            wire_messages[1]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );

        assert!(
            wire_messages[2]["content"][0]
                .get("cache_control")
                .is_none()
        );
        assert_eq!(
            wire_messages[2]["content"][1]["cache_control"]["type"],
            "ephemeral"
        );
    }

    #[test]
    fn cache_breakpoint_last_tool_only() {
        let model = Model::from_spec("github-copilot/claude-sonnet-4").unwrap();
        let tools = serde_json::json!([
            {"name": "tool1", "description": "first"},
            {"name": "tool2", "description": "second"},
            {"name": "tool3", "description": "third"}
        ]);
        let (body, _) = build_anthropic_messages_body(
            &model,
            &[Message::user("hello".into())],
            "system",
            &tools,
            ThinkingConfig::Off,
        );

        let wire_tools = body["tools"].as_array().unwrap();
        assert_eq!(wire_tools.len(), 3);

        assert!(wire_tools[0].get("cache_control").is_none());
        assert!(wire_tools[1].get("cache_control").is_none());

        assert_eq!(wire_tools[2]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn cache_breakpoint_system_block() {
        let model = Model::from_spec("github-copilot/claude-sonnet-4").unwrap();
        let (body, _) = build_anthropic_messages_body(
            &model,
            &[Message::user("hello".into())],
            "You are helpful",
            &serde_json::json!([]),
            ThinkingConfig::Off,
        );

        let system = body["system"].as_array().unwrap();
        assert_eq!(system.len(), 1);
        assert_eq!(system[0]["cache_control"]["type"], "ephemeral");
        assert_eq!(system[0]["text"], "You are helpful");
    }

    #[test]
    fn thinking_applied_after_shared_body_construction() {
        let model = Model::from_spec("github-copilot/claude-sonnet-4.6").unwrap();
        let messages = vec![
            Message::user("first".into()),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text {
                    text: "reply".into(),
                }],
                ..Default::default()
            },
        ];
        let (body, mode) = build_anthropic_messages_body(
            &model,
            &messages,
            "system",
            &serde_json::json!([]),
            ThinkingConfig::Adaptive,
        );

        assert_eq!(body["thinking"]["type"], "adaptive");
        assert_eq!(body["output_config"]["effort"], "high");
        assert_eq!(mode, MessagesThinkingMode::Adaptive);

        let wire_messages = body["messages"].as_array().unwrap();
        assert_eq!(
            wire_messages[1]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );

        let system = body["system"].as_array().unwrap();
        assert_eq!(system[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn tool_sanitization_removes_input_examples() {
        let model = Model::from_spec("github-copilot/claude-sonnet-4").unwrap();
        let tools = serde_json::json!([{
            "name": "bash",
            "description": "Run a command",
            "input_schema": {"type": "object"},
            "input_examples": [{"cmd": "pwd"}]
        }]);
        let (body, _) = build_anthropic_messages_body(
            &model,
            &[Message::user("hello".into())],
            "system",
            &tools,
            ThinkingConfig::Off,
        );

        let wire_tools = body["tools"].as_array().unwrap();
        assert!(wire_tools[0].get("input_examples").is_none());
        assert_eq!(wire_tools[0]["name"], "bash");
    }

    #[test]
    fn unsigned_thinking_normalized_before_body_construction() {
        let model = Model::from_spec("github-copilot/claude-sonnet-4").unwrap();
        let messages = vec![Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Thinking {
                    thinking: "my reasoning".into(),
                    signature: None,
                },
                ContentBlock::Text {
                    text: "answer".into(),
                },
            ],
            ..Default::default()
        }];
        let (body, _) = build_anthropic_messages_body(
            &model,
            &messages,
            "system",
            &serde_json::json!([]),
            ThinkingConfig::Off,
        );

        let content = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "my reasoning");
        assert_eq!(content[1]["type"], "text");
        assert_eq!(content[1]["text"], "answer");
    }

    #[test_case(TEST_PLACEHOLDER_THINKING, None; "placeholder text")]
    #[test_case("", Some("sig"); "empty thinking text")]
    #[test_case("valid", Some("sig@bad"); "signature with @")]
    #[test_case(TEST_PLACEHOLDER_THINKING, Some(""); "placeholder with empty sig")]
    fn drops_invalid_thinking_blocks(thinking: &str, signature: Option<&str>) {
        let blocks = vec![ContentBlock::Thinking {
            thinking: thinking.into(),
            signature: signature.map(|s| s.into()),
        }];
        assert!(normalize_for_claude(&blocks).is_empty());
    }

    #[test]
    fn keeps_valid_signed_thinking() {
        let blocks = vec![ContentBlock::Thinking {
            thinking: "actual reasoning".into(),
            signature: Some("clean-sig".into()),
        }];
        let result = normalize_for_claude(&blocks);

        assert_eq!(result.len(), 1);
        assert!(
            matches!(&result[0], ContentBlock::Thinking { thinking, signature }
                if thinking == "actual reasoning" && signature.as_deref() == Some("clean-sig"))
        );
    }

    #[test]
    fn converts_unsigned_thinking_to_text() {
        let blocks = vec![ContentBlock::Thinking {
            thinking: "valid reasoning".into(),
            signature: None,
        }];
        let result = normalize_for_claude(&blocks);

        assert_eq!(result.len(), 1);
        assert!(matches!(&result[0], ContentBlock::Text { text } if text == "valid reasoning"));
    }

    #[test]
    fn filters_mixed_blocks() {
        let blocks = vec![
            ContentBlock::Thinking {
                thinking: TEST_PLACEHOLDER_THINKING.into(),
                signature: None,
            },
            ContentBlock::Text {
                text: "actual text".into(),
            },
            ContentBlock::Thinking {
                thinking: "unsigned".into(),
                signature: None,
            },
            ContentBlock::Thinking {
                thinking: "signed".into(),
                signature: Some("good".into()),
            },
            ContentBlock::Thinking {
                thinking: "bad".into(),
                signature: Some("sig@bad".into()),
            },
        ];
        let result = normalize_for_claude(&blocks);

        assert_eq!(result.len(), 3);
        assert!(matches!(&result[0], ContentBlock::Text { text } if text == "actual text"));
        assert!(matches!(&result[1], ContentBlock::Text { text } if text == "unsigned"));
        assert!(
            matches!(&result[2], ContentBlock::Thinking { thinking, signature }
                if thinking == "signed" && signature.as_deref() == Some("good"))
        );
    }
}
