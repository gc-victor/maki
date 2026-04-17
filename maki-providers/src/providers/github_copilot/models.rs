use crate::model::{ModelEntry, ModelFamily, ModelPricing, ModelTier};
use crate::providers::github_copilot::platform::EndpointPath;

const GPT_PREFIX: &str = "gpt-";

/// Extracts the major version number from a lowercase GPT model ID.
/// Returns Some(version) for strings like "gpt-5", "gpt-5.2", "gpt-6-alpha".
/// Returns None if the model doesn't start with "gpt-" or has no valid version.
fn extract_gpt_version(model_id: &str) -> Option<u32> {
    let after_prefix = model_id.strip_prefix(GPT_PREFIX)?;
    let version_digits: String = after_prefix
        .split('.')
        .next()?
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    if version_digits.is_empty() {
        return None;
    }
    version_digits.parse::<u32>().ok()
}

/// Routes a model to its appropriate API endpoint based on family classification.
/// - Claude models use Anthropic's /v1/messages endpoint
/// - GPT-5+ models use OpenAI's /responses endpoint
/// - All other models (GPT-4 family, unknown, generic) use /chat/completions
pub(crate) fn endpoint_path_for_model(model_id: &str, family: ModelFamily) -> EndpointPath {
    // Normalize to lowercase for case-insensitive version detection
    let model_lower = model_id.to_ascii_lowercase();
    match family {
        ModelFamily::Claude => EndpointPath::V1Messages,
        ModelFamily::Gpt => {
            // Within the GPT family, only version 5+ uses the Responses API
            if extract_gpt_version(&model_lower).is_some_and(|v| v >= 5) {
                EndpointPath::Responses
            } else {
                EndpointPath::ChatCompletions
            }
        }
        // All other families (Generic, Gemini, Glm, Synthetic, unknown) default to chat completions
        _ => EndpointPath::ChatCompletions,
    }
}

pub(crate) fn models() -> &'static [ModelEntry] {
    &[
        // Weak tier: prefix name → family → number (low→high)
        ModelEntry {
            prefixes: &["claude-haiku-4.5"],
            tier: ModelTier::Weak,
            family: ModelFamily::Claude,
            default: false,
            pricing: ModelPricing::ZERO,
            max_output_tokens: 32_000,
            context_window: 144_000,
        },
        ModelEntry {
            prefixes: &["gemini-3-flash-preview"],
            tier: ModelTier::Weak,
            family: ModelFamily::Generic,
            default: false,
            pricing: ModelPricing::ZERO,
            max_output_tokens: 64_000,
            context_window: 128_000,
        },
        ModelEntry {
            prefixes: &["gpt-4o"],
            tier: ModelTier::Weak,
            family: ModelFamily::Gpt,
            default: false,
            pricing: ModelPricing::ZERO,
            max_output_tokens: 4_096,
            context_window: 128_000,
        },
        ModelEntry {
            prefixes: &["gpt-5-mini"],
            tier: ModelTier::Weak,
            family: ModelFamily::Gpt,
            default: true,
            pricing: ModelPricing::ZERO,
            max_output_tokens: 64_000,
            context_window: 264_000,
        },
        ModelEntry {
            prefixes: &["gpt-5.4-mini"],
            tier: ModelTier::Weak,
            family: ModelFamily::Gpt,
            default: false,
            pricing: ModelPricing::ZERO,
            max_output_tokens: 128_000,
            context_window: 400_000,
        },
        // Medium tier
        ModelEntry {
            prefixes: &["claude-sonnet-4"],
            tier: ModelTier::Medium,
            family: ModelFamily::Claude,
            default: false,
            pricing: ModelPricing::ZERO,
            max_output_tokens: 16_000,
            context_window: 216_000,
        },
        ModelEntry {
            prefixes: &["claude-sonnet-4.5"],
            tier: ModelTier::Medium,
            family: ModelFamily::Claude,
            default: false,
            pricing: ModelPricing::ZERO,
            max_output_tokens: 32_000,
            context_window: 144_000,
        },
        ModelEntry {
            prefixes: &["claude-sonnet-4.6"],
            tier: ModelTier::Medium,
            family: ModelFamily::Claude,
            default: true,
            pricing: ModelPricing::ZERO,
            max_output_tokens: 32_000,
            context_window: 200_000,
        },
        ModelEntry {
            prefixes: &["gemini-2.5-pro"],
            tier: ModelTier::Medium,
            family: ModelFamily::Generic,
            default: false,
            pricing: ModelPricing::ZERO,
            max_output_tokens: 64_000,
            context_window: 128_000,
        },
        ModelEntry {
            prefixes: &["grok-code-fast-1"],
            tier: ModelTier::Medium,
            family: ModelFamily::Generic,
            default: false,
            pricing: ModelPricing::ZERO,
            max_output_tokens: 64_000,
            context_window: 128_000,
        },
        ModelEntry {
            prefixes: &["gpt-5.2-codex"],
            tier: ModelTier::Medium,
            family: ModelFamily::Gpt,
            default: false,
            pricing: ModelPricing::ZERO,
            max_output_tokens: 128_000,
            context_window: 400_000,
        },
        ModelEntry {
            prefixes: &["gpt-5.3-codex"],
            tier: ModelTier::Medium,
            family: ModelFamily::Gpt,
            default: false,
            pricing: ModelPricing::ZERO,
            max_output_tokens: 128_000,
            context_window: 400_000,
        },
        // Strong tier
        ModelEntry {
            prefixes: &["claude-opus-4.5"],
            tier: ModelTier::Strong,
            family: ModelFamily::Claude,
            default: false,
            pricing: ModelPricing::ZERO,
            max_output_tokens: 32_000,
            context_window: 160_000,
        },
        ModelEntry {
            prefixes: &["claude-opus-4.6"],
            tier: ModelTier::Strong,
            family: ModelFamily::Claude,
            default: false,
            pricing: ModelPricing::ZERO,
            max_output_tokens: 64_000,
            context_window: 144_000,
        },
        ModelEntry {
            prefixes: &["claude-opus-4.7"],
            tier: ModelTier::Strong,
            family: ModelFamily::Claude,
            default: false,
            pricing: ModelPricing::ZERO,
            max_output_tokens: 64_000,
            context_window: 144_000,
        },
        ModelEntry {
            prefixes: &["gemini-3.1-pro-preview"],
            tier: ModelTier::Strong,
            family: ModelFamily::Generic,
            default: false,
            pricing: ModelPricing::ZERO,
            max_output_tokens: 64_000,
            context_window: 128_000,
        },
        ModelEntry {
            prefixes: &["gpt-4.1"],
            tier: ModelTier::Strong,
            family: ModelFamily::Gpt,
            default: false,
            pricing: ModelPricing::ZERO,
            max_output_tokens: 16_384,
            context_window: 128_000,
        },
        ModelEntry {
            prefixes: &["gpt-5.2"],
            tier: ModelTier::Strong,
            family: ModelFamily::Gpt,
            default: false,
            pricing: ModelPricing::ZERO,
            max_output_tokens: 64_000,
            context_window: 264_000,
        },
        ModelEntry {
            prefixes: &["gpt-5.4"],
            tier: ModelTier::Strong,
            family: ModelFamily::Gpt,
            default: true,
            pricing: ModelPricing::ZERO,
            max_output_tokens: 128_000,
            context_window: 400_000,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelTier;
    use std::collections::HashSet;

    const AUTHORIZED_MODELS: &[&str] = &[
        "claude-haiku-4.5",
        "gemini-3-flash-preview",
        "gpt-4o",
        "gpt-4.1",
        "gpt-5-mini",
        "gpt-5.4-mini",
        "claude-sonnet-4",
        "claude-sonnet-4.5",
        "claude-sonnet-4.6",
        "gemini-2.5-pro",
        "grok-code-fast-1",
        "gpt-5.2",
        "gpt-5.2-codex",
        "gpt-5.3-codex",
        "claude-opus-4.5",
        "claude-opus-4.6",
        "claude-opus-4.7",
        "gemini-3.1-pro-preview",
        "gpt-5.4",
    ];

    #[test]
    fn model_set_matches_authorized() {
        let actual: HashSet<String> = models()
            .iter()
            .flat_map(|e| e.prefixes.iter().map(|p| p.to_string()))
            .collect();
        let expected: HashSet<String> = AUTHORIZED_MODELS.iter().map(|s| s.to_string()).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn tier_defaults_exist() {
        assert!(
            models()
                .iter()
                .any(|e| e.tier == ModelTier::Weak && e.default)
        );
        assert!(
            models()
                .iter()
                .any(|e| e.tier == ModelTier::Medium && e.default)
        );
        assert!(
            models()
                .iter()
                .any(|e| e.tier == ModelTier::Strong && e.default)
        );
    }

    #[test]
    fn gpt_5_mini_is_weak() {
        let entry = models()
            .iter()
            .find(|e| e.prefixes.contains(&"gpt-5-mini"))
            .unwrap();
        assert_eq!(entry.tier, ModelTier::Weak);
    }

    #[test]
    fn claude_opus_is_strong() {
        for entry in models()
            .iter()
            .filter(|e| e.prefixes.iter().any(|p| p.starts_with("claude-opus")))
        {
            assert_eq!(entry.tier, ModelTier::Strong);
        }
    }

    #[test]
    fn specific_prefixes_resolve_correctly() {
        use crate::model::lookup_entry;
        let entries = models();
        assert!(
            lookup_entry(entries, "gpt-5.2-codex")
                .unwrap()
                .prefixes
                .contains(&"gpt-5.2-codex")
        );
        assert!(
            lookup_entry(entries, "gpt-5.4-mini")
                .unwrap()
                .prefixes
                .contains(&"gpt-5.4-mini")
        );
        assert!(
            lookup_entry(entries, "gpt-5.4")
                .unwrap()
                .prefixes
                .contains(&"gpt-5.4")
        );
    }
}
