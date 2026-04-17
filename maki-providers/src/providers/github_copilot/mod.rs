//! GitHub Copilot API provider.
//! Uses the GitHub Copilot Chat API with OpenAI-compatible models.

pub(crate) mod anthropic;
pub mod auth;
mod models;
pub(crate) mod openai;
pub mod platform;

pub(crate) use models::{endpoint_path_for_model, models};

use flume::Sender;
use maki_storage::DataDir;
use serde_json::Value;

use crate::model::Model;
use crate::provider::{BoxFuture, Provider};
use crate::providers::ResolvedAuth;
use crate::providers::anthropic::Anthropic;
use crate::providers::github_copilot::anthropic::MessagesThinkingMode;
use crate::providers::openai::OpenAi;
use crate::providers::openai_compat::OpenAiCompatProvider;
use crate::{AgentError, Message, ProviderEvent, StreamResponse, ThinkingConfig, Timeouts};

use platform::GitHubCopilotPlatform;

#[derive(Debug)]
struct StreamMetadata {
    endpoint: platform::EndpointPath,
    body: Value,
    anthropic_mode: Option<MessagesThinkingMode>,
}

static CONFIG: crate::providers::openai_compat::OpenAiCompatConfig =
    crate::providers::openai_compat::OpenAiCompatConfig {
        api_key_env: "",
        base_url: auth::FALLBACK_COPILOT_BASE_URL,
        max_tokens_field: "max_completion_tokens",
        include_stream_usage: false,
        provider_name: "GitHub Copilot",
    };

pub struct GitHubCopilot {
    platform: GitHubCopilotPlatform,
    compat: OpenAiCompatProvider,
    anthropic: Anthropic,
    openai: OpenAi,
}

impl GitHubCopilot {
    pub fn new() -> Result<Self, AgentError> {
        let storage = DataDir::resolve()?;
        let platform = GitHubCopilotPlatform::new(&storage)?;
        let compat = OpenAiCompatProvider::new(&CONFIG, Timeouts::default());
        let anthropic = Anthropic::with_auth(platform.share_auth_for_child(), Timeouts::default());
        let openai = OpenAi::with_auth(platform.share_auth_for_child(), Timeouts::default());
        Ok(Self {
            platform,
            compat,
            anthropic,
            openai,
        })
    }

    #[cfg(test)]
    pub fn from_parts(platform: GitHubCopilotPlatform, compat: OpenAiCompatProvider) -> Self {
        let anthropic = Anthropic::with_auth(platform.share_auth_for_child(), Timeouts::default());
        let openai = OpenAi::with_auth(platform.share_auth_for_child(), Timeouts::default());
        Self {
            platform,
            compat,
            anthropic,
            openai,
        }
    }

    fn build_stream_metadata(
        &self,
        model: &Model,
        messages: &[Message],
        system: &str,
        tools: &Value,
        thinking: ThinkingConfig,
        session_id: Option<&str>,
    ) -> StreamMetadata {
        use crate::providers::github_copilot::platform::EndpointPath;

        let endpoint = self.platform.select_endpoint_path(&model.id);
        let (body, anthropic_mode) = match endpoint {
            EndpointPath::Responses => {
                let mut body =
                    openai::build_responses_body(model, messages, system, tools, false, thinking);
                openai::apply_responses_extensions(&mut body, session_id);
                (body, None)
            }
            EndpointPath::V1Messages => {
                let (body, mode) = anthropic::build_anthropic_messages_body(
                    model, messages, system, tools, thinking,
                );
                (body, Some(mode))
            }
            EndpointPath::ChatCompletions => {
                let body = self.compat.build_body(model, messages, system, tools);
                (body, None)
            }
        };

        StreamMetadata {
            endpoint,
            body,
            anthropic_mode,
        }
    }

    fn current_auth(&self) -> ResolvedAuth {
        let mut auth = self.platform.current_auth();
        auth.base_url.get_or_insert_with(|| {
            crate::providers::github_copilot::auth::FALLBACK_COPILOT_BASE_URL.into()
        });
        auth
    }

    /// Use one auth snapshot per request to avoid auth/header races.
    fn current_auth_with_headers(&self, messages: &[Message]) -> ResolvedAuth {
        let mut auth = self.current_auth();
        auth.headers = GitHubCopilotPlatform::build_headers_from_auth(&auth, messages);
        auth
    }
}

impl Provider for GitHubCopilot {
    fn stream_message<'a>(
        &'a self,
        model: &'a Model,
        messages: &'a [Message],
        system: &'a str,
        tools: &'a Value,
        event_tx: &'a Sender<ProviderEvent>,
        thinking: ThinkingConfig,
        session_id: Option<&'a str>,
    ) -> BoxFuture<'a, Result<StreamResponse, AgentError>> {
        Box::pin(async move {
            use crate::providers::github_copilot::platform::EndpointPath;

            let meta =
                self.build_stream_metadata(model, messages, system, tools, thinking, session_id);

            match meta.endpoint {
                EndpointPath::Responses => {
                    self.platform
                        .with_auth_retry(|| async {
                            let auth = self.current_auth_with_headers(messages);
                            self.openai
                                .do_responses_with_parse(
                                    model,
                                    &meta.body,
                                    event_tx,
                                    &auth,
                                    openai::parse_usage,
                                )
                                .await
                        })
                        .await
                }
                EndpointPath::V1Messages => {
                    let Some(mode) = meta.anthropic_mode else {
                        return Err(crate::AgentError::Config {
                            message: "V1Messages endpoint requires thinking mode configuration"
                                .into(),
                        });
                    };
                    self.platform
                        .with_auth_retry(|| async {
                            let mut auth = self.current_auth_with_headers(messages);
                            anthropic::adjust_anthropic_beta_header(
                                &mut auth.headers,
                                mode,
                                &model.id,
                            );
                            let url = auth
                                .base_url
                                .as_ref()
                                .map(|base| format!("{}/v1/messages", base.trim_end_matches('/')));
                            self.anthropic
                                .do_stream_request_with_url(
                                    &meta.body,
                                    event_tx,
                                    url.as_deref(),
                                    &auth,
                                )
                                .await
                        })
                        .await
                }
                EndpointPath::ChatCompletions => {
                    self.platform
                        .with_auth_retry(|| async {
                            let auth = self.current_auth_with_headers(messages);
                            self.compat
                                .do_stream_with_path(
                                    model,
                                    &[] as &[(String, String)],
                                    &meta.body,
                                    event_tx,
                                    &auth,
                                    meta.endpoint.as_str(),
                                )
                                .await
                        })
                        .await
                }
            }
        })
    }

    fn list_models(&self) -> BoxFuture<'_, Result<Vec<String>, AgentError>> {
        Box::pin(async {
            Ok(models()
                .iter()
                .flat_map(|e| e.prefixes.iter())
                .map(|p| p.to_string())
                .collect())
        })
    }
}

#[cfg(test)]
pub(crate) mod test_constants {
    pub(crate) const GPT5_MODEL: &str = "gpt-5.4";
    pub(crate) const GPT5_SPEC: &str = "github-copilot/gpt-5.4";
    pub(crate) const CLAUDE_SPEC: &str = "github-copilot/claude-sonnet-4";
    pub(crate) const GPT4_SPEC: &str = "github-copilot/gpt-4o";

    pub(crate) const GEMINI_25_PRO_SPEC: &str = "github-copilot/gemini-2.5-pro";
    pub(crate) const GEMINI_3_PRO_SPEC: &str = "github-copilot/gemini-3-pro-preview";
    pub(crate) const GROK_CODE_FAST_SPEC: &str = "github-copilot/grok-code-fast-1";

    pub(crate) const FALLBACK_CHAT_MODEL_SPECS: &[&str] =
        &[GEMINI_25_PRO_SPEC, GEMINI_3_PRO_SPEC, GROK_CODE_FAST_SPEC];
}

#[cfg(test)]
mod wrapper_tests {
    use std::sync::{Arc, Mutex};

    use crate::Message;
    use crate::providers::ResolvedAuth;
    use crate::providers::github_copilot::auth::FALLBACK_COPILOT_BASE_URL;
    use crate::providers::github_copilot::platform::GitHubCopilotPlatform;
    use crate::providers::openai_compat::OpenAiCompatProvider;

    use super::test_constants::{CLAUDE_SPEC, FALLBACK_CHAT_MODEL_SPECS, GPT4_SPEC, GPT5_SPEC};
    use super::*;

    pub(crate) const TEST_TOKEN: &str = "test_copilot_token_abc123";
    pub(crate) const TEST_BASE_URL: &str = "https://proxy.individual.githubcopilot.com";

    pub(crate) fn make_auth(
        base_url: Option<&str>,
        auth_header: Option<&str>,
    ) -> Arc<Mutex<ResolvedAuth>> {
        let headers = auth_header.map(|h| ("authorization".into(), format!("Bearer {h}")));
        Arc::new(Mutex::new(ResolvedAuth {
            base_url: base_url.map(String::from),
            headers: headers.into_iter().collect(),
        }))
    }

    pub(crate) fn make_copilot(auth: Arc<Mutex<ResolvedAuth>>) -> GitHubCopilot {
        let platform = GitHubCopilotPlatform::with_auth(auth);
        let compat = OpenAiCompatProvider::new(&CONFIG, Timeouts::default());
        GitHubCopilot::from_parts(platform, compat)
    }

    pub(crate) fn user_message() -> Vec<Message> {
        vec![Message::user("hello".into())]
    }

    #[test]
    fn current_auth_falls_back_to_default_base_url() {
        let auth = make_auth(None, Some(TEST_TOKEN));
        let copilot = make_copilot(auth);
        let resolved = copilot.current_auth();
        assert_eq!(resolved.base_url, Some(FALLBACK_COPILOT_BASE_URL.into()));
    }

    #[test]
    fn auth_headers_include_required_copilot_headers() {
        let auth = make_auth(Some(TEST_BASE_URL), Some(TEST_TOKEN));
        let copilot = make_copilot(auth);
        let resolved = copilot.current_auth_with_headers(&user_message());

        let has_authorization = resolved
            .headers
            .iter()
            .any(|(k, _)| k.to_lowercase() == "authorization");
        assert!(has_authorization, "authorization header should be present");

        let has_editor_version = resolved
            .headers
            .iter()
            .any(|(k, _)| k.to_lowercase() == "editor-version");
        assert!(
            has_editor_version,
            "editor-version header should be present"
        );

        let has_copilot_integration_id = resolved
            .headers
            .iter()
            .any(|(k, _)| k.to_lowercase() == "copilot-integration-id");
        assert!(
            has_copilot_integration_id,
            "copilot-integration-id header should be present"
        );
    }

    #[test]
    fn gpt5_routes_to_responses() {
        let auth = make_auth(Some(TEST_BASE_URL), Some(TEST_TOKEN));
        let copilot = make_copilot(auth);
        let model = Model::from_spec(GPT5_SPEC).unwrap();

        let meta = copilot.build_stream_metadata(
            &model,
            &user_message(),
            "system",
            &serde_json::json!([]),
            ThinkingConfig::Off,
            None,
        );
        let (path, body) = (meta.endpoint, meta.body);

        assert_eq!(path.as_str(), "/responses");
        assert!(body.get("input").is_some());
        assert_eq!(body["model"], super::test_constants::GPT5_MODEL);
    }

    #[test]
    fn claude_routes_to_v1_messages() {
        let auth = make_auth(Some(TEST_BASE_URL), Some(TEST_TOKEN));
        let copilot = make_copilot(auth);
        let model = Model::from_spec(CLAUDE_SPEC).unwrap();

        let meta = copilot.build_stream_metadata(
            &model,
            &user_message(),
            "system",
            &serde_json::json!([]),
            ThinkingConfig::Off,
            None,
        );
        let (path, body) = (meta.endpoint, meta.body);

        assert_eq!(path.as_str(), "/v1/messages");
        assert!(body.get("messages").is_some());
        assert!(body.get("system").is_some());
    }

    #[test]
    fn gpt4_routes_to_chat_completions() {
        let auth = make_auth(Some(TEST_BASE_URL), Some(TEST_TOKEN));
        let copilot = make_copilot(auth);
        let model = Model::from_spec(GPT4_SPEC).unwrap();

        let meta = copilot.build_stream_metadata(
            &model,
            &user_message(),
            "system",
            &serde_json::json!([]),
            ThinkingConfig::Off,
            None,
        );
        let (path, body) = (meta.endpoint, meta.body);

        assert_eq!(path.as_str(), "/chat/completions");
        assert!(body.get("messages").is_some());
        assert!(body.get("system").is_none());
    }

    #[test]
    fn fallback_models_route_to_chat_completions() {
        let auth = make_auth(Some(TEST_BASE_URL), Some(TEST_TOKEN));
        let copilot = make_copilot(auth);

        for spec in FALLBACK_CHAT_MODEL_SPECS {
            let model = Model::from_spec(spec).unwrap();
            let meta = copilot.build_stream_metadata(
                &model,
                &user_message(),
                "system",
                &serde_json::json!([]),
                ThinkingConfig::Off,
                None,
            );
            let (path, body) = (meta.endpoint, meta.body);
            assert_eq!(path.as_str(), "/chat/completions", "{spec}");
            assert!(body.get("messages").is_some());
        }
    }

    #[test]
    fn responses_cache_key_behavior() {
        let auth = make_auth(Some(TEST_BASE_URL), Some(TEST_TOKEN));
        let copilot = make_copilot(auth);
        let model = Model::from_spec(GPT5_SPEC).unwrap();

        let cases = [(Some("session-abc"), true), (None, false)];

        for (session_id, expect_key) in cases {
            let meta = copilot.build_stream_metadata(
                &model,
                &user_message(),
                "system",
                &serde_json::json!([]),
                ThinkingConfig::Off,
                session_id,
            );
            let body = meta.body;

            if expect_key {
                let key = body["prompt_cache_key"].as_str().unwrap();
                assert!(key.starts_with("maki-"));
            } else {
                assert!(body.get("prompt_cache_key").is_none());
            }
        }
    }

    #[test]
    fn thinking_config_for_responses_path() {
        let auth = make_auth(Some(TEST_BASE_URL), Some(TEST_TOKEN));
        let copilot = make_copilot(auth);
        let model = Model::from_spec(GPT5_SPEC).unwrap();

        let cases = [
            (ThinkingConfig::Adaptive, true),
            (ThinkingConfig::Off, false),
        ];

        for (config, expect_reasoning) in cases {
            let meta = copilot.build_stream_metadata(
                &model,
                &user_message(),
                "system",
                &serde_json::json!([]),
                config,
                None,
            );
            let body = meta.body;

            if expect_reasoning {
                assert_eq!(body["reasoning"]["effort"], "high");
            } else {
                assert!(body.get("reasoning").is_none());
            }
        }
    }

    #[test]
    fn thinking_budget_uses_manual_mode() {
        let auth = make_auth(Some(TEST_BASE_URL), Some(TEST_TOKEN));
        let copilot = make_copilot(auth);
        let model = Model::from_spec(CLAUDE_SPEC).unwrap();

        let meta = copilot.build_stream_metadata(
            &model,
            &user_message(),
            "system",
            &serde_json::json!([]),
            ThinkingConfig::Budget(8192),
            None,
        );
        let body = meta.body;

        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["budget_tokens"], 8192);
    }

    #[test]
    fn v1_messages_uses_auth_with_headers() {
        let auth = make_auth(Some(TEST_BASE_URL), Some(TEST_TOKEN));
        let copilot = make_copilot(auth);

        let resolved = copilot.current_auth_with_headers(&user_message());

        let has_editor_version = resolved
            .headers
            .iter()
            .any(|(k, _)| k.to_lowercase() == "editor-version");
        let has_copilot_integration_id = resolved
            .headers
            .iter()
            .any(|(k, _)| k.to_lowercase() == "copilot-integration-id");
        let has_x_initiator = resolved
            .headers
            .iter()
            .any(|(k, _)| k.to_lowercase() == "x-initiator");

        assert!(
            has_editor_version,
            "Editor-Version header should be present"
        );
        assert!(
            has_copilot_integration_id,
            "Copilot-Integration-Id header should be present"
        );
        assert!(has_x_initiator, "X-Initiator header should be present");
    }
}
