use std::sync::{Arc, Mutex};

use maki_storage::DataDir;

use crate::providers::ResolvedAuth;
use crate::{ContentBlock, Message, Role};

const RESPONSES_PATH: &str = "/responses";
const V1_MESSAGES_PATH: &str = "/v1/messages";

const HEADER_INITIATOR: &str = "X-Initiator";
const HEADER_EDITOR_VERSION: &str = "Editor-Version";
const HEADER_EDITOR_PLUGIN_VERSION: &str = "Editor-Plugin-Version";
const HEADER_COPILOT_INTEGRATION_ID: &str = "Copilot-Integration-Id";
const HEADER_USER_AGENT: &str = "User-Agent";
const HEADER_OPENAI_INTENT: &str = "Openai-Intent";
const HEADER_COPILOT_VISION: &str = "Copilot-Vision-Request";
const HEADER_INTERACTION_TYPE: &str = "X-Interaction-Type";
const HEADER_AUTHORIZATION: &str = "authorization";
const HEADER_CONTENT_TYPE: &str = "content-type";

// VS Code-style headers required by the Copilot proxy.
pub const EDITOR_VERSION: &str = "vscode/1.96.2";
pub const EDITOR_PLUGIN_VERSION: &str = "copilot-chat/0.23.2";
pub const COPILOT_INTEGRATION_ID: &str = "vscode-chat";
const USER_AGENT: &str = "GitHubCopilotChat/0.23.2";
const INTERACTION_TYPE: &str = "chat";

const CONVERSATION_EDITS_INTENT: &str = "conversation-edits";
const CHAT_COMPLETIONS_PATH: &str = "/chat/completions";
const INITIATOR_USER: &str = "user";
const INITIATOR_AGENT: &str = "agent";

#[derive(Clone, Debug)]
pub enum EndpointPath {
    Responses,
    V1Messages,
    ChatCompletions,
}

impl EndpointPath {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Responses => RESPONSES_PATH,
            Self::V1Messages => V1_MESSAGES_PATH,
            Self::ChatCompletions => CHAT_COMPLETIONS_PATH,
        }
    }
}

/// Shared GitHub Copilot transport and auth state.
pub struct GitHubCopilotPlatform {
    auth: Arc<Mutex<ResolvedAuth>>,
    storage: Option<DataDir>,
    /// Test-only refresh hook.
    #[cfg(test)]
    test_refresh: Option<Box<dyn Fn() -> Result<(), crate::AgentError> + Send + Sync>>,
}

impl GitHubCopilotPlatform {
    /// Create a platform backed by stored OAuth state.
    pub fn new(dir: &DataDir) -> Result<Self, crate::AgentError> {
        let auth = super::auth::resolve(dir)?;
        Ok(Self {
            auth: Arc::new(Mutex::new(auth)),
            storage: Some(dir.clone()),
            #[cfg(test)]
            test_refresh: None,
        })
    }

    /// Create a platform with fixed auth for tests.
    #[cfg(test)]
    pub fn with_auth(auth: Arc<Mutex<ResolvedAuth>>) -> Self {
        Self {
            auth,
            storage: None,
            test_refresh: None,
        }
    }

    /// Create a platform with a custom refresh hook for tests.
    #[cfg(test)]
    pub fn with_auth_and_refresh<F>(auth: Arc<Mutex<ResolvedAuth>>, refresh: F) -> Self
    where
        F: Fn() -> Result<(), crate::AgentError> + Send + Sync + 'static,
    {
        Self {
            auth,
            storage: None,
            test_refresh: Some(Box::new(refresh)),
        }
    }

    /// Return whether auth headers are present.
    #[cfg(test)]
    pub fn has_auth(&self) -> bool {
        self.auth
            .lock()
            .map(|auth| !auth.headers.is_empty())
            .unwrap_or(false)
    }

    /// Return whether auth is storage-backed.
    #[cfg(test)]
    pub fn is_oauth(&self) -> bool {
        self.storage.is_some()
    }

    /// Return the current auth state, even from a poisoned lock.
    ///
    /// Keep stored credentials when a lock is poisoned. The caller still fills
    /// in the fallback base URL if needed.
    fn recover_auth(&self) -> ResolvedAuth {
        match self.auth.lock() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    /// Return the current auth state.
    pub fn current_auth(&self) -> ResolvedAuth {
        self.recover_auth()
    }

    /// Share auth state with child providers.
    pub(crate) fn share_auth_for_child(&self) -> Arc<Mutex<ResolvedAuth>> {
        self.auth.clone()
    }

    /// Build request headers from one auth snapshot.
    pub fn build_headers_from_auth(
        auth: &ResolvedAuth,
        messages: &[Message],
    ) -> Vec<(String, String)> {
        let auth_headers = &auth.headers;

        let mut headers = Vec::new();

        headers.push((HEADER_CONTENT_TYPE.into(), "application/json".into()));

        if let Some((_, auth_value)) = auth_headers.iter().find(|(k, _)| k == HEADER_AUTHORIZATION)
        {
            headers.push((HEADER_AUTHORIZATION.into(), auth_value.clone()));
        }

        headers.push((HEADER_USER_AGENT.into(), USER_AGENT.into()));
        headers.push((HEADER_EDITOR_VERSION.into(), EDITOR_VERSION.into()));
        headers.push((
            HEADER_EDITOR_PLUGIN_VERSION.into(),
            EDITOR_PLUGIN_VERSION.into(),
        ));
        headers.push((
            HEADER_COPILOT_INTEGRATION_ID.into(),
            COPILOT_INTEGRATION_ID.into(),
        ));
        headers.push((HEADER_INITIATOR.into(), infer_initiator(messages).into()));
        headers.push((
            HEADER_OPENAI_INTENT.into(),
            CONVERSATION_EDITS_INTENT.into(),
        ));

        headers.push((HEADER_INTERACTION_TYPE.into(), INTERACTION_TYPE.into()));

        if has_image_content(messages) {
            headers.push((HEADER_COPILOT_VISION.into(), "true".into()));
        }

        super::auth::sanitize_headers(&mut headers);
        headers
    }

    /// Choose the API path for a model.
    pub fn select_endpoint_path(&self, model_id: &str) -> EndpointPath {
        let entries = super::models();
        let family = crate::model::lookup_entry(entries, model_id)
            .map(|e| e.family)
            .unwrap_or(crate::model::ModelFamily::Generic);
        super::endpoint_path_for_model(model_id, family)
    }

    /// Retry once after refreshing auth on 401.
    pub async fn with_auth_retry<T, F, Fut>(&self, f: F) -> Result<T, crate::AgentError>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, crate::AgentError>>,
    {
        match f().await {
            Err(crate::AgentError::Api { status: 401, .. }) => {
                self.refresh_auth().await?;
                f().await
            }
            result => result,
        }
    }

    async fn refresh_auth(&self) -> Result<(), crate::AgentError> {
        #[cfg(test)]
        if let Some(ref test_refresh) = self.test_refresh {
            return test_refresh();
        }

        let storage = self
            .storage
            .clone()
            .ok_or_else(|| crate::AgentError::Config {
                message: "cannot refresh auth without storage".into(),
            })?;

        let auth_arc = self.auth.clone();

        smol::unblock(move || {
            let tokens = maki_storage::auth::load_tokens(&storage, super::auth::PROVIDER)
                .ok_or_else(|| crate::AgentError::Config {
                    message: "no tokens found for refresh".into(),
                })?;

            let fresh = super::auth::refresh_tokens(&tokens)?;
            maki_storage::auth::save_tokens(&storage, super::auth::PROVIDER, &fresh)?;

            let resolved = super::auth::resolve(&storage)?;
            match auth_arc.lock() {
                Ok(mut guard) => *guard = resolved,
                Err(poisoned) => *poisoned.into_inner() = resolved,
            }

            Ok::<(), crate::AgentError>(())
        })
        .await
    }
}

fn has_image_content(messages: &[Message]) -> bool {
    messages.iter().any(|m| {
        m.content
            .iter()
            .any(|b| matches!(b, ContentBlock::Image { .. }))
    })
}

fn infer_initiator(messages: &[Message]) -> &'static str {
    match messages.last() {
        Some(msg) if matches!(msg.role, Role::User) => INITIATOR_USER,
        _ => INITIATOR_AGENT,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use maki_storage::DataDir;

    use crate::providers::ResolvedAuth;
    use crate::providers::github_copilot::auth;
    use crate::providers::github_copilot::platform::GitHubCopilotPlatform;

    use crate::{AgentError, Message, Role};

    use super::{
        CHAT_COMPLETIONS_PATH, HEADER_AUTHORIZATION, HEADER_COPILOT_VISION, HEADER_INITIATOR,
        INITIATOR_AGENT, INITIATOR_USER, RESPONSES_PATH, V1_MESSAGES_PATH,
    };

    const TEST_COPILOT_TOKEN: &str = "copilot_test_token_abc123";
    const TEST_BASE_URL: &str = auth::FALLBACK_COPILOT_BASE_URL;

    fn sample_resolved_auth() -> ResolvedAuth {
        ResolvedAuth {
            base_url: Some(TEST_BASE_URL.into()),
            headers: vec![(
                HEADER_AUTHORIZATION.into(),
                format!("Bearer {TEST_COPILOT_TOKEN}"),
            )],
        }
    }

    fn user_messages() -> Vec<Message> {
        vec![Message::user("hello".into())]
    }

    fn agent_messages() -> Vec<Message> {
        vec![
            Message::user("hello".into()),
            Message {
                role: Role::Assistant,
                content: vec![],
                ..Default::default()
            },
        ]
    }

    #[test]
    fn initiator_user_when_last_message_is_user() {
        let auth = sample_resolved_auth();
        let headers = GitHubCopilotPlatform::build_headers_from_auth(&auth, &user_messages());
        let initiator = headers
            .iter()
            .find(|(k, _)| k == HEADER_INITIATOR)
            .map(|(_, v): &(_, _)| v.as_str());
        assert_eq!(initiator, Some(INITIATOR_USER));
    }

    #[test]
    fn initiator_agent_when_last_message_is_assistant() {
        let auth = sample_resolved_auth();
        let headers = GitHubCopilotPlatform::build_headers_from_auth(&auth, &agent_messages());
        let initiator = headers
            .iter()
            .find(|(k, _)| k == HEADER_INITIATOR)
            .map(|(_, v): &(_, _)| v.as_str());
        assert_eq!(initiator, Some(INITIATOR_AGENT));
    }

    #[test]
    fn auth_retry_triggers_on_401() {
        let auth = sample_resolved_auth();
        let platform = GitHubCopilotPlatform::with_auth(Arc::new(Mutex::new(auth)));
        let result: Result<String, AgentError> = smol::block_on(async {
            platform
                .with_auth_retry(|| async {
                    Err(AgentError::Api {
                        status: 401,
                        message: "Unauthorized".into(),
                    })
                })
                .await
        });
        assert!(result.is_err());
    }

    #[test]
    fn auth_retry_rereads_fresh_auth_after_401() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let auth = Arc::new(Mutex::new(ResolvedAuth {
            base_url: Some(TEST_BASE_URL.into()),
            headers: vec![("authorization".into(), "Bearer token_v1_initial".into())],
        }));

        let calls = Arc::new(Mutex::new(Vec::<String>::new()));
        let call_count = Arc::new(AtomicUsize::new(0));

        let auth_for_refresh = auth.clone();
        let platform = GitHubCopilotPlatform::with_auth_and_refresh(auth.clone(), move || {
            auth_for_refresh.lock().unwrap().headers =
                vec![("authorization".into(), "Bearer token_v2_refreshed".into())];
            Ok(())
        });

        let calls_for_closure = calls.clone();
        let count_for_closure = call_count.clone();

        let result: Result<String, AgentError> = smol::block_on(async {
            platform
                .with_auth_retry(|| {
                    let captured = calls_for_closure.clone();
                    let count = count_for_closure.clone();
                    let auth_ref = auth.clone();

                    async move {
                        let n = count.fetch_add(1, Ordering::SeqCst);

                        let header = {
                            let auth = auth_ref.lock().unwrap();
                            auth.headers
                                .iter()
                                .find(|(k, _)| k.eq_ignore_ascii_case("authorization"))
                                .map(|(_, v)| v.clone())
                                .unwrap_or_default()
                        };

                        captured.lock().unwrap().push(header);

                        if n == 0 {
                            Err(AgentError::Api {
                                status: 401,
                                message: "Unauthorized".into(),
                            })
                        } else {
                            Ok("success".into())
                        }
                    }
                })
                .await
        });

        assert_eq!(call_count.load(Ordering::SeqCst), 2);
        assert!(result.is_ok());

        let seen = calls.lock().unwrap();
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[0], "Bearer token_v1_initial");
        assert_eq!(seen[1], "Bearer token_v2_refreshed");
    }

    #[test]
    fn auth_retry_skips_on_non_401() {
        let auth = sample_resolved_auth();
        let platform = GitHubCopilotPlatform::with_auth(Arc::new(Mutex::new(auth)));
        let result: Result<String, AgentError> = smol::block_on(async {
            platform
                .with_auth_retry(|| async {
                    Err(AgentError::Api {
                        status: 400,
                        message: "Bad Request".into(),
                    })
                })
                .await
        });
        assert!(result.unwrap_err().user_message().contains("Bad Request"));
    }

    #[test]
    fn oauth_detection() {
        let auth = sample_resolved_auth();
        let platform_no_storage =
            GitHubCopilotPlatform::with_auth(Arc::new(Mutex::new(auth.clone())));
        assert!(!platform_no_storage.is_oauth());

        let platform_with_storage = GitHubCopilotPlatform {
            auth: Arc::new(Mutex::new(auth)),
            storage: Some(DataDir::resolve().unwrap()),
            test_refresh: None,
        };
        assert!(platform_with_storage.is_oauth());
    }

    #[test]
    fn has_auth_checks() {
        let auth = sample_resolved_auth();
        let platform = GitHubCopilotPlatform::with_auth(Arc::new(Mutex::new(auth)));
        assert!(platform.has_auth());

        let no_auth = ResolvedAuth {
            base_url: Some(TEST_BASE_URL.into()),
            headers: vec![],
        };
        let platform_no_auth = GitHubCopilotPlatform::with_auth(Arc::new(Mutex::new(no_auth)));
        assert!(!platform_no_auth.has_auth());
    }

    #[test]
    fn headers_contain_auth() {
        let auth = sample_resolved_auth();
        let headers = GitHubCopilotPlatform::build_headers_from_auth(&auth, &user_messages());
        let auth_header = headers
            .iter()
            .find(|(k, _)| k == HEADER_AUTHORIZATION)
            .map(|(_, v): &(_, _)| v.as_str());
        assert_eq!(auth_header.unwrap(), format!("Bearer {TEST_COPILOT_TOKEN}"));
    }

    #[test]
    fn headers_include_vision_for_images() {
        use crate::{ImageMediaType, ImageSource};
        let auth = sample_resolved_auth();
        let source = ImageSource::new(ImageMediaType::Png, std::sync::Arc::from("abc"));
        let headers = GitHubCopilotPlatform::build_headers_from_auth(
            &auth,
            &[Message::user_with_images("hi".into(), vec![source])],
        );
        assert!(
            headers
                .iter()
                .any(|(k, v)| k == HEADER_COPILOT_VISION && v == "true")
        );
    }

    #[test]
    fn headers_without_auth_skips_bearer() {
        let auth = ResolvedAuth {
            base_url: Some(TEST_BASE_URL.into()),
            headers: vec![],
        };
        let headers = GitHubCopilotPlatform::build_headers_from_auth(&auth, &user_messages());
        assert!(!headers.iter().any(|(k, _)| k == HEADER_AUTHORIZATION));
    }

    #[test]
    fn routing_claude_to_v1_messages() {
        use crate::model::ModelFamily;
        use crate::providers::github_copilot::endpoint_path_for_model;

        assert_eq!(
            endpoint_path_for_model("claude-sonnet-4", ModelFamily::Claude).as_str(),
            V1_MESSAGES_PATH
        );
    }

    #[test]
    fn routing_gpt5_to_responses() {
        use crate::model::ModelFamily;
        use crate::providers::github_copilot::endpoint_path_for_model;

        assert_eq!(
            endpoint_path_for_model("gpt-5.4", ModelFamily::Gpt).as_str(),
            RESPONSES_PATH
        );
    }

    #[test]
    fn routing_gpt4_to_chat_completions() {
        use crate::model::ModelFamily;
        use crate::providers::github_copilot::endpoint_path_for_model;

        assert_eq!(
            endpoint_path_for_model("gpt-4o", ModelFamily::Gpt).as_str(),
            CHAT_COMPLETIONS_PATH
        );
    }

    #[test]
    fn routing_generic_to_chat_completions() {
        use crate::model::ModelFamily;
        use crate::providers::github_copilot::endpoint_path_for_model;

        assert_eq!(
            endpoint_path_for_model("gemini-2.5-pro", ModelFamily::Generic).as_str(),
            CHAT_COMPLETIONS_PATH
        );
    }
}
