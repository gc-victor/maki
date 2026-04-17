use std::time::Duration;

use isahc::ReadResponseExt;
use maki_storage::DataDir;
use maki_storage::auth::{OAuthTokens, delete_tokens, load_tokens, save_tokens};
use serde::{Deserialize, Serialize};
use tracing::{debug, error, warn};

use super::platform::{COPILOT_INTEGRATION_ID, EDITOR_PLUGIN_VERSION, EDITOR_VERSION};
use crate::AgentError;
use crate::providers::{ResolvedAuth, urlenc};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const TOKEN_EXCHANGE_TIMEOUT: Duration = Duration::from_secs(30);

pub(crate) const PROVIDER: &str = "github_copilot";
const CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";
const DEVICE_CODE_URL: &str = "https://github.com/login/device/code";
const DEVICE_TOKEN_URL: &str = "https://github.com/login/oauth/access_token";
const COPILOT_TOKEN_URL: &str = "https://api.github.com/copilot_internal/v2/token";
const DEVICE_AUTH_URL: &str = "https://github.com/login/device";
const POLL_SAFETY_MARGIN: Duration = Duration::from_secs(3);
const POLL_TIMEOUT: Duration = Duration::from_secs(600);
const COPILOT_EXPIRY_BUFFER_MS: u64 = 300_000;
pub const FALLBACK_COPILOT_BASE_URL: &str = "https://api.githubcopilot.com";
const GH_COPILOT_TOKEN_ENV: &str = "GH_COPILOT_TOKEN";

#[derive(Debug, Deserialize)]
pub struct DeviceCodeResponse {
    pub device_code: String,
    pub user_code: String,
    pub verification_uri: String,
    pub expires_in: u64,
    pub interval: u64,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TokenPollResponse {
    pub access_token: Option<String>,
    pub token_type: Option<String>,
    pub scope: Option<String>,
    pub error: Option<String>,
    pub error_description: Option<String>,
    pub interval: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct CopilotTokenResponse {
    pub token: String,
    pub expires_at: u64,
    #[serde(rename = "proxy-ep")]
    pub proxy_ep: Option<String>,
}

fn build_device_code_body() -> String {
    format!(
        "client_id={}&scope={}",
        urlenc(CLIENT_ID),
        urlenc("read:user,copilot")
    )
}

fn http_client(timeout: Duration) -> Result<isahc::HttpClient, AgentError> {
    use isahc::config::Configurable;
    isahc::HttpClient::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(timeout)
        .build()
        .map_err(|e| AgentError::Config {
            message: format!("http client: {e}"),
        })
}

pub fn request_device_code() -> Result<DeviceCodeResponse, AgentError> {
    let client = http_client(TOKEN_EXCHANGE_TIMEOUT)?;
    let form_body = build_device_code_body();

    let request = isahc::Request::builder()
        .method("POST")
        .uri(DEVICE_CODE_URL)
        .header("content-type", "application/x-www-form-urlencoded")
        .header("accept", "application/json")
        .body(form_body.into_bytes())?;

    let mut resp = client.send(request).map_err(|e| AgentError::Config {
        message: format!("device code request: {e}"),
    })?;

    if resp.status().as_u16() != 200 {
        let body_text = resp.text().unwrap_or_else(|_| "unknown error".into());
        return Err(AgentError::Config {
            message: format!("device code request failed: {body_text}"),
        });
    }

    let body_text = resp.text()?;
    serde_json::from_str(&body_text).map_err(Into::into)
}

fn build_device_token_body(device_code: &str) -> String {
    format!(
        "client_id={}&device_code={}&grant_type={}",
        urlenc(CLIENT_ID),
        urlenc(device_code),
        urlenc("urn:ietf:params:oauth:grant-type:device_code")
    )
}

pub fn poll_device_token(device: &DeviceCodeResponse) -> Result<TokenPollResponse, AgentError> {
    let client = http_client(TOKEN_EXCHANGE_TIMEOUT)?;
    let poll_interval =
        Duration::from_secs(device.interval).max(Duration::from_secs(1)) + POLL_SAFETY_MARGIN;

    smol::block_on(poll_device_token_with(device, poll_interval, |request| {
        client.send(request).map_err(|e| AgentError::Config {
            message: format!("device token poll: {e}"),
        })
    }))
}

async fn poll_device_token_with<FSend>(
    device: &DeviceCodeResponse,
    poll_interval: Duration,
    mut send: FSend,
) -> Result<TokenPollResponse, AgentError>
where
    FSend: FnMut(isahc::Request<Vec<u8>>) -> Result<isahc::Response<isahc::Body>, AgentError>,
{
    let deadline = std::time::Instant::now() + POLL_TIMEOUT;
    let form_body = build_device_token_body(&device.device_code);
    let mut current_interval = poll_interval;

    loop {
        if std::time::Instant::now() > deadline {
            return Err(AgentError::Config {
                message: "device authorization timed out".into(),
            });
        }

        smol::Timer::after(current_interval).await;

        let request = isahc::Request::builder()
            .method("POST")
            .uri(DEVICE_TOKEN_URL)
            .header("content-type", "application/x-www-form-urlencoded")
            .header("accept", "application/json")
            .body(form_body.clone().into_bytes())?;

        let mut resp = send(request)?;
        let body_text = resp.text()?;
        let poll_result: TokenPollResponse = serde_json::from_str(&body_text)?;

        match classify_poll_response(&poll_result) {
            PollStatus::Success => return Ok(poll_result),
            PollStatus::Pending => {
                debug!("authorization pending, continuing poll");
            }
            PollStatus::SlowDown(new_interval) => {
                warn!("polling too fast, slowing down");
                current_interval = new_interval;
            }
            PollStatus::Error(msg) => {
                return Err(AgentError::Config {
                    message: format!("device authorization failed: {msg}"),
                });
            }
        }
    }
}

#[derive(Debug, PartialEq)]
enum PollStatus {
    Success,
    Pending,
    SlowDown(Duration),
    Error(String),
}

fn classify_poll_response(response: &TokenPollResponse) -> PollStatus {
    if response.access_token.is_some() {
        return PollStatus::Success;
    }

    if let Some(error) = &response.error {
        match error.as_str() {
            "authorization_pending" => PollStatus::Pending,
            "slow_down" => {
                let interval = response
                    .interval
                    .map(Duration::from_secs)
                    .unwrap_or_else(|| Duration::from_secs(10));
                PollStatus::SlowDown(interval + POLL_SAFETY_MARGIN)
            }
            _ => PollStatus::Error(format!(
                "{} - {}",
                error,
                response
                    .error_description
                    .as_deref()
                    .unwrap_or("unknown error")
            )),
        }
    } else {
        PollStatus::Error("no access token received".into())
    }
}

pub fn exchange_copilot_token(github_token: &str) -> Result<CopilotTokenResponse, AgentError> {
    let client = http_client(TOKEN_EXCHANGE_TIMEOUT)?;

    let request = isahc::Request::builder()
        .method("GET")
        .uri(COPILOT_TOKEN_URL)
        .header("authorization", format!("token {github_token}"))
        .header("accept", "application/json")
        .header("Editor-Version", EDITOR_VERSION)
        .header("Editor-Plugin-Version", EDITOR_PLUGIN_VERSION)
        .header("Copilot-Integration-Id", COPILOT_INTEGRATION_ID)
        .body(())?;

    let mut resp = client.send(request).map_err(|e| AgentError::Config {
        message: format!("copilot token exchange: {e}"),
    })?;

    if resp.status().as_u16() != 200 {
        let body_text = resp.text().unwrap_or_else(|_| "unknown error".into());
        return Err(AgentError::Config {
            message: format!("copilot token exchange failed: {body_text}"),
        });
    }

    let body_text = resp.text()?;
    serde_json::from_str(&body_text).map_err(Into::into)
}

pub fn refresh_tokens(tokens: &OAuthTokens) -> Result<OAuthTokens, AgentError> {
    let expired = tokens.is_expired();
    debug!(expired, "refreshing GitHub Copilot tokens");

    let github_token = &tokens.refresh;
    let copilot = exchange_copilot_token(github_token)?;
    Ok(into_oauth_tokens(github_token, copilot))
}

pub fn resolve(dir: &DataDir) -> Result<ResolvedAuth, AgentError> {
    if let Ok(env_token) = std::env::var(GH_COPILOT_TOKEN_ENV)
        && !env_token.is_empty()
    {
        debug!("using GitHub Copilot authentication from GH_COPILOT_TOKEN");
        return Ok(build_resolved_auth_from_token(&env_token));
    }

    if let Some(tokens) = load_tokens(dir, PROVIDER) {
        if !tokens.is_expired() {
            debug!("using GitHub Copilot authentication");
            return Ok(build_resolved_auth(&tokens));
        }
        match refresh_tokens(&tokens) {
            Ok(fresh) => {
                save_tokens(dir, PROVIDER, &fresh)?;
                debug!("using GitHub Copilot authentication (refreshed)");
                return Ok(build_resolved_auth(&fresh));
            }
            Err(e) => {
                warn!(error = %e, "GitHub Copilot token refresh failed, clearing stale tokens");
                delete_tokens(dir, PROVIDER).ok();
            }
        }
    }

    Err(AgentError::Config {
        message: "not authenticated, run `maki auth login github-copilot`".into(),
    })
}

pub fn login(dir: &DataDir) -> Result<(), AgentError> {
    let device = request_device_code()?;

    println!("Open this URL in your browser:\n\n  {DEVICE_AUTH_URL}\n");
    println!("Enter code: {}\n", device.user_code);
    println!("Waiting for authorization...");

    let token_resp = poll_device_token(&device).map_err(|e| {
        error!(error = %e, "GitHub device authorization failed");
        e
    })?;

    let github_token = token_resp.access_token.ok_or_else(|| AgentError::Config {
        message: "no access token received from GitHub".into(),
    })?;

    let copilot = exchange_copilot_token(&github_token).map_err(|e| {
        error!(error = %e, "GitHub Copilot token exchange failed");
        e
    })?;

    let tokens = into_oauth_tokens(&github_token, copilot);
    save_tokens(dir, PROVIDER, &tokens)?;
    println!("Authenticated successfully.");
    Ok(())
}

pub fn logout(dir: &DataDir) -> Result<(), AgentError> {
    if delete_tokens(dir, PROVIDER)? {
        println!("Logged out of GitHub Copilot.");
    } else {
        println!("Not currently logged in to GitHub Copilot.");
    }
    Ok(())
}

fn build_resolved_auth(tokens: &OAuthTokens) -> ResolvedAuth {
    let base_url = extract_proxy_ep(&tokens.access)
        .and_then(normalize_proxy_ep)
        .unwrap_or_else(|| FALLBACK_COPILOT_BASE_URL.into());

    ResolvedAuth {
        base_url: Some(base_url),
        headers: vec![("authorization".into(), format!("Bearer {}", tokens.access))],
    }
}

fn build_resolved_auth_from_token(token: &str) -> ResolvedAuth {
    let base_url = extract_proxy_ep(token)
        .and_then(normalize_proxy_ep)
        .unwrap_or_else(|| FALLBACK_COPILOT_BASE_URL.into());

    ResolvedAuth {
        base_url: Some(base_url),
        headers: vec![("authorization".into(), format!("Bearer {token}"))],
    }
}

const DISALLOWED_AUTH_HEADERS: &[&str] = &[
    "cookie",
    "x-auth-token",
    "x-api-key",
    "x-csrf-token",
    "set-cookie",
];

pub fn sanitize_headers(headers: &mut Vec<(String, String)>) {
    headers.retain(|(key, _)| {
        let key_lower = key.to_lowercase();
        !DISALLOWED_AUTH_HEADERS
            .iter()
            .any(|disallowed| key_lower == *disallowed)
    });
}

fn extract_proxy_ep(token: &str) -> Option<String> {
    use base64::Engine;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;

    let parts: Vec<&str> = token.split('.').take(4).collect();
    if parts.len() != 3 || parts.iter().any(|p| p.is_empty()) {
        return None;
    }

    let payload = URL_SAFE_NO_PAD.decode(parts[1]).ok()?;
    let json: serde_json::Value = serde_json::from_slice(&payload).ok()?;

    json.get("proxy-ep")
        .and_then(|v| v.as_str())
        .map(String::from)
}

fn normalize_proxy_ep(proxy_ep: String) -> Option<String> {
    let trimmed = proxy_ep.trim();
    if trimmed.is_empty() {
        return None;
    }

    let url = if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        trimmed.to_string()
    } else {
        format!("https://{}", trimmed)
    };

    let scheme_end = url.find("://")?;
    let scheme = &url[..scheme_end];
    let rest = &url[scheme_end + 3..];

    let host_port = rest.split('/').next()?;
    if host_port.is_empty() {
        return None;
    }

    Some(format!("{scheme}://{host_port}"))
}

fn into_oauth_tokens(github_token: &str, copilot: CopilotTokenResponse) -> OAuthTokens {
    OAuthTokens {
        access: copilot.token,
        refresh: github_token.into(),
        expires: copilot
            .expires_at
            .saturating_mul(1000)
            .saturating_sub(COPILOT_EXPIRY_BUFFER_MS),
        account_id: None,
    }
}

#[cfg(test)]
mod tests {
    use maki_storage::DataDir;
    use maki_storage::auth::{load_tokens, now_millis, save_tokens};
    use test_case::test_case;

    use super::*;

    pub(crate) const TEST_PROVIDER: &str = "github_copilot";

    pub(crate) fn sample_oauth_tokens() -> OAuthTokens {
        OAuthTokens {
            access: "copilot_xyz789".into(),
            refresh: "gho_abc123xyz".into(),
            expires: now_millis() + 3_600_000,
            account_id: None,
        }
    }

    #[test]
    fn device_code_response_parses() {
        let json = r#"{
            "device_code": "device-code-abc",
            "user_code": "ABCD-1234",
            "verification_uri": "https://github.com/login/device",
            "expires_in": 900,
            "interval": 5
        }"#;
        let parsed: DeviceCodeResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.device_code, "device-code-abc");
        assert_eq!(parsed.user_code, "ABCD-1234");
        assert_eq!(parsed.expires_in, 900);
    }

    #[test]
    fn token_poll_success_parses() {
        let json = r#"{
            "access_token": "gho_abc123xyz",
            "token_type": "bearer",
            "scope": "read:user,copilot"
        }"#;
        let parsed: TokenPollResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.access_token.unwrap(), "gho_abc123xyz");
        assert_eq!(parsed.error, None);
    }

    #[test]
    fn token_poll_error_parses() {
        let json = r#"{
            "error": "authorization_pending",
            "error_description": "The authorization request is still pending.",
            "interval": 5
        }"#;
        let parsed: TokenPollResponse = serde_json::from_str(json).unwrap();
        assert!(parsed.access_token.is_none());
        assert_eq!(parsed.error.as_deref(), Some("authorization_pending"));
    }

    #[test]
    fn copilot_token_response_parses() {
        let json = r#"{"token": "copilot_xyz789", "expires_at": 9999999999}"#;
        let parsed: CopilotTokenResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.token, "copilot_xyz789");
        assert_eq!(parsed.expires_at, 9999999999);
    }

    #[test]
    fn resolve_fails_without_auth() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = DataDir::from_path(tmp.path().to_path_buf());
        unsafe {
            std::env::remove_var(GH_COPILOT_TOKEN_ENV);
        }
        assert!(resolve(&dir).is_err());
    }

    #[test_case(vec![("authorization", "Bearer token"), ("content-type", "application/json")], 2)]
    #[test_case(vec![("authorization", "Bearer token"), ("cookie", "session=abc")], 1)]
    fn sanitize_headers_filters_disallowed(headers: Vec<(&str, &str)>, expected: usize) {
        let mut headers: Vec<(String, String)> = headers
            .into_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .collect();
        sanitize_headers(&mut headers);
        assert_eq!(headers.len(), expected);
    }

    #[test]
    fn extract_proxy_ep_from_valid_jwt() {
        use base64::Engine;
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        let payload = r#"{"proxy-ep":"https://proxy.example.com/v1/chat"}"#;
        let encoded = URL_SAFE_NO_PAD.encode(payload);
        let token = format!("header.{}.{}", encoded, "sig");
        assert_eq!(
            extract_proxy_ep(&token),
            Some("https://proxy.example.com/v1/chat".into())
        );
    }

    #[test_case("not.valid", None)]
    #[test_case("invalid", None)]
    #[test_case(".b.c", None)]
    #[test_case("a..c", None)]
    #[test_case("a.b.", None)]
    #[test_case("a.b.c.d", None)]
    fn extract_proxy_ep_error_cases(token: &str, expected: Option<String>) {
        assert_eq!(extract_proxy_ep(token), expected);
    }

    #[test_case("https://proxy.example.com/v1/chat", Some("https://proxy.example.com"))]
    #[test_case("proxy.example.com", Some("https://proxy.example.com"))]
    #[test_case("", None)]
    fn normalize_proxy_ep_cases(input: &str, expected: Option<&str>) {
        assert_eq!(normalize_proxy_ep(input.into()), expected.map(String::from));
    }

    #[test]
    fn into_oauth_tokens_saturates_on_small_expires() {
        let gh_token = "gho_test";
        let copilot = CopilotTokenResponse {
            token: "cop".into(),
            expires_at: 301,
            proxy_ep: None,
        };
        let tokens = into_oauth_tokens(gh_token, copilot);
        assert_eq!(tokens.expires, 1000);
    }

    #[test]
    fn resolved_auth_fallback_for_invalid_token() {
        let tokens = OAuthTokens {
            access: "not_a_valid_jwt".into(),
            refresh: "gho_test".into(),
            expires: now_millis() + 3_600_000,
            account_id: None,
        };
        let auth = build_resolved_auth(&tokens);
        assert_eq!(auth.base_url, Some(FALLBACK_COPILOT_BASE_URL.into()));
    }

    #[test]
    fn resolved_auth_uses_proxy_ep_from_token() {
        use base64::Engine;
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        let payload = r#"{"proxy-ep":"https://proxy.example.com/v1/chat"}"#;
        let encoded = URL_SAFE_NO_PAD.encode(payload);
        let token = format!("header.{}.{}", encoded, "sig");
        let tokens = OAuthTokens {
            access: token,
            refresh: "gho_test".into(),
            expires: now_millis() + 3_600_000,
            account_id: None,
        };
        let auth = build_resolved_auth(&tokens);
        assert_eq!(auth.base_url, Some("https://proxy.example.com".into()));
    }

    #[test]
    fn resolve_prefers_env_token() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = DataDir::from_path(tmp.path().to_path_buf());
        unsafe {
            std::env::set_var(GH_COPILOT_TOKEN_ENV, "env_token");
        }
        let auth = resolve(&dir).unwrap();
        unsafe {
            std::env::remove_var(GH_COPILOT_TOKEN_ENV);
        }
        assert_eq!(auth.base_url, Some(FALLBACK_COPILOT_BASE_URL.into()));
    }

    #[test]
    fn resolve_uses_stored_tokens() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = DataDir::from_path(tmp.path().to_path_buf());
        save_tokens(&dir, TEST_PROVIDER, &sample_oauth_tokens()).unwrap();
        unsafe {
            std::env::remove_var(GH_COPILOT_TOKEN_ENV);
        }
        let auth = resolve(&dir).unwrap();
        assert!(auth.headers.iter().any(|(k, _)| k == "authorization"));
        let header = auth
            .headers
            .iter()
            .find(|(k, _)| k == "authorization")
            .map(|(_, v)| v.as_str());
        assert!(header.unwrap().starts_with("Bearer "));
    }

    #[test]
    fn logout_clears_tokens() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = DataDir::from_path(tmp.path().to_path_buf());
        save_tokens(&dir, TEST_PROVIDER, &sample_oauth_tokens()).unwrap();
        assert!(load_tokens(&dir, TEST_PROVIDER).is_some());
        logout(&dir).unwrap();
        assert!(load_tokens(&dir, TEST_PROVIDER).is_none());
    }

    #[test]
    fn logout_ok_when_not_logged_in() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = DataDir::from_path(tmp.path().to_path_buf());
        assert!(logout(&dir).is_ok());
    }

    #[test]
    fn resolve_clears_stale_tokens_on_refresh_failure() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = DataDir::from_path(tmp.path().to_path_buf());
        let expired = OAuthTokens {
            access: "expired".into(),
            refresh: "invalid".into(),
            expires: now_millis() - 1000,
            account_id: None,
        };
        save_tokens(&dir, TEST_PROVIDER, &expired).unwrap();
        unsafe {
            std::env::remove_var(GH_COPILOT_TOKEN_ENV);
        }
        assert!(resolve(&dir).is_err());
        assert!(load_tokens(&dir, TEST_PROVIDER).is_none());
    }

    #[test]
    fn extract_proxy_ep_handles_invalid_json() {
        use base64::Engine;
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        let encoded = URL_SAFE_NO_PAD.encode("not json");
        let token = format!("header.{}.{}", encoded, "sig");
        assert_eq!(extract_proxy_ep(&token), None);
    }

    #[test]
    fn extract_proxy_ep_handles_missing_proxy_ep() {
        use base64::Engine;
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        let encoded = URL_SAFE_NO_PAD.encode(r#"{"other":"value"}"#);
        let token = format!("header.{}.{}", encoded, "sig");
        assert_eq!(extract_proxy_ep(&token), None);
    }

    #[test]
    fn extract_proxy_ep_handles_null_proxy_ep() {
        use base64::Engine;
        use base64::engine::general_purpose::URL_SAFE_NO_PAD;
        let encoded_payload = URL_SAFE_NO_PAD.encode(r#"{"proxy-ep":null}"#);
        let token = format!("header.{}.{}", encoded_payload, "signature");

        assert_eq!(extract_proxy_ep(&token), None);
    }

    #[test_case("https://", None)]
    #[test_case("http:/bad", Some("https://http:"))]
    fn normalize_proxy_ep_edge_cases(input: &str, expected: Option<&str>) {
        assert_eq!(normalize_proxy_ep(input.into()), expected.map(String::from));
    }

    #[test]
    fn device_code_body_contains_required_params() {
        let body = build_device_code_body();
        assert!(body.contains("client_id="));
        assert!(body.contains("scope="));
    }

    #[test]
    fn device_token_body_url_encodes_params() {
        let body = build_device_token_body("code with spaces");
        assert!(body.contains("device_code=code%20with%20spaces"));
    }

    #[test]
    fn classify_poll_response_success() {
        let response = TokenPollResponse {
            access_token: Some("token123".into()),
            token_type: Some("bearer".into()),
            scope: Some("read".into()),
            error: None,
            error_description: None,
            interval: None,
        };
        assert_eq!(classify_poll_response(&response), PollStatus::Success);
    }

    #[test]
    fn classify_poll_response_pending() {
        let response = TokenPollResponse {
            access_token: None,
            token_type: None,
            scope: None,
            error: Some("authorization_pending".into()),
            error_description: None,
            interval: None,
        };
        assert_eq!(classify_poll_response(&response), PollStatus::Pending);
    }

    #[test_case(Some(5), Duration::from_secs(8))]
    #[test_case(None, Duration::from_secs(13))]
    fn classify_poll_response_slow_down(interval: Option<u64>, expected: Duration) {
        let response = TokenPollResponse {
            access_token: None,
            token_type: None,
            scope: None,
            error: Some("slow_down".into()),
            error_description: None,
            interval,
        };
        assert!(
            matches!(classify_poll_response(&response), PollStatus::SlowDown(d) if d == expected)
        );
    }

    #[test]
    fn classify_poll_response_error_cases() {
        let unknown = TokenPollResponse {
            access_token: None,
            token_type: None,
            scope: None,
            error: Some("expired_token".into()),
            error_description: Some("token expired".into()),
            interval: None,
        };
        assert!(
            matches!(classify_poll_response(&unknown), PollStatus::Error(ref m) if m.contains("expired"))
        );

        let no_token = TokenPollResponse {
            access_token: None,
            token_type: None,
            scope: None,
            error: None,
            error_description: None,
            interval: None,
        };
        assert!(
            matches!(classify_poll_response(&no_token), PollStatus::Error(ref m) if m == "no access token received")
        );
    }

    #[test]
    fn poll_device_token_retries_pending() {
        use std::cell::RefCell;
        let device = DeviceCodeResponse {
            device_code: "d123".into(),
            user_code: "C123".into(),
            verification_uri: "https://github.com/login/device".into(),
            expires_in: 900,
            interval: 1,
        };
        let pending = r#"{"error":"authorization_pending"}"#;
        let success = r#"{"access_token":"gho_abc"}"#;

        let calls = RefCell::new(0);
        let mock_send = |_req: isahc::Request<Vec<u8>>| {
            let c = *calls.borrow();
            *calls.borrow_mut() = c + 1;
            let body = if c == 0 {
                isahc::Body::from(pending)
            } else {
                isahc::Body::from(success)
            };
            Ok::<_, AgentError>(isahc::Response::builder().status(200).body(body).unwrap())
        };

        let result = smol::block_on(poll_device_token_with(
            &device,
            Duration::from_millis(1),
            mock_send,
        ));
        assert!(result.is_ok());
        assert_eq!(result.unwrap().access_token, Some("gho_abc".into()));
    }

    #[test]
    fn poll_device_token_propagates_errors_from_send() {
        use std::cell::RefCell;
        let device = DeviceCodeResponse {
            device_code: "d789".into(),
            user_code: "C789".into(),
            verification_uri: "https://github.com/login/device".into(),
            expires_in: 900,
            interval: 1,
        };

        let call_count = RefCell::new(0);
        let mock_send = move |_req: isahc::Request<Vec<u8>>| {
            let count = *call_count.borrow();
            *call_count.borrow_mut() = count + 1;
            if count >= 1 {
                return Err(AgentError::Config {
                    message: "network error".into(),
                });
            }
            Ok::<_, AgentError>(
                isahc::Response::builder()
                    .status(200)
                    .body(isahc::Body::from(r#"{"error":"authorization_pending"}"#))
                    .unwrap(),
            )
        };

        let result = smol::block_on(poll_device_token_with(
            &device,
            Duration::from_millis(1),
            mock_send,
        ));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("network error"));
    }
}
