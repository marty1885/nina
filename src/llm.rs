use anyhow::{bail, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

// --- OpenAI-compatible types ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallInfo>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn tool_result(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(id.into()),
        }
    }

    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls.as_ref().is_some_and(|tc| !tc.is_empty())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallInfo {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<serde_json::Value>>,
    max_tokens: Option<u32>,
    temperature: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ChatMessage,
    finish_reason: Option<String>,
}

// --- LlmProvider trait ---

#[derive(Debug)]
pub enum LlmResponse {
    Text(String),
    /// (content_text, tool_calls) — content may accompany tool calls
    ToolCalls(Option<String>, Vec<ToolCallInfo>),
}

#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn complete(
        &self,
        messages: &[ChatMessage],
        tools: &[serde_json::Value],
    ) -> Result<LlmResponse>;
}

// --- Error classification ---

/// Returned (wrapped in anyhow::Error) when a provider fails for infrastructure reasons:
/// network error, timeout, HTTP 5xx, or HTTP 429. WeightedProvider uses this to decide
/// whether to fall back to another provider.
#[derive(Debug)]
pub struct InfrastructureError(pub String);

impl std::fmt::Display for InfrastructureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for InfrastructureError {}

// --- OpenAiProvider (OpenAI-compatible endpoint) ---

pub struct OpenAiProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
}

impl OpenAiProvider {
    pub fn new(base_url: &str, api_key: &str, model: &str) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .expect("Failed to build HTTP client");

        Self {
            client,
            base_url: base_url.into(),
            api_key: api_key.into(),
            model: model.into(),
        }
    }
}

#[async_trait]
impl LlmProvider for OpenAiProvider {
    async fn complete(
        &self,
        messages: &[ChatMessage],
        tools: &[serde_json::Value],
    ) -> Result<LlmResponse> {
        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages: messages.to_vec(),
            tools: if tools.is_empty() {
                None
            } else {
                Some(tools.to_vec())
            },
            max_tokens: Some(8192),
            temperature: Some(0.7),
        };

        let resp = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Accept", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                anyhow::Error::new(InfrastructureError(format!("Network error: {e}")))
            })?;

        let status = resp.status();

        if status.as_u16() == 429 {
            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(1);
            return Err(anyhow::Error::new(InfrastructureError(format!(
                "Rate limited (429), retry-after={}s [model={}]",
                retry_after, self.model
            ))));
        }

        if status.is_server_error() {
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow::Error::new(InfrastructureError(format!(
                "Server error {status} [model={}]: {body}",
                self.model
            ))));
        }

        if !status.is_success() {
            // 4xx client errors — surface immediately, switching provider won't help.
            let body = resp.text().await.unwrap_or_default();
            bail!("API error {status} [model={}]: {body}", self.model);
        }

        let completion: ChatCompletionResponse = resp.json().await?;
        let choice = completion
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No choices in response"))?;

        let finish = choice.finish_reason.as_deref().unwrap_or("");
        let tool_calls = choice.message.tool_calls.unwrap_or_default();
        // Guard: only treat as tool calls if the list is actually non-empty.
        // Some providers set finish_reason="tool_calls" but return a null/empty
        // tool_calls array — treat that as a plain text response to avoid
        // storing an assistant message with empty tool_calls that would orphan
        // any subsequent tool-result messages.
        if !tool_calls.is_empty() {
            let content = choice.message.content.filter(|s| !s.trim().is_empty());
            return Ok(LlmResponse::ToolCalls(content, tool_calls));
        }
        if finish == "tool_calls" {
            warn!("finish_reason=tool_calls but tool_calls list is empty; treating as text");
        }

        let text = choice.message.content.unwrap_or_default();
        Ok(LlmResponse::Text(text))
    }
}

// --- WeightedProvider ---

/// Routes requests across a weighted pool of providers.
/// On infrastructure errors (network, 5xx, 429) the failing provider is removed from
/// the pool for that request and the next is tried. Client errors (4xx except 429)
/// are surfaced immediately since switching providers won't help.
pub struct WeightedProvider {
    providers: Vec<(std::sync::Arc<dyn LlmProvider>, u32)>,
}

impl WeightedProvider {
    pub fn new(providers: Vec<(std::sync::Arc<dyn LlmProvider>, u32)>) -> Self {
        Self { providers }
    }
}

#[async_trait]
impl LlmProvider for WeightedProvider {
    async fn complete(
        &self,
        messages: &[ChatMessage],
        tools: &[serde_json::Value],
    ) -> Result<LlmResponse> {
        // Pool entries: (index into self.providers, weight)
        let mut pool: Vec<(usize, u32)> = self
            .providers
            .iter()
            .enumerate()
            .map(|(i, (_, w))| (i, *w))
            .collect();

        loop {
            if pool.is_empty() {
                return Err(anyhow::Error::new(InfrastructureError(
                    "All providers in this tier exhausted".into(),
                )));
            }

            // Sample a provider by weight. The rng is scoped so it is dropped
            // before the .await below (ThreadRng is !Send).
            let pool_idx = {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                let total: u32 = pool.iter().map(|(_, w)| *w).sum();
                if total == 0 {
                    bail!("All remaining providers have zero weight");
                }
                let mut r = rng.gen_range(0..total);
                pool.iter()
                    .position(|(_, w)| {
                        if r < *w {
                            true
                        } else {
                            r -= *w;
                            false
                        }
                    })
                    .unwrap_or(0)
            };

            let provider_idx = pool[pool_idx].0;
            let provider = &self.providers[provider_idx].0;

            debug!(provider_idx, "WeightedProvider: trying provider");

            match provider.complete(messages, tools).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    if e.downcast_ref::<InfrastructureError>().is_some() {
                        warn!(
                            provider_idx,
                            error = %e,
                            "Provider infrastructure error, trying next provider"
                        );
                        pool.remove(pool_idx);
                        continue;
                    }
                    // Client error or other — surface immediately.
                    return Err(e);
                }
            }
        }
    }
}

// --- TieredProvider ---

/// Routes requests through tiers in strict priority order.
/// The primary tier is tried first (internally weighted). Only if every provider
/// in a tier fails with an infrastructure error does the next tier get tried.
/// Client errors (4xx except 429) surface immediately — switching tiers won't help.
pub struct TieredProvider {
    tiers: Vec<std::sync::Arc<dyn LlmProvider>>,
}

impl TieredProvider {
    pub fn new(tiers: Vec<std::sync::Arc<dyn LlmProvider>>) -> Self {
        Self { tiers }
    }
}

#[async_trait]
impl LlmProvider for TieredProvider {
    async fn complete(
        &self,
        messages: &[ChatMessage],
        tools: &[serde_json::Value],
    ) -> Result<LlmResponse> {
        let mut last_err = anyhow::anyhow!("No tiers configured");
        for (i, tier) in self.tiers.iter().enumerate() {
            match tier.complete(messages, tools).await {
                Ok(response) => return Ok(response),
                Err(e) if e.downcast_ref::<InfrastructureError>().is_some() => {
                    warn!(tier = i, error = %e, "Tier exhausted, trying next tier");
                    last_err = e;
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_message_system_fields() {
        let m = ChatMessage::system("you are helpful");
        assert_eq!(m.role, "system");
        assert_eq!(m.content.as_deref(), Some("you are helpful"));
        assert!(m.tool_calls.is_none());
        assert!(m.tool_call_id.is_none());
    }

    #[test]
    fn chat_message_user_fields() {
        let m = ChatMessage::user("hello there");
        assert_eq!(m.role, "user");
        assert_eq!(m.content.as_deref(), Some("hello there"));
        assert!(m.tool_calls.is_none());
        assert!(m.tool_call_id.is_none());
    }

    #[test]
    fn chat_message_tool_result_fields() {
        let m = ChatMessage::tool_result("call_abc", "42");
        assert_eq!(m.role, "tool");
        assert_eq!(m.content.as_deref(), Some("42"));
        assert_eq!(m.tool_call_id.as_deref(), Some("call_abc"));
        assert!(m.tool_calls.is_none());
    }

    #[test]
    fn has_tool_calls_false_when_none() {
        let m = ChatMessage::user("test");
        assert!(!m.has_tool_calls());
    }

    #[test]
    fn has_tool_calls_false_for_empty_vec() {
        let m = ChatMessage {
            role: "assistant".into(),
            content: None,
            tool_calls: Some(vec![]),
            tool_call_id: None,
        };
        assert!(!m.has_tool_calls());
    }

    #[test]
    fn has_tool_calls_true_when_populated() {
        let m = ChatMessage {
            role: "assistant".into(),
            content: None,
            tool_calls: Some(vec![ToolCallInfo {
                id: "id1".into(),
                call_type: "function".into(),
                function: FunctionCall {
                    name: "shell".into(),
                    arguments: "{}".into(),
                },
            }]),
            tool_call_id: None,
        };
        assert!(m.has_tool_calls());
    }

    #[test]
    fn chat_message_serialization_roundtrip() {
        let m = ChatMessage::user("round-trip test");
        let json = serde_json::to_string(&m).unwrap();
        let back: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(back.role, m.role);
        assert_eq!(back.content, m.content);
    }

    #[test]
    fn tool_calls_skipped_in_serialization_when_none() {
        let m = ChatMessage::user("no tools");
        let json = serde_json::to_string(&m).unwrap();
        assert!(!json.contains("tool_calls"));
    }

    // --- WeightedProvider tests ---

    struct AlwaysFailProvider;
    struct AlwaysSucceedProvider;

    #[async_trait]
    impl LlmProvider for AlwaysFailProvider {
        async fn complete(
            &self,
            _messages: &[ChatMessage],
            _tools: &[serde_json::Value],
        ) -> Result<LlmResponse> {
            Err(anyhow::Error::new(InfrastructureError("simulated infra failure".into())))
        }
    }

    #[async_trait]
    impl LlmProvider for AlwaysSucceedProvider {
        async fn complete(
            &self,
            _messages: &[ChatMessage],
            _tools: &[serde_json::Value],
        ) -> Result<LlmResponse> {
            Ok(LlmResponse::Text("ok".into()))
        }
    }

    #[tokio::test]
    async fn weighted_provider_falls_back_on_infra_error() {
        use std::sync::Arc;

        let providers: Vec<(Arc<dyn LlmProvider>, u32)> = vec![
            (Arc::new(AlwaysFailProvider), 50),
            (Arc::new(AlwaysSucceedProvider), 50),
        ];
        let weighted = WeightedProvider::new(providers);

        // Run multiple times — regardless of which provider is sampled first,
        // the result should always be Ok("ok") because FailProvider falls back.
        for _ in 0..10 {
            let result = weighted.complete(&[ChatMessage::user("test")], &[]).await;
            assert!(result.is_ok());
            match result.unwrap() {
                LlmResponse::Text(t) => assert_eq!(t, "ok"),
                _ => panic!("Expected text response"),
            }
        }
    }

    #[tokio::test]
    async fn weighted_provider_all_fail_returns_exhausted_error() {
        use std::sync::Arc;

        let providers: Vec<(Arc<dyn LlmProvider>, u32)> = vec![
            (Arc::new(AlwaysFailProvider), 50),
            (Arc::new(AlwaysFailProvider), 50),
        ];
        let weighted = WeightedProvider::new(providers);
        let result = weighted.complete(&[ChatMessage::user("test")], &[]).await;
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("exhausted"),
            "Error should mention exhausted"
        );
    }

    #[tokio::test]
    async fn weighted_provider_surfaces_client_error_immediately() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU32, Ordering};

        struct CountingClientErrorProvider {
            calls: Arc<AtomicU32>,
        }

        #[async_trait]
        impl LlmProvider for CountingClientErrorProvider {
            async fn complete(
                &self,
                _messages: &[ChatMessage],
                _tools: &[serde_json::Value],
            ) -> Result<LlmResponse> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                bail!("client error: 400 bad request")
            }
        }

        let call_count = Arc::new(AtomicU32::new(0));
        let providers: Vec<(Arc<dyn LlmProvider>, u32)> = vec![
            (
                Arc::new(CountingClientErrorProvider { calls: call_count.clone() }),
                100,
            ),
        ];
        let weighted = WeightedProvider::new(providers);
        let result = weighted.complete(&[ChatMessage::user("test")], &[]).await;

        assert!(result.is_err());
        // Should have been called exactly once — no retry on client errors.
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
        let err_str = result.unwrap_err().to_string();
        assert!(err_str.contains("400"), "Got: {err_str}");
    }

    // --- TieredProvider tests ---

    #[tokio::test]
    async fn tiered_provider_uses_primary_when_it_succeeds() {
        use std::sync::Arc;
        let tiers: Vec<Arc<dyn LlmProvider>> = vec![
            Arc::new(AlwaysSucceedProvider),
            Arc::new(AlwaysFailProvider), // fallback — should never be reached
        ];
        let tiered = TieredProvider::new(tiers);
        let result = tiered.complete(&[ChatMessage::user("test")], &[]).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn tiered_provider_falls_through_to_fallback_on_infra_exhaustion() {
        use std::sync::Arc;
        // Primary tier: always fails with infra error → exhausted.
        // Fallback tier: succeeds.
        let primary_tier: Arc<dyn LlmProvider> =
            Arc::new(WeightedProvider::new(vec![(Arc::new(AlwaysFailProvider), 100)]));
        let fallback_tier: Arc<dyn LlmProvider> = Arc::new(AlwaysSucceedProvider);

        let tiered = TieredProvider::new(vec![primary_tier, fallback_tier]);
        let result = tiered.complete(&[ChatMessage::user("test")], &[]).await;
        assert!(result.is_ok());
        match result.unwrap() {
            LlmResponse::Text(t) => assert_eq!(t, "ok"),
            _ => panic!("Expected text response"),
        }
    }

    #[tokio::test]
    async fn tiered_provider_surfaces_client_error_without_trying_fallback() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU32, Ordering};

        struct CountingClientErrorProvider {
            calls: Arc<AtomicU32>,
        }

        #[async_trait]
        impl LlmProvider for CountingClientErrorProvider {
            async fn complete(
                &self,
                _messages: &[ChatMessage],
                _tools: &[serde_json::Value],
            ) -> Result<LlmResponse> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                bail!("client error: 400 bad request")
            }
        }

        struct CountingSucceedProvider {
            calls: Arc<AtomicU32>,
        }

        #[async_trait]
        impl LlmProvider for CountingSucceedProvider {
            async fn complete(
                &self,
                _messages: &[ChatMessage],
                _tools: &[serde_json::Value],
            ) -> Result<LlmResponse> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                Ok(LlmResponse::Text("fallback ok".into()))
            }
        }

        let primary_calls = Arc::new(AtomicU32::new(0));
        let fallback_calls = Arc::new(AtomicU32::new(0));

        let tiers: Vec<Arc<dyn LlmProvider>> = vec![
            Arc::new(CountingClientErrorProvider { calls: primary_calls.clone() }),
            Arc::new(CountingSucceedProvider { calls: fallback_calls.clone() }),
        ];
        let tiered = TieredProvider::new(tiers);
        let result = tiered.complete(&[ChatMessage::user("test")], &[]).await;

        assert!(result.is_err());
        assert_eq!(primary_calls.load(Ordering::SeqCst), 1);
        assert_eq!(fallback_calls.load(Ordering::SeqCst), 0, "Fallback must not be called on client error");
    }

    #[tokio::test]
    async fn tiered_provider_all_tiers_exhausted_returns_infra_error() {
        use std::sync::Arc;
        let primary_tier: Arc<dyn LlmProvider> =
            Arc::new(WeightedProvider::new(vec![(Arc::new(AlwaysFailProvider), 100)]));
        let fallback_tier: Arc<dyn LlmProvider> =
            Arc::new(WeightedProvider::new(vec![(Arc::new(AlwaysFailProvider), 100)]));

        let tiered = TieredProvider::new(vec![primary_tier, fallback_tier]);
        let result = tiered.complete(&[ChatMessage::user("test")], &[]).await;
        assert!(result.is_err());
        // Error should be an InfrastructureError (all tiers are infra-exhausted)
        let err = result.unwrap_err();
        assert!(
            err.downcast_ref::<InfrastructureError>().is_some(),
            "Expected InfrastructureError, got: {err}"
        );
    }
}
