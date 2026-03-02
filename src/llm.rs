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

// --- DeepInfra implementation ---

pub struct DeepInfraProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
}

impl DeepInfraProvider {
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
impl LlmProvider for DeepInfraProvider {
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

        // Retry logic: 4 attempts, exponential backoff
        let mut last_err = String::new();
        for attempt in 0..4 {
            if attempt > 0 {
                debug!(attempt, "Retrying LLM request");
            }

            let resp = self
                .client
                .post(format!("{}/chat/completions", self.base_url))
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Accept", "application/json")
                .json(&request)
                .send()
                .await;

            let resp = match resp {
                Ok(r) => r,
                Err(e) => {
                    last_err = format!("HTTP error: {e}");
                    let delay = std::time::Duration::from_millis(500 * 2u64.pow(attempt));
                    tokio::time::sleep(delay).await;
                    continue;
                }
            };

            let status = resp.status();
            if status.as_u16() == 429 {
                // Parse Retry-After header if present
                let retry_after = resp
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(1);
                let delay = std::time::Duration::from_secs(retry_after.max(1));
                warn!(attempt, retry_after_secs = retry_after, "Rate limited, backing off");
                tokio::time::sleep(delay).await;
                last_err = "Rate limited (429)".into();
                continue;
            }

            if !status.is_success() {
                let body = resp.text().await.unwrap_or_default();
                bail!("API error {status}: {body}");
            }

            let completion: ChatCompletionResponse = resp.json().await?;
            let choice = completion
                .choices
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("No choices in response"))?;

            let finish = choice.finish_reason.as_deref().unwrap_or("");
            if finish == "tool_calls" || choice.message.has_tool_calls() {
                let content = choice.message.content.filter(|s| !s.trim().is_empty());
                let tool_calls = choice.message.tool_calls.unwrap_or_default();
                return Ok(LlmResponse::ToolCalls(content, tool_calls));
            }

            let text = choice.message.content.unwrap_or_default();
            return Ok(LlmResponse::Text(text));
        }

        bail!("LLM request failed after 4 attempts: {last_err}");
    }
}
