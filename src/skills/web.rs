use crate::tools::web_fetch::WebFetchTool;
use async_trait::async_trait;
use rig::tool::Tool;
use serde::Deserialize;
use std::sync::Arc;
use tracing::info;

pub struct WebSkill {
    web_fetch: WebFetchTool,
    http_client: Arc<reqwest::Client>,
    brave_api_key: Option<String>,
}

impl WebSkill {
    pub fn new(http_client: Arc<reqwest::Client>) -> Self {
        let brave_api_key = std::env::var("BRAVE_API_KEY").ok();
        Self {
            web_fetch: WebFetchTool::new(http_client.clone()),
            http_client,
            brave_api_key,
        }
    }
}

#[derive(Deserialize)]
struct WebSearchArgs {
    query: String,
}

#[derive(Deserialize)]
struct BraveResponse {
    web: Option<BraveWebResults>,
}

#[derive(Deserialize)]
struct BraveWebResults {
    results: Vec<BraveResult>,
}

#[derive(Deserialize)]
struct BraveResult {
    title: String,
    url: String,
    description: Option<String>,
}

#[async_trait]
impl super::Skill for WebSkill {
    fn name(&self) -> &str {
        "Web"
    }

    fn description(&self) -> Option<&str> {
        Some("Fetch web pages and search the web")
    }

    async fn tool_definitions(&self) -> Vec<serde_json::Value> {
        let d = self.web_fetch.definition(String::new()).await;
        let mut defs = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": d.name,
                "description": d.description,
                "parameters": d.parameters,
            }
        })];

        defs.push(serde_json::json!({
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web using Brave Search. Returns titles, URLs, and snippets.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        }));

        defs
    }

    async fn call(&self, tool_name: &str, args: &str, _ctx: &crate::channel::CallContext) -> Option<String> {
        match tool_name {
            "web_fetch" => Some(crate::tools::dispatch(&self.web_fetch, args).await),
            "web_search" => Some(self.web_search(args).await),
            _ => None,
        }
    }
}

impl WebSkill {
    async fn web_search(&self, args_json: &str) -> String {
        let api_key = match &self.brave_api_key {
            Some(k) => k,
            None => return "web search not configured".into(),
        };

        let args: WebSearchArgs = match serde_json::from_str(args_json) {
            Ok(a) => a,
            Err(e) => return format!("Invalid arguments: {e}"),
        };

        info!(query = %args.query, "web_search tool");

        let resp = self
            .http_client
            .get("https://api.search.brave.com/res/v1/web/search")
            .header("X-Subscription-Token", api_key.as_str())
            .query(&[("q", &args.query), ("count", &"5".to_string())])
            .send()
            .await;

        let resp = match resp {
            Ok(r) => r,
            Err(e) => return format!("Search request failed: {e}"),
        };

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return format!("Search API error {status}: {body}");
        }

        let brave: BraveResponse = match resp.json().await {
            Ok(b) => b,
            Err(e) => return format!("Failed to parse search results: {e}"),
        };

        let results = match brave.web {
            Some(w) => w.results,
            None => return "No results found.".into(),
        };

        if results.is_empty() {
            return "No results found.".into();
        }

        results
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let snippet = r.description.as_deref().unwrap_or("(no snippet)");
                format!("{}. {}\n   {}\n   {}", i + 1, r.title, r.url, snippet)
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}
