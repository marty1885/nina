use crate::identity;
use crate::memory::MemoryStore;
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::Arc;

// --- Remember Tool ---

pub struct RememberTool {
    store: Arc<MemoryStore>,
}

impl RememberTool {
    pub fn new(store: Arc<MemoryStore>) -> Self {
        Self { store }
    }
}

#[derive(Deserialize)]
pub struct RememberArgs {
    pub content: String,
    pub category: Option<String>,
    pub source: Option<String>,
}

#[derive(Debug)]
pub struct MemoryError(pub String);

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for MemoryError {}

impl Tool for RememberTool {
    const NAME: &'static str = "remember";
    type Error = MemoryError;
    type Args = RememberArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "remember".into(),
            description: "Store a piece of information in long-term memory. Use this to remember facts about the user, preferences, or anything worth recalling later. Use category='core' for evergreen facts that should always surface prominently.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The information to remember"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["core", "conversation"],
                        "description": "Memory category. 'core' for stable, high-importance facts (always surfaces preferentially). 'conversation' for session-contextual facts (default)."
                    },
                    "source": {
                        "type": "string",
                        "enum": ["user", "agent"],
                        "description": "Who provided this memory. 'user' if the user stated it directly (default). 'agent' if inferred."
                    }
                },
                "required": ["content"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<String, Self::Error> {
        let category = args.category.as_deref().unwrap_or("conversation");
        let source = args.source.as_deref().unwrap_or("user");
        self.store
            .remember(&args.content, category, source)
            .await
            .map_err(|e| MemoryError(format!("Failed to store memory: {e}")))?;
        Ok("Remembered.".into())
    }
}

// --- Recall Tool ---

pub struct RecallTool {
    store: Arc<MemoryStore>,
}

impl RecallTool {
    pub fn new(store: Arc<MemoryStore>) -> Self {
        Self { store }
    }
}

#[derive(Deserialize)]
pub struct RecallArgs {
    pub query: String,
}

impl Tool for RecallTool {
    const NAME: &'static str = "recall";
    type Error = MemoryError;
    type Args = RecallArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "recall".into(),
            description: "Search long-term memory for information. Uses both keyword and semantic search with recency weighting. Core memories surface preferentially. Returns up to 5 relevant memories.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in memory"
                    }
                },
                "required": ["query"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<String, Self::Error> {
        self.store
            .recall(&args.query)
            .await
            .map_err(|e| MemoryError(format!("Failed to recall: {e}")))
    }
}

// --- Remember Daily Tool ---

pub struct RememberDailyTool {
    context_dir: PathBuf,
}

impl RememberDailyTool {
    pub fn new(context_dir: PathBuf) -> Self {
        Self { context_dir }
    }
}

#[derive(Deserialize)]
pub struct RememberDailyArgs {
    pub content: String,
}

impl Tool for RememberDailyTool {
    const NAME: &'static str = "remember_daily";
    type Error = MemoryError;
    type Args = RememberDailyArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "remember_daily".into(),
            description: "Append a note to today's daily log file. Use this for session-specific context, task progress, things to follow up on, or anything relevant to today. Separate from long-term semantic memory.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The note to append to today's daily log"
                    }
                },
                "required": ["content"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<String, Self::Error> {
        identity::append_daily_note(&self.context_dir, &args.content)
            .await
            .map_err(|e| MemoryError(format!("Failed to append daily note: {e}")))?;
        Ok("Noted.".into())
    }
}

// --- Update Memory Tool ---

pub struct UpdateMemoryTool {
    store: Arc<MemoryStore>,
}

impl UpdateMemoryTool {
    pub fn new(store: Arc<MemoryStore>) -> Self {
        Self { store }
    }
}

#[derive(Deserialize)]
pub struct UpdateMemoryArgs {
    pub query: String,
    pub new_content: Option<String>,
}

impl Tool for UpdateMemoryTool {
    const NAME: &'static str = "update_memory";
    type Error = MemoryError;
    type Args = UpdateMemoryArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "update_memory".into(),
            description: "Correct or retract a stored memory. Search for the memory by query, then either correct it (provide new_content to supersede) or retract it (omit new_content to mark as retracted). Use this when a previously stored fact is wrong or no longer valid.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find the memory to update or retract"
                    },
                    "new_content": {
                        "type": "string",
                        "description": "Replacement content. If provided, the old memory is superseded by this new one. If omitted, the memory is retracted."
                    }
                },
                "required": ["query"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<String, Self::Error> {
        self.store
            .update_memory(&args.query, args.new_content)
            .await
            .map_err(|e| MemoryError(format!("Failed to update memory: {e}")))
    }
}
