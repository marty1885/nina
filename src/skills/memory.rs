use crate::memory::MemoryStore;
use crate::tools::memory::{RecallTool, RememberDailyTool, RememberTool, UpdateMemoryTool};
use async_trait::async_trait;
use rig::tool::Tool;
use std::path::PathBuf;
use std::sync::Arc;

pub struct MemorySkill {
    remember: RememberTool,
    recall: RecallTool,
    remember_daily: RememberDailyTool,
    update_memory: UpdateMemoryTool,
}

impl MemorySkill {
    pub fn new(memory_store: Arc<MemoryStore>, context_dir: PathBuf) -> Self {
        Self {
            remember: RememberTool::new(memory_store.clone()),
            recall: RecallTool::new(memory_store.clone()),
            remember_daily: RememberDailyTool::new(context_dir),
            update_memory: UpdateMemoryTool::new(memory_store),
        }
    }
}

#[async_trait]
impl super::Skill for MemorySkill {
    fn name(&self) -> &str {
        "Memory"
    }

    fn description(&self) -> Option<&str> {
        Some("Persistent semantic memory with daily notes")
    }

    fn system_prompt_addition(&self) -> Option<String> {
        Some(
            "Persistent memory: remember(fact, category?) stores long-term facts \
             (category: core|conversation), remember_daily(note) for session logs, \
             update_memory(query, new?) to correct or retract, recall(query) to search. \
             Core memories surface preferentially."
                .into(),
        )
    }

    async fn tool_definitions(&self) -> Vec<serde_json::Value> {
        let tools: Vec<_> = vec![
            self.remember.definition(String::new()).await,
            self.recall.definition(String::new()).await,
            self.remember_daily.definition(String::new()).await,
            self.update_memory.definition(String::new()).await,
        ];
        tools
            .into_iter()
            .map(|d| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": d.name,
                        "description": d.description,
                        "parameters": d.parameters,
                    }
                })
            })
            .collect()
    }

    async fn call(&self, tool_name: &str, args: &str, _ctx: &crate::channel::CallContext) -> Option<String> {
        match tool_name {
            "remember" => Some(crate::tools::dispatch(&self.remember, args).await),
            "recall" => Some(crate::tools::dispatch(&self.recall, args).await),
            "remember_daily" => Some(crate::tools::dispatch(&self.remember_daily, args).await),
            "update_memory" => Some(crate::tools::dispatch(&self.update_memory, args).await),
            _ => None,
        }
    }
}
