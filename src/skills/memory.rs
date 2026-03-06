use crate::memory::MemoryStore;
use crate::tools::memory::{RecallTool, RememberDailyTool, RememberTool, UpdateMemoryTool};
use crate::tools::memory::{RecallArgs, RememberArgs, RememberDailyArgs, UpdateMemoryArgs};
use async_trait::async_trait;
use rig::tool::Tool;
use std::path::PathBuf;
use std::sync::Arc;

pub struct MemorySkill {
    store: Arc<MemoryStore>,
    remember: RememberTool,
    recall: RecallTool,
    remember_daily: RememberDailyTool,
    update_memory: UpdateMemoryTool,
}

impl MemorySkill {
    pub fn new(memory_store: Arc<MemoryStore>, context_dir: PathBuf) -> Self {
        Self {
            store: memory_store.clone(),
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
            "Memory heuristics — use remember() for: unresolved problems (something wasn't solved \
             by end of conversation); emotional texture (how someone feels about something — stress, \
             frustration, excitement); their world (names and relationships that come up: \"my manager \
             Sarah\", \"the thing with my landlord\"); preferences revealed incidentally (tool they hate, \
             way they prefer to work); decisions that got made. Low threshold: if it has weight and could \
             matter next time, store it. remember_daily() for things that only matter in the next day or two — task progress, session follow-ups; it expires quickly. \
             Don't narrate the act of storing. recall() proactively when a reference might connect to \
             past context. update_memory(query, new?) to correct or retract. category='core' for stable \
             evergreen facts. What NOT to store: chitchat with no content, things already in context, \
             anything you'd remember for the next hour anyway."
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

    async fn call(&self, tool_name: &str, args: &str, ctx: &crate::channel::CallContext) -> Option<String> {
        let identity_id = ctx.identity_id;
        match tool_name {
            "remember" => {
                let args: RememberArgs = match serde_json::from_str(args) {
                    Ok(a) => a,
                    Err(e) => return Some(format!("Invalid arguments: {e}")),
                };
                let category = args.category.as_deref().unwrap_or("conversation");
                let source = args.source.as_deref().unwrap_or("user");
                Some(match self.store.remember(&args.content, category, source, identity_id).await {
                    Ok(()) => "Remembered.".into(),
                    Err(e) => format!("Tool error: {e}"),
                })
            }
            "recall" => {
                let args: RecallArgs = match serde_json::from_str(args) {
                    Ok(a) => a,
                    Err(e) => return Some(format!("Invalid arguments: {e}")),
                };
                Some(match self.store.recall(&args.query, identity_id).await {
                    Ok(r) => r,
                    Err(e) => format!("Tool error: {e}"),
                })
            }
            "remember_daily" => {
                let args: RememberDailyArgs = match serde_json::from_str(args) {
                    Ok(a) => a,
                    Err(e) => return Some(format!("Invalid arguments: {e}")),
                };
                Some(match crate::identity::append_daily_note(
                    &self.remember_daily.context_dir(),
                    &args.content,
                ).await {
                    Ok(()) => "Noted.".into(),
                    Err(e) => format!("Tool error: {e}"),
                })
            }
            "update_memory" => {
                let args: UpdateMemoryArgs = match serde_json::from_str(args) {
                    Ok(a) => a,
                    Err(e) => return Some(format!("Invalid arguments: {e}")),
                };
                Some(match self.store.update_memory(&args.query, args.new_content, identity_id).await {
                    Ok(r) => r,
                    Err(e) => format!("Tool error: {e}"),
                })
            }
            _ => None,
        }
    }
}
