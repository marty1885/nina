pub mod current_time;
pub mod relation;
pub mod files;
pub mod lua;
pub mod markdown;
pub mod memory;
pub mod reminders;
pub mod sessions_send;
pub mod shell;
pub mod tts;
pub mod web;

use async_trait::async_trait;

use crate::channel::CallContext;

/// A self-contained capability that contributes tools and prompt additions.
#[async_trait]
pub trait Skill: Send + Sync {
    /// Human-readable name.
    fn name(&self) -> &str;

    /// Optional short description of this skill's capabilities.
    fn description(&self) -> Option<&str> {
        None
    }

    /// Tool definitions in OpenAI function-calling format.
    async fn tool_definitions(&self) -> Vec<serde_json::Value>;

    /// Dispatch a tool call by name. Returns Some(result) if handled, None if not.
    async fn call(&self, tool_name: &str, args: &str, ctx: &CallContext) -> Option<String>;

    /// Optional text to append to system prompt at startup.
    /// Defaults to a one-liner using `description()` if provided.
    fn system_prompt_addition(&self) -> Option<String> {
        self.description()
            .map(|d| format!("**{}**: {}", self.name(), d))
    }
}

/// Collects skills and provides unified tool definitions + dispatch.
pub struct SkillRegistry {
    skills: Vec<Box<dyn Skill>>,
}

impl SkillRegistry {
    pub fn new() -> Self {
        Self { skills: Vec::new() }
    }

    pub fn register(&mut self, skill: Box<dyn Skill>) {
        tracing::info!(skill = skill.name(), "Registered skill");
        self.skills.push(skill);
    }

    /// All tool definitions from all skills.
    pub async fn definitions(&self) -> Vec<serde_json::Value> {
        let mut defs = Vec::new();
        for skill in &self.skills {
            defs.extend(skill.tool_definitions().await);
        }
        defs
    }

    /// Try to dispatch a tool call across all skills. First match wins.
    pub async fn call(&self, tool_name: &str, args: &str, ctx: &CallContext) -> Option<String> {
        for skill in &self.skills {
            if let Some(result) = skill.call(tool_name, args, ctx).await {
                return Some(result);
            }
        }
        None
    }

    /// Collect system prompt additions from all skills.
    pub fn system_prompt_additions(&self) -> Vec<String> {
        self.skills
            .iter()
            .filter_map(|s| s.system_prompt_addition())
            .collect()
    }
}
