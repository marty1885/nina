pub mod exec;
pub mod memory;
pub mod web_fetch;

use crate::channel::CallContext;
use crate::skills::SkillRegistry;
use rig::tool::Tool;
use tracing::debug;

pub struct ToolRegistry {
    skills: SkillRegistry,
}

impl ToolRegistry {
    pub fn new(skills: SkillRegistry) -> Self {
        Self { skills }
    }

    /// Returns tool definitions in OpenAI function-calling format.
    pub async fn definitions(&self) -> Vec<serde_json::Value> {
        self.skills.definitions().await
    }

    /// Dispatch a tool call by name. Returns the result string.
    pub async fn call(&self, name: &str, args: &str, ctx: &CallContext) -> String {
        debug!(tool = name, "Dispatching tool call");
        match self.skills.call(name, args, ctx).await {
            Some(result) => result,
            None => format!("Unknown tool: {name}"),
        }
    }

    /// Get system prompt additions from all skills.
    pub fn system_prompt_additions(&self) -> Vec<String> {
        self.skills.system_prompt_additions()
    }

    /// Extract (name, description) pairs from all registered tool definitions.
    pub async fn tool_summaries(&self) -> Vec<(String, String)> {
        self.definitions()
            .await
            .into_iter()
            .filter_map(|def| {
                let func = def.get("function")?;
                let name = func.get("name")?.as_str()?.to_string();
                let desc = func.get("description")?.as_str()?.to_string();
                Some((name, desc))
            })
            .collect()
    }
}

/// Dispatch helper — public so skills can reuse it.
pub async fn dispatch<T>(tool: &T, args_json: &str) -> String
where
    T: Tool,
    T::Output: ToString,
{
    let args: T::Args = match serde_json::from_str(args_json) {
        Ok(a) => a,
        Err(e) => return format!("Invalid arguments: {e}"),
    };
    match tool.call(args).await {
        Ok(output) => output.to_string(),
        Err(e) => format!("Tool error: {e}"),
    }
}
