use crate::tools::exec::ExecTool;
use async_trait::async_trait;
use rig::tool::Tool;

pub struct ShellSkill {
    exec: ExecTool,
}

impl ShellSkill {
    pub fn new() -> Self {
        Self { exec: ExecTool }
    }
}

#[async_trait]
impl super::Skill for ShellSkill {
    fn name(&self) -> &str {
        "Shell execution"
    }

    fn description(&self) -> Option<&str> {
        Some("Run bash commands on the host system")
    }

    async fn tool_definitions(&self) -> Vec<serde_json::Value> {
        let d = self.exec.definition(String::new()).await;
        vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": d.name,
                "description": d.description,
                "parameters": d.parameters,
            }
        })]
    }

    async fn call(&self, tool_name: &str, args: &str, _ctx: &crate::channel::CallContext) -> Option<String> {
        if tool_name != "exec" {
            return None;
        }
        Some(crate::tools::dispatch(&self.exec, args).await)
    }
}
