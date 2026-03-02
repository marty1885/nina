use async_trait::async_trait;
use chrono::Local;

use crate::channel::CallContext;

pub struct CurrentTimeSkill;

impl CurrentTimeSkill {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl super::Skill for CurrentTimeSkill {
    fn name(&self) -> &str {
        "Current time"
    }

    fn description(&self) -> Option<&str> {
        Some("Returns the current date and time")
    }

    async fn tool_definitions(&self) -> Vec<serde_json::Value> {
        vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "current_time",
                "description": "Returns the current date, time, and day of week.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        })]
    }

    async fn call(&self, tool_name: &str, _args: &str, _ctx: &CallContext) -> Option<String> {
        if tool_name != "current_time" {
            return None;
        }

        let now = Local::now();
        Some(now.format("%A, %B %-d, %Y — %H:%M %Z").to_string())
    }
}
