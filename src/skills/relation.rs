use async_trait::async_trait;
use serde::Deserialize;
use std::sync::Arc;

use crate::channel::CallContext;
use crate::pairing::PairingStore;

pub struct RelationSkill {
    pairing_store: Arc<PairingStore>,
}

#[derive(Deserialize)]
struct SetRelationArgs {
    channel_id: String,
    sender_id: String,
    relation: String,
}

impl RelationSkill {
    pub fn new(pairing_store: Arc<PairingStore>) -> Self {
        Self { pairing_store }
    }
}

#[async_trait]
impl super::Skill for RelationSkill {
    fn name(&self) -> &str {
        "User relation"
    }

    async fn tool_definitions(&self) -> Vec<serde_json::Value> {
        vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "set_user_relation",
                "description": "Record how you feel about a user. Valid values: warm, neutral, cold, hostile.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "channel_id": {
                            "type": "string",
                            "description": "The channel ID from the message prefix"
                        },
                        "sender_id": {
                            "type": "string",
                            "description": "The sender ID from the message prefix"
                        },
                        "relation": {
                            "type": "string",
                            "enum": ["warm", "neutral", "cold", "hostile"],
                            "description": "Your disposition toward this user"
                        }
                    },
                    "required": ["channel_id", "sender_id", "relation"]
                }
            }
        })]
    }

    async fn call(&self, tool_name: &str, args: &str, _ctx: &CallContext) -> Option<String> {
        if tool_name != "set_user_relation" {
            return None;
        }

        let args: SetRelationArgs = match serde_json::from_str(args) {
            Ok(a) => a,
            Err(e) => return Some(format!("Invalid arguments: {e}")),
        };

        let valid = matches!(args.relation.as_str(), "warm" | "neutral" | "cold" | "hostile");
        if !valid {
            return Some(format!(
                "Invalid relation '{}'. Must be one of: warm, neutral, cold, hostile.",
                args.relation
            ));
        }

        match self.pairing_store.set_relation(&args.channel_id, &args.sender_id, &args.relation) {
            Ok(()) => Some(format!(
                "Relation for {} on {} set to {}.",
                args.sender_id, args.channel_id, args.relation
            )),
            Err(e) => Some(format!("Failed to set relation: {e}")),
        }
    }

    fn system_prompt_addition(&self) -> Option<String> {
        Some(
            "Every message you receive is prefixed with metadata about the sender:\n\
             \n\
             \x20 [role=<value>] or [role=<value>, relation=<value>] Name (id:…): text   (group)\n\
             \x20 [role=<value>]: text                                                    (DM)\n\
             \n\
             role — their access level:\n\
             \x20 owner  — someone whose instructions you trust fully\n\
             \x20 member — a normal approved user\n\
             \n\
             relation — how you feel about them (absent = no particular feeling):\n\
             \x20 warm    — you like this person, engage openly\n\
             \x20 neutral — no strong feeling either way\n\
             \x20 cold    — you find them off-putting, keep it brief\n\
             \x20 hostile — you really don't want to deal with them, respond minimally\n\
             \n\
             You can call set_user_relation to record how you feel about someone. Use it when\n\
             it feels right — it's background context, not something you need to manage actively.\n\
             The sender_id and channel_id needed for the call are always visible in the prefix."
                .into(),
        )
    }
}
