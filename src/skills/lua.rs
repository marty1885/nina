use async_trait::async_trait;
use mlua::{HookTriggers, Lua, StdLib, Value};
use serde::Deserialize;
use tracing::info;

/// Maximum Lua VM instructions before aborting (~tens of milliseconds on modern hardware).
const MAX_INSTRUCTIONS: u32 = 10_000_000;
/// Maximum bytes of printed output returned to the caller.
const MAX_OUTPUT: usize = 4096;

pub struct LuaSkill;

impl LuaSkill {
    pub fn new() -> Self {
        Self
    }

    fn eval(code: &str) -> String {
        // Only safe standard libraries — no io, os, package, debug, coroutine.
        let lua = match Lua::new_with(
            StdLib::MATH | StdLib::STRING | StdLib::TABLE | StdLib::UTF8,
            mlua::LuaOptions::default(),
        ) {
            Ok(lua) => lua,
            Err(e) => return format!("[error] failed to init lua: {e}"),
        };

        // Hard instruction-count ceiling.
        lua.set_hook(
            HookTriggers {
                every_nth_instruction: Some(MAX_INSTRUCTIONS),
                ..Default::default()
            },
            |_lua, _debug| {
                Err(mlua::Error::RuntimeError(
                    "instruction limit exceeded".to_string(),
                ))
            },
        );

        // Shared buffer for print() and error() output.
        let output = std::sync::Arc::new(std::sync::Mutex::new(String::new()));

        // print() → appends to output buffer
        let out_ref = output.clone();
        let print_fn = lua.create_function(move |_lua, args: mlua::MultiValue| {
            let mut buf = out_ref.lock().unwrap();
            let parts: Vec<String> = args
                .iter()
                .map(|v| lua_value_to_string(v))
                .collect();
            buf.push_str(&parts.join("\t"));
            buf.push('\n');
            Ok(())
        });
        if let Ok(f) = print_fn {
            let _ = lua.globals().set("print", f);
        }

        // Belt-and-suspenders: nil out anything that shouldn't be reachable
        // even though StdLib flags already exclude these.
        for name in &["io", "os", "require", "dofile", "loadfile", "package", "debug"] {
            let _ = lua.globals().set(*name, Value::Nil);
        }

        // Run the code — on error, append the error to the output buffer
        // so the agent always sees partial print output + the error together.
        if let Err(e) = lua.load(code).exec() {
            let mut buf = output.lock().unwrap();
            if !buf.is_empty() && !buf.ends_with('\n') {
                buf.push('\n');
            }
            buf.push_str(&format!("[error] {e}"));
        }

        let mut result = output.lock().unwrap().clone();
        result = result.trim_end().to_string();

        if result.is_empty() {
            result = "(no output — use print() to show results)".to_string();
        } else if result.len() > MAX_OUTPUT {
            let end = result.floor_char_boundary(MAX_OUTPUT);
            result.truncate(end);
            result.push_str("\n[truncated]");
        }

        result
    }
}

#[derive(Deserialize)]
struct LuaEvalArgs {
    code: String,
}

#[async_trait]
impl super::Skill for LuaSkill {
    fn name(&self) -> &str {
        "Lua evaluator"
    }

    fn description(&self) -> Option<&str> {
        Some("Sandboxed Lua 5.4 for precise arithmetic and calculations")
    }

    async fn tool_definitions(&self) -> Vec<serde_json::Value> {
        vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "lua_eval",
                "description": "Evaluate a Lua 5.4 snippet. Sandboxed: no io/os/require/filesystem. Only math, string, table, utf8 stdlib. Use print() for output. Aborts after 10M instructions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Lua 5.4 code to run. Use print() to output values."
                        }
                    },
                    "required": ["code"]
                }
            }
        })]
    }

    async fn call(&self, tool_name: &str, args: &str, _ctx: &crate::channel::CallContext) -> Option<String> {
        if tool_name != "lua_eval" {
            return None;
        }
        let parsed: LuaEvalArgs = match serde_json::from_str(args) {
            Ok(a) => a,
            Err(e) => return Some(format!("Invalid arguments: {e}")),
        };
        info!(bytes = parsed.code.len(), "lua_eval");
        Some(Self::eval(&parsed.code))
    }
}

fn lua_value_to_string(v: &Value) -> String {
    match v {
        Value::Nil => "nil".to_string(),
        Value::Boolean(b) => b.to_string(),
        Value::Integer(i) => i.to_string(),
        Value::Number(n) => format!("{n}"),
        Value::String(s) => s.to_str().map(|s| s.to_string()).unwrap_or_else(|_| "(non-utf8)".to_string()),
        _ => "(value)".to_string(),
    }
}
