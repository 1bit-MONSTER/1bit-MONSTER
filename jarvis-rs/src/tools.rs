//! Tool invocation with explicit permission gating. Port of jarvis/tools.py.
//!
//! Every tool is "safe" (read-only, always runs) or "sensitive" (mutates
//! local state -- only runs if the caller's request explicitly opted in via
//! `allow_write: true`). Every call is appended to a local-only audit log --
//! nothing here is ever transmitted off-box.

use crate::rag::{audit_log_path, KnowledgeBase};
use crate::routing::MODEL_ROUTING;
use crate::time_util::now_iso;
use serde_json::{json, Value};
use std::io::Write;

pub const SYSTEM_PROMPT_TOOLS: &str = "You have access to tools, but only these four —
never invent a tool name that isn't listed here:
- search_knowledge(query: str) — search the local knowledge base
- get_time() — current local date/time
- list_models() — list available local models
- add_note(title: str, content: str) — save a fact/note to the local knowledge base (requires write permission)

Only call a tool when the task genuinely needs one of these four things
(looking something up, saving a note, checking the time/model list). For
anything else — including math, reasoning, writing, or general knowledge —
answer directly yourself; do not call a tool.

To use one of the four tools, respond with ONLY a single line:
TOOL_CALL: {\"name\": \"<tool_name>\", \"arguments\": {...}}";

#[derive(Clone)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

fn audit(kb: &KnowledgeBase, name: &str, arguments: &Value, allowed: bool, result_summary: &str) {
    let path = audit_log_path(&kb.root);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let entry = json!({
        "ts": now_iso(),
        "tool": name,
        "arguments": arguments,
        "allowed": allowed,
        "result_summary": result_summary,
    });
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(f, "{entry}");
    }
}

/// Brace-matched extraction of the first `TOOL_CALL: {...}` directive.
/// Brace-matched rather than regex-captured to the closing brace -- a model
/// that emits multiple TOOL_CALL lines (small models do) would otherwise
/// have its first call's JSON swallow everything up to the last line's
/// closing brace, producing invalid JSON that silently fails to parse.
pub fn parse_tool_call(text: &str) -> Option<ToolCall> {
    let marker = "TOOL_CALL:";
    let marker_pos = text.find(marker)?;
    let after = &text[marker_pos + marker.len()..];
    let brace_offset = after.find('{')?;
    let start = marker_pos + marker.len() + brace_offset;
    let bytes = text.as_bytes();
    let mut depth = 0i32;
    let mut end = None;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let end = end?;
    let json_str = &text[start..=end];
    let call: Value = serde_json::from_str(json_str).ok()?;
    let name = call.get("name")?.as_str()?.to_string();
    let arguments = call.get("arguments").cloned().unwrap_or_else(|| json!({}));
    Some(ToolCall { name, arguments })
}

pub fn format_tool_followup(result: &Value, allowed: bool) -> String {
    if !allowed {
        let reason = result.get("error").and_then(|e| e.as_str()).unwrap_or("not permitted");
        return format!(
            "Tool call was denied ({reason}). Do not attempt another tool call — answer the original question directly using your own knowledge instead."
        );
    }
    format!("Tool result: {result}\n\nNow answer the original question using this result.")
}

/// Execute a tool under the permission gate. Always audited. Returns
/// (result, allowed) exactly like the Python run_tool.
pub fn run_tool(kb: &KnowledgeBase, name: &str, arguments: Value, allow_write: bool) -> (Value, bool) {
    let is_sensitive = name == "add_note";
    let is_known = matches!(name, "search_knowledge" | "get_time" | "list_models" | "add_note");

    if !is_known {
        audit(kb, name, &arguments, false, "unknown tool");
        return (json!({ "error": format!("unknown tool: {name}") }), false);
    }

    if is_sensitive && !allow_write {
        audit(kb, name, &arguments, false, "blocked: write permission not granted");
        return (
            json!({ "error": format!("tool '{name}' requires write permission (pass allow_write: true)") }),
            false,
        );
    }

    let result = match name {
        "search_knowledge" => {
            let q = arguments.get("query").and_then(|v| v.as_str()).unwrap_or("");
            let results = kb.search(q, 5);
            json!({ "results": results })
        }
        "get_time" => json!({ "time": now_iso() }),
        "list_models" => {
            let models: Vec<&str> = MODEL_ROUTING.keys().copied().collect();
            json!({ "models": models })
        }
        "add_note" => {
            let title = arguments.get("title").and_then(|v| v.as_str()).unwrap_or("note");
            let content = arguments.get("content").and_then(|v| v.as_str()).unwrap_or("");
            match kb.add_document(title, content) {
                Ok(path) => json!({ "saved": path }),
                Err(e) => json!({ "error": e.to_string() }),
            }
        }
        _ => unreachable!(),
    };
    audit(kb, name, &arguments, true, &result.to_string().chars().take(200).collect::<String>());
    (result, true)
}
