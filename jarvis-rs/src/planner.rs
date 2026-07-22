//! Multi-step task planning across the local model roster. Port of
//! jarvis/planner.py.
//!
//! A small, fast model (qwen3:0.6b) decomposes the request into an ordered
//! list of subtasks. Each subtask is routed to whichever local model
//! actually fits it (vision -> qwen3vl, heavy reasoning -> the larger
//! model, everything else -> the fast default). Tool calls inside a subtask
//! go through the same permission gate as single-turn chat. The final step
//! synthesizes all subtask outputs into one answer.

use crate::rag::KnowledgeBase;
use crate::routing::{self, Backend};
use crate::tools::{format_tool_followup, parse_tool_call, run_tool, SYSTEM_PROMPT_TOOLS};
use regex::Regex;
use serde_json::{json, Value};
use std::sync::LazyLock;

const PLANNER_MODEL: &str = "qwen3:0.6b";
const SYNTH_MODEL: &str = "qwen3.6:35b";

static SUBTASK_MODEL_HINTS: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    vec![
        (Regex::new(r"(?i)\b(image|photo|picture|diagram|screenshot)\b").unwrap(), "qwen3vl:4b"),
        (Regex::new(r"(?i)\b(reason|analy[sz]e|compare|plan|strategy|deep|complex)\b").unwrap(), SYNTH_MODEL),
    ]
});

fn plan_prompt(request: &str) -> String {
    format!(
        "Break the following request into 2-5 concrete subtasks, ordered by\nexecution. Reply with ONLY a JSON list of strings, e.g.\n[\"look up X\", \"compute Y from the result\", \"summarize for the user\"].\nIf the request is a single simple step, reply with a one-item list.\n\nRequest: {request}"
    )
}

async fn chat_one(client: &reqwest::Client, model_id: &str, messages: Value, max_tokens: i64) -> String {
    let route = routing::resolve_model(client, model_id).await;
    if route.backend == Backend::Gpu {
        let r = routing::ollama_chat(client, &route.model, &messages, max_tokens, 0.7).await;
        return match r.get("error") {
            Some(e) => format!("[error: {e}]"),
            None => r.get("response").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        };
    }
    let r = routing::flm_chat(client, &route.model, &messages, max_tokens, 0.7).await;
    if let Some(e) = r.get("error") {
        return format!("[error: {e}]");
    }
    r.get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string()
}

fn pick_model(subtask_text: &str) -> &'static str {
    for (re, model_id) in SUBTASK_MODEL_HINTS.iter() {
        if re.is_match(subtask_text) {
            return model_id;
        }
    }
    PLANNER_MODEL
}

async fn make_plan(client: &reqwest::Client, request: &str) -> Vec<String> {
    let msgs = json!([{ "role": "user", "content": plan_prompt(request) }]);
    let raw = chat_one(client, PLANNER_MODEL, msgs, 200).await;
    let re = Regex::new(r"(?s)\[.*\]").unwrap();
    let Some(m) = re.find(&raw) else { return vec![request.to_string()] };
    match serde_json::from_str::<Vec<Value>>(m.as_str()) {
        Ok(steps) => {
            let steps: Vec<String> = steps
                .into_iter()
                .filter_map(|s| s.as_str().map(|s| s.trim().to_string()))
                .filter(|s| !s.is_empty())
                .take(5)
                .collect();
            if steps.is_empty() {
                vec![request.to_string()]
            } else {
                steps
            }
        }
        Err(_) => vec![request.to_string()],
    }
}

pub struct StepResult {
    pub step: String,
    pub model: String,
    pub tool_call: Option<Value>, // {"name":..,"allowed":..,"result":..}
    pub output: String,
}

async fn run_step(client: &reqwest::Client, kb: &KnowledgeBase, step: &str, allow_write: bool) -> StepResult {
    let model_id = pick_model(step);
    let msgs = json!([
        { "role": "system", "content": SYSTEM_PROMPT_TOOLS },
        { "role": "user", "content": step },
    ]);
    let mut reply = chat_one(client, model_id, msgs.clone(), 300).await;

    if let Some(call) = parse_tool_call(&reply) {
        let (result, allowed) = run_tool(kb, &call.name, call.arguments.clone(), allow_write);
        let mut follow = msgs.as_array().unwrap().clone();
        follow.push(json!({ "role": "assistant", "content": reply }));
        follow.push(json!({ "role": "user", "content": format_tool_followup(&result, allowed) }));
        reply = chat_one(client, model_id, Value::Array(follow), 300).await;
        return StepResult {
            step: step.to_string(),
            model: model_id.to_string(),
            tool_call: Some(json!({ "name": call.name, "allowed": allowed, "result": result })),
            output: reply,
        };
    }

    StepResult { step: step.to_string(), model: model_id.to_string(), tool_call: None, output: reply }
}

pub async fn run_plan(client: &reqwest::Client, kb: &KnowledgeBase, request: &str, allow_write: bool) -> Value {
    let plan = make_plan(client, request).await;
    let mut step_results = Vec::with_capacity(plan.len());
    for s in &plan {
        step_results.push(run_step(client, kb, s, allow_write).await);
    }

    // Always synthesize, even for a single-step plan: a subtask model's raw
    // reply can ignore or misreport a tool result it just received (small/
    // fast models do this), so grounding synthesis in the actual tool
    // output (not just the subtask model's paraphrase) matters here too.
    let fmt = |r: &StepResult| {
        let mut line = format!("- {}: {}", r.step, r.output);
        if let Some(tc) = &r.tool_call
            && tc.get("allowed").and_then(|a| a.as_bool()).unwrap_or(false) {
                line.push_str(&format!("\n  (tool {} actually returned: {})", tc["name"], tc["result"]));
            }
        line
    };
    let synthesis_input = step_results.iter().map(fmt).collect::<Vec<_>>().join("\n");
    let synth_msgs = json!([{
        "role": "user",
        "content": format!(
            "Original request: {request}\n\nSubtask results:\n{synthesis_input}\n\nGive one concise final answer to the original request."
        ),
    }]);
    let final_answer = chat_one(client, SYNTH_MODEL, synth_msgs, 400).await;

    json!({
        "plan": plan,
        "steps": step_results.iter().map(|r| json!({
            "step": r.step,
            "model": r.model,
            "tool_call": r.tool_call,
            "output": r.output,
        })).collect::<Vec<_>>(),
        "answer": final_answer,
    })
}
