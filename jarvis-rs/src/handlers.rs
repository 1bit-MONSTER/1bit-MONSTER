//! HTTP request handlers. Port of jarvis/server.py's `class H` dispatch
//! table -- endpoint paths, JSON shapes, and SSE framing all match exactly
//! so 1bit Mobile and the /chat web UI need zero client-side changes.

use crate::audio_out;
use crate::planner;
use crate::routing::{self, Backend};
use crate::state::SharedState;
use crate::tools::{format_tool_followup, parse_tool_call, run_tool, SYSTEM_PROMPT_TOOLS};
use crate::ui::CHAT_HTML;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

fn html(body: &'static str) -> Response {
    (
        StatusCode::OK,
        [("Content-Type", "text/html; charset=utf-8")],
        body,
    )
        .into_response()
}

fn j(status: StatusCode, body: Value) -> Response {
    (status, Json(body)).into_response()
}

// ── GET handlers ─────────────────────────────────────────────────────────

pub async fn chat_page() -> Response {
    html(CHAT_HTML)
}

pub async fn health() -> Response {
    j(StatusCode::OK, json!({ "status": "ok" }))
}

pub async fn audio_devices() -> Response {
    let spk = audio_out::find_external_speaker().await;
    let devices = audio_out::list_playback_devices().await;
    j(
        StatusCode::OK,
        json!({
            "devices": devices,
            "external_speaker_connected": spk.is_some(),
            "active": spk,
        }),
    )
}

pub async fn list_models() -> Response {
    let data: Vec<Value> = routing::MODEL_ROUTING
        .iter()
        .map(|(id, route)| {
            let backend = match route.backend {
                Backend::Npu => "npu",
                Backend::NpuVision => "npu_vision",
                Backend::Gpu => "gpu",
                Backend::Unified => "unified",
            };
            json!({ "id": id, "object": "model", "backend": backend })
        })
        .collect();
    j(StatusCode::OK, json!({ "data": data }))
}

/// Voice pack listing/loading/activation are part of the custom
/// voice-cloning engine (jarvis/voice/engine.py), ported in Phase 3 of the
/// Rust-port plan (candle-based codec decoder). Stubbed here with an
/// honest empty/unavailable response rather than silently 404ing, so
/// callers can distinguish "no packs yet" from "endpoint doesn't exist".
pub async fn voice_packs_list() -> Response {
    j(StatusCode::OK, json!({ "voices": [], "active": Value::Null }))
}

pub async fn voice_packs_post() -> Response {
    j(StatusCode::NOT_IMPLEMENTED, json!({ "error": "voice pack loading lands in Phase 3 of the Rust port" }))
}

pub async fn voice_activate() -> Response {
    j(StatusCode::NOT_IMPLEMENTED, json!({ "error": "voice pack loading lands in Phase 3 of the Rust port" }))
}

/// STT lands in Phase 2 (whisper-rs). Stubbed for now.
pub async fn audio_transcriptions() -> Response {
    j(StatusCode::NOT_IMPLEMENTED, json!({ "error": "STT lands in Phase 2 of the Rust port (whisper-rs)" }))
}

/// TTS lands in Phase 3 (Piper subprocess + candle codec decoder). Stubbed
/// for now -- the Python server remains the source of truth for
/// /v1/audio/speech (and hence the physical-speaker demo) until then.
pub async fn audio_speech() -> Response {
    j(StatusCode::NOT_IMPLEMENTED, json!({ "error": "TTS lands in Phase 3 of the Rust port" }))
}

pub async fn knowledge_list(State(state): State<SharedState>) -> Response {
    let mut entries = Vec::new();
    for f in state.kb.all_files() {
        let Ok(text) = std::fs::read_to_string(&f) else { continue };
        let mut title = String::new();
        let mut tags: Vec<String> = Vec::new();
        for l in text.lines() {
            if let Some(rest) = l.strip_prefix("# ") {
                title = rest.trim().to_string();
            }
            if let Some(rest) = l.strip_prefix("tags:") {
                tags = rest
                    .trim()
                    .trim_start_matches('[')
                    .trim_end_matches(']')
                    .split(',')
                    .map(|x| x.trim().trim_matches('\'').to_string())
                    .filter(|x| !x.is_empty())
                    .collect();
            }
        }
        let rel = f.strip_prefix(&state.kb.root).unwrap_or(&f).to_string_lossy().to_string();
        let category = rel.split('/').next().unwrap_or("").to_string();
        let name = f.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        entries.push(json!({
            "path": rel,
            "title": if title.is_empty() { name } else { title },
            "category": category,
            "tags": tags,
            "size": text.len(),
        }));
    }
    j(StatusCode::OK, json!({ "entries": entries }))
}

// ── POST handlers ────────────────────────────────────────────────────────

pub async fn knowledge_search(State(state): State<SharedState>, Json(body): Json<Value>) -> Response {
    let query = body.get("query").and_then(|v| v.as_str()).unwrap_or("");
    let max_results = body.get("max_results").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
    let results = state.kb.search(query, max_results);
    j(StatusCode::OK, json!({ "results": results }))
}

pub async fn knowledge_upload(State(state): State<SharedState>, headers: HeaderMap, body: Bytes) -> Response {
    let content_type = headers.get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("");
    if content_type.contains("multipart") {
        let Some(boundary) = crate::multipart_util::boundary_from_content_type(content_type) else {
            return j(StatusCode::BAD_REQUEST, json!({ "error": "no boundary" }));
        };
        let mut filename = "doc.txt".to_string();
        let content = crate::multipart_util::extract_file_part(&body, &boundary, b"Content-Disposition");
        if let Some(fname) = find_filename_in_body(&body, &boundary) {
            filename = fname;
        }
        let Some(content) = content else {
            return j(StatusCode::BAD_REQUEST, json!({ "error": "empty" }));
        };
        let content_str = String::from_utf8_lossy(&content).to_string();
        return match state.kb.add_document(&filename, &content_str) {
            Ok(path) => j(StatusCode::OK, json!({ "path": path })),
            Err(e) => j(StatusCode::INTERNAL_SERVER_ERROR, json!({ "error": e.to_string() })),
        };
    }
    let Ok(d) = serde_json::from_slice::<Value>(&body) else {
        return j(StatusCode::BAD_REQUEST, json!({ "error": "json" }));
    };
    let filename = d.get("filename").and_then(|v| v.as_str()).unwrap_or("note.md");
    let content = d.get("content").and_then(|v| v.as_str()).unwrap_or("");
    match state.kb.add_document(filename, content) {
        Ok(path) => j(StatusCode::OK, json!({ "path": path })),
        Err(e) => j(StatusCode::INTERNAL_SERVER_ERROR, json!({ "error": e.to_string() })),
    }
}

fn find_filename_in_body(body: &[u8], boundary: &str) -> Option<String> {
    let delim = format!("--{boundary}");
    let delim = delim.as_bytes();
    let text = body;
    let mut start = 0;
    while let Some(pos) = windows_find(&text[start..], delim) {
        let part_start = start + pos + delim.len();
        let part_end = windows_find(&text[part_start..], delim).map(|p| part_start + p).unwrap_or(text.len());
        let part = &text[part_start..part_end];
        if windows_find(part, b"Content-Disposition").is_some()
            && let Some(fname) = crate::multipart_util::extract_filename(part) {
                return Some(fname);
            }
        start = part_end;
        if part_end >= text.len() {
            break;
        }
    }
    None
}

fn windows_find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

pub async fn agent_plan(State(state): State<SharedState>, Json(body): Json<Value>) -> Response {
    let Some(request) = body.get("request").and_then(|v| v.as_str()).filter(|s| !s.is_empty()) else {
        return j(StatusCode::BAD_REQUEST, json!({ "error": "request required" }));
    };
    let allow_write = body.get("allow_write").and_then(|v| v.as_bool()).unwrap_or(false);
    let session_id = body.get("session_id").and_then(|v| v.as_str()).unwrap_or("default");
    let _ = state.kb.save_turn(session_id, "user", request);
    let result = planner::run_plan(&state.client, &state.kb, request, allow_write).await;
    if let Some(answer) = result.get("answer").and_then(|v| v.as_str()) {
        let _ = state.kb.save_turn(session_id, "assistant", answer);
    }
    j(StatusCode::OK, result)
}

// ── /v1/chat/completions ────────────────────────────────────────────────

fn sse_single(content: &str) -> Response {
    let chunk = format!(
        "data: {}\n\ndata: [DONE]\n\n",
        json!({ "choices": [{ "delta": { "content": content }, "index": 0 }] })
    );
    (
        StatusCode::OK,
        [("Content-Type", "text/event-stream")],
        chunk,
    )
        .into_response()
}

async fn resolve_tool_call(
    state: &SharedState,
    model_id: &str,
    msgs: &[Value],
    content: String,
    max_tokens: i64,
    temp: f64,
    allow_write: bool,
    backend: Backend,
) -> (String, Option<Value>) {
    let Some(call) = parse_tool_call(&content) else {
        return (content, None);
    };
    let (result, allowed) = run_tool(&state.kb, &call.name, call.arguments.clone(), allow_write);
    let mut follow: Vec<Value> = msgs.to_vec();
    follow.push(json!({ "role": "assistant", "content": content }));
    follow.push(json!({ "role": "user", "content": format_tool_followup(&result, allowed) }));
    let follow_v = Value::Array(follow);
    let r2 = match backend {
        Backend::Unified => routing::unified_chat(&state.client, model_id, &follow_v, max_tokens, temp).await,
        _ => routing::flm_chat(&state.client, model_id, &follow_v, max_tokens, temp).await,
    };
    let final_content = if r2.get("error").is_none() {
        r2.get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or(&content)
            .to_string()
    } else {
        content
    };
    (final_content, Some(json!({ "name": call.name, "allowed": allowed, "result": result })))
}

#[allow(clippy::too_many_arguments)]
async fn npu_or_unified_chat(
    state: &SharedState,
    model: &str,
    msgs: &[Value],
    stream: bool,
    max_tokens: i64,
    temp: f64,
    session_id: &str,
    use_tools: bool,
    allow_write: bool,
    backend: Backend,
) -> Response {
    let msgs_v = Value::Array(msgs.to_vec());
    let r = match backend {
        Backend::Unified => routing::unified_chat(&state.client, model, &msgs_v, max_tokens, temp).await,
        _ => routing::flm_chat(&state.client, model, &msgs_v, max_tokens, temp).await,
    };
    if r.get("error").is_some() {
        return j(StatusCode::BAD_GATEWAY, r);
    }
    let content = r
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();
    let mut tool_info = None;
    let mut c = content;
    if use_tools {
        let (nc, ti) = resolve_tool_call(state, model, msgs, c, max_tokens, temp, allow_write, backend).await;
        c = nc;
        tool_info = ti;
    }
    if !c.is_empty() {
        let _ = state.kb.save_turn(session_id, "assistant", &c);
    }
    if stream {
        sse_single(&c)
    } else if let Some(ti) = tool_info {
        j(
            StatusCode::OK,
            json!({ "tool_call": ti, "choices": [{ "index": 0, "message": { "role": "assistant", "content": c }, "finish_reason": "stop" }] }),
        )
    } else {
        j(StatusCode::OK, r)
    }
}

async fn vis_chat(state: &SharedState, model: &str, msgs: &[Value], stream: bool, max_tokens: i64, temp: f64) -> Response {
    let mut parts_text = Vec::new();
    for m in msgs {
        let content = m.get("content").cloned().unwrap_or(Value::Null);
        let parts: Vec<Value> = match content {
            Value::Array(a) => a,
            other => vec![other],
        };
        for p in parts {
            if let Some(obj) = p.as_object() {
                parts_text.push(obj.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string());
            } else if let Some(s) = p.as_str() {
                parts_text.push(s.to_string());
            } else if p.is_null() {
                parts_text.push(String::new());
            } else {
                parts_text.push(p.to_string());
            }
        }
    }
    let mut txt = parts_text.join("\n");
    if txt.trim().is_empty() {
        txt = "Describe this image.".to_string();
    }
    let flat_msgs: Vec<Value> = msgs
        .iter()
        .map(|m| json!({ "role": m.get("role").cloned().unwrap_or(json!("user")), "content": txt }))
        .collect();
    let r = routing::flm_chat(&state.client, model, &Value::Array(flat_msgs), max_tokens, temp).await;
    if r.get("error").is_some() {
        return j(StatusCode::BAD_GATEWAY, r);
    }
    if stream {
        let content = r
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("");
        sse_single(content)
    } else {
        j(StatusCode::OK, r)
    }
}

#[allow(clippy::too_many_arguments)]
async fn gpu_chat(
    state: &SharedState,
    model: &str,
    msgs: &[Value],
    stream: bool,
    max_tokens: i64,
    temp: f64,
    session_id: &str,
    use_tools: bool,
    allow_write: bool,
) -> Response {
    if stream {
        let rx = routing::ollama_chat_stream(state.client.clone(), model.to_string(), Value::Array(msgs.to_vec()), max_tokens, temp);
        let kb = state.kb.clone();
        let session_id = session_id.to_string();
        let stream_body = async_stream::stream! {
            let mut rx = rx;
            let mut full = String::new();
            while let Some(chunk) = rx.recv().await {
                if let Some(delta) = chunk.get("choices").and_then(|c| c.get(0)).and_then(|c| c.get("delta")).and_then(|d| d.get("content")).and_then(|c| c.as_str()) {
                    full.push_str(delta);
                }
                let line = format!("data: {chunk}\n\n");
                yield Ok::<_, std::io::Error>(Bytes::from(line));
            }
            yield Ok::<_, std::io::Error>(Bytes::from("data: [DONE]\n\n"));
            if !full.is_empty() {
                let _ = kb.save_turn(&session_id, "assistant", &full);
            }
        };
        return (
            StatusCode::OK,
            [("Content-Type", "text/event-stream")],
            Body::from_stream(stream_body),
        )
            .into_response();
    }

    let r = routing::ollama_chat(&state.client, model, &Value::Array(msgs.to_vec()), max_tokens, temp).await;
    if r.get("error").is_some() {
        return j(StatusCode::BAD_GATEWAY, r);
    }
    let mut c = r.get("response").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let mut tool_info = None;
    if use_tools
        && let Some(call) = parse_tool_call(&c) {
            let (result, allowed) = run_tool(&state.kb, &call.name, call.arguments.clone(), allow_write);
            let mut follow: Vec<Value> = msgs.to_vec();
            follow.push(json!({ "role": "assistant", "content": c }));
            follow.push(json!({ "role": "user", "content": format_tool_followup(&result, allowed) }));
            let r2 = routing::ollama_chat(&state.client, model, &Value::Array(follow), max_tokens, temp).await;
            if r2.get("error").is_none() {
                c = r2.get("response").and_then(|v| v.as_str()).unwrap_or(&c).to_string();
            }
            tool_info = Some(json!({ "name": call.name, "allowed": allowed, "result": result }));
        }
    if !c.is_empty() {
        let _ = state.kb.save_turn(session_id, "assistant", &c);
    }
    let mut out = json!({
        "id": format!("c-{}", &uuid::Uuid::new_v4().simple().to_string()[..8]),
        "object": "chat.completion",
        "model": model,
        "choices": [{ "index": 0, "message": { "role": "assistant", "content": c }, "finish_reason": "stop" }],
    });
    if let Some(ti) = tool_info {
        out["tool_call"] = ti;
    }
    j(StatusCode::OK, out)
}

pub async fn chat_completions(State(state): State<SharedState>, Json(body): Json<Value>) -> Response {
    let mid_raw = body.get("model").and_then(|v| v.as_str()).map(|s| s.to_string());
    let mut msgs: Vec<Value> = body.get("messages").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    let stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let max_tokens = body.get("max_tokens").and_then(|v| v.as_i64()).unwrap_or(256);
    let temp = body.get("temperature").and_then(|v| v.as_f64()).unwrap_or(0.7);
    let rag = body.get("rag").and_then(|v| v.as_bool()).unwrap_or(true);
    let session_id = body.get("session_id").and_then(|v| v.as_str()).unwrap_or("default").to_string();
    let use_tools = body.get("tools").and_then(|v| v.as_bool()).unwrap_or(true) && !stream;
    let allow_write = body.get("allow_write").and_then(|v| v.as_bool()).unwrap_or(false);

    // ── Local multi-turn memory: recall this session's prior turns
    // server-side, independent of whatever history the client itself sent.
    let history = state.kb.get_recent_conversation(&session_id, 10);
    if !history.is_empty() {
        let mut merged: Vec<Value> = history.iter().map(|t| json!({ "role": t.role, "content": t.content })).collect();
        merged.extend(msgs);
        msgs = merged;
    }
    let user_q = msgs
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user") && m.get("content").map(|c| c.is_string()).unwrap_or(false))
        .and_then(|m| m.get("content").and_then(|c| c.as_str()))
        .map(|s| s.to_string());
    if let Some(q) = &user_q {
        let _ = state.kb.save_turn(&session_id, "user", q);
    }

    let has_tool_system = msgs.iter().any(|m| {
        m.get("role").and_then(|r| r.as_str()) == Some("system")
            && m.get("content").and_then(|c| c.as_str()).map(|c| c.contains("TOOL_CALL")).unwrap_or(false)
    });
    if use_tools && !has_tool_system {
        msgs.insert(0, json!({ "role": "system", "content": SYSTEM_PROMPT_TOOLS }));
    }

    let mid = match mid_raw {
        Some(m) if m != "auto" => m,
        _ => {
            let has_img = msgs.iter().any(|m| m.get("content").map(|c| c.is_array()).unwrap_or(false));
            if has_img {
                "qwen3vl:4b".to_string()
            } else {
                let ln: usize = msgs.iter().filter_map(|m| m.get("content").and_then(|c| c.as_str())).map(|s| s.len()).sum();
                if ln < 500 { "qwen3:0.6b".to_string() } else { "qwen3.5:9b".to_string() }
            }
        }
    };

    if rag {
        let q = msgs
            .iter()
            .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
            .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
            .next_back()
            .map(|s| s.to_string());
        if let Some(q) = q
            && !q.is_empty() {
                let ctx = state.kb.get_knowledge_context(&q, 3);
                if !ctx.is_empty()
                    && let Some(m) = msgs.iter_mut().find(|m| {
                        m.get("role").and_then(|r| r.as_str()) == Some("user") && m.get("content").map(|c| c.is_string()).unwrap_or(false)
                    }) {
                        let orig = m.get("content").and_then(|c| c.as_str()).unwrap_or("").to_string();
                        m["content"] = json!(format!("{ctx}\n\nQ: {orig}"));
                    }
            }
    }

    let route = routing::resolve_model(&state.client, &mid).await;
    match route.backend {
        Backend::NpuVision => vis_chat(&state, &route.model, &msgs, stream, max_tokens, temp).await,
        Backend::Gpu => gpu_chat(&state, &route.model, &msgs, stream, max_tokens, temp, &session_id, use_tools, allow_write).await,
        Backend::Unified => {
            npu_or_unified_chat(&state, &route.model, &msgs, stream, max_tokens, temp, &session_id, use_tools, allow_write, Backend::Unified).await
        }
        Backend::Npu => {
            npu_or_unified_chat(&state, &route.model, &msgs, stream, max_tokens, temp, &session_id, use_tools, allow_write, Backend::Npu).await
        }
    }
}
