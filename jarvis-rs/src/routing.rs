//! Outbound proxying to the NPU/unified/GPU backends. Port of jarvis/routing.py.
//! All three backends are already separate HTTP servers (npu_xrt/FLM bridge,
//! tools/unified_server.cpp, Ollama) — this module just forwards to them,
//! same as the Python version did with urllib.

use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::LazyLock;
use tokio::sync::mpsc;

pub const NPU_URL: &str = "http://127.0.0.1:52625";
pub const OLLAMA_URL: &str = "http://127.0.0.1:11434";
// tools/unified_server.cpp — the native engine's own OpenAI-compatible server
// (npu_xrt/hip_gpu/mamba1_gpu/vulkan_gpu/cpu, auto-selected per model, live
// per-request model switching via the standard "model" field).
pub const UNIFIED_URL: &str = "http://127.0.0.1:8088";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Backend {
    Npu,
    NpuVision,
    Gpu,
    Unified,
}

#[derive(Clone)]
pub struct Route {
    pub backend: Backend,
    pub model: String, // flm_model / ollama_model / unified_model, whichever applies
}

pub static MODEL_ROUTING: LazyLock<HashMap<&'static str, Route>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    let add = |m: &mut HashMap<&'static str, Route>, id, backend, model: &str| {
        m.insert(id, Route { backend, model: model.to_string() });
    };
    add(&mut m, "qwen3:0.6b", Backend::Npu, "qwen3:0.6b");
    add(&mut m, "qwen3:1.7b", Backend::Npu, "qwen3:1.7b");
    add(&mut m, "qwen3:4b", Backend::Npu, "qwen3:4b");
    add(&mut m, "bonsai:1.7b", Backend::Npu, "bonsai:1.7b");
    add(&mut m, "gemma4:e2b", Backend::Npu, "gemma4:e2b");
    add(&mut m, "phi4-mini:4b", Backend::Npu, "phi4-mini:4b");
    add(&mut m, "qwen3.6:35b", Backend::Npu, "qwen3.6:35b");
    add(&mut m, "qwen3vl:4b", Backend::NpuVision, "qwen3vl-it:4b");
    add(&mut m, "qwen3-vl:4b", Backend::NpuVision, "qwen3vl-it:4b");
    add(&mut m, "qwen3.5:9b", Backend::Gpu, "qwen3.5:9b");
    add(&mut m, "llama3.1:8b", Backend::Gpu, "llama3.1:8b");
    add(&mut m, "deepseek-r1:8b", Backend::Gpu, "deepseek-r1:8b");
    add(&mut m, "qwen2.5:7b", Backend::Gpu, "qwen2.5:7b");
    add(&mut m, "mistral:7b", Backend::Gpu, "mistral:7b");
    add(&mut m, "gpt-oss:20b", Backend::Gpu, "gpt-oss:20b");
    add(&mut m, "llama3.2-vision", Backend::Gpu, "llama3.2-vision");
    // Zyphra family, served via tools/unified_server.cpp. blackmamba-1.5b/2.8b
    // are deliberately excluded -- no usable tokenizer vocab in that GGUF
    // conversion yet (issue #590), unified_server falls back to raw token IDs.
    add(&mut m, "zr1:1.5b", Backend::Unified, "ZR1 1.5B");
    add(&mut m, "zamba2:1.2b", Backend::Unified, "zamba2-1.2b-instruct-v2-q4_0");
    add(&mut m, "zamba2:2.7b", Backend::Unified, "zamba2-2.7b-instruct-v2-q4_0");
    add(&mut m, "zamba2:7b", Backend::Unified, "zamba2-7b-instruct-v2-q4_0");
    add(&mut m, "zaya1:8b", Backend::Unified, "ZAYA1 8B");
    m
});

pub async fn resolve_model(client: &reqwest::Client, model_id: &str) -> Route {
    if let Some(r) = MODEL_ROUTING.get(model_id) {
        return r.clone();
    }
    if let Ok(resp) = client
        .get(format!("{OLLAMA_URL}/api/tags"))
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await
        && let Ok(v) = resp.json::<Value>().await
            && let Some(models) = v.get("models").and_then(|m| m.as_array()) {
                for m in models {
                    if let Some(name) = m.get("name").and_then(|n| n.as_str())
                        && name.contains(model_id) {
                            return Route { backend: Backend::Gpu, model: model_id.to_string() };
                        }
                }
            }
    Route { backend: Backend::Npu, model: model_id.to_string() }
}

fn err(prefix: &str, e: impl std::fmt::Display) -> Value {
    json!({ "error": format!("{prefix}: {e}") })
}

pub async fn flm_chat(client: &reqwest::Client, model: &str, messages: &Value, max_tokens: i64, temp: f64) -> Value {
    let payload = json!({ "model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temp, "stream": false });
    match client
        .post(format!("{NPU_URL}/v1/chat/completions"))
        .json(&payload)
        .timeout(std::time::Duration::from_secs(120))
        .send()
        .await
    {
        Ok(resp) => resp.json::<Value>().await.unwrap_or_else(|e| err("NPU", e)),
        Err(e) => err("NPU", e),
    }
}

pub async fn unified_chat(client: &reqwest::Client, model: &str, messages: &Value, max_tokens: i64, temp: f64) -> Value {
    // Live model-switch on the server side means the first request for a
    // not-currently-active model pays a reload cost -- larger Zyphra sizes
    // (7B) can take well over a minute. Generous timeout, same as Python.
    let payload = json!({ "model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temp, "stream": false });
    match client
        .post(format!("{UNIFIED_URL}/v1/chat/completions"))
        .json(&payload)
        .timeout(std::time::Duration::from_secs(180))
        .send()
        .await
    {
        Ok(resp) => resp.json::<Value>().await.unwrap_or_else(|e| err("unified", e)),
        Err(e) => err("unified", e),
    }
}

pub async fn ollama_chat(client: &reqwest::Client, model: &str, messages: &Value, max_tokens: i64, temp: f64) -> Value {
    // /api/chat (not /api/generate) -- takes the full messages array natively.
    let payload = json!({
        "model": model, "messages": messages, "stream": false,
        "options": { "num_predict": max_tokens, "temperature": temp },
    });
    match client
        .post(format!("{OLLAMA_URL}/api/chat"))
        .json(&payload)
        .timeout(std::time::Duration::from_secs(300))
        .send()
        .await
    {
        Ok(resp) => match resp.json::<Value>().await {
            Ok(data) => {
                let content = data.get("message").and_then(|m| m.get("content")).cloned().unwrap_or(json!(""));
                json!({ "response": content })
            }
            Err(e) => err("GPU", e),
        },
        Err(e) => err("GPU", e),
    }
}

/// Streams `{"choices":[{"delta":{"content":...}}]}` chunks over the returned
/// channel, mirroring the Python generator in ollama_chat_stream.
pub fn ollama_chat_stream(
    client: reqwest::Client,
    model: String,
    messages: Value,
    max_tokens: i64,
    temp: f64,
) -> mpsc::Receiver<Value> {
    let (tx, rx) = mpsc::channel(32);
    tokio::spawn(async move {
        let payload = json!({
            "model": model, "messages": messages, "stream": true,
            "options": { "num_predict": max_tokens, "temperature": temp },
        });
        let resp = match client
            .post(format!("{OLLAMA_URL}/api/chat"))
            .json(&payload)
            .timeout(std::time::Duration::from_secs(300))
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.send(json!({ "error": e.to_string() })).await;
                return;
            }
        };
        let mut stream = resp.bytes_stream();
        let mut buf = String::new();
        use futures_util::StreamExt;
        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(_) => break,
            };
            buf.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(pos) = buf.find('\n') {
                let line = buf[..pos].trim().to_string();
                buf.drain(..=pos);
                if line.is_empty() {
                    continue;
                }
                let data: Value = match serde_json::from_str(&line) {
                    Ok(d) => d,
                    Err(_) => continue,
                };
                let content = data.get("message").and_then(|m| m.get("content")).and_then(|c| c.as_str()).unwrap_or("");
                let done = data.get("done").and_then(|d| d.as_bool()).unwrap_or(false);
                if !content.is_empty() {
                    let _ = tx.send(json!({ "choices": [{ "delta": { "content": content }, "index": 0 }] })).await;
                }
                if done {
                    return;
                }
            }
        }
    });
    rx
}
