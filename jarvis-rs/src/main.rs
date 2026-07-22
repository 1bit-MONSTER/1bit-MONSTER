mod audio_out;
mod beacon;
mod handlers;
mod multipart_util;
mod planner;
mod rag;
mod routing;
mod state;
mod stt;
mod time_util;
mod tools;
mod tts;
mod ui;
mod voice;

use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use axum::Router;
use clap::Parser;
use state::AppState;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};

const MAX_BODY_SIZE: usize = 16 * 1024 * 1024;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = default_port())]
    port: u16,
    /// Disable the LAN auto-discovery UDP broadcast.
    #[arg(long)]
    no_beacon: bool,
}

fn default_port() -> u16 {
    std::env::var("JARVIS_PORT").ok().and_then(|v| v.parse().ok()).unwrap_or(8080)
}

fn knowledge_dir() -> std::path::PathBuf {
    std::env::var("JARVIS_KNOWLEDGE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            std::path::PathBuf::from(home).join("jarvis/data/knowledge")
        })
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let kb = rag::KnowledgeBase::new(knowledge_dir()).expect("failed to initialize knowledge base directory");
    let client = reqwest::Client::builder().build().expect("failed to build HTTP client");

    let packs_dir = {
        // jarvis/voice/engine.py: Path(__file__).parent.parent.parent / "voice" / "packs"
        // i.e. <repo_root>/voice/packs, relative to wherever this binary is run from.
        std::path::PathBuf::from("voice/packs")
    };
    let mut voice_engine = voice::VoiceEngine::new(packs_dir);
    let packs = voice_engine.list_available_packs();
    if let Some(first) = packs.first() {
        if let Some(path) = first.get("path").and_then(|p| p.as_str()) {
            match voice_engine.load_pack(std::path::Path::new(path)) {
                Ok(name) => println!("  Voice engine: {name} loaded"),
                Err(e) => println!("  Voice engine: no pack loaded ({e})"),
            }
        }
    } else {
        println!("  Voice engine: no voice packs found (run jarvis/voice/record.py)");
    }

    let state = Arc::new(AppState { client, kb: Arc::new(kb), voice: tokio::sync::Mutex::new(voice_engine) });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([axum::http::Method::GET, axum::http::Method::POST, axum::http::Method::OPTIONS])
        .allow_headers(Any);

    let app = Router::new()
        .route("/", get(handlers::chat_page))
        .route("/chat", get(handlers::chat_page))
        .route("/health", get(handlers::health))
        .route("/live", get(handlers::health))
        .route("/v1/audio/devices", get(handlers::audio_devices))
        .route("/v1/models", get(handlers::list_models))
        .route("/v1/voice/packs", get(handlers::voice_packs_list).post(handlers::voice_packs_post))
        .route("/v1/voice/activate", post(handlers::voice_activate))
        .route("/v1/knowledge", get(handlers::knowledge_list))
        .route("/v1/audio/transcriptions", post(handlers::audio_transcriptions))
        .route("/v1/audio/speech", post(handlers::audio_speech))
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/api/chat", post(handlers::chat_completions))
        .route("/v1/knowledge/search", post(handlers::knowledge_search))
        .route("/v1/knowledge/upload", post(handlers::knowledge_upload))
        .route("/v1/agent/plan", post(handlers::agent_plan))
        .layer(DefaultBodyLimit::max(MAX_BODY_SIZE))
        .layer(cors)
        .with_state(state);

    let spk = audio_out::find_external_speaker().await;
    match &spk {
        Some(d) => println!("  Speaker: {} ({})", d.name, d.alsa_id),
        None => println!("  Speaker: none connected -- TTS replies stay silent locally until one is plugged in"),
    }

    if !args.no_beacon {
        beacon::start_beacon(args.port, std::time::Duration::from_secs_f64(3.0)).await;
    }

    println!("JARVIS (rust) @ http://localhost:{}/chat", args.port);

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", args.port)).await.expect("failed to bind port");
    axum::serve(listener, app).await.expect("server error");
}
