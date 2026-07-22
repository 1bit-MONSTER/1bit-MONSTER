//! 1bit — terminal coding agent for the 1bit.systems NPU+GPU+CPU inference stack.
//!
//! A single static binary that replaces the old TypeScript wrapper (which depended
//! on pi's npm packages).  Handles NPU stack management, interactive chat, config,
//! and engine builds — all in one Rust binary with no Node.js dependency.
//!
//! Architecture:
//!   1bit chat        → interactive REPL → NPU API (onebitd proxy → bitnet_decode)
//!   1bit up          → spawn onebitd + bitnet_decode daemon
//!   1bit down        → kill NPU processes
//!   1bit status      → check NPU stack health
//!   1bit build       → compile NPU engine from source
//!   1bit config      → view / set settings

mod config;
mod npu;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::process::{Command, Stdio};
use std::time::Duration;
use tracing::error;

// ── CLI Definition ───────────────────────────────────────────────

const VERSION: &str = "0.1.0";
const BANNER: &str = r"
  ╔══════════════════════════════════════════╗
  ║                                          ║
  ║    ██   ██████╗  ██╗  ████████╗         ║
  ║    ██   ██╔══██╗  ██║  ╚══██╔══╝        ║
  ║    ██   ██████╔╝  ██║     ██║           ║
  ║    ██   ██╔══██╗  ██║     ██║           ║
  ║   ██████ ██████╔╝  ██║     ██║          ║
  ║   ╚═════ ╚═════╝   ╚═╝     ╚═╝          ║
  ║                                          ║
  ║       NPU-native coding agent            ║
  ║    50 TOPS · 94 tok/s · 0 cloud          ║
  ║              vVERSION                     ║
  ╚══════════════════════════════════════════╝
";

#[derive(Parser, Debug)]
#[command(
    name = "1bit",
    version = VERSION,
    about = "NPU-native coding agent for 1bit.systems",
    long_about = "Terminal coding agent for the 1bit.systems NPU+GPU+CPU inference stack.
Zero cloud, zero Python — runs entirely on your AMD Strix Halo NPU."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Prompt to send directly (non-interactive)
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    prompt: Vec<String>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start interactive agent session (default)
    Chat {
        /// Model to use
        #[arg(short, long)]
        model: Option<String>,
    },

    /// Start NPU stack (onebitd + bitnet_decode daemon)
    Up {
        /// Path to model file
        #[arg(short, long)]
        model: Option<String>,

        /// Path to bitnet_decode binary
        #[arg(long)]
        bitnet_decode: Option<String>,
    },

    /// Stop NPU stack
    Down,

    /// Show NPU stack status
    Status,

    /// Build NPU engine from source
    Build {
        /// Build directory
        #[arg(short, long, default_value = "engine/npu")]
        dir: String,
    },

    /// View or set configuration
    Config {
        /// Key to get or set (e.g. "theme", "npu.api_port")
        key: Option<String>,

        /// Value to set (omit to get current value)
        value: Option<String>,
    },
}

// ── Main ─────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "onebit=info".into()),
        )
        .init();

    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Chat { model }) => cmd_chat(model.as_deref()).await?,
        Some(Commands::Up { model, bitnet_decode }) => {
            cmd_up(model.as_deref(), bitnet_decode.as_deref()).await?
        }
        Some(Commands::Down) => cmd_down().await?,
        Some(Commands::Status) => cmd_status().await?,
        Some(Commands::Build { dir }) => cmd_build(dir)?,
        Some(Commands::Config { key, value }) => cmd_config(key.as_deref(), value.as_deref())?,
        None => {
            // No subcommand + trailing args = chat prompt
            if !cli.prompt.is_empty() {
                let prompt = cli.prompt.join(" ");
                cmd_chat_once(&prompt).await?;
            } else {
                cmd_chat(None).await?;
            }
        }
    }

    Ok(())
}

// ── Command: chat (interactive REPL) ─────────────────────────────

async fn cmd_chat(model_override: Option<&str>) -> Result<()> {
    let settings = config::Settings::load()?;
    let model = model_override
        .map(|s| s.to_string())
        .unwrap_or(settings.default_model.clone());

    println!("{}", BANNER.replace("VERSION", VERSION));

    // Check NPU health
    let client = npu::NpuClient::new(&settings.npu_endpoint)?;
    match client.health_check().await {
        Ok(true) => println!("  ✅ NPU stack is online — ask me anything.\n"),
        Ok(false) => {
            println!("  ℹ️  NPU stack is not running. Type /up to start it.\n");
        }
        Err(e) => {
            println!("  ⚠️  Could not reach NPU: {e}\n");
        }
    }

    // Initialize rustyline
    let mut rl = rustyline::DefaultEditor::new()
        .context("Failed to initialize readline editor")?;

    println!("  Type /help for commands, /exit to quit.\n");

    loop {
        let readline = rl.readline("1bit> ");
        match readline {
            Ok(line) => {
                let trimmed = line.trim().to_string();
                if trimmed.is_empty() {
                    continue;
                }
                rl.add_history_entry(&trimmed).ok();

                match handle_chat_command(&trimmed, &client, &model).await {
                    Action::Continue => continue,
                    Action::Break => break,
                    Action::Send => {
                        if !client.health_check().await.unwrap_or(false) {
                            println!("  ⚠️  NPU stack not running. Type /up to start.\n");
                            continue;
                        }
                        println!("  🤔 Thinking...");
                        match client
                            .chat(
                                &model,
                                vec![npu::Message {
                                    role: "user".into(),
                                    content: trimmed,
                                }],
                                None,
                            )
                            .await
                        {
                            Ok(response) => {
                                println!("\n  {}\n", response);
                            }
                            Err(e) => {
                                println!("  ⚠️  Error: {e}\n");
                            }
                        }
                    }
                }
            }
            Err(rustyline::error::ReadlineError::Interrupted)
            | Err(rustyline::error::ReadlineError::Eof) => {
                println!("\n  Goodbye.\n");
                break;
            }
            Err(e) => {
                error!("Readline error: {e}");
                break;
            }
        }
    }

    Ok(())
}

/// One-shot chat (for `1bit "prompt"` or `1bit -- "prompt"`)
async fn cmd_chat_once(prompt: &str) -> Result<()> {
    let settings = config::Settings::load()?;
    let client = npu::NpuClient::new(&settings.npu_endpoint)?;

    if !client.health_check().await.unwrap_or(false) {
        anyhow::bail!("NPU stack is not running. Start it with `1bit up`");
    }

    let response = client
        .chat(
            &settings.default_model,
            vec![npu::Message {
                role: "user".into(),
                content: prompt.to_string(),
            }],
            None,
        )
        .await?;

    println!("{}", response);
    Ok(())
}

enum Action {
    Continue,
    Break,
    Send,
}

async fn handle_chat_command(line: &str, _client: &npu::NpuClient, _model: &str) -> Action {
    let parts: Vec<&str> = line.trim().split_whitespace().collect();
    let verb = parts.first().copied().unwrap_or("");

    match verb {
        "/help" => {
            println!(
                "
  Commands:
    /help              Show this help
    /status            Check NPU stack status
    /up                Start NPU stack
    /down              Stop NPU stack
    /clear             Clear the screen
    /models            List available models
    /exit              Exit 1bit chat
  "
            );
            Action::Continue
        }
        "/status" => {
            match cmd_status().await {
                Ok(()) => {}
                Err(e) => println!("  ⚠️  Status error: {e}"),
            }
            Action::Continue
        }
        "/up" => {
            match cmd_up(None, None).await {
                Ok(()) => println!("  ✅ NPU stack started"),
                Err(e) => println!("  ⚠️  Error: {e}"),
            }
            Action::Continue
        }
        "/down" => {
            match cmd_down().await {
                Ok(()) => println!("  ✅ NPU stack stopped"),
                Err(e) => println!("  ⚠️  Error: {e}"),
            }
            Action::Continue
        }
        "/clear" => {
            print!("\x1B[2J\x1B[1;1H");
            println!("{}", BANNER.replace("VERSION", VERSION));
            Action::Continue
        }
        "/models" => {
            let settings = config::Settings::load().ok();
            let endpoint = settings
                .as_ref()
                .map(|s| s.npu_endpoint.clone())
                .unwrap_or_else(|| "http://127.0.0.1:9090/v1".to_string());
            match npu::NpuClient::new(&endpoint) {
                Ok(client) => match client.list_models().await {
                    Ok(models) => {
                        if models.is_empty() {
                            println!("  ℹ️  No models available");
                        } else {
                            println!("  Available models:");
                            for m in models {
                                println!("    • {m}");
                            }
                        }
                    }
                    Err(e) => println!("  ⚠️  {e}"),
                },
                Err(e) => println!("  ⚠️  {e}"),
            }
            Action::Continue
        }
        "/exit" | "/quit" => {
            println!("  Goodbye.\n");
            Action::Break
        }
        _ if verb.starts_with('/') => {
            println!("  Unknown command: {verb}. Try /help");
            Action::Continue
        }
        _ => Action::Send,
    }
}

// ── Command: up ──────────────────────────────────────────────────

async fn cmd_up(model: Option<&str>, bitnet_decode: Option<&str>) -> Result<()> {
    let settings = config::Settings::load()?;
    let api_port = settings.npu.api_port;

    println!("  🚀 Starting 1bit NPU stack...\n");

    // Check if port is already in use
    if is_port_in_use(api_port).await {
        println!("  ✅ onebitd already running on port {api_port}");
    } else {
        let decode_path = bitnet_decode
            .map(|s| s.to_string())
            .unwrap_or(settings.npu.bitnet_decode_path.clone());

        let model_path = model
            .map(|s| s.to_string())
            .unwrap_or_else(|| "./model.h1b".to_string());

        println!("  Starting onebitd (bitnet_decode proxy)...");
        let mut cmd = Command::new(&decode_path);
        cmd.arg(&model_path)
            .arg("--server")
            .arg(api_port.to_string())
            .arg("--bind")
            .arg("127.0.0.1")
            .stdout(Stdio::null())
            .stderr(Stdio::null());

        if settings.npu.tune_prefill {
            cmd.arg("--tune-prefill");
        }
        if let Some(v) = settings.npu.prefill_variant {
            cmd.arg("--prefill-variant").arg(v.to_string());
        }
        if settings.npu.fp16_weights {
            cmd.arg("--fp16-weights");
        }

        // Set ROCm env vars
        cmd.env("HSA_OVERRIDE_GFX_VERSION", "11.5.1");
        cmd.env("HSA_ENABLE_SDMA", "0");

        match cmd.spawn() {
            Ok(mut child) => {
                // Detach
                child.stdin = None;
                println!("  ✅ bitnet_decode started (pid {}) on port {api_port}", child.id());
            }
            Err(e) => {
                println!("  ⚠️  Failed to start bitnet_decode: {e}");
                println!("       Is rocm-cpp installed? Try `1bit build`");
            }
        }
    }

    // Wait briefly and check
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Verify it's running
    let client = npu::NpuClient::new(&settings.npu_endpoint)?;
    match client.health_check().await {
        Ok(true) => {
            let models = client.list_models().await.unwrap_or_default();
            println!(
                "  ✅ NPU API responding ({} model{})",
                models.len(),
                if models.len() == 1 { "" } else { "s" }
            );
        }
        Ok(false) => {
            println!("  ⚠️  NPU API not yet ready (still starting?)");
        }
        Err(e) => {
            println!("  ⚠️  Health check error: {e}");
        }
    }

    println!("\n  📍 API:       http://127.0.0.1:{api_port}/v1");
    println!("  📍 Health:    http://127.0.0.1:{api_port}/health\n");

    Ok(())
}

// ── Command: down ────────────────────────────────────────────────

async fn cmd_down() -> Result<()> {
    println!("  🛑 Stopping 1bit NPU stack...\n");

    let ports = [9090, 13305, 9000];
    let mut any_killed = false;

    for port in ports {
        // Try fuser first
        let output = Command::new("fuser")
            .args(["-k", &format!("{port}/tcp")])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();

        if let Ok(status) = output {
            if status.success() {
                println!("  ✅ Killed process on port {port}");
                any_killed = true;
            }
        }

        // Also try pkill -f bitnet_decode and onebitd
        for proc in &["bitnet_decode", "onebitd"] {
            let _ = Command::new("pkill")
                .args(["-f", proc])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
    }

    if any_killed {
        println!("\n  ✅ NPU stack stopped");
    } else {
        println!("  ℹ️  No NPU processes were running");
    }

    Ok(())
}

// ── Command: status ──────────────────────────────────────────────

async fn cmd_status() -> Result<()> {
    let settings = config::Settings::load()?;

    println!("  ┌─ 1bit NPU Stack Status ──────────────────────────┐\n");

    // Check NPU API health
    let client = npu::NpuClient::new(&settings.npu_endpoint)?;
    match client.health_check().await {
        Ok(true) => {
            println!("  ✅  NPU API (port {})     — running", settings.npu.api_port);
            match client.list_models().await {
                Ok(models) => {
                    if !models.is_empty() {
                        let display: Vec<&str> = models.iter().map(|s| s.as_str()).take(5).collect();
                        let suffix = if models.len() > 5 { "..." } else { "" };
                        println!("      Models: {}{}", display.join(", "), suffix);
                    }
                }
                Err(_) => {}
            }
        }
        Ok(false) => {
            println!("  ❌  NPU API (port {})     — not running", settings.npu.api_port);
        }
        Err(e) => {
            println!("  ❌  NPU API (port {})     — error: {e}", settings.npu.api_port);
        }
    }

    // Check other ports
    for &(port, name) in &[(13305, "Lemond (Chat UI)"), (9000, "Lemond WebSocket")] {
        if is_port_in_use(port).await {
            println!("  ✅  {name:<27} port {port} — running");
        } else {
            println!("  ❌  {name:<27} port {port} — not running");
        }
    }

    println!("\n  └──────────────────────────────────────────────────┘");
    println!("  📍 API:   http://127.0.0.1:{}/v1", settings.npu.api_port);
    println!("  📍 Chat:  http://127.0.0.1:13305/\n");

    Ok(())
}

// ── Command: build ───────────────────────────────────────────────

fn cmd_build(dir: &str) -> Result<()> {
    println!("  🔨 Building NPU engine from source...\n");

    // Look for build scripts
    let build_sh = format!("{dir}/build_npu.sh");
    let cmake_dir = dir;

    if std::path::Path::new(&build_sh).exists() {
        println!("  Found build_npu.sh — running...");
        let status = Command::new("bash")
            .arg(&build_sh)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .context("Failed to run build script")?;

        if status.success() {
            println!("\n  ✅ NPU engine build complete");
        } else {
            anyhow::bail!("Build script exited with status {status}");
        }
    } else if std::path::Path::new(cmake_dir).join("CMakeLists.txt").exists() {
        // cmake build
        println!("  Running cmake...");
        let status = Command::new("cmake")
            .args(["-B", "build", "-G", "Ninja"])
            .current_dir(cmake_dir)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .context("cmake failed")?;

        if !status.success() {
            anyhow::bail!("cmake configuration failed");
        }

        println!("  Running ninja...");
        let status = Command::new("ninja")
            .arg("-j")
            .arg(num_cpus().to_string())
            .current_dir(format!("{cmake_dir}/build"))
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .context("ninja build failed")?;

        if status.success() {
            println!("\n  ✅ NPU engine build complete");
        } else {
            anyhow::bail!("Build failed with status {status}");
        }
    } else {
        anyhow::bail!("No build script or CMakeLists.txt found in {dir}");
    }

    Ok(())
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

// ── Command: config ──────────────────────────────────────────────

fn cmd_config(key: Option<&str>, value: Option<&str>) -> Result<()> {
    let mut settings = config::Settings::load()?;

    match (key, value) {
        // Show all
        (None, None) => {
            println!("  1bit Configuration\n");
            println!("  Theme:                {}", settings.theme);
            println!("  Default provider:     {}", settings.default_provider);
            println!("  Default model:        {}", settings.default_model);
            println!("  NPU endpoint:         {}", settings.npu_endpoint);
            println!("  Thinking level:       {}", settings.thinking_level);
            println!(
                "  Packages:             {}",
                if settings.packages.is_empty() {
                    "none".to_string()
                } else {
                    settings.packages.join(", ")
                }
            );
            println!("\n  NPU settings:");
            println!("    bitnet_decode:      {}", settings.npu.bitnet_decode_path);
            println!("    daemon path:        {}", settings.npu.daemon_path);
            println!("    tune prefill:       {}", settings.npu.tune_prefill);
            println!(
                "    prefill variant:    {}",
                settings
                    .npu
                    .prefill_variant
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "auto".to_string())
            );
            println!("    fp16 weights:       {}", settings.npu.fp16_weights);
            println!("    API port:           {}", settings.npu.api_port);
            println!("    Lemond port:        {}", settings.npu.lemond_port);
            println!(
                "\n  Config file: {}",
                config::settings_path().unwrap_or_default().display()
            );
        }

        // Get single key
        (Some(k), None) => {
            match settings.get(k) {
                Ok(val) => println!("{k} = {val}"),
                Err(e) => println!("  ⚠️  {e}"),
            }
        }

        // Set key=value
        (Some(k), Some(v)) => {
            settings.set(k, v)?;
            println!("  ✅ {k} = {v}");
        }

        // Value without key (shouldn't happen via clap but handle it)
        (None, Some(v)) => {
            println!("  ⚠️  Cannot set value without a key: {v}");
            println!("       Usage: 1bit config <key> <value>");
        }
    }

    Ok(())
}

// ── Helpers ──────────────────────────────────────────────────────

/// Check if a TCP port is in use by attempting a connection.
async fn is_port_in_use(port: u16) -> bool {
    use tokio::net::TcpStream;
    TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .is_ok()
}
