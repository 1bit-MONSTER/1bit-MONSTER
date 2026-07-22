//! Piper TTS fallback. Port of jarvis/tts.py -- needs no ML runtime port at
//! all, it already just shells out to the `piper` CLI; this does the same.

use std::process::Stdio;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

fn piper_voices_dir() -> String {
    std::env::var("PIPER_VOICES_DIR").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        format!("{home}/piper-voices")
    })
}

fn jarvis_venv() -> String {
    std::env::var("JARVIS_VENV").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        format!("{home}/jarvis-env")
    })
}

/// Synthesizes `text` via Piper, returning a complete WAV file (mono,
/// 16-bit, 22050 Hz -- Piper's `--output-raw` gives headerless s16le PCM at
/// that rate, so a WAV header is added here exactly like the Python version
/// did with the `wave` module).
pub async fn synthesize_speech(text: &str, voice: &str) -> Option<Vec<u8>> {
    let model_path = format!("{}/{voice}.onnx", piper_voices_dir());
    if !std::path::Path::new(&model_path).exists() {
        return None;
    }
    let venv_piper = format!("{}/bin/piper", jarvis_venv());
    let piper_bin = if std::path::Path::new(&venv_piper).exists() { venv_piper } else { "piper".to_string() };

    let mut child = Command::new(&piper_bin)
        .args(["--model", &model_path, "--output-raw"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;

    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(text.as_bytes()).await;
    }
    let output = tokio::time::timeout(std::time::Duration::from_secs(30), child.wait_with_output())
        .await
        .ok()?
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let pcm = output.stdout;
    let spec = hound::WavSpec { channels: 1, sample_rate: 22050, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut buf = std::io::Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut buf, spec).ok()?;
        for chunk in pcm.chunks_exact(2) {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            writer.write_sample(sample).ok()?;
        }
        writer.finalize().ok()?;
    }
    Some(buf.into_inner())
}
