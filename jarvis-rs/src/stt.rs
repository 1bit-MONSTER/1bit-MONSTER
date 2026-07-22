//! Speech-to-text. Port of jarvis/stt.py, swapping faster-whisper
//! (CTranslate2 "tiny"/int8/CPU) for whisper-rs (whisper.cpp bindings) with
//! the equivalent ggml "tiny" model -- same size class, same CPU-only
//! inference, no language pinned (multilingual auto-detect either way).

use hound::WavReader;
use std::sync::OnceLock;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

static WHISPER_CTX: OnceLock<Option<WhisperContext>> = OnceLock::new();

fn model_path() -> String {
    std::env::var("WHISPER_MODEL_PATH").unwrap_or_else(|_| "models/ggml-tiny.bin".to_string())
}

fn get_context() -> Option<&'static WhisperContext> {
    WHISPER_CTX
        .get_or_init(|| {
            let path = model_path();
            match WhisperContext::new_with_params(&path, WhisperContextParameters::default()) {
                Ok(ctx) => Some(ctx),
                Err(e) => {
                    eprintln!("  STT: failed to load whisper model at {path}: {e}");
                    None
                }
            }
        })
        .as_ref()
}

/// Decodes WAV bytes (any sample format hound supports) into 16kHz mono f32
/// PCM samples, resampling if necessary. whisper.cpp requires 16kHz mono.
fn wav_to_f32_16k_mono(wav_bytes: &[u8]) -> Result<Vec<f32>, String> {
    let cursor = std::io::Cursor::new(wav_bytes);
    let mut reader = WavReader::new(cursor).map_err(|e| e.to_string())?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().filter_map(|s| s.ok()).map(|s| s as f32 / max).collect()
        }
    };
    // Downmix to mono if needed.
    let mono: Vec<f32> = if spec.channels > 1 {
        samples
            .chunks(spec.channels as usize)
            .map(|c| c.iter().sum::<f32>() / c.len() as f32)
            .collect()
    } else {
        samples
    };
    // Linear-interpolation resample to 16kHz, matching the precision level
    // Python's pipeline already operated at (it also just used ffmpeg's
    // resampler upstream of this, not a high-quality one here either).
    if spec.sample_rate == 16000 {
        Ok(mono)
    } else {
        let ratio = 16000.0 / spec.sample_rate as f32;
        let new_len = (mono.len() as f32 * ratio) as usize;
        let mut out = Vec::with_capacity(new_len);
        for i in 0..new_len {
            let src_pos = i as f32 / ratio;
            let idx = src_pos as usize;
            let frac = src_pos - idx as f32;
            let a = mono.get(idx).copied().unwrap_or(0.0);
            let b = mono.get(idx + 1).copied().unwrap_or(a);
            out.push(a + (b - a) * frac);
        }
        Ok(out)
    }
}

pub fn transcribe_audio(wav_bytes: &[u8]) -> String {
    let Some(ctx) = get_context() else {
        return "[transcription error: whisper model not loaded]".to_string();
    };
    let audio = match wav_to_f32_16k_mono(wav_bytes) {
        Ok(a) => a,
        Err(e) => return format!("[transcription error: {e}]"),
    };
    let mut state = match ctx.create_state() {
        Ok(s) => s,
        Err(e) => return format!("[transcription error: {e}]"),
    };
    // beam_size=1 in the Python version maps to greedy sampling here.
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_print_progress(false);
    params.set_print_special(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    if let Err(e) = state.full(params, &audio) {
        return format!("[transcription error: {e}]");
    }

    let num_segments = state.full_n_segments().unwrap_or(0);
    let mut text_parts = Vec::new();
    for i in 0..num_segments {
        if let Ok(seg) = state.full_get_segment_text(i) {
            text_parts.push(seg);
        }
    }
    let text = text_parts.join(" ").trim().to_string();
    if text.is_empty() {
        "[silence]".to_string()
    } else {
        text
    }
}
