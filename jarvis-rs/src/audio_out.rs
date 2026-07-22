//! Local speaker playback for jarvis's TTS output. Port of jarvis/audio_out.py.
//!
//! Lets the server mirror synthesized speech out through a physical speaker
//! attached to the box jarvis runs on, independent of whatever the calling
//! client does with the returned audio bytes. Entirely dormant (no-op, no
//! error) until a non-onboard playback device is actually present.

use serde::Serialize;
use std::process::Stdio;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

const ONBOARD_HINTS: [&str; 2] = ["hd-audio generic", "hdmi"];

#[derive(Serialize, Clone, Debug)]
pub struct PlaybackDevice {
    pub card: i32,
    pub device: i32,
    pub name: String,
    pub device_name: String,
    pub is_onboard: bool,
    pub alsa_id: String,
}

/// Parses `aplay -l` into a list of playback devices, e.g.:
/// "card 2: Phone [USB Speaker Phone], device 0: USB Audio [USB Audio]"
pub async fn list_playback_devices() -> Vec<PlaybackDevice> {
    let out = match Command::new("aplay").arg("-l").output().await {
        Ok(o) => o,
        Err(_) => return Vec::new(),
    };
    let text = String::from_utf8_lossy(&out.stdout);
    let re = regex::Regex::new(r"^card (\d+): \S+ \[(.*?)\], device (\d+): (.*?) \[").unwrap();
    let mut devices = Vec::new();
    for line in text.lines() {
        let Some(caps) = re.captures(line) else { continue };
        let card: i32 = caps[1].parse().unwrap_or(0);
        let name = caps[2].to_string();
        let device: i32 = caps[3].parse().unwrap_or(0);
        let device_name = caps[4].to_string();
        let is_onboard = ONBOARD_HINTS.iter().any(|h| name.to_lowercase().contains(h));
        devices.push(PlaybackDevice {
            card,
            device,
            name,
            device_name,
            is_onboard,
            alsa_id: format!("plughw:CARD={card},DEV={device}"),
        });
    }
    devices
}

/// First non-onboard playback device (i.e. a plugged-in USB speaker), or None.
pub async fn find_external_speaker() -> Option<PlaybackDevice> {
    list_playback_devices().await.into_iter().find(|d| !d.is_onboard)
}

/// Play WAV bytes on a local ALSA device. No-op if none is available.
/// Fire-and-forget by default -- runs in a background task so it never adds
/// latency to the HTTP response that carries the same audio back to the
/// calling client.
pub async fn play_wav_local(wav_bytes: Vec<u8>, device: Option<PlaybackDevice>, blocking: bool) -> bool {
    let dev = match device {
        Some(d) => Some(d),
        None => find_external_speaker().await,
    };
    let Some(dev) = dev else { return false };

    let run = move || async move {
        let mut child = match Command::new("aplay")
            .args(["-q", "-D", &dev.alsa_id, "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(c) => c,
            Err(_) => return,
        };
        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(&wav_bytes).await;
        }
        let _ = tokio::time::timeout(std::time::Duration::from_secs(60), child.wait()).await;
    };

    if blocking {
        run().await;
    } else {
        tokio::spawn(run());
    }
    true
}
