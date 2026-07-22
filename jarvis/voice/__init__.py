"""Zaya Co-Host Voice Module — Phase 1.

Offline voice-clone training tooling. The runtime engine that loads
.voice packs and serves synthesis (formerly engine.py here) now lives in
jarvis-rs/src/voice.rs, a candle-nn port -- see jarvis-rs/README or the
top-level jarvis/README.md. This package retains only the pieces that
stay Python: sample recording and training.

Components:
    record.py   — Record high-quality voice samples (24kHz mono)
    train.py    — Train voice clone: codec + ZAYA adapter → .voice pack
    codec.py    — Shared AudioCodec definition used by train.py/dataset.py
                  (its decode-only inference path is separately ported to
                  jarvis-rs/src/voice.rs for the running server)

Architecture (agnostic):
    Audio → [Codec Encoder] → tokens → [Any LLM] → tokens → [Codec Decoder] → Audio
                               ↑  Voice Pack (speaker embedding)  ↑

Quick start:
    python3 jarvis/voice/record.py --name bcloud --session 1
    python3 jarvis/voice/train.py --samples ./voice/samples/bcloud/ --name bcloud
"""
