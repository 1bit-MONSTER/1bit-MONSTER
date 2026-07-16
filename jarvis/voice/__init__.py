"""Zaya Co-Host Voice Module — Phase 1.

Voice cloning, synthesis, and pack management for the Zaya Co-Host platform.

Components:
    record.py   — Record high-quality voice samples (48kHz)
    train.py    — Train voice clone from samples → .voice pack
    engine.py   — Load .voice packs, synthesize speech via ONNX/PyTorch

Quick start:
    python3 jarvis/voice/record.py --name bcloud --session 1
    python3 jarvis/voice/train.py --samples ./voice/samples/bcloud/ --name bcloud
"""

from jarvis.voice.engine import VoiceEngine, VoicePack

__all__ = ["VoiceEngine", "VoicePack"]
