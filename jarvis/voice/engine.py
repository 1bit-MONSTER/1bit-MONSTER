"""
Voice inference engine — Zaya Co-Host Phase 1.2

Loads a .voice pack and runs text-to-speech inference.
Supports both ONNX (fast path) and PyTorch (fallback) backends.

Usage:
  from jarvis.voice.engine import VoiceEngine
  engine = VoiceEngine()
  engine.load_pack("./voice/packs/bcloud.voice")
  audio = engine.synthesize("Hello, I'm your AI co-host!")
"""

import json, os, struct, tarfile, tempfile, time, shutil
from pathlib import Path
from typing import Optional

import numpy as np

VOICE_PACKS_DIR = Path(__file__).parent.parent.parent / "voice" / "packs"

# ── ONNX Runtime Inference ─────────────────────────────────────────────

class OnnxTTS:
    """ONNX-based TTS engine (fast path, CPU/NPU)."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.session = None
        self._load()

    def _load(self):
        try:
            import onnxruntime as ort
            providers = []
            # Try NPU first (XDNA), then ROCm, then CPU
            try:
                providers.append(("ROCMExecutionProvider", {"device_id": 0}))
            except:
                pass
            providers.append("CPUExecutionProvider")
            
            self.session = ort.InferenceSession(
                str(self.model_path), providers=providers
            )
            self.input_names = [i.name for i in self.session.get_inputs()]
            self.output_names = [o.name for o in self.session.get_outputs()]
            print(f"    ONNX engine loaded: {len(self.input_names)} inputs, "
                  f"{len(self.output_names)} outputs")
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")

    def synthesize(self, text: str, speaker_embedding: np.ndarray,
                   speed: float = 1.0) -> np.ndarray:
        if not self.session:
            raise RuntimeError("ONNX model not loaded")
        
        # Tokenize text to input IDs
        tokens = np.array([ord(c) for c in text[:200]], dtype=np.int64)
        tokens = np.pad(tokens, (0, max(0, 200 - len(tokens))),
                       constant_values=0)[:200].reshape(1, -1)
        
        feed = {
            "text": tokens,
            "speaker_embeddings": speaker_embedding.reshape(1, -1).astype(np.float32),
        }
        if "speed" in self.input_names:
            feed["speed"] = np.array([speed], dtype=np.float32)

        result = self.session.run(self.output_names, feed)
        audio = result[0].flatten()
        return audio


# ── PyTorch TTS (Fallback) ─────────────────────────────────────────────

class TorchTTS:
    """PyTorch-based TTS engine (fallback, CPU)."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model = None
        self._load()

    def _load(self):
        try:
            from transformers import XTTSForTextToSpeech
            self.model = XTTSForTextToSpeech.from_pretrained(
                str(self.model_dir), device_map="cpu"
            )
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model: {e}")

    def synthesize(self, text: str, speaker_embedding: np.ndarray,
                   speed: float = 1.0) -> np.ndarray:
        if not self.model:
            raise RuntimeError("PyTorch model not loaded")
        
        import torch
        with torch.no_grad():
            output = self.model.generate(
                text=text,
                speaker_embeddings=torch.from_numpy(speaker_embedding).unsqueeze(0),
                speed=speed,
            )
        return output.cpu().numpy().flatten()


# ── Voice Pack Loader ───────────────────────────────────────────────────

class VoicePack:
    """Loaded and parsed .voice pack."""

    def __init__(self, path: Path):
        self.path = path
        self.metadata: dict = {}
        self.speaker_embedding: Optional[np.ndarray] = None
        self.tts_engine = None
        self._tmp_dir: Optional[Path] = None

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *args):
        self.cleanup()

    def load(self):
        """Extract and load a .voice pack."""
        self._tmp_dir = Path(tempfile.mkdtemp())
        
        with tarfile.open(self.path, "r:gz") as tar:
            tar.extractall(self._tmp_dir)

        # Load metadata
        meta_path = self._tmp_dir / "metadata.json"
        if meta_path.exists():
            self.metadata = json.loads(meta_path.read_text())

        # Load speaker embedding
        embed_path = self._tmp_dir / "speaker.pt"
        if embed_path.exists():
            import torch
            self.speaker_embedding = torch.load(embed_path, map_location="cpu",
                                                weights_only=True).numpy()
        else:
            self.speaker_embedding = np.random.randn(512).astype(np.float32)

        # Load TTS engine
        onnx_path = self._tmp_dir / "model.onnx"
        model_dir = self._tmp_dir / "model"

        if onnx_path.exists():
            self.tts_engine = OnnxTTS(onnx_path)
        elif model_dir.exists():
            self.tts_engine = TorchTTS(model_dir)

        print(f"  Voice pack loaded: {self.metadata.get('name', 'unknown')}")
        print(f"    Model:   {self.metadata.get('model', 'unknown')}")
        print(f"    Engine:  {'ONNX' if onnx_path.exists() else 'PyTorch'}")

    def synthesize(self, text: str, speed: float = 1.0) -> np.ndarray:
        if not self.tts_engine:
            raise RuntimeError("No TTS engine loaded")
        return self.tts_engine.synthesize(
            text, self.speaker_embedding, speed
        )

    def cleanup(self):
        if self._tmp_dir:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None


# ── High-Level Engine ──────────────────────────────────────────────────

class VoiceEngine:
    """Manages multiple voice packs and provides synthesis API."""

    def __init__(self):
        self.voices: dict[str, VoicePack] = {}
        self.active_voice: Optional[str] = None
        self.sample_rate = 24000  # XTTS-v2 native

    def list_available_packs(self) -> list[dict]:
        """List all .voice packs in the packs directory."""
        packs = []
        if not VOICE_PACKS_DIR.exists():
            return packs
        for f in sorted(VOICE_PACKS_DIR.glob("*.voice")):
            # Quick metadata read without full load
            try:
                with tarfile.open(f, "r:gz") as tar:
                    meta = tar.extractfile("metadata.json")
                    if meta:
                        info = json.loads(meta.read())
                        info["path"] = str(f)
                        info["loaded"] = f.stem in self.voices
                        packs.append(info)
            except:
                packs.append({"name": f.stem, "path": str(f), "loaded": False})
        return packs

    def load_pack(self, path: str | Path, name: Optional[str] = None):
        """Load a voice pack."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Voice pack not found: {path}")
        
        pack = VoicePack(path)
        pack.load()
        voice_name = name or pack.metadata.get("name", path.stem)
        self.voices[voice_name] = pack
        if self.active_voice is None:
            self.active_voice = voice_name
        return voice_name

    def unload(self, name: str):
        """Unload a voice pack."""
        if name in self.voices:
            self.voices[name].cleanup()
            del self.voices[name]
            if self.active_voice == name:
                self.active_voice = next(iter(self.voices.keys()), None)

    def activate(self, name: str):
        """Set active voice for synthesis."""
        if name not in self.voices:
            raise KeyError(f"Voice '{name}' not loaded. Available: {list(self.voices.keys())}")
        self.active_voice = name

    def synthesize(self, text: str, voice: Optional[str] = None,
                   speed: float = 1.0) -> tuple[np.ndarray, int]:
        """Synthesize text to audio waveform.
        
        Returns:
            Tuple of (audio_samples: np.ndarray, sample_rate: int)
        """
        v = voice or self.active_voice
        if not v or v not in self.voices:
            raise RuntimeError(f"No voice loaded. Load a voice pack first.")
        audio = self.voices[v].synthesize(text, speed)
        return audio, self.sample_rate

    def cleanup_all(self):
        """Unload all voices."""
        for name in list(self.voices.keys()):
            self.unload(name)


# ── Quick test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=str, required=True,
                        help="Path to .voice pack")
    parser.add_argument("--text", type=str, default="Hello, I'm your AI co-host! Testing one two three.",
                        help="Text to synthesize")
    parser.add_argument("--output", type=str, default="/tmp/test_voice.wav",
                        help="Output WAV path")
    args = parser.parse_args()

    import wave
    engine = VoiceEngine()
    name = engine.load_pack(args.pack)
    audio, sr = engine.synthesize(args.text)
    
    with wave.open(args.output, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    
    print(f"\n  Audio saved: {args.output}")
    print(f"  Duration: {len(audio) / sr:.1f}s")
    engine.cleanup_all()
