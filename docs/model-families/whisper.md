# Whisper — Speech-to-Text

OpenAI Whisper V3 Turbo. Provides the speech-to-text (STT) stage of the [JARVIS pipeline](../jarvis.md) — full FFT/STFT + encoder-decoder path through GPU HIP kernels, with an NPU xclbin build ready.

## Models

| Model | Backend(s) | Status |
|-------|------------|:------:|
| **Whisper-V3-Turbo** | NPU (5 xclbins) / GPU HIP | 🔄 |

## Notes

- **NPU:** Whisper-V3-Turbo build stanza ready (5 xclbins).
- **GPU HIP:** FFT/STFT kernels validated.
- **GPU Vulkan:** not yet. **CPU:** universal GGUF backend.

**See also:** [JARVIS pipeline](../jarvis.md) · [full model support detail](../wiki/models.md) · [all families](README.md)
