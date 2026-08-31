# Hacker News launch post — copy-paste ready

**Post as a "Show HN" on https://news.ycombinator.com/submit**

---

## Title (pick one, 80-char limit)

1. `Show HN: Reverse-engineered AMD's XDNA2 NPU in 4 days — any LLM, one C++ binary`
2. `Show HN: One C++ binary runs any LLM on NPU/GPU/CPU — zero Python`
3. `Show HN: 100% of HuggingFace LLMs run on one MIT C++ binary, zero Python`

Recommended: **#1** (the reverse-engineering hook is what HN clicks on).

---

## Story (paste into the "text" box; 1,995 chars — under the 2,000 limit)

**1bit.MONSTER — one C++ binary, any LLM, any hardware, zero Python. MIT.**

AMD shipped a 50 TOPS XDNA 2 NPU locked behind a closed runtime (FastFlowLM): 22 proprietary .so files, 209 xclbin bitstreams, zero docs. I took it apart in 4 days — RSA-2048-signed firmware, mailbox protocol decoded by hand with a disassembler, ftrace and bpftrace — and replaced the whole stack with open C++. Unedited session logs:  https://github.com/1bit-MONSTER/1bit-MONSTER/blob/main/docs/journey.md

What came out of it:

* One busybox-style binary (`build/1bit`) that auto-detects the model and runs it on whatever you have: XDNA2 NPU, GPU (HIP/CUDA/Metal/Vulkan), or CPU (AVX-512/scalar). No config files, no per-model glue.
* 554 architecture tokens mapping 1,798 HuggingFace arch strings — 317,310 / 317,310 text-generation checkpoints on the hub (100%) land on a supported token. 16+ families: Zyphra, Qwen, Llama, Mistral, Gemma, Phi, Falcon, OLMo, Granite, SmolLM, DeepSeek, GPT-OSS, Kimi, BitNet/Bonsai, Whisper. Reads GGUF, 1BP, ONNX, H1B, safetensors.
* Zero Python at runtime — pure C++26. No interpreter, no venv, nothing to babysit.
* JARVIS: a fully-local voice pipeline (mic → STT → LLM → TTS → speaker), one subcommand.
* Lemonade v11.8.1 vendored (15-backend SDK). ~600 hours of engineering, public from day one, MIT.

Numbers: Q1_0 HIP kernel = 24-33x faster prompt processing on gfx1151; 4,172 t/s prompt on Bonsai-1.7B; the NPU runtime replacement is now byte-identical to the closed original, 2x on the 35B.

Site (docs, model families, benchmarks, the full RE writeup): https://1bit.monster

Honest caveats: young project — NPU support targets Strix Halo-class XDNA2 SKUs today; GPU/CPU paths are the battle-tested ones. Issues and contributions welcome (MIT).

Try it:
git clone https://github.com/1bit-MONSTER/1bit-MONSTER && cd 1bit-MONSTER && cmake -B build && cmake --build build
./build/1bit zaya -m model.1bp -p "Hello world"

---

## Posting notes

- **URL vs text**: paste the URL (https://github.com/1bit-MONSTER/1bit-MONSTER) into the URL field OR leave it out and use the text above — the text is self-contained either way. If you include the URL, HN shows the story text below it; that's fine too.
- **Timing**: weekday 09:00–11:00 ET / 14:00–16:00 UTC is the classic high-visibility window.
- **First comment**: add a short comment right after posting (HN sorts new posts with comments up) — e.g. "Happy to answer questions about the NPU RE or the 1-bit kernels — the session logs are linked above."
- **Don't ask for upvotes** (HN guideline) — the story stands on its own.
- The stats in the story are pulled from the current engine state (554 tokens / 1,798 arch strings) — if the census re-sweeps, refresh the numbers from the site before posting.
