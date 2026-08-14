# PLAN: 1bit.MONSTER — one binary, full model catalog (500+ models)

**Date:** 2026-08-14 · **Status:** plan — bring-up arc complete, catalog expansion next
**Master log:** `docs/research/onebit-modular-research.md` (pilots 1–22, 2026-08-13)
**References:** `docs/wiki/models.md` (canonical support doc) · `docs/research/modular-channel-summary.md`

## Why this is now possible (what was decoded)

| Asset | Status | Evidence |
|---|---|---|
| Modular/MAX architecture-registry pattern | Decoded — 500+ models = ~50 arch classes + HF `config.json` matching, NOT 500 ports | `docs/research/onebit-modular-research.md` §1–5 |
| Modular code graph / Mojo attribute-expression system | Decoded (MLIR attribute meta-layer, De Bruijn indices, depth-aware replacer) | wiki SRC-2026-08-12-00x + `modular-channel-summary.md` #3 |
| Mojo rocblas/hipblaslt/miopen bindings | Decoded (CDNA2/3/4 targets, gfx90a/942/950) | wiki obs 2026-08-12-modular |
| Mojo GPU puzzles | Completed — 35 puzzles / 70 tests on BOTH AMD boxes (strix gfx1151: 59/11, ryzen gfx1201: 56/0/14-skip) | research §6 |
| XDNA/NPU exec flow | Decoded (raw-ioctl GEMM byte-identical to XRT; 1 firmware-check line open) | wiki obs 2026-08-10, `npu_re_workspace` |
| **HF-native loading** | **PROVEN** — raw HF checkpoints discovered→arch-mapped→routed→loaded→**generation bit-identical to torch** | research §8–30, commit `45054ab1` |

## Current state (post-pilots 1–22)

- **19 families validated end-to-end** (see the manifest tiers for exact gate standards — full 20/20 vs torch, Q8-oracle, numpy-exact, and the honest near-tie/degenerate caveats): 14 full vs torch (f32) — llama, qwen2, qwen3, gemma3, granite (MoE), mistral-7B, phi-3-mini, olmo, gpt2, falcon, opt, gptj, gptneo, **step1 (2026-08-15, sqrt-ALiBi, no RoPE — the 2,882-checkpoint Step1MoE census class is dense-mislabeled pretrain runs)**; exaone vs llama.cpp Q8; **gptoss (20B, 407 checkpoints) vs a numpy port of the authoritative modeling_gpt_oss.py on the real checkpoint** — the memory-blocked family now runs via packed-MXFP4 per-expert dequant (engine ≡ reference, logits max|diff| 6.7e-5). 4 more families numpy-exact top-8 but no 20/20 (internlm2, minicpm, gptneox, codegen — near-tie/degenerate). 5/6 torch families bit-exact logits (0.000); gemma3 at the chaos-bound f32 floor (argmax correct, logits carry model-amplified rounding — proven unreachable for any independent impl).
- **80 fixture checks green** (`Testing/run_all.sh` 10/10): arch mapping (30), discovery (5), router (11), dtype decode (11), sharded reader (6), rotation table (17).
- **Arch registry:** `rcpp_arch_t` 24 tokens (gpt2/gptneox/codegen/gptj/gptoss/opt landed during the sweep); LLaMA-layout breadth added (openelm, nemotron, minicpm, baichuan, exaone, solar, internlm, xverse); unknown archs now fail LOUDLY (`RCPP_ARCH_UNKNOWN` sentinel, no silent BITNET).
- **Formats:** GGUF, H1B, 1BP/Q4NX, safetensors (incl. sharded + MoE fused/stacked), ONNX.
- **Catalog today:** 19 documented families / ~47 models / 37 NPU FLM / 37 1BP on HF.

## The gap (why we're not at 500 yet)

Coverage + process, not architecture: every new HF family needs (1) arch-string→enum mapping, (2) per-family tensor-name / norm / rope / activation quirks (the real work — each family's `config.json` differs), (3) validation. MAX's trick is declarative graphs over one kernel library + agent-driven bring-up; we have the registry + kernels + proven validation harness, we lack the per-family quirk table at scale.

## Path to MONSTER (in order)

1. **Rotate the bring-up loop into an agent skill.** The pilot pipeline (arch mapping → discovery fixture → router check → real-checkpoint e2e vs torch) is now scriptable; codify it as a repo skill so adding a family is a guided process, not archaeology. `skills/1bit-writer` is the seed.
2. **Per-family quirk table → validated families list.** Standing deck (research §27–30): zamba/mamba/whisper/kimi/olmo real-prompt validation; MoE real-prompt via engine tokenizer (htok workstream); gemma3 sliding-window masking >512 tok — **IMPLEMENTED 2026-08-15** (attention window mask mirrors the reference; long-context e2e pending fixture); gpt2 custom tensor map; **gptoss sliding-window attention >128 tok (packed-MXFP4 path validated 2026-08-14; mask landed 2026-08-15)**. Each lands with a rotation-table + e2e check.
3. **Breadth sweep: bulk-mint the LLaMA-layout long tail** (one-line arch mappings; verified against the arch self-check) — biggest models-per-line-of-code ratio.
4. **Publish the arch→checkpoint table** (MAX's marketing artifact) in `docs/wiki/models.md`.
5. **Catalog ops:** refresh the 37 1BP HF catalog + 37 NPU FLM map as families land.

## Open questions (need user input)

- **"Modular gets M3 working on 700MB" — RESOLVED 2026-08-14:** it's the container, not a chip/model. modular.com/pricing: the free Self-Hosted Community Edition is "one container, under 700MB" carrying MAX + Mojo + the hundreds-of-models catalog. Models are data (pulled from HF), the platform stays tiny. **Our standing: `build/1bit` is 68MB (+ flm-real 106MB) — already ~10x under their container.** The MONSTER lesson isn't size, it's the distribution pattern: free self-hosted community edition + models-as-data + OpenAI-compatible API. (Bonus from the pricing page: their hosted endpoint list now includes MiniMax M3, DeepSeek V4, Qwen 3.7-Max, GLM 5.1/5.2, Kimi K2.6, Nemotron 3, GPT-OSS 120B, Gemma 4 31B — useful MONSTER-family watchlist.)
- **Where should scattered home-dir knowledge live from now on?** Proposal: `docs/research/` (logs) + `docs/wiki/` (canonical) + wiki (immutable sources) — three places, no more. `~/research-papers/`, `~/npu_re_workspace`, `~/xdna-driver`, `~/max-xdna-backend` stay as working dirs, referenced by the logs.
- **MONSTER as product name? — RESOLVED 2026-08-14:** yes. The product is now **1bit MONSTER**, repo `1bit-MONSTER/1bit-MONSTER`, domain `1bit.monster`. "One Bit Systems" / `1bit-systems` is the predecessor identity; this repo is being transferred/renamed to become the new `1bit-MONSTER/1bit-MONSTER` (GitHub-native transfer, keeps stars/issues/PR/Actions history intact). Gaia (third_party/gaia) and the old hand-rolled `onebit` agent CLI were both removed in the same pass — JARVIS (`tools/jarvis/`) is the one agent/voice pipeline going forward.
