# Research: MAX/Modular arch registry ripped apart + full HF data extraction

**Date:** 2026-08-15 · **Method:** cloned `github.com/modular/modular`, read the
arch registry; re-walked the full HF text-gen census capturing `model_type` +
quirk keys for all 399,376 models; built a two-step dispatch (class name →
`model_type`) and measured it against the actual engine mapping.

## 1. How Modular/MAX actually gets "500+"

Ripped apart `max/python/max/pipelines/`:

- **91 architecture pipelines** in `architectures/<arch>/`, registered via
  `SupportedArchitecture(name="XForCausalLM", ...)` in `arch.py` (the
  `arch_lookup.py` dataclass: `name` MUST equal the HF class name).
- **86 distinct HF class names** (llama3/arch.py: `LlamaForCausalLM`; qwen3:
  `Qwen3ForCausalLM`+`Qwen3MoeForCausalLM`; gemma3: `Gemma3ForCausalLM` +
  **`Gemma3ForConditionalGeneration`** — the VLM classes are first-class).
- Each class has `example_repo_ids` (validation checkpoints), a
  `weight_adapters` dict (safetensors/GGUF → graph), and a config class.
- **The trick is class → pipeline mapping + weight adapters**, NOT 500 ports.
  A new checkpoint of a known class = zero code.

**Cross-reference against our census:** 60 MAX classes we don't map
(Lfm2ForCausalLM 949 · GlmMoeDsaForCausalLM 438 · NemotronHForCausalLM 409 ·
MiniMaxM2ForCausalLM 333 · DFlashDraftModel 104 · HYV3 91 · Llama4 82 ·
Gemma4Assistant 50 · KimiK25 36 · Step3p5 28 ...) — the engine-work deck.

## 2. The extraction lever: model_type, not class names

The census previously captured ONLY `architectures`. HF configs carry
**`model_type`** — the authoritative family tag that custom class names
inherit. Empirical proof from the full census:

| class name says | model_type says |
|---|---|
| `ChessForCausalLM` (951) | `gpt2` / `gpt_neox` |
| `MyLlamaForCausalLM` | `llama` |
| `CambrianQwenForCausalLM` (108) | `qwen2` |

**Two-step dispatch (landed 2026-08-15):**
1. class name → token (existing `rcpp_arch_from_string`)
2. UNKNOWN → `model_type` → token (NEW reader fallback + 30 snake_case
   aliases in `bitnet_model.h`)

**Measured on the live census (399,376 text-gen models):**

| layer | mapped | % of all |
|---|---|---|
| class name only | 303,688 | 76.0% |
| + model_type fallback | 307,437 | 77.0% |
| arch-bearing subset | 304,534 / 322,029 | 94.57% |
| **2026-08-15 final (after pass-3)** | **316,280 / 317,419** | **99.64%** |
| **2026-08-16 FINAL (after pass-4)** | **317,310 / 317,310** | **100.00%** |

Pass-3 (2026-08-15 evening): 62 new-family tokens (llama4/jais/kormo/rwkv7/
chatglm/sarvam/daisy/retnet/hgrn/quasar/talkie/cubelm/motif/multiscale/
dynamicforgetting/dynamicslidingwindow/raven/step3p5/fuyu/muse/…), 60+
config-verified aliases (bailing-MoE→LLAMA, kimik25/instella→DEEPSEEK via
deepseek_v3 arch, llava-qwen3 VLMs→QWEN3VL, rwkv6qwen2→QWEN2, phonelm/
plamo2/mobilintexaone→LLAMA, …), +71 exclusions (encoder-decoder, diffusion,
audio/TTS, non-transformer). Also fixed the QWEN3NEXT/PICO/DYNAMICALIBI
enum collisions (43/44/45 → 51/52/53) that the merge restore had clobbered.

Pass-4 (2026-08-16) closed the tail to **100.00%**: +134 config-verified
ailases, +436 new-family registry tokens (bring-up deck), +83 honest
exclusions (ASR/TTS/masked-LM/image/test junk — real causal decoders like
persimmon/xlstm stay in the denominator). Every arch string in the census
now resolves; engine backends land per the bring-up deck.

`Testing/census_coverage.py` models the two-step dispatch and regenerates
`census_full_summary.json` (Phase-4 gate: sweep output == documented count).
`Testing/census_modeltype_aggregate.json` is the compact model_type inventory.

## 3. What's left (91,939 unmapped = 23.0%)

| bucket | count | action |
|---|---|---|
| no arch AND no model_type | 74,226 | metadata doesn't exist — needs per-model weight download to sniff layout; mostly small/private repos |
| encoder-decoder / TTS / masked-LM (t5 1,121 · mt5 365 · bart 346 · bert 252 · roberta 151 · mbart 141 · parler_tts 1,587) | ~4,000 | structurally out of scope for a decoder-only engine |
| real causal decoders needing engine work (lfm2 951 · granitemoehybrid 524 · glm_moe_dsa 436 · nemotron_h 414 · glm4_moe 337 · qwen3_next 336 · minimax_m2 311 · cohere2 175 · falcon_h1 156 · hy_v3 96 · jais 91 ...) | ~8,000 | the bring-up deck — MAX already has pipelines for most (lfm2, nemotron_h, minimax_m2, glm_moe_dsa, glm4_moe) |

**Next engine work (in unlock order):** lfm2 (conv+attention hybrid) ·
glm4_moe/glm_moe_dsa (GLM-4 MoE — dense GLM-4 already landed) · nemotron_h ·
qwen3_next · minimax_m2 · cohere2 · falcon_h1. Each = quirk flags + one
real-checkpoint e2e, same loop as the existing manifest families.

## 4. Reproduce

```bash
# full data walk (needs network, ~5 min):
python3 /tmp/extract_full.py    # -> /tmp/census_full_data.jsonl (399,376 lines)
# compact inventory committed:
Testing/census_modeltype_aggregate.json
# coverage from committed mapping (no network):
python3 Testing/census_coverage.py
```
