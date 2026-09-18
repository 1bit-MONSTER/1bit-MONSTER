# RESULTS — dual-engine concurrency, and five family-resolution fixes — 2026-09-16

Two results, both measured on strixhalo. The first is the largest throughput finding in the
project; the second is a class of bug that had been presenting as one mystery.

---

## 1. The dual NPU engine beats FLM on aggregate throughput

The NPU runs **multiple engines concurrently at full per-stream speed**. FLM's server
serializes. Qwen3-0.6B, 256-token prompt, 32 decode tokens, identical command per stream:

| configuration | solo | concurrent (per stream) | **aggregate** | scaling |
|---|---|---|---|---|
| unified, 1 × 0.6B | 76 tok/s | — | 76 tok/s | 1.00x |
| unified, 2 × 0.6B (trial 1) | 76 | 80 / 76 | **156 tok/s** | **2.05x** |
| unified, 2 × 0.6B (trial 2) | 76 | 79 / 79 | **158 tok/s** | **2.08x** |
| unified, 3 × 0.6B | 76 | 74 / 74 / 79 | **227 tok/s** | **2.99x** |
| unified, 0.6B + 4B | 79 / 18 | 79 / 19 | 98 tok/s | ~1.01x of the solos |
| unified, 0.6B + 8B | 81 / 11 | 81 / 11 | 92 tok/s | ~1.00x of the solos |
| **runlist, 2 × 0.6B** | **91** | **92 / 92** | **184 tok/s** | **2.02x** |

**Per-stream throughput does not degrade.** Two concurrent engines each run at ~79 tok/s
against a 76 tok/s solo baseline — slightly *faster* than solo — and three hold 74–79 each.
Aggregate scales essentially linearly, across model sizes and on both the unified and runlist
paths. In the mixed rows each stream holds its own solo rate within noise (18 → 19 for the 4B;
11 → 11 for the 8B), so a large model and a small one do not contend.

**FLM does not do this.** Against the running `flm serve` (:8098, qwen3.6-moe:35b-a3b), 40
completion tokens per request:

| | wall clock | tokens | aggregate |
|---|---|---|---|
| 1 request | 5.66 s | 40 | 7.1 tok/s |
| 2 concurrent | **11.31 s** | 80 | **7.1 tok/s** |

Two concurrent requests take exactly 2x the wall clock and aggregate throughput is unchanged.
**It queues.**

### Why this matters

Every single-stream comparison in this repo understates the engine, because FLM only serves
one stream. On the metric a server is actually judged on, the engine wins by 2–3x with
existing code. Concurrency is a first-class parity axis and had not been measured.

### Caveats

- The FLM comparison is different models (its serve runs the 35B). What is compared is the
  **scaling property**, which is a property of the serving layer; the absolute rates are not
  compared.
- Cross-model concurrency on our side was tested only for 0.6B/4B/8B.
- `hwctx_limit 16`, `context_limit 64`; the driver is not the constraint.

---

## 2. Five fixes, four of them one class

**Every one of the four was a Qwen3 constant applied to every model.** They all produced the
*same* symptom — a timeout or a divergence on non-Qwen3 families — which is why they read as
one mystery rather than four bugs, and why each fix exposed the next: every site sits earlier
on the path than the previous one's success.

| commit | bug | effect |
|---|---|---|
| `91cfc0fd6` | `sess_model_dir(H)` mapped **every** H to a Qwen3 directory, so a non-Qwen3 model loaded a Qwen3 `layer.xclbin`. Nanbeige (H=2560) took Qwen3-4B's; Phi-4 (H=3072) fell through to 0.6B's. | Nanbeige **0.14 → 27 tok/s** (FLM 21); Phi-4 **→ 24 tok/s** (FLM 19) |
| `4a269430e` | `shipped_elf_dir(H)` seeded the ELF cache from the same H table, planting **Qwen3 ELFs in another model's cache** — Nanbeige's cache was created with 4,249 symlinks to Qwen3-4B's. | second, independent cause of the same timeout |
| `1bdfc33ee` | the `gen_layer_elfs` spawn passed no family, so it defaulted to `qwen3` and threw `Unsupported intermediate size` on every other family. | a non-Qwen3 model could not run from the **default path** at all |
| `36292e1a8` | `is_eos_token()` hardcoded Qwen3's 151643/151645, so no non-Qwen3 decode could stop on its own EOS. | non-Qwen3 decode ran to the token budget |
| `963da4d5b` | the generic `Gemma` prefix test swallowed **Gemma4** and sent it to `gemma_text`, which asserts on its weights and **aborts the process** — the oracle saw no output at all. | Gemma4 now reports a capability gap instead of crashing |

`36292e1a8` keeps **both** Qwen3 ids deliberately: its `tokenizer_config.json` lists only
`<|im_end|>` (151645), but answers also end with `<|endoftext|>` (151643), which is why that
second id existed. Taking the config alone would have regressed Qwen3.

Also `6e2d617c9`: the BO-capture tool wrote each syncing buffer in full — 512 MB weight BOs,
40 per forward — and filled `/tmp` completely, which wedged every agent tool on the box
because they all stage scratch there. Now capped, and refuses a tmpfs destination.

---

## 3. Verification

| | result |
|---|---|
| dense Qwen3 @1k/2k/4k, unified vs runlist vs FLM | prefill faster in **12/12** model x context pairs (0.66–0.88x the time); decode match-or-beat **12/12** |
| Qwen3-0.6B answer-level oracle | **20/20** |
| Phi-4-mini answer-level oracle | **17/20** vs FLM 20/20 (reference self-check 20/20) |
| engine after untracking the captures | unchanged — **99 tok/s** |

---

## 4. Accuracy is NOT established for most families

| family | throughput | accuracy |
|---|---|---|
| Qwen3 dense | ✅ verified | **20/20** |
| Phi-4-mini | ✅ 25 vs FLM 19 tok/s | **17/20** vs 20/20 |
| Nanbeige4.1-3B | ✅ 27 vs FLM 21 tok/s | ❌ **unscoreable** — reference arm fails |
| Llama-3.2-1B | runs at 67 tok/s, rc=0 | ❌ **1/20**, cause unknown |
| Gemma3-1B/4B | — | ❌ **cannot generate ELFs** |
| Gemma4-E2B/E4B | — | ❌ **cannot generate ELFs** |
| 35B MoE | — | ❌ captured ELF is not a complete layer |

- **Llama** loads, runs at full speed, exits 0, and answers wrongly. **No error surfaces.**
  The `head_dim` hardcode at `npu_runlist_bridge.cpp:63` was the obvious candidate — the same
  file's own line 614 warns *"head_dim is NOT always 128 (LFM2 and Llama-3.2 use 64)"* —
  **tested and refuted: 1/20 → 0/20.** Reverted.
- **Gemma** has two distinct causes: no `gemma4e` path exists in `gen_layer_elfs`
  (`gemma4e_npu.hpp` declares a `causal_lm` with `prefill()`, not the `<family>_npu_sequence`
  API the tool templates over), and the vendored `gemma_text` class aborts on Gemma3's
  weights with `blocks_per_row <= 63`.
- **35B MoE**: the captured `layer.xclbin` ELF consumes no expert weights in **any** order.
  FLM's `qwen3_6_moe_npu` has no source in the vendored tree and every kernel reports the name
  `MLIR_AIE`, so the remaining route is reversing a closed binary.
- **Nanbeige's oracle is invalid**: the FLM reference arm scores 2/14 — a reasoning model
  whose `<think>` block never terminates in the budget. Its `0/14` native cannot be read as
  accuracy.

---

## 5. Four hypotheses tested and refuted (recorded so they are not re-derived)

1. **The KV re-pack causes the unified failure.** Refuted: Phi-4's strides already match, so
   the re-pack never runs, and it fails identically.
2. **The unified handoff needs a larger token budget for Nanbeige.** Refuted: 512 tokens, still
   3/13 on the reference.
3. **Llama's failure is the `head_dim` hardcode.** Refuted: 1/20 → 0/20.
4. **The ERT contention result** (throughput falling 83 → 180 → 280 ms/tok) **does not
   reproduce** in any of five configurations, including its own 0.6B + 8B contender mix. Its
   own root cause — `timeout_in_sec=2` TDR teardown — explains it; at 15, contention costs
   nothing measurable.

---

## 6. Method note

This work produced **seven** corrections, six of them from comparisons or diagnoses I
controlled and had not independently validated — including a published claim that a family
answered a trivial prompt wrongly, which a proper oracle then refuted (it answered correctly).

**A difference is guilty until the extraction is re-derived independently — and a diagnosis is
guilty until an experiment separates it.** The engine work in this record needed none of those
corrections: four readable bugs, each with a reproduced symptom and a verified fix.
