# WS-13 — gatman45 XDNA2 cross-validation collab (external, Discord)

**Status:** 🔄 in progress — waiting on gatman45's packed-buffer artifacts
**Surface:** real Discord, 1bit-MONSTER guild `1542589029630087278`, #help forum thread `1545394554956423269`
**Counterpart:** gatman45 (Discord user `497141650812960779`) — Windows-side XDNA2 / Ryzen AI NPU reverse-engineer (ggml-xdna class stack)
**Owner:** npu · **Updated:** 2026-09-05
**Local copy of posted reply:** `research/ws13-gatman45-xdna-collab/1bit-monster-reply.txt`

---

## TL;DR

gatman45 is independently running an INT4 attention-output GEMV on **Windows** (his repo `repo_0808` @ `18b583a`) against the same XDNA2 kernel class we build on **Linux**. He posted a help-forum thread; our bot engaged (independent-validation offer); he responded with a full cross-validated root-cause analysis of a NaN cascade. We replied with agreement on the verdict + an H1 (host/kernel tile-geometry ABI mismatch) assessment + concrete independent checks we can run. **Next step: he exports the packed-buffer dump and we do the byte-level expected-vs-actual diff.**

## Why this matters to us

Same hardware (XDNA2 / Strix Halo-class), same stack class (llama.cpp → ggml → runtime → XRT → NPU), same failure taxonomy ("COMPLETED ≠ correct", geometry-baked-at-compile-time vs runtime request). His Windows findings are a free second data point on geometry-contract bugs we also chase on Linux — and he's working Qwen3.5-9B layer composition (GDN), overlapping our Qwen3.5 GateDeltaNet / Zaya work. The collaboration is Windows↔Linux cross-validation, per his ask: *independent review, not belief.*

## Thread timeline

| When (UTC) | Author | Content |
|---|---|---|
| 2026-09-04 11:26 | gatman45 | Thread opener: ~1y of Windows XDNA2 work; stack LLM/llama.cpp → ggml-xdna → runtime → XRT → Windows NPU driver → XDNA2 → AIE → DMA. Fused path (16 layers/backend call) faster but numerically unsolved; per-op path verified but high overhead. Wants independent technical review/repro. |
| 2026-09-04 11:26 | gatman45 | Looking for: XDNA/XDNA2/Ryzen AI, AIE/AIE-ML/MLIR/IRON, XRT internals, Windows kernel/driver, llama.cpp/ggml, quantization/tensor layouts, runtime design. Discovered earlier high-throughput results passed an insufficient harness → treats throughput and correctness separately. |
| 2026-09-05 02:09 | 1bit-MONSTER (bot) | Our engagement (`1545616871343132683`): Linux side on XDNA2; IRON/MLIR-AIE kernels built+verified; xdna-driver/XRT built from source; layer batching into one `xrt::runlist` (≥2.25); Q4_K→NPU-tile double-quant ~10% rel-L2; bit-exact ggml dequant ports (Q4_K/Q5_K/Q6_K/Q8_0); Qwen3.5 GateDeltaNet layer composition; fused-vs-per-op wall (naive scalar fused dequant+mm 55× slower → JIT specialization); **offer: independent validation** — send fused-path correctness delta and we'll say repack vs sync vs kernel bug. |
| 2026-09-05 16:46 | gatman45 | **Response** (`1545837640748900482`) — attachment `message.txt` (34,396 B; CDN content later dead: "no longer available"): "FULL CROSS-VALIDATED ANALYSIS — NaN / FlowKV / QKV16 / INT4 8-column failure" (05/09/2026). Recovered via operator paste; key points below. |
| 2026-09-05 ~17:35 | 1bit-MONSTER (bot) | Our reply (`1545894273357185154`) + attachment `1bit-monster-reply.txt` (3,244 B; archived here). See below. |

Discord thread messages are read via the bot token (`~/fluxer-mcp/state/discord-bot.json`) — REST API, real Discord (see `~/okf/systems/1bit-monster/references/discord-access.md`). Single-message GET works with a **bot** token; use `?around=` listing otherwise.

## gatman45's analysis (message.txt, condensed)

Test config: `repo_0808` @ commit `18b583a`; **Llama-3.2-1B Q4_0**; QKV16 + FlowKV era + GEMV_INT4(+V2) + SWIGLU_INT4; FlowKV H16_KV4, `kv_h += num_cols`; 624 CONT calls.

**Two-failure split (his core finding):**
- **FAILURE A** — INT4 attention-output GEMV `K2048/N2048/8col` produces **catastrophic garbage (±1e36) in output columns 0–3**; cols 4–7 correct/plausible. First observable corruption.
- **FAILURE B** — FlowKV then switches to **uniform 0x7F81** (persistent across CONT #2..#624); later QKV16 Q-source shows 0x7FC1/0xFFC1 variants (cascade, not primary).

**Refuted as primary:** K/V NaN propagation (K_bad=0, V_bad=0), unstable BO addresses (stable: bo_k 0x3F28000, bo_v 0x3F48000, bo_q 0x3620000, bo_out 0x3621000), dead dispatch (every CONT dispatched/completed ~0.8–1.3 ms), FlowKV era loop (BF16 control clean: 288 dispatches, 0 NaN, sane text).

**Key control:** disabling v2 → v1 runs → same failure. So not a v2 binary bug → shared component (host repack / geometry metadata / strides). BF16 interleaving clean; only INT4 path corrupts.

**H1 (VERY HIGH): host INT4 packing vs compiled kernel tile ABI mismatch.** For K2048/N2048/8col: reported **host m_input = 8** vs **kernel tsi = 4** (BF16-style capacity calc 60416/8192 = 7.375 → tsi 4; packed-fragment calc `frag = K/2 + (K/32)·2` = 1152 B → 60416/(2·1152) = 26.2 → m_input 8). Correlation: K12288/N4096/8col (tsi=2, m_input=2) is internally consistent and clean. **Exact byte/offset mismatch not yet demonstrated** — that is the open proof.

Other ranked hypotheses: H2 HIGH (8-col output stride / column mapping / BD mapping — fits the 4-of-8 split), H3 HIGH (scale/group boundary), H4 MED (accumulator/persistent INT4 state), H5 MED (cross-xclbin/AIE context interaction — candidate for persistent 0x7F81), H6 LOW (FlowKV loop), H7 VERY LOW (QKV16).

**His 10 decisive tests (essence):** dump packed INT4 buffer & derive expected layout; zero-input synthetic GEMV; deterministic INT4 pattern; 4col vs 8col same weights/input; v1 vs v2 on identical packed input; minimal FlowKV interleaving matrix; BO canaries; log every column offset; accumulator-init check; disassembly/xclbin geometry audit (kernel name, MD5, K/N/cols/tsi/tile_out).

**Governance verdict:** reject "FlowKV primary"; accept "INT4 GEMV 8col first"; treat 0x7F81 as downstream persistent failure; proof = byte layout → DMA → GEMV numerics → FlowKV → QKV16 → layer → semantic → perf. NaN guards = safety net only. Fix order: repair GEMV first, never patch 0x7F81 first. Do NOT hard-code K2048 — fix the general geometry contract.

Full original text not archived here (was pasted into the pi session by the operator; CDN copy dead). Re-ask gatman45 for the file if needed.

## Our reply (verbatim in `1bit-monster-reply.txt`)

Core positions sent:
1. **Agree with the split**; retire the "FlowKV broken" model; NaN guards are a net, not a fix.
2. **Shape-dependence** (K12288 clean vs K2048 bad, v1+v2 both fail) ⇒ common geometry-contract bug, not kernel implementation.
3. **Failure-class parallel from our side:** M=128-baked tiling vs runtime M (REG_M can't resize → deadlock) — loud; his misalign-and-compute-garbage is quiet and worse. "COMPLETED ≠ correct" stays governance.
4. **H1 support:** if tsi and m_input describe the same K-grouping it's a contract violation; input-tile dim = row-replication width; 8-row host groups vs 4-row kernel ⇒ scale/weight reads land on every second group boundary ⇒ ±1e36 in half the lanes is the expected signature. 4-of-8 split also fits H2 as a second fault. Fix `select_gemv_tiles()` with packed-fragment capacity model for INT4, not a K2048 special case.
5. **Q4_0 note:** frag formula is Q4_0-consistent (2-byte fp16 scale / 32 rows); Q4_K (256-block, 6-bit scales) is where we measured compounding ~10% rel-L2 from double repack; Q4_0 has fewer places to hide → clean byte-level target.

## Commitments made to gatman45 (agent obligations — follow through if he delivers)

1. **Packed-buffer diff:** given his host packed buffer + scale stream for K2048/N2048/8col, derive expected layout against our ggml dequant reference and diff byte-by-byte — no kernel execution needed. (Note: our existing reference set is Q4_K/Q5_K/Q6_K/Q8_0; his case is **Q4_0** — Q4_0 reference may need to be added; it is the simplest format.)
2. **Synthetic GEMV on our harness:** zero-input + deterministic-pattern for the same geometry; report whether our compiler also selects tsi=4 for K2048 (independent data point on the selector logic).
3. **Descriptor checklist share (if useful):** NoC DMA 32B/8-dword vs Memory-Module 24B/6-dword descriptors, IRON 4D-DMA stride semantics — for his BO-canary/column-offset work.

## Artifacts we asked gatman45 for (a–d) — proves or kills H1

- (a) packed-buffer hexdump with scale offsets,
- (b) actual `select_gemv_tiles()` output for K2048/N2048/8col (verify the 7.375→4 and 26.22→8 figures vs current `compile.py`),
- (c) xclbin MD5 + timestamp for the 4col and 8col binaries (cache staleness),
- (d) confirmation v1 and v2 consumed byte-identical packed input.

## How to resume (next agent)

1. Check the Discord thread (`1545394554956423269`, via bot token REST — see discord-access reference) for a new reply from gatman45.
2. If he delivered (a)–(d): run the packed-buffer expected-vs-actual diff (commitment 1) and the synthetic-GEMV selector check (commitment 2) on strixhalo.
3. Reply in-thread with findings; keep FlowKV parked until the GEMV fault is proven fixed; return to 9B helper-kernel chain after.
4. Update this file (status, artifacts received, results) and `research/TRACKING.md`.
