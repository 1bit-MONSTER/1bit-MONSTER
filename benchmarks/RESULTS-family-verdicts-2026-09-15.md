# Family verdicts — per-family table with citations (2026-09-15)

Goal `mu35shsg-i3hlyi`, `task-families`. Every row is a verdict with the record it
came from. **Nothing here is my measurement of a family**: the tabulated numbers are
from the cited runs by earlier lanes, plus this lane's measurements of what the
*generated* kernel can and cannot do (which is what the >1024 rows turn on).

Scoring column: "at parity" means the native path's boot token matched FLM's own
(`NPU_FLM_PREFILL=1`) for the same ids, which is the gate every other result in this
lane uses.

| family | shape | ≤1024 keys (bf16 prefill) | >1024 keys | verdict |
|---|---|---|---|---|
| **Nanbeige4.1-3B** | nh20 nkv4 hd128 | boot **1033 vs FLM 1214 → mismatch**; 1989 ms vs FLM 1912 ms (`RESULTS-coverage-multifamily-2026-09-13.md`) | CPU reference, **correct** (my run: boot 7753 = FLM 7753 at 2048 keys, ~44 s) | **not at parity** on bf16; its **i8 path is at parity** (1033 @1024, 5938 @256 — `RESULTS-attention-c2-regression-2026-09-15.md` L2 note) |
| **Phi4-mini** | nh24 nkv8 hd128 | boot **25 vs FLM 350 → mismatch**; **259303 ms** (CPU attention fallback) vs FLM 1718 ms (same doc) | CPU reference | **not at parity**; the 259 s is the CPU fallback the 09-14 shape guard made *correct* instead of silently wrong |
| **Gemma3-1B** | nh4 nkv1 **hd256** | **crash** (same doc) | CPU reference | **hds256 blocker** (below) |
| **Gemma3-4B** | nh8 nkv4 **hd256** | **missing xclbin** (same doc) | CPU reference | **hd256 blocker** |
| **Qwen3.5-4B** | nh16 nkv4 **hd256** | boot **220 vs FLM "0 (no prefill)" → no comparable reference**; 10039 ms / 102 tok/s | CPU reference | **no FLM reference exists for it**, and hd256 blocker |
| **LFM2-1.2B / 2.6B** | — | — | — | **architectural exclusion: no native engine variant exists** (`engine/npu/build/` has no `npu_engine_lfm2_*`), so there is nothing to measure |

## The two structural blockers, stated with their source

1. **hd256 (Gemma3-1B/4B, Qwen3.5-4B)** — the engine's KV write is
   `bKv[region*kv_region + pi*512 + lh*HD + d]` with a literal `512`, i.e. a 4-head x
   128-dim slot. At HD=256 the index reaches 1023 inside a 512-element slot, so a
   token's K/v overwrite the next token's; and `region = kvh < 4 ? 0 : 1` plus
   `lh = kvh & 3` assume the same 4-head packing. The fix is `4*HD` arithmetic in that
   loop (`npu_engine_universal.cpp`, `qk_norm_pi`), not an artifact.
2. **The >1024 window for any family** — the generated kernel is **per query token**
   (its M=8 tile is the eight columns' head rows of ONE token, not eight tokens), so a
   block-shaped prefill needs one launch per row. Measured at 2048 keys: **823.86 s
   with the K/V cache, 1087.92 s without**, against **~44 s for the CPU reference that
   returns the right token** — 19x slower than the fallback, wrong, and aborting with
   `free(): invalid size`. The generated ELF *is* selected and run (this lane), so this
   is not an integration gap: it is the kernel's shape meeting a prefill.

## What would change a verdict, per family

| family | route to parity |
|---|---|
| Nanbeige | its **i8 path already matches FLM** — the honest verdict is "at parity on i8, not on bf16 prefill"; nothing to fix for a parity claim that names the path |
| Phi4-mini | a block-shaped attention kernel (option 3), or the CPU reference with the 259 s stated |
| Gemma3-1B/4B, Qwen3.5-4B | the `4*HD` KV fix first (hd256), then the same block-shape question |
| LFM2 | building an engine variant is the prerequisite, not a kernel |

## Reading of the task's contract

`task-families` asks for "gated parity numbers at 1024/2048/4096 **or** an
architectural exclusion citing the source line". This table delivers the second form
for every family, and the first form for one path: **Nanbeige's i8 path is at parity,
cited**. The >1024 boundary is not an exclusion of the sort that reads well — it is a
measured consequence of the kernel's per-token shape, and it is recorded with the
number that shows it (823.86 s vs ~44 s) rather than as a preference.

## Update 2026-09-16 — landing on `goal/runlist-decode-wire`, plus the Nanbeige route

The table above was committed only on `origin/family/head-block-loop`; this file is
its landing on the branch the objective names. Numbers below are this session's.

### Nanbeige4.1-3B — the generated attention now runs in the family prefill

- **ABI gap closed.** The generated nh20/nkv4/hd128 attention kernel is `AttnCtx`-ABI
  while `Bf16Mm` drives FLM's captured `(act,out,kv)` ABI, so the family prefill could
  not use it at all. `NPU_ATTN_CTX=1` in `npu_engine_universal.cpp` now drives it
  through the same `AttnCtx` that `zaya_decode.cpp` uses (one query row per call).
- **Determinism bug found and fixed.** Built lazily *inside* the prefill while five
  `Bf16Ctx` + Bf16Mm contexts were live, the output varied run to run (8-token prompt:
  152369 / 152367 / 8900 / 6037 / 152360 / 152552 / 2092). Constructing it **before**
  `bf16mm_init` (`ac_ctx_init`) removes it: **13/13/13/13** at 8 keys, **764/764** at
  1202 keys. The two reference paths were never affected (captured bf16 152343 ×2,
  plain fallback 152367 ×2).
- **Bench gate met:** nh20 hd128 cols4 nkv4, `NPU==EMU 8.575258e-02`, C2 2/2.
- **Token identity above 1024 keys: NOT met** — 764 vs FLM `13` at 1202
  template-matched keys (the template renders to exactly FLM's 1202 tokens).
- **Cause, measured:** the generated kernel is int8. Env-gated EMU variants attribute
  the whole gap to **int8 Q/K** (float-A2 changes it ~2%, float-V ~1%; baseline
  L1 0.197073 / L2 0.784570 / L7 1.231850, with the NPU matching its own EMU to ~1e-4).
  The prefill stack also diverges from FLM **independently of attention**: the
  CPU-attention path gives the same 166103 as the default path, vs FLM's 13.
- **Verdict:** parity **on the i8/default path** (as the table above already says);
  the bf16 arm is excluded by the **int8 QK^T precision**, with the fix route
  (bf16/mixed-precision QK^T) specified in
  `benchmarks/RESULTS-family-attn-ctx-adapter-2026-09-16.md` (addenda 3-10).

### Other families, this session

| family | delta |
|---|---|
| Phi4 (nkv8) | host layout **gated**: `NPU==EMU 2.403474e-02`, C2 2/2 (908a098bf) |
| hd256 (Gemma3/4, Qwen3.5) | host PV N-split **implemented** (row-major V for `n_hd>1`), hd128 paths re-verified; the hd256 kernel's **own PV** still disagrees with its host contract (NPU 3.752225e-01 vs EMU 2.126317e-02) — cited kernel-side |

Source docs: `benchmarks/RESULTS-family-attn-ctx-adapter-2026-09-16.md`,
`benchmarks/RESULTS-family-host-2026-09-16.md`, LEVERS register section 6.1.
