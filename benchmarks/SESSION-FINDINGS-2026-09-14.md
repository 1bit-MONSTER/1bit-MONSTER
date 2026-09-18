# Session findings — native NPU prefill/decode vs FastFlowLM (2026-09-12 → 09-14)

Consolidated record of everything established in this session on Qwen3-0.6B and
the rest of the dense Qwen3 family. Where a number is **not** trustworthy it says
so. Nothing here should be quoted without checking the caveat that applies.

---

## 1. The starting claim was false

The crashed agent had recorded "native @1k prefill 2087.7 tok/s, beats FLM by
~40%", verified by "boot == the `NPU_ATTN_CPU=1` reference". Both halves fail:

- The bf16 prefill layer body was **two hardcoded 128-row GEMM batches**
  (`ensure_a` staged `A[0..127]` and `A[128..255]`; host loops were
  `int h0 = npt < 128 ? npt : 128;` and `for (pi = 128; pi < npt; pi++)`). Rows
  ≥ 256 were never computed. `NPU_PREFILL_MAX=1024` ran a 256-token pipeline and
  reported the wall time as 1024-token throughput.
- `NPU_ATTN_CPU=1` is the *same* broken bf16 pipeline with only the attention
  swapped, so the gate compared two variants of the same wrong path.

Re-gated against the byte-exact `NPU_RUNLIST=1` int8 path and
`NPU_FLM_PREFILL=1` (which agree with each other): bf16 returned `boot=44402` at
1024 tokens where both trusted paths said `25`. At 256 all three agreed (1614).

Withdrawn in `ca02e75ac`; evidence in
`RESULTS-bf16-prefill-CORRECTION-2026-09-12.md`.

## 2. The attention blocker was a capture problem, not a kernel problem

The engine embedded `attn_mha_256_nh16.elf` (26 928 B), captured from a **short**
prefill. It was correct to ~512 keys and silently wrong beyond, which capped
native prefill and made @1k unreachable. Multiple workarounds failed: generated
`gen(0,1024)` ELFs (wrong, ~1200x slower at 223 s), per-chunk chunking (wrong
past 2 full chunks), CPU attention (correct, ~15 s).

**The fix was to capture FLM's own kernel at the right context.**
`npu-infer/tools/capture/run_qwen3_prefill` drives the real FLM runtime's batched
`prefill()`; under `LD_PRELOAD=cap_interposer.so` it dumps every `xrt::elf` FLM
builds. Each capture **self-validates**: the harness prints FLM's `GREEDY_NEXT`,
which matched the trusted token (25 at 1024, 220 at 2048) independently.

Installed captures (all in `engine/npu/xclbins/`):

| file | bytes | shape | context |
|---|---|---|---|
| `attn_mha_1024_nh16.elf` | 98 848 | nh16 | 1024 |
| `attn_mha_2048_nh16.elf` | 194 736 | nh16 | 2048 |
| `attn_mha_1024_nh32.elf` | 177 696 | nh32 | 1024 |
| `attn_mha_2048_nh32.elf` | 352 432 | nh32 | 2048 |
| `attn_mha_1024_nh20_hd128.elf` | 177 728 | nh20 | 1024 |

**Capture length semantics (measured):** a capture at length L serves contexts
**≤ L**. A 1024-context ELF used at npt=2048 returned `19841` (wrong); a
2048-context ELF used at npt=1024 returned `25` (correct). So the range cannot be
extended by passing more keys.

## 3. Three engine fixes

1. **Row tiling.** All four GEMM stages (QKV/O/GU/D) walk the prompt in 256-row
   blocks. SiLU output moved to its own buffer — writing it back into `bA` is
   only safe when every block's kernel has already been launched, which a block
   loop cannot guarantee (`bA[pi*IM]` overlaps unread rows of `bA[pi*H]` for
   `H < IM`).
2. **Half of every GEMM launch was wasted.** `get_mm_app` generates the sequence
   with M=256, but `ensure_a` staged 128 real rows and zero-padded to 256, so each
   run computed 128 useful rows plus 128 zeros — and 256 rows took two launches.
   Staging 256 contiguous rows and reading back `g_run_rows` rows took npt=1024
   from 906 ms to ~710 ms.
3. **Cross-block pipelining** (by the concurrent session in this worktree):
   block `i` on batch slot `i & 1`, block `i+2` issued before waiting, so the next
   block's GEMM overlaps the current block's host readback. 1024: 709→525-565 ms;
   2048: 1295→858-912 ms.

## 4. Two correctness traps found the hard way

### 4a. Captured ELFs are (shape, context) specific — guard both

The long-context slot selected on context alone, so Qwen3-4B/8B (nh32) picked up
`attn_mha_2048_nh16.elf` at npt=2048 and returned **112103** where the trusted
path says **220**. The slot now requires a matching pair and otherwise returns
`nullptr` → the caller's CPU attention reference (slow but correct).

| shape | ≤256 | (256,1024] | (1024,2048] | >2048 |
|---|---|---|---|---|
| nh16 (0.6B, 1.7B) | embedded | nh16 1k | nh16 2k | CPU |
| nh32 (4B, 8B) | embedded | nh32 1k | nh32 2k | CPU |
| other (nh20, hd64…) | per-file | per-file | CPU | CPU |

### 4b. `max_l` mismatch produces size-identical but WRONG layer ELFs

The per-context layer ELFs in `npu-infer/captures/txn-elfs*` were all generated
with **`max_l=8192`**. Generating extra contexts with the tool's default
`max_l=32768` yields files of the **same byte size but different content** (the
KV region stride baked into the stream differs), and they produce wrong tokens
while still running.

Determined by regenerating one known context and comparing sha256:

```
max_l=2048  4b7d67df…   max_l=8192  7f0af706…  <-- MATCH (the committed file)
max_l=4096  8c7167a1…   max_l=16384 23213e56…   max_l=32768 cd6fe743…
```

All four dense-Qwen3 dirs are `max_l=8192`; the added ranges were regenerated
with it and now match a fresh regeneration byte-for-byte. **This invalidated an
earlier "decode cliff fix" of mine**, which had been measured at 67 tok/s on
32768-generated ELFs — fast but wrong. Verify any new layer ELF by regenerating a
known context and diffing the hash before trusting it.

## 5. Correctness gates (the only trustworthy numbers under contention)

Gate = boot token equal to the byte-exact `NPU_RUNLIST=1` int8 path.

| model | npt | boot | trusted |
|---|---|---|---|
| 0.6B | 256 / 512 / 1024 / 2048 | 1614 / 220 / 25 / 220 | same |
| 1.7B | 1024 / 2048 | 220 / 220 | same |
| 4B | 1024 / 2048 | 220 / 220 | same |
| 8B | 1024 / 2048 | 220 / 220 | same |

Cross-checked at 2048 for all four: bf16 prefill boot == runlist decode `[1]`.

320 and 768 differ from the int8 path by near-tied argmaxes between the bf16 and
int8 decompositions (bf16 with CPU attention returns the same values), not
attention bugs.

## 6. Measured performance (matched context, free device)

`flm_parity.sh` with `FLM_PARITY_TRUE_NATIVE=1` and a **tokenizer-exact
1024-token prompt. Matching the prompt matters:** the harness's default prompt
tokenizes to 2088 tokens while `flm bench` runs with `max_length=1024`, so the
default invocation compares 2088-context native against 1024-context FLM.

| model | decode tok/s | prefill tok/s | TTFT (s) |
|---|---|---|---|
| Qwen3-0.6B | 79.3 / 77.6 | **1905.2** / 1100.9 | **0.537** / 0.718 |
| Qwen3-1.7B | 40.0 / 40.3 | **1319.3** / 766.6 | **0.776** / 1.030 |
| Qwen3-4B | 19.0 / 19.0 | **673.9** / 408.8 | **1.520** / 1.931 |
| Qwen3-8B | 11.0 / 10.8 | **469.7** / 294.9 | **2.180** / 2.677 |

Native beats on-box FLM on prefill (+59…+73%) and TTFT (19…25% faster) at every
dense Qwen3 size, with a ~1.6-1.7x prefill lead. Decode ties at 1.7B/4B and wins
slightly at 0.6B/8B. Two full passes: native varies <1% on prefill, <1.5% on
TTFT, identical on decode.

This **supersedes** `RESULTS-coverage-qwen3-dense-2026-09-13.md`, which reports
native behind FLM on prefill for 1.7B/4B/8B — those numbers predate the
pipelining. A SUPERSEDED banner was added there.

## 7. Operational hazards (all cost time this session)

1. **Concurrent agents share this worktree and the NPU.** Decode measured
   **29 tok/s and 87 tok/s on identical commands seconds apart** while another
   session ran `npu_engine_qwen3_8b`. Only the boot-token gates are
   contention-independent. Check `ps` before trusting any timing, and re-run.
2. **First run after a rebuild can return a wrong token.** `boot=19` at npt=256
   once, then 1614 on three re-runs plus a CPU-attention run. Re-run a single
   failing gate before calling it a regression — `benchmarks/gate-check.sh`
   documents that this mistake already caused three reverted commits.
3. **The capture harness publishes enormous per-sync dumps.** A 0.6B/2048 capture
   wrote **35 GB**; the 4B/2048 attempt faulted the NPU
   (`aie2_dump_ctx: Fatal error task ID: 0`) and afterwards returned `boot=328`
   at npt=256 until the driver reset the context. The same capture with
   **`CAP_NO_SYNC=1` succeeded cleanly** (`GREEDY_NEXT: 220`, 3 GB). Always set
   it; it does not affect the `xrt::elf` hook that produces the attention ELFs.
4. **`NPU_XCLBIN_DIR` must be exported.** The engine's fallback resolves into
   another worktree (`/home/bcloud/1bit-MONSTER-pi/...`) if unset.
5. The engine prints `=== Prefill <n> ===` on some paths and
   `=== Prefill <n> [bf16] ===` on the bf16 path; `flm_parity.sh`'s npt regex
   required the bare form and silently reported `n/a`. Fixed.

## 8. What is still open

- **Long-context decode for 4B/8B.** 8B measured 5 tok/s at 2048 against FLM's
  10.38, and 4B 10 vs 17.8 — but those runs overlapped another session's work, so
  treat them as unverified. Re-measure on an idle device. 0.6B/1.7B at 2048 also
  need re-measuring for the same reason.
- **Prefill past 2048.** Contexts 2049..2200 are generated; beyond that needs
  another generation pass (and the matching attention capture).
- **Other families.** The capture technique is per-family: the engine now loads
  nh20 (Nanbeige) and has an hd64 nh32 file for Llama, but their gates have not
  been re-run in this session.
- **TTFT is measured, not modelled.** Native TTFT equals its prefill wall time;
  FLM's reported "prefill speed" is not `tokens/TTFT`, so the two metrics
  disagree by construction. Compare TTFT to TTFT.

## 9. Provenance

Mixed session. From this session: the withdrawal, the 1024/2048 nh16 captures,
the 2048 nh32 capture, row tiling, the M=256 staging fix, the shape+context
guard, the `max_l` discovery, the per-context ELF range work, and the
re-measurement. From a concurrent session in the same worktree: the cross-block
pipelining, the nh32/nh20 1024 captures, and `benchmarks/gate-check.sh`.
