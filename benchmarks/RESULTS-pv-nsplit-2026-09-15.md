# PV N-split implemented — hd > 128 now builds (2026-09-15)

Task `task-families` of goal `mu35shsg-i3hlyi` (family coverage lane). Branch
`family/head-block-loop`, worktree `~/wt/family-head-block`, based on
`4929ae2b1`.

**Status: BUILD-verified on CPU. NOT yet device-verified** — accel0 was held by the
correctness lane (`mu34scbf-4ffm1o`) for the whole window, so the NPU half of the
verification is deliberately open. Nothing here is claimed as a parity number.

## What was blocking

`n1_core_attn.py` asserted `K <= n`, because the PV output tile was the head dim:
`C_ty = (m, n)` is both the QK^T output tile and the PV output tile, so 128 was the
ceiling. The register cites the measured failure this assert replaced: `-K 256
-N 512 -c 8` produced a clean 90192 B xclbin computing **128 of 256 head dims**,
silently (`RESULTS-attention-c2-regression-2026-09-15.md:56-70`). Affected
families: Gemma3-1B/4B and Gemma4 (hd256), Qwen3.5-4B (hd256, and nh16 on top).

## What changed

| file | change |
|---|---|
| `engine/npu/generators/n1_core_attn.py` | `n_hd = K // n` head-dim tiles instead of the `K <= n` assert; C2 object FIFOs gain depth `n_hd`; the core's PV phase and both sequence feeds loop `hi`-major; the V B-tile becomes a `(k,n)` slice of a `K`-wide row (`sizes=[1,1,k,n]`, `strides=[4,4,K,1]`); the C2 writeback emits `n_hd` tasks at `c*(M*K) + hi*(M*n)`. `n_hd == 1` keeps the original single-tile code on every one of those paths. |
| `engine/npu/generators/build_attn.sh` | `NPU_ATTN_K` (default 128), `NPU_ATTN_COLS`, `NPU_ATTN_HEADS`, and `NPU_ATTN_XCLBIN` / `NPU_ATTN_INSTS` overrides so a candidate build never touches the tracked `xclbins/attn{.xclbin,_insts.txt}`. |

The host-side C2 layout is **unchanged**: the tiles of column `c` still land in the
same flat `(M,K)` region at `c*(M*K) + hi*(M*n)`, so a host reader that already
reads `K` columns needs no layout change (its correctness is one of the open
device checks below).

## The stride trap, hit again

The first hd256 build failed:

```
design.mlir:2768:9: error: 'aie.dma_bd' op Stride 2 is 1 elements * 1 bytes = 1 bytes,
                          which is not divisible by 4.
Error: NPU lowering pipeline failed
```

`verifyStridesWraps` checks **every** stride for 4-byte divisibility even when its
dimension has size 1 and the stride is never applied. The V B-tile was written
`strides=[1,1,K,1]`; `strides=[4,4,K,1]` is semantically identical and legal. This
is the same trap the L1 lane documented for the strided A2 writeback
(`device-claim-goal-lane-2026-09-15.txt`), which is why the fix was one line rather
than a redesign.

## Verification done (CPU only)

| check | result |
|---|---|
| hd128 / N=512 / cols 8 rebuilt from the patched generator | `attn_insts.txt` = **`f3d0a132bde24a60`**, byte-identical to the shipped file (`cmp`), so the single-tile path is untouched |
| hd256 / N=512 (`n_hd`=2, `n_grp`=1 path) | builds: **104528 B** xclbin, insts `e5bd92c64dca07fd` |
| hd256 / N=1024 (`n_hd`=2, chunked path) | builds: **101840 B** xclbin, insts `000532f9ecdd97d4` |
| python/`bash -n` syntax | clean; no tracked artifact written during any of the above (all outputs under `/tmp`) |

## Open (needs the device, in this order)

1. Standalone bench, per the L1 verification recipe: `NPU_ATTN_MAX_SEQ=<N> /tmp/ck <xclbin> <insts> <N> 2`
   → 2/2 non-zero C2 **and** NPU `max_abs_err` == EMU `max_abs_err` to the digit.
   The EMU path mirrors the chunked core host-side (`npu_attn_ctx.h`), so the
   host-side contract can be checked before the NPU run.
2. Engine-driven token identity for Gemma3-1B (hd256, nh4) against the family's
   FLM reference, which is the family verdict this unblocks.
3. Only then: the same for Qwen3.5-4B, which needs the head-block loop
   (nh16 > cols 8) **in addition** to this N-split.

Do not treat the build success as parity: the register's own rule applies — a
number without its gate is a claim, and this document makes none.
