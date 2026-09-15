# Head-block loop implemented — nh > cols and hd > 128 now build (2026-09-15)

Task `task-families` of goal `mu35shsg-i3hlyi`. Branch `family/head-block-loop`,
worktree `~/wt/family-head-block`, on top of the PV N-split commit `10a766324`.

**Status: BUILD-verified on CPU. NOT device-verified** — accel0 was held by the
correctness lane (`mu34scbf-4ffm0o`'s dense 20-prompt run) for the whole window.
No parity number is claimed anywhere in this document.

## What was blocking (the last structural gap)

The design is one core column per q head, fed from q row `c*K_FRAME`. A model with
more heads than columns therefore got a kernel for only the first `cols` heads:
the xclbin built, loaded and ran, and the host silently received the wrong heads.
Affected: Nanbeige nh20, Phi4 nh24, Qwen3.5-4B nh16.

## What changed (`engine/npu/generators/n1_core_attn.py`)

| piece | change |
|---|---|
| head blocks | `n_hpass = H / cols` with `assert H % cols == 0` (a partial block would read past the q BO). `--heads` is now the real head count, not an alias for `--cols`. |
| GQA | `--nkv` added; `gqa = cols / nkv` asserted to divide; every `cc // 4` becomes `(hp*n_aie_cols + cc) // gqa` |
| host sequence | the whole feed is wrapped in `for hp in range(n_hpass)`; q-tile offsets become `(hp*cols + c)*K_FRAME + ki*k`; the C2 writeback goes to `(hp*cols + c)*(M*K)` |
| params row | `PARAM_ROW = 15 if H <= 15 else H` — row 15 is the padding row the params ride in today, so `H <= 15` keeps the old offset and `H > 15` moves the params out of the head range |
| BO sizes | q frame `max(16, PARAM_ROW+1) * K_FRAME`; C2 `n_heads * M * K` (was `n_aie_cols * M * K`) |

`n_hpass == 1` is the original single-pass path and stays byte-identical: the
Python-level loops unroll and every offset/expression collapses to the old form.

## Verification done (CPU only)

| case | shape | result |
|---|---|---|
| **single-pass guard** | hd128, nh8, cols8, nkv2, N512 | xclbin 95440 B; `attn_insts.txt` **byte-identical** to the shipped file (`cmp` clean, `f3d0a132bde24a60`, 45936 B) |
| Nanbeige | hd128, **nh20**, cols4, nkv4, N512 | builds, 51130 B xclbin, insts `a82c3cb21cdb7b7d` |
| Phi4-mini | hd128, **nh24**, cols8, nkv8, N512 | builds, 95440 B xclbin, insts `c025631a75051df8` |
| Qwen3.5-4B | **hd256**, **nh16**, cols8, nkv4, N512 | builds (both changes at once), 104528 B xclbin, insts `c2ec8b63b7670a28` |
| Nanbeige @2k | hd128, nh20, cols4, nkv4, **N1024** (chunked + head blocks) | builds, 55802 B xclbin, insts `d84841898ad30d94` |

## A harness bug worth recording (caught before it produced a wrong claim)

The first verification pass reported the guard as DIFFERS while printing the correct
hash. Cause: `NPU_ATTN_INSTS=/tmp/hb_$name_insts.txt` — bash parses `$name_insts`
as one identifier, so every build wrote the same `/tmp/hb_.txt` and the `cmp` was
comparing a file that did not exist. Fixed with `${name}` in braces and a fresh
prefix. This is the same failure class the correctness-lane goal lists as a binding
rule ("assert the extraction belongs to the run you think it does") — a stale or
shared output path is the cheapest way to manufacture a false result.

## Host side (`AttnCtx` + the bench) — made here, compile-verified

`src/npu_attn_ctx.h` now checks shapes against the kernel's *structure* instead of
pinning them to the one built configuration:

| was | now |
|---|---|
| `if (nq != 8 \|\| nkv != 2 \|\| hd != 128) unsupported` | `nq % cols == 0` (one column = one head), `nkv` divides `cols`, `hd` a multiple of 128 and `<= K_FRAME` |
| `cols` implicit (8) | `cols` from `NPU_ATTN_COLS`, and `scrsz` scales with `cols` (the A2 scratch is per column, reused per pass) |
| params hard-wired to row 15 | `PARAM_ROW = 15` for `nq <= 15`, else row `nq`; `qsz = max(16, PARAM_ROW+1) * K_FRAME` |
| C2 `nq*8*hd` | unchanged — it already matches the generator's `n_heads*M*K` |

For the shipped shapes (`nq=8, nkv=2, hd=128, cols=8`) every one of these
collapses to the previous value, so the existing lane is behaviourally identical.
`AttnCtx` is included only by `zaya_decode.cpp` and the bench, **not** by the
dense `Bf16Mm` path.

`tools/attn_kernel_bench.cpp` takes `CK_NQ` / `CK_NKV` / `CK_HD` from the
environment (its ground truth was already shape-agnostic), and the whole thing
compiles: `g++ -std=c++17 -O2 -mavx2 -I src -I generators -o /tmp/ck2
tools/attn_kernel_bench.cpp -lxrt_coreutil -lxrt_core -laiebu -luuid -ldl`.

**Still open, and now the only engine-side gap:** the dense path that the family
measurements go through (`Bf16Mm` in `npu_engine_bf16_mm.h`, whose attention
selection is the thing the L1 lane patched for nh20/nh24) must pass the real
`nq`/`nkv`/`hd`/`cols` and size its q and C2 BOs the same way before a family can
be driven end to end.

## Open (device, in this order)

1. Standalone bench for one new shape: `NPU_ATTN_MAX_SEQ=<N> /tmp/ck <xclbin> <insts> <N> 2`
   → 2/2 non-zero C2 and NPU `max_abs_err` == EMU `max_abs_err` to the digit. The
   `ck` bench hardcodes NQ/NKV/HD, so it needs its own rebuild for nh20/nh24/hd256.
2. The engine-side changes above, then the family token-identity check against each
   family's FLM reference (Nanbeige `nanbeige4.1:3b`, Phi4 `phi4-mini-it:4b`).
3. Only then can the family verdict table carry numbers rather than "builds".
