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

## Paired host-side changes (NOT made here — engine untouched)

The generator's `runtime_sequence` signature changed for H > cols, so the engine
side must move with it before any of this can be measured:

1. q BO: `max(16, PARAM_ROW + 1) * K_FRAME` bytes, i.e. all H head rows plus the
   params row (params move to row `H` when `H > 15`).
2. C2 BO: `n_heads * M * K` int32 instead of `n_aie_cols * M * K`.
3. Feed `n_hpass` passes per attention call, and the KV head for global head h is
   `h / gqa`, not `h / 4`.
4. Both live in the `Bf16Mm` attention path (`npu_engine_bf16_mm.h`) and its
   caller, not in the generator.

## Open (device, in this order)

1. Standalone bench for one new shape: `NPU_ATTN_MAX_SEQ=<N> /tmp/ck <xclbin> <insts> <N> 2`
   → 2/2 non-zero C2 and NPU `max_abs_err` == EMU `max_abs_err` to the digit. The
   `ck` bench hardcodes NQ/NKV/HD, so it needs its own rebuild for nh20/nh24/hd256.
2. The engine-side changes above, then the family token-identity check against each
   family's FLM reference (Nanbeige `nanbeige4.1:3b`, Phi4 `phi4-mini-it:4b`).
3. Only then can the family verdict table carry numbers rather than "builds".
