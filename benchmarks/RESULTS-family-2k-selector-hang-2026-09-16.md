# Family bf16 attention: the nh20 ELF loads, but selecting it hangs the prefill

Goal `mu35shsg-i3hlyi`, tasks `family-host` / `family-nanbeige`. Date 2026-09-16
~03:20 ADT. Follows the loader fix (commit `c066f1621`,
`RESULTS-bf16mm-attn-loader-fix-2026-09-16.md`).

## What now works

With the loader fix, `NPU_PREFILL_BF16=1` on Nanbeige loads the generated
shape-specific ELF — the artifact `task-family-nanbeige` needed:

```
Bf16Mm: attention ELF loaded (482960 B): .../attn_mha_2048_nh20_hd128.elf
Bf16Mm: attention ELF loaded (154528 B): .../attn_mha_1024_nh20_hd128.elf
```

The loader searches `attn_mha_<tok>_nh<NH>_hd<HD>.elf` first, so the nh20 files
are found by filename, no code change per family.

## What does not work: the selector is nh16/nh32-only

`run_attn`'s slot picker refuses the shape ELF for nh20:

```cpp
else if (attn_tokens > 1024)
    // Only nh16/nh32 have 2k captures. Any other shape must fall through
    // to the CPU reference rather than borrow the nh32 kernel ...
    kern = nh16 ? attn_kernel2k : nh32 ? attn_kernel2k32 : nullptr;
```

Nanbeige (qout 2560 → nh20) matches neither, so `kern = nullptr` and the run
prints `bf16 attn: npt 1500 outside the verified envelope (<=256, or exactly 512)
— CPU attn_omp fallback` and is correct-but-slow (`Prefill: 54521ms`).

## The experiment: allow it, and it hangs

Patched the picker to use the slot's kernel when the shape-specific file was loaded
for this shape (`attn_shaped2k`), rebuilt, and re-ran the same 1500-key Nanbeige
`NPU_PREFILL_BF16=1` command:

```
Bf16Mm: attention ELF loaded (482960 B): .../attn_mha_2048_nh20_hd128.elf
=== Prefill 1500 [bf16] ===
  L1 L2 L3 ... L11        <- and stops
```

The process sat at layer 11 for >25 minutes (two polls 10 minutes apart, tail
unchanged) and had to be killed; the CPU-attention run of the same prompt finishes
in 52 s. So the generated nh20/2048 ELF, which **passes its bench gate**
(NPU==EMU at seq=2048, `RESULTS-family-attn-nh20-gate-2026-09-16.md`), does not
satisfy the engine's bf16 prefill call contract and wedges the device.

That is consistent with the existing rule in this code: a captured/generated
attention kernel is only valid for the exact context it was built for, and the
bf16 prefill calls attention in 256-row blocks. The bench drives one `(M=8, N=2048)`
call; the engine drives a different call shape (rows/keys/region). Which of those
differs is the next measurement.

## State

- The loader fix is kept (commit `c066f1621`) — it is verified on the supported
  models (0.6B bf16 600-key tokens identical to `NPU_RUNLIST=1`).
- The `attn_shaped2k` selector experiment is **reverted**; the binaries were
  rebuilt from the reverted source, so the tree behaves as before (CPU fallback for
  nh20 >1024) and nothing hangs.
- `task-family-nanbeige`'s token identity above 1024 keys is therefore **not yet
  met**, and the bench-gate half stands. The next step is to compare the two call
  contracts directly — dump the `run_attn` arguments (rows, keys, region stride,
  BO sizes) at npt=1500 and compare them with what the generated ELF's stream
  expects — before the picker is re-enabled.
