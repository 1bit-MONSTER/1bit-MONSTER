# The bf16 attention-ELF loader was compiled out under g++ — now it runs

Goal `mu35shsg-i3hlyi`, tasks `family-host` / `beyond8192` support. Date 2026-09-16
~02:50 ADT. No NPU number in this doc lacks its command.

## The defect

`engine/npu/src/npu_engine_bf16_mm.h` embeds FLM's fixed 256-token nh16/nh32
attention ELFs and gates **the entire long-context attention loader** on that:

```cpp
#if defined(__has_embed)
#  if __has_embed("../xclbins/attn_mha_256_nh16.elf")
#    define BF16MM_HAS_ATTN_ELF 1
...
#ifdef BF16MM_HAS_ATTN_ELF
    ... load_attn_elf(...)   // shape-specific attn_mha_<tok>_nh<NH>_hd<HD>.elf
#endif
```

`__has_embed` is a **Clang** feature; the engine is built with `g++`
(`CXX="${CXX:-g++}"` in `build_npu.sh`). Under g++ `defined(__has_embed)` is false,
so `BF16MM_HAS_ATTN_ELF` is never defined and the loader — including the
shape-specific filename search that would let a family drop in
`attn_mha_2048_nh20_hd128.elf` — is compiled out. The `Bf16Mm` bf16 prefill path
then always falls back to the CPU attention reference for npt>256, and the only
sign was its absence: no `Bf16Mm:` line in the log.

## The fix

Compile the loader unconditionally and gate it on `attn.xclbin` existing at
runtime; keep the embedded-ELF construction under `BF16MM_HAS_ATTN_ELF`
(never true under g++, harmless under clang). A missing `attn.xclbin` now prints a
warning and disables the loader instead of failing the arm.

## Verification

**Short prompt** (`NPU_BF16=1`, 5 tokens, France): the loader now reports every
runtime ELF and the answer is right:

```
Bf16Mm: attention ELF loaded (98848 B): .../attn_mha_1024_nh16.elf
Bf16Mm: attention ELF loaded (194736 B): .../attn_mha_2048_nh16.elf
... 256/1024/2048/4096/8192 nh16 + nh32 ...
Prefill 5 [bf16]   Prefill: 302ms [GEMM 23ms, attn 126ms, conv+other 297ms]
=== 11.8 ms/tok (84 tok/s) ===  -> " Paris, and the capital of Italy is"
```

**Long prompt** — the path this fix actually changes (npt>256 now takes the NPU
1024-context ELF instead of CPU attention), same 600-token id file, 4 greedy
tokens:

| arm | first tokens | ms/tok |
|---|---|---|
| `NPU_BF16=1` (patched, uses `attn_mha_1024_nh16.elf`) | **16 62 15 1889** | 12.1 (83 tok/s) |
| `NPU_RUNLIST=1` (byte-exact reference) | **16 62 15 1889** | 9.2 (109 tok/s) |

Identical output. The prefill dropped from the CPU-attention cost to
`attn 177ms` for 600 keys.

## Scope

This fixes the **`Bf16Mm`** bf16 path (Qwen3 dense). The Nanbeige family bf16 arm
runs a different implementation — `Bf16Ctx` with the per-op `final_bf16_*`
xclbins — and its log contains no `Bf16Mm:` lines, so the nh20/2048 ELF built and
installed for `task-family-nanbeige` is still not reached there. That wiring (or
proving the family path needs its own hook) is the remaining `task-family-host`
item. What is fixed here is real and gated: the supported models' bf16 prefill now
uses the runtime attention ELFs their filenames already existed for.
