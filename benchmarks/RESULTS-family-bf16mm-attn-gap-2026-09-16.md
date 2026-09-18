# Family >1024: the bench gate is met, and the engine gap is a compile-time one

Goal `mu35shsg-i3hlyi`, tasks `family-nanbeige` / `family-host`. Date 2026-09-16
~02:40 ADT. Follows `RESULTS-family-attn-nh20-gate-2026-09-16.md` (whose FAIL was
corrected there: the gate PASSES with a current bench binary).

## Done this round

**1. The nh20/nkv4 bench gate passes** (corrected in the gate doc): NPU == EMU at
seq=2048 (`max_abs_err=8.575258e-02`, C2 2/2) and seq=513 (`2.377548e-01`, C2 2/2).

**2. The nh20/2048 preemptible ELF is now built and installed.** The family
`build_attn.sh` already exposes `NPU_ATTN_ELF`; built with

```
NPU_ATTN_N=2048 NPU_ATTN_COLS=4 NPU_ATTN_HEADS=20 NPU_ATTN_NKV=4 \
NPU_ATTN_ELF=/tmp/attn_mha_2048_nh20_hd128.elf bash build_attn.sh
# -> attn_mha_2048_nh20_hd128.elf, 482960 B, ELF32 / OS-ABI 45 / ABI 2 (xrt::elf)
```

and installed at `engine/npu/xclbins/attn_mha_2048_nh20_hd128.elf`, which is the
exact shape-specific name `Bf16Mm::load_attn_elf` searches for
(`attn_mha_<tokens>_nh<NH>_hd<HD>.elf`, tokens=2048, nh = attn_qout/attn_hd =
2560/128 = 20).

**3. The engine drives Nanbeige at 1500 keys — but not through that ELF.**

```
NPU_BF16=1 NPU_ATTN_KV_REGION=4194304 NPU_PREFILL_MAX=2048 \
  ./engine/npu/build/npu_engine_nanbeige4_1_3b \
  ~/.config/flm/models/Nanbeige4.1-3B-NPU2/model.q4nx 4 /tmp/nb_ids1500.txt
# -> === Prefill 1500 [fallback] ===  Prefill: 54521ms (36 ms/tok)
#    [0] boot=51752   rc=0
```

## Why the ELF is not used: the loader is compiled out

`npu_engine_bf16_mm.h`:

```cpp
#if defined(__has_embed)
#  if __has_embed("../xclbins/attn_mha_256_nh16.elf")
#    define BF16MM_HAS_ATTN_ELF 1
...
#ifdef BF16MM_HAS_ATTN_ELF
    ... load_attn_elf(...)   // the whole shape-specific loader, incl. nh20
#endif
```

The Nanbeige run log contains **zero** `Bf16Mm:` lines — neither
"attention ELF loaded" nor "no 2048-context attention ELF" — which means the
entire loader block (the `#ifdef BF16MM_HAS_ATTN_ELF` region at line 189) was
compiled out: the compile-time `__has_embed` probe of
`../xclbins/attn_mha_256_nh16.elf` did not resolve from the translation unit, so
`BF16MM_HAS_ATTN_ELF` is undefined and the shape-specific name is never even
searched. The 1500-key run therefore used the CPU attention reference (36 ms/tok
prefill), which is correct-but-slow — the same honest status
`RESULTS-family-attention-shape-2026-09-14.md` records.

So the remaining work is not kernel math and not the host packing in `AttnCtx`
(that is verified); it is that **`Bf16Mm`'s attention-ELF loader must actually
compile and run**, so the installed `attn_mha_2048_nh20_hd128.elf` is found. That
is the concrete next edit: make the `__has_embed` path resolve (or gate the block
on the file's presence at runtime instead of compile time), rebuild
`npu_engine_nanbeige4_1_3b`, and only then is the family token identity against
`flm run nanbeige4.1:3b` above 1024 keys a real test.

`NPU_ATTN_KV_REGION=4194304` (the contract's 4096-bucket override for the
2 MB-undersized H table) was set on the run above; it is only meaningful once the
NPU attention path is reached.
