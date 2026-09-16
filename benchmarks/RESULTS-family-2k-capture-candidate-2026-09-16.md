# Family nh20 attention: a captured ELF runs, the token identity is inconclusive

Goal `mu35shsg-i3hlyi`, tasks `family-host` / `family-nanbeige`. Date 2026-09-16
~03:55 ADT. Follows `RESULTS-family-2k-selector-hang-2026-09-16.md`.

## What was tried (route C: use FLM's captured nh20 ELF)

The generated nh20 ELF uses the generator/`AttnCtx` ABI; `Bf16Mm` drives FLM's
`(act, out, kv)` ABI, so route (B) is a kernel-ABI adapter, not a small packing
tweak. Route (C) is the practical one, and the loader fix (`c066f1621`) now lets
it be tried. Re-enabled the shape-specific 2k selector (`attn_shaped2k`) and swept
the four ELF sizes that appear in `cap2048` but not `cap1024`
(`{20064, 69344, 266464, 308736}`), installed one at a time as
`attn_mha_2048_nh20_hd128.elf`, with a 180 s bound each:

| captured ELF size | result |
|---:|---|
| 20064 | hung (L3), killed |
| 69344 | hung (L2), killed |
| 266464 | hung (L3), killed |
| **308736** | **rc=0, boot=152349, `Prefill 1500 [bf16]` 1.299 ms/tok (attn 442 ms)** |

So **308736 is the one that runs**; the generated ELF and the other three hang
the device. That is a real step: the family can be driven through the engine with
a captured 2k kernel at 1500 keys.

## Why the identity is not claimed

The I1-clean reference is the engine's own FLM arm
(`NPU_FLM_PREFILL=1 NPU_FLM_DECODE=1 NPU_RUNLIST=0`) on the same id file. On the
same 1500-token prompt:

```
native (308736):  boot=152349
FLM reference:    boot=13
```

They do not match, and a KV-region sweep on the native arm does not reconcile
them:

| `NPU_ATTN_KV_REGION` | native boot |
|---|---|
| default (4194304) | 152398 |
| 524288 | 153643 |
| 1048576 | 152629 |
| 2097152 | 152398 |
| 4194304 | 152349 |
| 6291456 (H=2560 per the code comment) | 152349 |

`boot=13` from the FLM arm is suspicious (it is a special/newline id, not a
content token), which is why the mismatch is reported as **inconclusive**, not as a
native failure. The existing record also notes the native Nanbeige boot is
nondeterministic (1214 / 131718 / 145029 from the same command), so a single-run
token comparison is not yet a gate.

## State

- The `attn_shaped2k` selector experiment is **reverted again** and the Nanbeige
  binary rebuilt from the reverted source, so the tree is back to the CPU fallback
  and nothing hangs. The verified loader fix stays.
- The captured 2k ELF that runs is preserved at
  `/tmp/nbcap/cap2048/elf_0012_308736.bin` with its run log `/tmp/nbcand_308736.log`
  and `/tmp/nb_native1500.log`.
- Next: establish a trustworthy >1024 reference for Nanbeige (the FLM arm's
  `boot=13` must be explained or replaced by FLM's own runtime / the byte-exact CPU
  attention), then re-enable the selector with `elf_0012_308736.bin` and claim the
  identity. Do not re-enable the selector before the reference is sound.

## DECISIVE (2026-09-16 04:05 ADT): 308736 is not the correct kernel either

Same 1500-key prompt, same bf16 path, same one decode token, deterministic
(`NPU_SEED=1 NPU_GREEDY=1`):

| arm | boot |
|---|---|
| `NPU_PREFILL_BF16=1 NPU_ATTN_CPU=1` (byte-exact CPU attention reference) | **456** |
| `NPU_PREFILL_BF16=1` with `attn_mha_2048_nh20_hd128.elf = elf_0012_308736.bin` | **152349** |
| engine FLM arm (`NPU_FLM_PREFILL/DECODE=1 NPU_RUNLIST=0`) | 13 |
| Bf16Ctx path (`NPU_BF16=1`, separate implementation) | 51752 |

The CPU attention is the trustworthy reference (it is the path the engine falls
back to and it is deterministic here), and 308736 does **not** match it: the
candidate runs the device but computes the wrong attention. So **all four**
`cap2048`-but-not-`cap1024` candidates are eliminated — 20064 / 69344 / 266464
hang the device, 308736 runs but is numerically wrong.

Conclusion for this route: either the correct nh20 2048 attention ELF is not among
those four sizes (the size differential is not a sufficient identifier), or the
`Bf16Mm` nh20 KV/region contract is also wrong independent of the kernel. Until
that is separated, the family >1024 identity cannot be claimed. The generated
`n1_core_attn.py` ELF is a different ABI from `Bf16Mm`'s (act/out/kv vs
Q/KT/C2/V/SCR), so route (B) remains an adapter, not a packing tweak.

State unchanged and safe: the shape-specific 2k selector is reverted, the Nanbeige
binary is rebuilt, and the verified loader fix (`c066f1621`) stays.
