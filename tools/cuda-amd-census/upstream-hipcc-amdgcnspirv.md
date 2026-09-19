# Upstream-ready writeup: `hipcc --offload-arch=amdgcnspirv` (ROCm / TheRock)

**Status: NOT A DEFECT — documented retraction. No issue should be filed.**
**Date:** 2026-09-19 · **Harness:** `tools/cuda-amd-census/` · **Goal:** `mu7uk3un-pf0g4n`

This document exists because an ordered goal step required "an upstream-ready writeup" of a claimed
`hipcc` defect. **The defect does not exist**, so the honest deliverable is this record: a report
written to the standard a maintainer would expect, whose conclusion is that the tool is behaving
correctly and that **the original claim traced to a mislabeled command**, not to any `hipcc`
behaviour.

Filing this as a bug report would be actively misleading. It is kept in-repo as the retraction
record and as the reproduction anyone can re-run.

## Summary

`hipcc` **honours** `--offload-arch=amdgcnspirv` and emits a genuinely portable SPIR-V target. An
earlier claim in this repository — that `hipcc` "exits 0 and silently substitutes native arch
detection" — was **false**. It was produced by a command that **omitted the flag entirely** while
its `echo` label asserted the flag was present.

## Environment

| Item | Value |
|---|---|
| Toolchain A | TheRock `_rocm_sdk_devel` `10.1.0a20260822`, HIP `7.16.26332`, AMD clang `23.0.0git` |
| Toolchain B | TheRock `_rocm_sdk_devel` `10.1.0a20260910`, HIP `7.16.26362`, AMD clang `24.0.0git` |
| GPU | AMD Radeon RX 9070 XT (`gfx1201`); host also exposes a Granite Ridge iGPU (`gfx1036`) |
| OS | Ubuntu 26.04.1 LTS |

## Reproduction

```bash
D=<therock>/lib/python3.14/site-packages/_rocm_sdk_devel
L=$D/lib/llvm/bin

cat > /tmp/t.cu <<'EOF'
#include <hip/hip_runtime.h>
__global__ void k(int*o){ if(threadIdx.x==0) o[0]=7; }
int main(){ int *d=nullptr; if(hipMalloc(&d,4)!=hipSuccess) return 1;
  k<<<1,1>>>(d); hipDeviceSynchronize(); hipFree(d); return 0; }
EOF

# The correct invocation: pass the flag.
$D/bin/hipcc -O3 --offload-arch=amdgcnspirv -I$D/include -isystem $D/include /tmp/t.cu -o /tmp/a.out
$L/llvm-objdump --offloading /tmp/a.out
```

## Expected vs actual

| Invocation | Embedded device target | Verdict |
|---|---|---|
| `hipcc --offload-arch=amdgcnspirv` | `hip-spirv64-amd-amdhsa--amdgcnspirv` | ✅ **correct** — portable, no concrete gfx |
| `clang++ -x hip --offload-arch=amdgcnspirv` | `hip-spirv64-amd-amdhsa--amdgcnspirv` | ✅ correct (the AMD blog's method) |
| `hipcc` **with no `--offload-arch`** | `hip-amdgcn-amd-amdhsa--gfx1201` + `hip-amdgcn-amd-amdhsa--gfx1036` | ✅ correct — documented *native detection* behaviour |

Verified across **four** invocation styles (`.cpp`/`.cu` × with/without `-x hip`) on **both** clang
23 and clang 24. `hipcc` emits the portable `amdgcnspirv` target in every case where the flag is
passed, and falls back to native detection only when it is not.

## Root cause of the original false claim: a mislabeled command

The original claim **traced to a mislabeled command** — the label asserted the flag was passed while
the command omitted it entirely:

```bash
echo "=== A) hipcc --offload-arch=amdgcnspirv ==="      # <-- label asserts the flag IS passed
$H -O3 -std=c++17 -I$D/include -isystem $D/include k.cpp -o /tmp/gate   # <-- it is NOT
```

`hipcc` then did the **correct** thing: with no `--offload-arch` it detected the locally installed
GPUs and embedded `gfx1201` + `gfx1036`. That expected output was attributed to the
`amdgcnspirv` flag, and the result was published as a fail-open trap. It was a **misattribution
caused by a mislabeled command**, not a measurement error — the same failure class as the project's earlier "census ran a
55-commits-stale tree" and an overlay that landed one directory too deep.

## Conclusion

- **No defect.** No change is requested of ROCm/TheRock/hipcc. The original claim traced to a
  mislabeled command, so this is a retraction of a claim about hipcc rather than a report on it.
- **No workaround is needed.** `hipcc --offload-arch=amdgcnspirv` is as valid as raw `clang++`;
  earlier guidance in this repository to "use raw `clang++`, never `hipcc`" has been corrected.
- **What was retained from the original work, and is real:** that `amdgcnspirv` + ZCFS work on this
  toolchain. One binary (`md5 4777733dd83a89dfe429f838932a8e94`), built once on ryzen, ran correctly
  on **both** `gfx1201` (token 1201) and `gfx1151` (token 1151), each checked against an independent
  host-side reference derived from `gcnArchName`, with `clock64()` device-execution witnesses.
- **Also real and retained:** `__builtin_amdgcn_is_invocable` returns an opaque
  `__amdgpu_feature_predicate_t` that clang refuses to cast or store — it is legal only in a boolean
  control-flow context, exactly as AMD's SPIR-V blog describes. The WMMA builtin name on this
  toolchain is `__builtin_amdgcn_wmma_f32_16x16x32_f16`, not the `...x16x16x16...` mnemonic shape.

## Lesson recorded for this repository

An `echo` label is not evidence that its command matched it. Before publishing a defect claim:
read the command that actually ran, re-run the variant matrix, and confirm the **control case**
behaves as predicted. Here the control — `hipcc` with no flag producing native targets — was
mistaken for the bug.
