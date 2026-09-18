# 4-column engine kernels: built, bit-identical, +15% under concurrency — and exactly what still blocks 2x — 2026-09-18

Follow-on to `RESULTS-4col-spatial-partitioning-2026-09-18.md` (IRON's `flm.GEMM` co-scheduled at
**2.06x** on disjoint 4-column partitions). This step pushes the same idea into the **engine**: build
the engine's own bf16 GEMM kernels 4-wide, run them, and measure what two instances do.

## The kernels build and the engine accepts them

The engine has its own bf16 GEMM path (`NPU_BF16=1` → `final_bf16_<proj>_K<K>_N<N>.xclbin` +
`insts_bf16_*.txt` from `$NPU_XCLBIN_DIR`), produced by `generators/build_bf16_xclbins.sh` /
`n1_core_bf16_v1.py`, which already takes a column count (`-c`).

One thing was missing: the generator hardcodes `aie.device(npu2)` in the emitted MLIR, so a `-c 4`
design *still stamped `column_width=8`*. Emitting `aie.device(npu2_4col)` instead fixes that — the
design visibly uses columns 0–3, and the descriptor now tells the truth:

```
final_bf16_QKV_K1024_N4096.xclbin  84384 B   column_width=4  start=['1']
```

Built for the Qwen3-0.6B shapes (QKV K=1024/N=4096, O 2048/1024, GU 1024/6144, D 3072/1024) into
`~/xclbins-4col-dev` with `benchmarks/flm_engine_4col_build.sh` (patched copy of the stock script,
so `$G` still resolves; nothing in the repo's generator is modified).

**Correctness: bit-identical.** `NPU_BF16=1 NPU_XCLBIN_DIR=~/xclbins-4col-dev` on the 1k prompt
returns the same 16 tokens as the 8-column build, with prefill 510 ms vs 522 ms and decode
12.3 ms/tok (81 tok/s) vs 12.6 (79).

## What two instances do (1k prompt, 256 tokens, paired runs)

| kernel set | single | two at once (A / B) | aggregate | vs solo |
|---|---:|---|---:|---:|
| **4-column** | 77 tok/s | 38 + 40 | **78 tok/s** | **1.01x** |
| 8-column (control) | 77 tok/s | 34 + 34 | 68 tok/s | 0.88x |

Both instances in both configurations produced token streams **identical to the single run** (256
tokens each), so correctness is unaffected by concurrency.

So the 4-column set is a real improvement under load (+15% aggregate, and instances keep ~50% of
solo against ~44%), **but it is nowhere near the 2x the IRON probe showed** — and the reason is
diagnosable, not mysterious.

## Why not 2x: only the prefill is ours

The 4-column kernels cover the **prefill GEMMs**. The rest of the engine's hot path is still
FastFlowLM's full-width binaries:

| phase | kernel | width |
|---|---|---|
| prefill GEMMs (QKV/O/GU/D) | ours, rebuilt here | **4 columns** |
| attention (prefill) | `attn_mha_*.elf` | 8 columns |
| **decode (whole-layer, per token)** | **FLM's `layer.xclbin`** | **8 columns** |
| dequant | FLM's `dequant.xclbin` | 8 columns |

At 256 decode tokens the decode dominates the wall time, and a context built from FLM's 8-column
`layer.xclbin` occupies the whole array — so two instances share it no matter how narrow our GEMMs
are. That is the 1.01x: one device's worth of decode, split evenly. The +15% over the control comes
from the prefill and the reduced partition contention on our side.

**The remaining item is therefore a 4-column whole-layer decode kernel.** IRON has the building
blocks (`dequant`, `mha`, `rms_norm`, `rope`, `silu`, `swiglu_prefill`); FLM's fused `layer` has no
IRON equivalent, so this is a design task — and it is now the *only* thing between the engine and
the 2x that the GEMM probe already proves the hardware can do.

## Chess: installed and licensed, but not drivable from this flow

The user asked for Chess (`xchesscc`) to be used. Status after trying:

```
/home/bcloud/Xilinx/2026.1/Vitis/aietools/bin/xchesscc  → chesscc version X-2025.06#865883355f
/home/bcloud/Xilinx2025/2025.2/Vitis/aietools/bin/xchesscc → V-2024.06
~/.Xilinx/Xilinx.lic (47 KB); XILINXD_LICENSE_FILE / LM_LICENSE_FILE exported in ~/.bashrc
```

`xchesscc` itself is not in `aietools/bin`'s PATH dependencies — the mlir-aie `aiecc` requires
**`xchesscc_wrapper`**, which exists only in the IRON/mlir-aie source trees
(`~/iron/bin/xchesscc_wrapper`, `~/mlir-aie-main/tools/chess-clang/xchesscc_wrapper`), not in any
Vitis install. With the wrapper on PATH the build gets further and fails inside the wrap step:

```
chess-clang: error: no such file or directory: 'design.mlir.prj/main_input.chesslinked.ll'
xchesscc Failed No such device
```

…with the license exported explicitly, so it is a **wrapper/flow mismatch, not a license problem**.
IRON's own `aiecc` (`~/iron/bin/aiecc`) is the matching half for that wrapper but has a different
CLI (`--no-compile-host`, `--aie-generate-npu-insts` unknown), so it cannot drive this engine's
generator output. `engine/npu/README.md` already says the same thing from the other side: *"Peano
(LLVM-based, shipped with MLIR-AIE toolchain) — produces correct xclbins. Recommended for all GEMM
builds."* Everything in this note is therefore a **Peano** build, and Chess stays available for the
attention kernel as soon as a matching aiecc+wrapper pair is settled.

## Reproduce

```bash
# build the engine's bf16 kernels 4-wide (narrow device in the MLIR, so the partition is truthful)
OUT=~/xclbins-4col-dev bash ~/build4col_dev.sh QKV 1024 4096 4   # also O 2048 1024 / GU 1024 6144 / D 3072 1024
# run the engine on them (bit-identical tokens vs the 8-column set)
NPU_BF16=1 NPU_XCLBIN_DIR=~/xclbins-4col-dev NPU_GREEDY=1 \
    ./engine/npu/build/npu_engine_qwen3_0_6b ~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx 16 /tmp/p_1k.txt
# the concurrency comparison
bash ~/engine4col.sh
```
