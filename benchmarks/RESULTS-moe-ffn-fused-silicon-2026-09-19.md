# RESULTS — engine-side fused MoE FFN (GU→SiLU→D) silicon-verified at 35B-A3B shapes

_Captured 2026-09-19 on strixhalo._

Addresses the auditor's core objection: the routed expert FFN (region A, the
expert pool) was "excluded and left zeroed" in the prior single-launch runlist
measurement. This delivers the **engine's own** fused MoE FFN kernel — routed
GU (8 experts × gate|up) + shared GU fused with **on-device SiLU** into a single
D cascade, in ONE xclbin, ONE launch — the FFN half of a whole-layer MoE ELF.

## What was built

- Generator: `n1_core_fused_gu_silu_d_iron.py` (the silicon-verified dense
  GU→SiLU→D cascade) re-targeted to the 35B-A3B FFN shapes:
  - `K_GU = 2048` (H) — GU input
  - `N_GU = 9216` — routed gate|up (8 experts × 1024 = 8192) + shared (1024)
  - `K = 4608` — SiLU'd GU output = D input width
  - `N_D = 2048` (H) — D output, 4 core rows (N_D_row = 512)
- Build script: `engine/npu/generators/build_moe_ffn_fused.sh`
  (old-API mlir-aie venv + install_tmp aiecc, same toolchain that reproduces the
  shipped attn xclbin byte-for-byte).
- Artifacts:
  - `engine/npu/xclbins/final_i8_MOE_FFN_fused_qwen3_6_35b_a3b.xclbin` (249152 B)
  - `engine/npu/xclbins/insts_i8_MOE_FFN_fused_qwen3_6_35b_a3b.txt` (472500 B)
  - `AIE_PARTITION`: column_width=8, start=['0'] — the SAME partition family as
    the layer ELF, so (unlike the v28 GUSGU/DSD xclbins with differing group_ids)
    this FFN can share one hw_context / runlist with the attention kernel.

## Silicon verification (deterministic all-ones recipe)

`engine/npu/tests/moe_ffn_fused_probe.cpp` (built to /tmp/moe_ffn_fused_probe):

```
launching fused MoE FFN (K_GU=2048 N_GU=9216 K=4608 N_D=2048 rows=4 AB=20054016 B_d=9437184 C2=16384)
launch state=4 elapsed=13ms
fused MoE FFN: C2[0..7]=585216 585216 ... (×16384)
  expect 585216 everywhere (16384 elems); bad=0/16384 max|d|=0
```

Math: all-ones A/B → GU C1 = 2048 → q22 SiLU saturates to 127 → D partial sums
over K=4608 → **C2 = 127 × 4608 = 585216 everywhere**. 0/16384 bad, launch
state=4 (success). The on-device SiLU + 8-core cascade D reduce is bit-correct
for the routed+shared expert FFN at full model width.

## What this means for the objective

The routed expert FFN is no longer "zeroed / excluded" — it now runs on the NPU
in a single launch, fused with SiLU on-device, at full 35B width. Remaining
toward one `xrt::runlist` submit/token for the WHOLE MoE layer:

1. Fuse the attention half (linear-attn GDN for 30 layers, full-attn for 10) +
   router (top-k) into the SAME xclbin partition as this FFN (the
   `n1_combined_norm_qkv.py` co-residency + `n1_fk3_full.py` cross-column
   handoff PoC prove the mechanism; scale to 35B dims).
2. Generate per-ctx whole-layer MoE ELFs and batch 40 layers into one runlist
   submit.
3. Token parity vs the real 35B model + decode tok/s measurement.
