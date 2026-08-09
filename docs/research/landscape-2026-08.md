# Research Landscape — NPU / LLM / AI (sweep 2026-08-08)

Sweep via academia-mcp (arXiv + Semantic Scholar). Focus: NPU inference, LLM
quantization, 1-bit models, EEG foundation models. All directly relevant to
the 1bit.systems stack (Strix Halo XDNA NPU, Q4NX pivot, spec-decode, TUH EEG).

## NPU / on-device inference

| Paper | ID | Why it matters |
|---|---|---|
| **IRONSmith: Visual Dataflow Design Env for AMD Ryzen AI NPUs** | 2607.10944 | Visual IDE → IRON Python → runs on the Ryzen AI NPU. We have Vitis 2026.1 installed — this is the highest-level NPU programming path yet. |
| **STEEL: Sparsity-Aware Fused Attention on AMD XDNA NPU** | 2607.09385 | Fused attention for long-sequence inference on laptop SoCs — matches our attention kernel work (NPU + GPU). |
| **Striking the Balance: GEMM Optimization Across Ryzen AI NPU Generations** | 2512.13282 | GEMM perf across NPU gens — our WMMA/tiled-GEMV kernels map directly. |

## LLM quantization / 1-bit

| Paper | ID | Why it matters |
|---|---|---|
| **R2Q: Robust 2-Bit LLMs via Residual Refinement Quantization** | 2511.21736 | 2-bit formats beyond naive Q2 — the Q4NX/ternary direction. |
| **PTQTP: Post-Training Quantization to Trit-Planes** | 2509.16989 | Ternary-plane PTQ — the 1.58-bit family our engine targets. |
| **The Era of 1-bit LLMs (BitNet b1.58)** | 2402.17764 | The namesake paper. ~50 citing works per S2 (rate-limited, likely 500+). |

## EEG (TUH direction)

| Paper | ID | Why it matters |
|---|---|---|
| **OmniEEG-Bench: Standardized Benchmark for EEG Foundation Models** | 2606.00815 | We just got TUH EEG access — this is the evaluation harness for EEG FMs. |

## Other angles worth pulling next
- Speculative decoding (repo has spec-decode): 2412.00061, 2605.01106
- Hybrid Mamba/attention (Zamba2 lineage): search "hybrid Mamba attention LLM 2026"

## Suggested next actions
1. Read IRONSmith + STEEL full text (Vitis 2026.1 is installed — reproduce their demos on this box).
2. Track OmniEEG-Bench for when TUH EEG data lands in B2.
3. Pull R2Q/PTQTP details into the Q4NX converter notes.

## Online sweep (web, 2026-08-08)

### TileFuse — closest match to our stack (arXiv 2606.11357)
Fused mixed-precision kernel library for **AMD XDNA2 NPUs** targeting GEMM/GEMV
in quantized LLM inference. Brings **AWQ-style W4A16/W8A16** onto XDNA2 "rather
than forcing the model to be reshaped around an NPU-specific quantization
scheme" — exactly the Q4NX/NPU2 wall we hit (official weights required, custom
conversions degenerate). Fuses unpack+dequant+GEMM/GEMV in one flow; 4x8 AIE
array GEMV dataflow; GEMM dims to 32K. Results: +121.6% GEMM, +281% GEMV vs
fp32; >2x perf/energy vs iGPU; -64.6% energy end-to-end on Ryzen AI laptops.
→ Read this one. It's the blueprint for making custom quantized models run on
our NPU without the NPU2-formula requirement.

### AMD first-party: GPT-OSS-20B on Ryzen AI NPUs (amd.com, Mar 2026)
Efficient **MoE inference on Strix/Halo** — 20B open-weight MoE on our exact
SoC class. Direct reference for the Zaya 74B MoE NPU path.

### HeteroMosaic (arXiv 2607.12839)
Heterogeneous execution (NPU+iGPU+CPU) for energy-efficient edge LLM inference
— matches our hybrid GPU+NPU design.

### Ternary/1.58-bit adoption gap (arXiv 2607.12839-v1 companion / springer)
Ternary quantization reduces memory but real-world adoption is limited by
performance degradation — the case for mixed-precision (TileFuse) over pure
ternary.

### Ecosystem
- github.com/amd/iron — AMD's IRON Python framework (IRONSmith backend)
- Local-NPU LLM enterprise endpoint strategy pieces (currentstack.io)
- GPU vs TPU vs NPU energy systematic review (arXiv 2606.11357 companion)

## Priority reading
1. **TileFuse** (pull full text next — the Q4NX pivot's missing piece)
2. AMD GPT-OSS-20B article (MoE on our SoC)
3. HeteroMosaic (hybrid execution — our GPU+NPU hybrid backend)
