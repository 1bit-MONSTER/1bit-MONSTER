> **📜 Hackathon submission** — This document was created for the AMD Radeon Hackathon 2026-07 and reflects the project state at that time (August 2026). All figures are validated measurements from `site/benchmarks.json` / `benchmarks/` — see the [README](../README.md) and [current benchmarks](../docs/wiki/performance.md) for up-to-date data.
>
# AMD AI DevMaster Hackathon — Track 3
## Submission Checklist: AI Acceleration & Performance

Team: **1bit.MONSTER**
Project: **1bit.MONSTER — Open-Source AMD XDNA 2 NPU + ROCm GPU Kernels**
Track: **Track 3 — AI Acceleration & Performance**

---

## Submission Materials

| # | Deliverable | Status | File/Link |
|---|------------|--------|-----------|
| 1 | Project Specification Document | ✅ Complete | `hackathon/spec-document-track3.md` |
| 2 | Project Source Code | ✅ Complete | https://github.com/1bit-MONSTER/1bit-MONSTER |
| 3 | Demo Video | ⬜ To record | See `hackathon/demo-script-track3.md` |
| 4 | PPT / Poster | ✅ Below | Key slides in this document |

---

## Project Summary

**1bit.MONSTER — Open-Source AMD XDNA 2 NPU Stack + Custom ROCm GPU Kernels**

Full reverse-engineering of AMD's proprietary XDNA 2 NPU stack + custom ROCm HIP kernels for LLM inference.

**Key achievements:**
- **Reverse-engineered XDNA 2 NPU in 4 days** — 22 proprietary `.so` libraries disassembled, 209 xclbin bitstreams traced, 87.8 MB closed binary → 1.5 MB open-source
- **Native NPU INT8 GEMM** — 22/22 shapes, 0/10000 errors verified on hardware; Qwen3.6-35B-A3B at 11.66 tok/s on XDNA 2 (FLM, measured)
- **First open-source Mamba1 GPU backend** — BlackMamba 1.5B at 79.4 tok/s
- **Fused ternary kernels at 433 tok/s** (Q1 GEMV), 420 tok/s (fused TQ2) and 318 tok/s (Vulkan ZINC)
- **Token Router** — per-layer dispatch across NPU + GPU + CPU with auto-failover
- **TheRock 7.15.0a** — first project to adopt and validate AMD's nightly HIP SDK

**Hardware**: AMD Ryzen AI Max+ 395 (Strix Halo), Radeon 8060S GPU (gfx1151), 32 XDNA 2 NPU tiles, 128 GB unified LPDDR5X
