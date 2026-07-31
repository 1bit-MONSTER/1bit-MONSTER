# Ideas Parking Lot

Adopted-adjacent ideas from the research archive. Revisit after Phase 0 + first wave of landings.

| # | Idea | Source | Why parked | Revisit trigger |
|---|------|--------|-----------|-----------------|
| I-01 | **TriRoute-style joint routing** (attention resolution + expert selection + KV bit-width per token) | TriRoute 2607.06601 | Needs router unification first (P0.2, WS-09) | Router merged |
| I-02 | **W2 gate/up projections only** — 10-20% memory win for Zaya 74B | Recover-LoRA 2606.04238 | Needs ppl harness to verify data-free claim | WS-00 P0 done |
| I-03 | **PolyQ channel-permutation** for CPU SIMD/LUT kernels | PolyQ 2607.14618 | CPU backend already fast (417 tok/s) | WS-04 sweep shows a gap |
| I-04 | **BITEMBED ternary embedders** on NPU | BITEMBED 2606.25674 | Product question, not engineering | Someone asks for embeddings |
| I-05 | **BitDistill task fine-tuning** (FP16 → 1.58-bit for tasks) | BitDistill 2510.13998 | Training-side; we're an inference engine | 1BP catalog needs a new class |
| I-06 | **Spectra scaling laws** (data-vs-params for ternary) | Spectra 1.1 2506.23025 | Model procurement, not engine | Next model acquisition |
| I-07 | **LUT-LLM memory-compute** | LUT-LLM 2511.06174 | FPGA-specific | If an FPGA backend happens |
| I-08 | **TOM/VitaLLM ternary-ASIC patterns** | TOM 2602.20662, VitaLLM 2605.00320 | We don't build silicon | "XDNA-2 successor" doc |
| I-09 | **GSQ 2-3 bpp scalar format** | GSQ 2604.18556 | No user demand for a new format yet | Q2_K quality becomes the blocker |
| I-10 | **Compression-MoE interaction survey** (citation) | 2607.20981 | Citation only | Any "integrated > independent" claim |
| I-11 | **CAT-Q PTQ ternarization of open models** (512 samples, 235B scale) | CAT-Q 2606.26650 | Needs conversion pipeline decision (WS-05) | 1BP v2 lands |
| I-12 | **SANTA sampled-value attention** | SANTA 2605.01910 | Research-stage | 32k+ context is a product requirement |
| I-13 | **NVLLM / AME-PIM in-storage inference** | 2604.25699, 2604.27808 | No NAND-attached compute on Strix Halo | Never — document as dead end |
| I-14 | **AutoScale-NPU cloud pooling** | 2607.16488 | Cloud-side; we're edge-first | Multi-NPU server product |
| I-15 | **ExTernD full format (μ>1 pure)** — skip; only the correction-plane hybrid is actionable | WS-05 FINDINGS | Direct ternary wins at equal bitrate | — |
