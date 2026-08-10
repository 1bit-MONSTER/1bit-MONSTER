# AMD: Accelerating GPT-OSS-20B on AMD Ryzen AI NPUs (Mar 2026)
URL: https://www.amd.com/en/developer/resources/technical-articles/2026/accelerating-gpt-oss-20b-on-amd-ryzen-ai-npus.html

## The playbook for MoE LLMs on Ryzen AI NPUs (Strix + Halo)
- Model: GPT-OSS-20B MoE, INT4-quantized ONNX (Microsoft cpu-int4-rtn-block-32), BF16 activations.
- **QMoE offload**: CPU does top-K routing + token grouping; NPU executes only the SELECTED experts as batched GEMMs (not all-experts-parallel). This is exactly the #1529 'stream expert load' direction in 1bit-systems.
- **Dynamic expert mmap loading**: per-layer expert count + per-model dynamic-layer count session options; decode phase interleaves load with NPU compute. Mem scales 13.7GB->2.7GB at 13.4->2.4 TPS.
- **GQA on NPU**: FlashAttention-style tiling + online softmax; prefill chunking (OGA); KV cache in BF16 (halves cache).
- Results (Strix/Halo, Ryzen AI 1.7, INT4+BF16): TTFT 0.54s@128tok -> 3.23s@2048; 12.6->10.9 TPS; 13.6-14.0 GB.
- Stack: ONNX Runtime GenAI + RyzenAI-SW (github.com/amd/RyzenAI-SW, LLM-examples/oga_inference); model: huggingface.co/amd/gpt-oss-20b-onnx-ryzenai-npu.

## Relevance to 1bit.systems
- Validates the hybrid CPU-router + NPU-expert architecture (our batched_moe + npu backend).
- Dynamic expert streaming = our #1529 work; AMD ships session options for the same tradeoff.
- INT4 + BF16 activations on NPU = the Q4NX/BF16 pattern FLM already uses.
- The 74B Zaya MoE path should mirror this: CPU gating, NPU expert GEMMs, mmap'd expert streaming.
