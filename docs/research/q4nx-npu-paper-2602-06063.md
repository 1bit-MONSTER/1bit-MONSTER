# Q4NX + NPU prefill/decode architecture — external reference (arXiv 2602.06063)

Source: Du, Yu, Ni, Cai, Yang, Wei, Xu (Clemson / U Rhode Island),
"Mapping Gemma3 onto an Edge Dataflow Architecture", arXiv:2602.06063
(cs.DC, 27 Jan 2026). Read 2026-09-03. No open-source release yet (GitHub
link hidden for anonymous review; toolchains "to be released"). Built on
AMD's OPEN mlir-aie + IRON (the same toolchain on strixhalo), NOT
FastFlowLM. First end-to-end Gemma3 1B/4B (+4B vision) on the AMD Ryzen AI
NPU (NPU2/XDNA2, 32 CTs — same architecture class as Strix Halo).

## Q4NX format (canonical spec from the paper, §3.1.1)

- Block-wise dequant: `ŵ_i = d_g·wq_i + m_g`, group size g=32
  (wq = int4 weight in 0..15, d_g = bf16 scale, m_g = bf16 min offset)
- Projection weights tiled into **32×256 blocks**; each block stores
  32×256 int4 weights + 256 bf16 scales + 256 bf16 min-offsets =
  **5120 bytes** = the Q4NX tile. Only bf16 MM is native on NPU2; offsets
  pre-converted to bf16 offline.
- Dequant engine (prefill): each CT dequantizes a Q4NX block and writes the
  full-precision result to DDR for the bf16 MM.

⚠ Our models' .q4nx files are a FastFlowLM VARIANT that is NOT the clean
canonical tile: expert tensors = 32768×5120 (5120-tile row structure ✓) but
the linear-attn 8704-B-row tensors (qkv/ssm_out) = 2048×8704 and are not
5120-tiled; byte patterns in the tiles mix bf16 and int data in ways the
canonical [512sc][512zp][4096pk] layout does not explain. The R50 pool
spec (4736-window rows, byte-verified 100%) is the empirical ground truth
for the decode-side expert/linear layout; treat the canonical Q4NX spec as
a reference only, not a byte-exact description of these files.

## Prefill vs decode architecture (explains our buffer observations)

- **Prefill = compute-bound**: dequantize Q4NX → bf16, write to DDR,
  run tiled bf16 MMs. FlowQKV = chunked/pipelined attention overlapping
  data movement with compute across CTs.
- **Decode = memory-bound**: weights STAY QUANTIZED; FusedDQP fuses
  dequant+projection into the MVM (Y=W×A, W tiled 32×256, dequant in
  16×8 sub-blocks streamed into the vector processor, one L1 load+store
  per sub-block). FlowKV re-structures KV for read-bandwidth saturation.
- Mapping: A broadcast to all CTs; each CT owns output-row blocks.

### Why this matters for the 35B work
The never-synced (map-written) qkv payloads we isolated (R50-R54 byte-flip
differentials) fit the PREFILL path: dequantized-to-bf16 projection
intermediates written host-side without xrtBOSync. The byte-verified
quantized layouts (pool + 5MB BO, R50) = the DECODE-side kernel format.
The 16×8 / 32×256 tiling granularity matches the 8/16/32-row block
structure seen in the captured BOs. qkv/2MB-BO format archaeology (R51-54)
was partly confounded by reading mixed int4+bf16 content as uniform int16.

## Benchmarks (reference point; Gemma3 1B/4B, Ryzen AI 7 350 = Krackan-class)

- Prefill: up to 5.2× vs iGPU, 33.5× vs CPU (TTFT)
- Decode: up to 4.8× vs iGPU, 2.2× vs CPU (esp. long sequences)
- Vision tower (4B): 1.7× vs iGPU
- Energy: up to 67.2× / 222.9× TPS/W vs iGPU / CPU

Not directly comparable to strixhalo numbers (different model + silicon
class) but a cross-platform calibration for the NPU-vs-iGPU/CPU deltas.
