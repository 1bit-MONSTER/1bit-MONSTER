# q4nx_assemble.py — owning the FastFlowLM `model.q4nx` standard

**Status: validated end-to-end 2026-09-03.** First-party, dependency-light
GGUF → `model.q4nx` assembler. No torch, no einops, no amd-quark, no
FastFlowLM binary: numpy + the vendored pure-python gguf-py only.

## Why this exists

The FLM NPU2 runlist harness (`test_qwen3_npu`) consumes FastFlowLM's
`model.q4nx` weight container. The Qwen family weights we publish on HF
(`bong-water-water-bong/*-1BP`) are `.1bp` — the engine's 1BP container
(`"1BP\0"` + 256-byte binary header). The harness cannot load them: it reads
the file as a safetensors-style container (8-byte LE header length, then
JSON metadata), and a 1BP header is not JSON. Even the payload Q4NX tile
layout differs (1BP = row-major unsigned nibbles + row-major scales; FLM
`model.q4nx` = lane-swizzled unsigned nibbles + group-major scales/zp —
feeding 1BP bytes to the FLM reader corrupts ~99% of elements).

So the family cannot ride the FLM runlist lane until its weights are emitted
as `model.q4nx`. The converter FastFlowLM ships does this but is a heavy
torch/einops/quark stack. This tool is our owned replacement: same output,
fewer deps.

## Verified byte-level facts (2026-09-03, Qwen3-0.6B)

Reference: `~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx` (683,820,832 B,
the file the real harness runs). Source: `~/models/Qwen3-0.6B-Q4_K_M.gguf`.

| check | result |
|---|---|
| container: first 8 B = LE u64 header length (34584 = 0x8718; the hexdump's `18 87` is the length, not a "0x1887 magic") | ✓ |
| header JSON: 311 tensors, `BF16` × 114 (embed + norms), `I8` × 197 (matmuls, shape `[n_tiles, 5120]`) | ✓ identical |
| `q4nx_assemble.py` output vs vendored converter output (`/tmp/q4nx-repro-q4km/model.q4nx`) | **311/311 tensors byte-identical** |
| real NPU run of OUR assembled file (`test_qwen3_npu --model qwen3:0.6b`) | coherent output, **94.3 t/s decode**, 25.6 t/s prefill |
| embed/norm regions vs the official FastFlowLM/Qwen3-0.6B-NPU2 reference | byte-identical (they share the same Q6_K-quantized embed → same dequant) |

Note: the on-box reference's I8 payloads differ from any Q4_K_M→Q4_1
conversion (including FastFlowLM's own converter) — its embed matches
Q6_K-dequant byte-exact, so the reference itself was quantized from a
Q4_K_M-class GGUF with a different imatrix. That is a source-quality
difference, not a format one; our container/tile layout is byte-compatible
and NPU-verified.

## Layout spec (as emitted)

Per 5120-byte I8 row = one 32×256 tile, tensor = grid `(rows/32)×(cols/256)`:

```
[0:512]    bf16 scales  — group-major: scales[(g*32 + lr)*2], g=col/32
[512:1024] bf16 mins    — same layout
[1024:5120] packed int4 — unsigned nibbles, lane-swizzled:
          lane = row/16, byte = lane*2048 + col*8 + (row%16)/2,
          low nibble = even row
```

Weights come from the GGUF as-is:
- `token_embd.weight` → dequant → BF16 raw (vocab-major), shape `[n_vocab, n_embd]`
- 1-D norms (F32/F16/BF16) → BF16 raw
- float 2-D matmuls (F32/F16/BF16) → BF16 raw (matches the vendored
  converter — a pure-BF16 GGUF yields an all-BF16 `model.q4nx`, which is
  what the vendored tool does too; the NPU lane needs a quantized source)
- quantized matmuls (Q4_K/Q6_K/Q8_0/…) → dequant → **bf16 round-trip** →
  requant Q4_1 (`gguf.quantize`) → unpack `(d,m,qw)` → Q4NX tile pack

Two details that made the port byte-exact (found empirically, both easy to
get wrong):
1. **bf16 round-trip before requant**: dequantized f32 is rounded to bf16
   (torch `.to(bfloat16)` semantics) *before* the Q4_1 quantize.
2. **unpack nibble order**: within a 32-value Q4_1 block the vendored
   `unpack_q4_1` emits all 16 low nibbles first, then all 16 high nibbles
   (its `reshape(n,−1,1,16) >> [0,4]` broadcasts the shift over the byte
   axis) — not interleaved.
3. **header padding**: the JSON header is space-padded so `(8+len) % 8 == 0`
   (safetensors data alignment).

## Usage

```bash
python3 fastflowlm_analysis/q4nx_assemble.py \
    Qwen3-0.6B-Q4_K_M.gguf /tmp/out \
    --arch-config third_party/FLM_Q4NX_Converter/configs/qwen3.json
# writes /tmp/out/model.q4nx (+ copy tokenizer.json + config.json from the
# model dir the harness reads, e.g. ~/.config/flm/models/Qwen3-0.6B-NPU2/)
```

`--arch-config` maps GGUF tensor names (`blk.N.attn_q.weight`) to model
names (`model.layers.N.self_attn.q_proj.weight`) per architecture; configs
exist for qwen2/2vl/3/3vl/3.5, llama, gemma3/4, phi4, lfm2, gpt-oss,
nanbeige.

## To unblock the family

For each family member with a quantized GGUF available, run
`q4nx_assemble.py` and drop the output into a model dir the FLM harness
resolves (config.json + model.q4nx + tokenizer.json). Alternatively, for
models only published as `.1bp`, dequantize through the engine's
`NpuOnebpModel`/`get_tensor_f32` first, emit an F32 GGUF, then assemble —
or extend the name-map config to ingest 1BP directly.
