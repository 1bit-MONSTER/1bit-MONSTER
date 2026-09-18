# Qwen3.6-35B-A3B-NPU2 load_weights segfault — qwen3_6_reorder_cpy overflow

**Repo**: ROCm/FastFlowLM (libqwen3_6_moe_npu.so, XRT backend)
**Model**: Qwen3.6-35B-A3B-NPU2 (config flm_version 0.9.45)
**Repro**: instantiate `qwen3_6_moe_npu(config, &npu, 4096)` + `load_weights(Q4NX)`
-> SIGSEGV in `__memcpy_avx512_unaligned_erms`, 7 s after load starts.

## Stack (gdb)
```
#0 __memcpy_avx512_unaligned_erms
#1 qwen3_6_moe_desc::qwen3_6_reorder_cpy(unsigned char*, buffer<unsigned char>&,
     int, flm_dtype_t, int) [clone .constprop.2]
     <- libqwen3_6_moe_npu.so
#2 qwen3_6_moe_desc::load_linear_weights(int, Q4NX&, buffer<unsigned char>&,
     buffer<biovault::bfloat16_t>&, buffer<biovault::bfloat16_t>&)
#3 qwen3_6_moe_npu::Impl::load_weights(Q4NX&)
```

## Evidence
- The crash is the 4th `qwen3_6_reorder_cpy` call (all dtype=8, n=64):
  `CALL 1 size=2048  CALL 2 size=2048  CALL 3 size=512  CALL 4 size=4096 -> SIGSEGV`
- The faulting memcpy gets `rdx = 0xF5E56C80` (-169867392) — a garbage length
  from the reorder's internal chunk arithmetic (signed overflow), reading past
  the source buffer (a stack `buffer<unsigned char>`).
- The 4th tensor is a plain BF16 vector (2048 elements = 4096 B) among the
  layer's small BF16/F32 weights (input_layernorm / post_attention_layernorm /
  ssm_* / shared_expert_gate). The reorder's tile math assumes the
  I8-tile stride even for the BF16 vector path (its internal per-chunk stride
  computes to 9216 = the linear-attn addr_qk stride).
- All 733 tensors are present + correctly shaped in model.q4nx (verified by
  parsing the JSON header: ssm_a F32[32], ssm_dt.bias F32[32], ssm_conv1d
  BF16[4,8192], qkv_proj I8[256,8,8704], shared_expert_gate BF16[2048], ...).
  The file is NOT malformed.

## Impact
`qwen3_6_moe_npu` cannot load any Qwen3.6-35B model — the hybrid
linear/full-attention MoE family is dead on load. This also blocks
third-party engines (npu-infer) that use the runtime as their verification
reference for the 35B.

## Suggested fix
In `qwen3_6_moe_desc::load_linear_weights` / `qwen3_6_reorder_cpy`, compute
the reorder chunk count for BF16/F32 vector tensors from the tensor's actual
byte count (`size * dtype_size`) instead of the I8-tile-derived stride; guard
the memcpy length against overflow before copying.
