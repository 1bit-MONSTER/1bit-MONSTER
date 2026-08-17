#!/usr/bin/env python3
"""Generate a tiny synthetic ZAYA .gguf (8B-era hybrid topology: attention +
MoE on EVERY layer) for validating zaya_forward_batch vs zaya_forward (issues
#1712/#1713). Random weights — no pretrained values needed, the test only
compares batch vs single-token logits on the same weights.

Usage: python3 zaya_mini.py out.gguf
Then:   build/tools/gguf_to_onebp out.gguf out.1bp  (or cmake --build build --target gguf_to_onebp)
Then:   ./build/zaya_batch_check out.1bp
"""
import os
import numpy as np
from gguf import GGUFWriter, GGMLQuantizationType

H, NQ, NKV, HD, L, V = 256, 4, 2, 64, 2, 512
QD, KD, QKV = NQ*HD, NKV*HD, NQ*HD+NKV*HD
N_EXP, N_FF, RTR_H, N_EXP_T = 2, 256, 256, 3
GC = QKV // (NQ + NKV)          # group width for cca_conv_grp (qkv/6)
NROT = HD // 2

def w(*shape, seed, fan_in=None):
    # Xavier/Glorot-style: std = 1/sqrt(fan_in) keeps activations O(1)
    # through layers (unscaled randoms explode to fp16 inf -> NaN silu).
    rng = np.random.default_rng(seed)
    if fan_in is None:
        fan_in = shape[-1] if len(shape) >= 2 else 1
    return (rng.standard_normal(shape) / np.sqrt(fan_in)).astype(np.float32)

def w1(n, seed):
    rng = np.random.default_rng(seed)
    return (1.0 + 0.02 * rng.standard_normal(n)).astype(np.float32)  # near-unit norms

def f16(a):  return a.astype(np.float16)
def f32(a):  return a

out = os.environ.get("OUT", "/tmp/zaya_mini.gguf")
wr = GGUFWriter(out, "zaya")
wr.add_context_length(2048)
wr.add_embedding_length(H)
wr.add_block_count(L)
wr.add_head_count(NQ)
wr.add_head_count_kv(NKV)
wr.add_uint32("zaya.attention.key_length", HD)
wr.add_uint32("zaya.attention.value_length", HD)
wr.add_layer_norm_rms_eps(1e-5)
wr.add_rope_dimension_count(NROT)
wr.add_rope_freq_base(5000000.0)
wr.add_expert_count(N_EXP)
wr.add_expert_used_count(1)
wr.add_feed_forward_length(2 * N_FF)     # gate_up stacked rows (loader ignores)
wr.add_uint32("zaya.expert_feed_forward_length", N_FF)
wr.add_uint32("zaya.ssm.conv_kernel", 2)
wr.add_uint32("zaya.ssm.state_size", 2 * QKV + H)
wr.add_uint32("zaya.ssm.inner_size", 1)
wr.add_bos_token_id(2)
wr.add_eos_token_id(2)
wr.add_vocab_size(V)

seed = 1
wr.add_tensor("token_embd.weight", f16(w(V, H, seed=seed)), raw_dtype=GGMLQuantizationType.F16); seed += 1
wr.add_tensor("output_norm.weight", f32(w1(H, seed=seed)), raw_dtype=GGMLQuantizationType.F32); seed += 1
wr.add_tensor("input_hidden_states_scale.weight", f32(np.ones(H, np.float32)), raw_dtype=GGMLQuantizationType.F32); seed += 1
wr.add_tensor("input_hidden_states_scale.bias", f32(np.zeros(H, np.float32)), raw_dtype=GGMLQuantizationType.F32); seed += 1

for il in range(L):
    p = f"blk.{il}."
    # attention group (8B-era names)
    wr.add_tensor(p + "attn_norm.weight", f32(w1(H, seed=seed)), raw_dtype=GGMLQuantizationType.F32); seed += 1
    wr.add_tensor(p + "attn_q.weight", f16(w(QD, H, seed=seed)), raw_dtype=GGMLQuantizationType.F16); seed += 1
    wr.add_tensor(p + "attn_k.weight", f16(w(KD, H, seed=seed)), raw_dtype=GGMLQuantizationType.F16); seed += 1
    wr.add_tensor(p + "cca_val_proj1.weight", f16(w(KD//2, H, seed=seed)), raw_dtype=GGMLQuantizationType.F16); seed += 1
    wr.add_tensor(p + "cca_val_proj2.weight", f16(w(KD//2, H, seed=seed)), raw_dtype=GGMLQuantizationType.F16); seed += 1
    wr.add_tensor(p + "attn_output.weight", f16(w(H, QD, seed=seed)), raw_dtype=GGMLQuantizationType.F16); seed += 1
    wr.add_tensor(p + "ssm_conv1d.weight", f32(w(QKV, 2, seed=seed)), raw_dtype=GGMLQuantizationType.F32); seed += 1
    wr.add_tensor(p + "ssm_conv1d.bias", f32(w1(QKV, seed=seed)), raw_dtype=GGMLQuantizationType.F32); seed += 1
    # cca_conv_grp: real-checkpoint layout numpy (qkv, gc, 2) — GGUF ne
    # [2, gc, qkv], time-steps interleaved (llama.cpp PR #23112 create_tensor
    # {d_conv, qkv/n_groups, qkv}). gguf_to_onebp reorders it to the loader's
    # t-major [2, gc, qkv] blocks.
    wr.add_tensor(p + "cca_conv_grp.weight", f32(w(QKV, GC, 2, seed=seed, fan_in=2 * GC)), raw_dtype=GGMLQuantizationType.F32); seed += 1
    wr.add_tensor(p + "cca_conv_grp.bias", f32(w1(QKV, seed=seed)), raw_dtype=GGMLQuantizationType.F32); seed += 1
    wr.add_tensor(p + "cca_k_scale.weight", f32(np.ones(NKV, np.float32)), raw_dtype=GGMLQuantizationType.F32); seed += 1
    wr.add_tensor(p + "res_scale_hs.weight", f32(np.full(H, 0.1, np.float32)), raw_dtype=GGMLQuantizationType.F32); seed += 1
    wr.add_tensor(p + "res_scale_hs.bias", f32(np.full(H, 0.1, np.float32)), raw_dtype=GGMLQuantizationType.F32); seed += 1
    wr.add_tensor(p + "res_scale_res.weight", f32(np.full(H, 0.1, np.float32)), raw_dtype=GGMLQuantizationType.F32); seed += 1
    wr.add_tensor(p + "res_scale_res.bias", f32(np.full(H, 0.1, np.float32)), raw_dtype=GGMLQuantizationType.F32); seed += 1
    # MoE group
    wr.add_tensor(p + "ffn_gate_inp.weight", f32(w(RTR_H, H, seed=seed)), raw_dtype=GGMLQuantizationType.F32); seed += 1
    wr.add_tensor(p + "ffn_gate_inp.bias", f32(w1(RTR_H, seed=seed)), raw_dtype=GGMLQuantizationType.F32); seed += 1
    wr.add_tensor(p + "ffn_norm.weight", f32(w1(RTR_H, seed=seed)), raw_dtype=GGMLQuantizationType.F32); seed += 1
    wr.add_tensor(p + "ffn_gate.weight", f32(w(RTR_H, RTR_H, seed=seed)), raw_dtype=GGMLQuantizationType.F32); seed += 1
    wr.add_tensor(p + "ffn_gate.bias", f32(w1(RTR_H, seed=seed)), raw_dtype=GGMLQuantizationType.F32); seed += 1
    wr.add_tensor(p + "zaya_router_mlp2.weight", f32(w(RTR_H, RTR_H, seed=seed)), raw_dtype=GGMLQuantizationType.F32); seed += 1
    wr.add_tensor(p + "zaya_router_mlp2.bias", f32(w1(RTR_H, seed=seed)), raw_dtype=GGMLQuantizationType.F32); seed += 1
    wr.add_tensor(p + "zaya_router_mlp4.weight", f32(w(RTR_H, N_EXP_T, seed=seed)), raw_dtype=GGMLQuantizationType.F32); seed += 1
    wr.add_tensor(p + "zaya_router_mlp4.bias", f32(w1(N_EXP_T, seed=seed)), raw_dtype=GGMLQuantizationType.F32); seed += 1
    wr.add_tensor(p + "zaya_router_biases.weight", f32(w1(N_EXP_T, seed=seed)), raw_dtype=GGMLQuantizationType.F32); seed += 1
    wr.add_tensor(p + "ffn_gate_up_exps.weight", f16(w(N_EXP, 2*N_FF, H, seed=seed)), raw_dtype=GGMLQuantizationType.F16); seed += 1
    wr.add_tensor(p + "ffn_down_exps.weight", f16(w(N_EXP, H, N_FF, seed=seed)), raw_dtype=GGMLQuantizationType.F16); seed += 1
    wr.add_tensor(p + "res_scale_hs.mlp.weight", f32(np.full(H, 0.1, np.float32)), raw_dtype=GGMLQuantizationType.F32); seed += 1
    wr.add_tensor(p + "res_scale_hs.mlp.bias", f32(np.full(H, 0.1, np.float32)), raw_dtype=GGMLQuantizationType.F32); seed += 1
    wr.add_tensor(p + "res_scale_res.mlp.weight", f32(np.full(H, 0.1, np.float32)), raw_dtype=GGMLQuantizationType.F32); seed += 1
    wr.add_tensor(p + "res_scale_res.mlp.bias", f32(np.full(H, 0.1, np.float32)), raw_dtype=GGMLQuantizationType.F32); seed += 1

wr.write_header_to_file()
wr.write_kv_data_to_file()
wr.write_tensors_to_file()
wr.close()
print(f"wrote {out}")
