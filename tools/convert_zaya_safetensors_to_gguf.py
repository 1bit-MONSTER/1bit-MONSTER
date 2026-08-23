#!/usr/bin/env python3
"""Convert ZAYA safetensors → GGUF for the 1bit-monster llama.cpp runtime.

Handles both naming eras:
  - ZAYA1-base (Nov 2025): alternating layers (even=CCA attn, odd=MoE),
      self_attn.qkv.{linear_q,linear_k,val_proj1,val_proj2,conv_qk.0,conv_qk.1,temp},
      zaya_block.router.*, zaya_block.experts.local_experts.{e}.linear_fc{1,2},
      model.res_scale.* (final), model.final_norm
  - ZAYA1-8B / ZAYA1-74B-preview (May 2026): EVERY layer is hybrid (attn + MoE),
      self_attn.qkv_proj.{q_proj,k_proj,v_proj_current,v_proj_delayed,conv_qk_depthwise,conv_qk_grouped},
      self_attn.qk_norm.temp, mlp.gate.{down_proj,router_mlp.{norm,fc1,fc2,out_proj},balancing_biases},
      mlp.experts.{down_proj,gate_up_proj} (pre-stacked), post_{attention,mlp}_residual_scale.*,
      model.input_hidden_states_{scale,bias}, model.norm

GGUF tensor names follow the canonical llama.cpp ZAYA mapping (upstream PR #23112) so the
runtime's create_tensor() calls match: ssm_conv1d (conv_qk.0 / conv_qk_depthwise, squeeze(1)+transpose),
cca_conv_grp, cca_val_proj1/2, cca_k_scale, attn_q/attn_k/attn_output, ffn_gate_inp, ffn_norm,
ffn_gate, zaya_router_mlp2/4, zaya_router_biases, zaya_router_eda, ffn_down_exps, ffn_gate_up_exps,
res_scale_hs/res_scale_res (+_mlp per layer), res_scale_hs/res_scale_res (final, base only),
input_hidden_states_scale/bias (8B, model-level), post_attention_norm (8B, per layer).

Usage: python3 tools/convert_zaya_safetensors_to_gguf.py <model_dir> <output.gguf>
Self-check (no weights needed): python3 tools/convert_zaya_safetensors_to_gguf.py --check <model.safetensors.index.json>
"""
import sys, os, json, pathlib
import numpy as np
from safetensors import safe_open
from gguf import GGUFWriter, GGMLQuantizationType
from gguf.vocab import LlamaHfVocab

# ── config keys used for metadata (fallbacks for older configs) ──
def cfg_int(cfg, *names, default=0):
    for n in names:
        v = cfg.get(n)
        if v is not None:
            return int(v)
    return default

# ── HF safetensors name → (GGUF name template, transform, kind) ──
# kind: 'global' | 'layer' ; transform: 'T' (transpose) | 'squeeze1T' | 'permute210' | 'flat' | 'none'
# We match the CURRENT (8B-era) names and the base-era names via alternates.

def map_tensor(st_name, layer_idx, cfg):
    """Return (gguf_name, transform) or None if the tensor is not consumed by the runtime."""
    def layer(gguf_suffix, transform='T', lid=layer_idx):
        return f"blk.{lid}.{gguf_suffix}", transform

    def global_(gguf_name, transform='T'):
        return gguf_name, transform

    if st_name == 'model.embed_tokens.weight':
        return global_('token_embd.weight', 'none')
    if st_name in ('model.final_norm.weight', 'model.norm.weight'):
        return global_('output_norm.weight', 'none')
    # base-era final res scale → RES_SCALE_HS_FINAL / RES_SCALE_RES_FINAL ("res_scale_hs"/"res_scale_res", no blk)
    if st_name == 'model.res_scale.hidden_states_scale':
        return global_('res_scale_hs.weight', 'none')
    if st_name == 'model.res_scale.hidden_states_bias':
        return global_('res_scale_hs.bias', 'none')
    if st_name == 'model.res_scale.residual_scale':
        return global_('res_scale_res.weight', 'none')
    if st_name == 'model.res_scale.residual_bias':
        return global_('res_scale_res.bias', 'none')
    # 8B-era model-level input scale (applied to embeddings; runtime applies when present)
    if st_name == 'model.input_hidden_states_scale':
        return global_('input_hidden_states_scale.weight', 'none')
    if st_name == 'model.input_hidden_states_bias':
        return global_('input_hidden_states_scale.bias', 'none')

    if st_name.endswith('.input_layernorm.weight') or st_name.endswith('.input_norm.weight'):
        return layer('attn_norm.weight', 'none')
    if st_name.endswith('.post_attention_layernorm.weight'):
        return layer('post_attention_norm.weight', 'none')

    # ── attention (even layers in base; every layer in 8B) ──
    attn_prefixes = ('self_attn.', )
    if '.self_attn.' in st_name:
        base = 'self_attn.qkv.'
        new  = 'self_attn.qkv_proj.'
        if base + 'linear_q.weight' in st_name:     return layer('attn_q.weight', 'none')
        if base + 'linear_k.weight' in st_name:     return layer('attn_k.weight', 'none')
        if base + 'val_proj1.weight' in st_name:    return layer('cca_val_proj1.weight', 'none')
        if base + 'val_proj2.weight' in st_name:    return layer('cca_val_proj2.weight', 'none')
        if base + 'conv_qk.0.weight' in st_name:    return layer('ssm_conv1d.weight', 'squeeze1')
        if base + 'conv_qk.0.bias' in st_name:      return layer('ssm_conv1d.bias', 'none')
        if base + 'conv_qk.1.weight' in st_name:    return layer('cca_conv_grp.weight', 'none')
        if base + 'conv_qk.1.bias' in st_name:      return layer('cca_conv_grp.bias', 'none')
        if base + 'temp' in st_name:                return layer('cca_k_scale.weight', 'none')
        if new + 'q_proj.weight' in st_name:        return layer('attn_q.weight', 'none')
        if new + 'k_proj.weight' in st_name:        return layer('attn_k.weight', 'none')
        if new + 'v_proj_current.weight' in st_name:return layer('cca_val_proj1.weight', 'none')
        if new + 'v_proj_delayed.weight' in st_name:return layer('cca_val_proj2.weight', 'none')
        if new + 'conv_qk_depthwise.weight' in st_name: return layer('ssm_conv1d.weight', 'squeeze1')
        if new + 'conv_qk_depthwise.bias' in st_name:   return layer('ssm_conv1d.bias', 'none')
        if new + 'conv_qk_grouped.weight' in st_name:   return layer('cca_conv_grp.weight', 'none')
        if new + 'conv_qk_grouped.bias' in st_name:     return layer('cca_conv_grp.bias', 'none')
        if 'qk_norm.temp' in st_name:               return layer('cca_k_scale.weight', 'none')
        if 'o_proj.weight' in st_name:              return layer('attn_output.weight', 'none')

    # ── residual scaling (per layer) ──
    if '.res_scale.hidden_states_scale' in st_name:     return layer('res_scale_hs.weight', 'none')
    if '.res_scale.hidden_states_bias' in st_name:      return layer('res_scale_hs.bias', 'none')
    if '.res_scale.residual_scale' in st_name:          return layer('res_scale_res.weight', 'none')
    if '.res_scale.residual_bias' in st_name:           return layer('res_scale_res.bias', 'none')
    if '.post_attention_residual_scale.hidden_states_scale' in st_name: return layer('res_scale_hs.weight', 'none')
    if '.post_attention_residual_scale.hidden_states_bias' in st_name:  return layer('res_scale_hs.bias', 'none')
    if '.post_attention_residual_scale.residual_scale' in st_name:      return layer('res_scale_res.weight', 'none')
    if '.post_attention_residual_scale.residual_bias' in st_name:       return layer('res_scale_res.bias', 'none')
    if '.post_mlp_residual_scale.hidden_states_scale' in st_name:       return layer('res_scale_hs_mlp.weight', 'none')
    if '.post_mlp_residual_scale.hidden_states_bias' in st_name:        return layer('res_scale_hs_mlp.bias', 'none')
    if '.post_mlp_residual_scale.residual_scale' in st_name:            return layer('res_scale_res_mlp.weight', 'none')
    if '.post_mlp_residual_scale.residual_bias' in st_name:             return layer('res_scale_res_mlp.bias', 'none')

    # ── MoE router (base: zaya_block.router.*, 8B: mlp.gate.*) ──
    if '.router.down_proj.weight' in st_name or '.gate.down_proj.weight' in st_name:
        return layer('ffn_gate_inp.weight', 'none')
    if '.router.down_proj.bias' in st_name or '.gate.down_proj.bias' in st_name:
        return layer('ffn_gate_inp.bias', 'none')
    if '.router.rmsnorm_eda.weight' in st_name or '.router_mlp.norm.weight' in st_name:
        return layer('ffn_norm.weight', 'none')
    if '.router.router_mlp.0.weight' in st_name or '.router_mlp.fc1.weight' in st_name:
        return layer('ffn_gate.weight', 'none')
    if '.router.router_mlp.0.bias' in st_name or '.router_mlp.fc1.bias' in st_name:
        return layer('ffn_gate.bias', 'none')
    if '.router.router_mlp.2.weight' in st_name or '.router_mlp.fc2.weight' in st_name:
        return layer('zaya_router_mlp2.weight', 'none')
    if '.router.router_mlp.2.bias' in st_name or '.router_mlp.fc2.bias' in st_name:
        return layer('zaya_router_mlp2.bias', 'none')
    if '.router.router_mlp.4.weight' in st_name or '.router_mlp.out_proj.weight' in st_name:
        return layer('zaya_router_mlp4.weight', 'none')
    if '.router.router_mlp.4.bias' in st_name or '.router_mlp.out_proj.bias' in st_name:
        return layer('zaya_router_mlp4.bias', 'none')
    if '.router.balancing_biases' in st_name or '.gate.balancing_biases' in st_name:
        return layer('zaya_router_biases.weight', 'none')
    # base-era EDA: zaya_block.router.router_states_scale IS used by the base model
    if '.router.router_states_scale' in st_name:
        return layer('zaya_router_eda.weight', 'none')
    # 8B/74B-era: mlp.gate.router_states_scale exists in the checkpoint but the reference
    # model does NOT use it (module attr is None) — skipping avoids enabling the EDA path
    if '.gate.router_states_scale' in st_name:
        return None

    # ── experts ──
    if '.mlp.experts.down_proj' in st_name:      # 8B: (n_expert, n_embd, n_ff); ggml mul_mat A(ne0=n_ff=contraction) reads [e][r][c] → keep AS-IS
        return layer('ffn_down_exps.weight', 'none')
    if '.mlp.experts.gate_up_proj' in st_name:   # 8B: (n_expert, 2*n_ff, n_embd) → ne [n_embd, 2*n_ff, n_expert] ✓ as-is
        return layer('ffn_gate_up_exps.weight', 'none')
    if '.local_experts.' in st_name:             # base: per-expert, stacked at conversion time
        if st_name.endswith('.linear_fc1.weight'):
            return layer('ffn_gate_up_exps.weight', 'stack_raw')   # base: fc1 = fused gate+up; stack dim0 of raw (n_expert, 2*n_ff, n_embd)
        if st_name.endswith('.linear_fc2.weight'):
            return layer('ffn_down_exps.weight', 'stack_T')   # base: fc2 = down; per-expert .T then stack dim0

    return None  # not consumed by runtime (e.g. base's post_attention_layernorm variants, unused tensors)


def transform(t, kind):
    """Apply the GGUF orientation transform to a loaded tensor (numpy).

    llama.cpp stores ne in REVERSE numpy order, so create_tensor({a,b}) means
    numpy (b,a) — i.e. HF (out, in) orientation, NO transpose for linear layers.
    """
    if kind == 'squeeze1':
        return np.squeeze(t, axis=1)     # Conv1d {C,1,K} → {C,K}  (numpy (n_qk, d_conv))
    if kind == 'transpose021':
        return t.transpose(0, 2, 1)      # (n_expert, n_ff, n_embd) → (n_expert, n_embd, n_ff)
    if kind == 'stack_raw':
        return t  # stacked as-is on dim 0
    if kind == 'stack_T':
        return t  # caller transposes per-expert
    return t


def load_tensor(model_dir, shard, name):
    path = os.path.join(model_dir, shard)
    try:
        with safe_open(path, framework='np') as f:
            return f.get_tensor(name)
    except TypeError:
        import torch
        with safe_open(path, framework='pt') as f:
            return f.get_tensor(name).to(torch.float16).cpu().numpy()


def to_f16(t):
    import torch
    if isinstance(t, np.ndarray):
        if t.dtype == np.float32:
            return t.astype(np.float16)
        if t.dtype == np.float16:
            return t
        if t.dtype == np.dtype('bfloat16'):
            return torch.from_numpy(t).to(torch.float16).numpy()
        return t.astype(np.float32).astype(np.float16)
    return t.to(torch.float16).numpy()


def write_vocab(writer, model_dir):
    """Write tokenizer.ggml.* KVs from HF tokenizer.json (Gemma3 BPE, llama.cpp 'gemma' format)."""
    import json
    tok_path = os.path.join(model_dir, 'tokenizer.json')
    cfg_path = os.path.join(model_dir, 'tokenizer_config.json')
    if not os.path.exists(tok_path):
        print("  WARNING: no tokenizer.json; writing raw vocab_size KV only")
        return None
    t = json.load(open(tok_path))
    vocab = t['model']['vocab']
    merges_raw = t['model'].get('merges', [])
    # new HF format stores merges as [left, right] pairs; llama.cpp wants "left right" strings
    if merges_raw and isinstance(merges_raw[0], (list, tuple)):
        merges = [' '.join(pair) for pair in merges_raw]
    else:
        merges = merges_raw
    added = t.get('added_tokens', [])
    ids = {tok: tid for tok, tid in vocab.items()}
    for at in added:
        ids[at['content']] = at['id']
    n = len(ids)
    toks = [''] * n
    for tok, tid in ids.items():
        toks[tid] = tok
    types = [1] * n  # normal
    for at in added:
        if at['id'] < n:
            types[at['id']] = 3  # control (special tokens)
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    def sid(name):
        v = cfg.get(name)
        if isinstance(v, dict):
            v = v.get('content')
        if v is None:
            return 0
        return ids.get(v, 0)
    bos, eos, unk, pad = sid('bos_token'), sid('eos_token'), sid('unk_token'), sid('pad_token')
    if unk < n:
        types[unk] = 2  # unknown
    writer.add_tokenizer_model('gemma4')
    writer.add_token_list(toks)
    writer.add_token_types(types)
    writer.add_token_merges(merges)
    writer.add_bos_token_id(bos)
    writer.add_eos_token_id(eos)
    writer.add_unk_token_id(unk)
    writer.add_pad_token_id(pad)
    writer.add_add_bos_token(True)
    writer.add_add_eos_token(False)
    writer.add_vocab_size(n)
    print(f"  vocab: {n} tokens (gemma4 BPE), bos={bos} eos={eos} unk={unk} merges={len(merges)}")
    return n


def self_check(index_path):
    """Verify every safetensors key maps to a runtime-consumed GGUF tensor (no silent drops)."""
    with open(index_path) as f:
        idx = json.load(f)
    cfg = {}
    cfg_path = os.path.join(os.path.dirname(index_path), 'config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    wm = idx['weight_map']
    unmapped, mapped = [], []
    for k in sorted(wm):
        parts = k.split('.')
        lid = 0
        if len(parts) >= 3 and parts[:2] == ['model', 'layers']:
            lid = int(parts[2])
        r = map_tensor(k, lid, cfg)
        if r is None:
            unmapped.append(k)
        else:
            mapped.append(r)
    print(f"mapped: {len(mapped)}  unmapped: {len(unmapped)}")
    for k in unmapped:
        print(f"  SKIP: {k}")
    # required tensors per the runtime's create_tensor() calls, derived from ACTUAL layer keys
    n_layers = cfg_int(cfg, 'num_hidden_layers', 'n_layer', default=0)
    names = {r[0] for r in mapped}
    required = {'token_embd.weight', 'output_norm.weight'}
    attn_req = {'attn_q.weight', 'attn_k.weight', 'attn_output.weight', 'ssm_conv1d.weight',
                'cca_conv_grp.weight', 'cca_val_proj1.weight', 'cca_val_proj2.weight', 'cca_k_scale.weight'}
    moe_req = {'ffn_gate_inp.weight', 'ffn_norm.weight', 'ffn_gate.weight', 'zaya_router_mlp2.weight',
               'zaya_router_mlp4.weight', 'ffn_down_exps.weight', 'ffn_gate_up_exps.weight'}
    for k in wm:
        parts = k.split('.')
        if len(parts) >= 3 and parts[:2] == ['model', 'layers']:
            lid = parts[2]
            if '.self_attn.' in k:
                required |= {f'blk.{lid}.{t}' for t in attn_req}
            if '.mlp.experts' in k or '.zaya_block.experts' in k:
                required |= {f'blk.{lid}.{t}' for t in moe_req}
    missing = required - names
    if missing:
        print("MISSING REQUIRED TENSORS:")
        for m in sorted(missing):
            print(f"  {m}")
        sys.exit(1)
    print("self-check OK")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--check':
        self_check(sys.argv[2])
        return
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_dir> <output.gguf>   |   {sys.argv[0]} --check <index.json>")
        sys.exit(1)

    model_dir, output_path = sys.argv[1], sys.argv[2]
    with open(os.path.join(model_dir, 'config.json')) as f:
        cfg = json.load(f)
    with open(os.path.join(model_dir, 'model.safetensors.index.json')) as f:
        idx = json.load(f)
    wm = idx['weight_map']
    shards = sorted(set(wm.values()))

    n_layers = cfg_int(cfg, 'num_hidden_layers', default=40)
    hidden   = cfg_int(cfg, 'hidden_size', default=2048)
    n_heads  = cfg_int(cfg, 'num_attention_heads', default=8)
    n_kv     = cfg_int(cfg, 'num_key_value_heads', default=2)
    head_dim = cfg_int(cfg, 'head_dim', default=128)
    vocab    = cfg_int(cfg, 'vocab_size', default=262272)
    norm_eps = float(cfg.get('norm_epsilon', 1e-5))
    rope_base = float(cfg.get('rotary_base', cfg.get('rope_theta', 1000000.0)))
    partial_rotary = float(cfg.get('partial_rotary_factor', 0.5))
    d_conv   = cfg_int(cfg, 'cca_time0', default=2)
    topk     = cfg_int(cfg, 'num_experts_per_tok', 'moe_router_topk', default=1)

    n_qk = n_heads * head_dim + n_kv * head_dim
    state_size = 2 * n_qk + hidden  # runtime asserts n_embd_s() == 2*n_qk + n_embd

    # expert count: infer from stacked experts or base local_experts keys
    n_experts = 16
    for k in wm:
        if '.local_experts.' in k:
            import re
            m = re.search(r'\.local_experts\.(\d+)\.', k)
            if m:
                n_experts = max(n_experts, int(m.group(1)) + 1)

    # n_ff_exp: prefer the actual router down_proj width over config guess
    n_ff_exp = cfg_int(cfg, 'zaya_mlp_expansion', default=256)
    for probe in ('mlp.gate.down_proj.weight', 'zaya_block.router.down_proj.weight'):
        if probe in wm:
            try:
                t = load_tensor(model_dir, wm[probe], probe)
                n_ff_exp = t.shape[0]
                print(f"  n_ff_exp inferred from {probe}: {n_ff_exp}")
            except Exception as e:
                print(f"  WARNING: could not infer n_ff_exp from {probe}: {e}")
            break

    print(f"  Model: {cfg.get('_name_or_path', os.path.basename(model_dir))}")
    print(f"  arch=zaya  layers={n_layers}  hidden={hidden}  heads={n_heads}/{n_kv}  head_dim={head_dim}")
    print(f"  partial_rotary={partial_rotary} → rope dims={int(partial_rotary*head_dim)}")
    print(f"  conv_kernel={d_conv}  state_size={state_size}  experts={n_experts}  topk={topk}")

    writer = GGUFWriter(output_path, 'zaya')
    writer.add_block_count(n_layers)
    writer.add_context_length(cfg_int(cfg, 'max_position_embeddings', default=32768))
    writer.add_embedding_length(hidden)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv)
    writer.add_uint32('zaya.attention.key_length', head_dim)
    writer.add_uint32('zaya.attention.value_length', head_dim)
    writer.add_layer_norm_rms_eps(norm_eps)
    writer.add_rope_dimension_count(int(partial_rotary * head_dim))
    writer.add_rope_freq_base(rope_base)
    writer.add_expert_count(n_experts)
    writer.add_expert_used_count(topk)
    writer.add_feed_forward_length(cfg_int(cfg, 'moe_intermediate_size', 'ffn_hidden_size', default=hidden * 4))
    writer.add_uint32('zaya.ssm.conv_kernel', d_conv)
    writer.add_uint32('zaya.ssm.state_size', state_size)
    writer.add_uint32('zaya.ssm.inner_size', 1)
    writer.add_uint32('zaya.expert_feed_forward_length', n_ff_exp)

    # vocab from tokenizer.json (Gemma3 tokenizer for all Zaya releases)
    vocab_size = write_vocab(writer, model_dir) or vocab

    # ── tensor pass ──
    args_f32 = '--f32' in sys.argv
    expert_stack = {}  # (layer, gguf_suffix) -> list of per-expert tensors (base era)
    expert_stack_kind = {}
    plan = []
    for st_name in wm:
        r = map_tensor(st_name, 0, cfg)
        if r is None:
            continue
        gguf_name, kind = r
        lid = 0
        # extract layer id for blk tensors
        parts = st_name.split('.')
        if len(parts) >= 3 and parts[0] == 'model' and parts[1] == 'layers':
            lid = int(parts[2])
            gguf_name = gguf_name.replace('blk.0.', f'blk.{lid}.')
        if kind in ('stack_raw', 'stack_T'):
            expert_stack.setdefault((lid, gguf_name), []).append(st_name)
            expert_stack_kind[(lid, gguf_name)] = kind
        else:
            plan.append((st_name, gguf_name, kind))

    t0 = __import__('time').time()
    total = 0
    def add_t(gguf_name, tensor):
        nonlocal total
        # llama.cpp convention: norm weights, scales and biases are F32 (they are
        # multiplied/added directly against the F32 activation stream); matmuls stay F16
        is_small = ('norm' in gguf_name or 'scale' in gguf_name or 'biases' in gguf_name
                    or gguf_name.endswith('.bias') or 'k_scale' in gguf_name
                    or gguf_name.startswith('input_hidden_states'))
        writer.add_tensor(gguf_name, tensor.astype(np.float32) if is_small or args_f32 else to_f16(tensor))
        total += 1
        if total % 100 == 0:
            print(f"    {total} tensors ({__import__('time').time()-t0:.0f}s)...")

    print("  Global + layer tensors...")
    for st_name, gguf_name, kind in plan:
        t = load_tensor(model_dir, wm[st_name], st_name)
        if gguf_name == 'token_embd.weight' and t.shape[0] > vocab_size:
            # HF embeddings may exceed the tokenizer vocab (reserved/multimodal tokens)
            t = t[:vocab_size]
            print(f"  token_embd trimmed {t.shape[0]} → {vocab_size}")
        add_t(gguf_name, transform(t, kind))

    print("  Base-era expert stacking...")
    for (lid, gguf_name), st_list in sorted(expert_stack.items()):
        kind = expert_stack_kind.get((lid, gguf_name))
        if kind == 'stack_raw':
            tensors = [load_tensor(model_dir, wm[s], s) for s in sorted(st_list)]
        else:  # stack_T
            tensors = [load_tensor(model_dir, wm[s], s).T for s in sorted(st_list)]
        stacked = np.stack(tensors, axis=0)
        add_t(gguf_name, stacked)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    size = os.path.getsize(output_path)
    print(f"  ✅ Done: {output_path} ({size/1e9:.2f} GB)  tensors={total}")
    print(f"  Run self-check on a fresh index: python3 tools/convert_zaya_safetensors_to_gguf.py --check model.safetensors.index.json")


if __name__ == '__main__':
    main()
