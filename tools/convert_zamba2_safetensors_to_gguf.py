#!/usr/bin/env python3
"""Convert Zamba2 safetensors to GGUF format (F16).

Usage: python3 tools/convert_zamba2_safetensors_to_gguf.py <model_dir> <output.gguf>
"""
import sys, os, json, time, re
import numpy as np
import torch
from safetensors import safe_open
from gguf import GGUFWriter

GLOBAL_MAP = {
    'model.embed_tokens.weight':        'token_embd.weight',
    'model.final_layernorm.weight':     'output_norm.weight',
}

# SSM layer (mamba) tensor mapping
SSM_MAP = {
    'input_layernorm.weight':                      'attn_norm.weight',
    'mamba.in_proj.weight':                        'ssm_in.weight',
    'mamba.conv1d.weight':                         'ssm_conv1d.weight',
    'mamba.conv1d.bias':                           'ssm_conv1d.bias',
    'mamba.A_log':                                 'ssm_a',
    'mamba.D':                                     'ssm_d',
    'mamba.dt_bias':                                'ssm_dt.bias',
    'mamba.norm.weight':                           'ssm_norm.weight',
    'mamba.out_proj.weight':                       'ssm_out.weight',
}

# Hybrid layer extra tensors (beyond SSM)
# In safetensors, hybrid layers use mamba_decoder.mamba.* for SSM and shared_transformer.* for attn/ffn
HYBRID_MAMBA_MAP = {
    'mamba_decoder.input_layernorm.weight':        'attn_norm.weight',
    'mamba_decoder.mamba.in_proj.weight':          'ssm_in.weight',
    'mamba_decoder.mamba.conv1d.weight':           'ssm_conv1d.weight',
    'mamba_decoder.mamba.conv1d.bias':             'ssm_conv1d.bias',
    'mamba_decoder.mamba.A_log':                   'ssm_a',
    'mamba_decoder.mamba.D':                       'ssm_d',
    'mamba_decoder.mamba.dt_bias':                  'ssm_dt.bias',
    'mamba_decoder.mamba.norm.weight':             'ssm_norm.weight',
    'mamba_decoder.mamba.out_proj.weight':         'ssm_out.weight',
}

HYBRID_ATTN_MAP = {
    'shared_transformer.self_attn.q_proj.weight':  'attn_q.weight',
    'shared_transformer.self_attn.k_proj.weight':  'attn_k.weight',
    'shared_transformer.self_attn.v_proj.weight':  'attn_v.weight',
    'shared_transformer.self_attn.o_proj.weight':  'attn_output.weight',
    # input_layernorm / pre_ff_layernorm handled explicitly in load_shared_block
    # (post_attention_norm.weight = concat norm, ffn_norm.weight = pre-FFN norm)
    'shared_transformer.feed_forward.down_proj.weight': 'ffn_down.weight',
    # gate_up_proj needs splitting into gate + up
}


def load_tensor(model_dir, shard, name):
    path = os.path.join(model_dir, shard)
    try:
        with safe_open(path, framework='np') as f:
            return f.get_tensor(name)
    except TypeError:
        with safe_open(path, framework='pt') as f:
            t = f.get_tensor(name)
            return t.to(torch.float16).cpu().numpy()


def to_f16(t):
    if isinstance(t, np.ndarray):
        if t.dtype == np.float32:
            return t.astype(np.float16)
        if t.dtype == np.float16:
            return t
        return t.astype(np.float32).astype(np.float16)
    return t


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_dir> <output.gguf>")
        sys.exit(1)

    model_dir = sys.argv[1]
    output_path = sys.argv[2]

    with open(os.path.join(model_dir, 'config.json')) as f:
        cfg = json.load(f)
    
    # Handle nested text_config
    if 'text_config' in cfg:
        tcfg = cfg['text_config']
    else:
        tcfg = cfg

    with open(os.path.join(model_dir, 'model.safetensors.index.json')) as f:
        idx = json.load(f)

    wm = idx['weight_map']

    n_layers = int(tcfg.get('num_hidden_layers', cfg.get('num_hidden_layers', 38)))
    hidden = int(tcfg.get('hidden_size', cfg.get('hidden_size', 2048)))
    n_heads = int(tcfg.get('num_attention_heads', cfg.get('num_attention_heads', 32)))
    n_kv = int(tcfg.get('num_key_value_heads', cfg.get('num_key_value_heads', 32)))
    vocab = int(tcfg.get('vocab_size', cfg.get('vocab_size', 32000)))
    head_dim = int(tcfg.get('attention_head_dim', tcfg.get('kv_channels', tcfg.get('head_dim', hidden // n_heads))))

    max_seq = int(tcfg.get('max_position_embeddings', cfg.get('max_position_embeddings', 4096)))
    norm_eps = float(tcfg.get('rms_norm_eps', tcfg.get('norm_epsilon', 1e-5)))
    rope_base = float(tcfg.get('rope_theta', tcfg.get('rope_freq_base', 10000.0)))

    # SSM params
    ssm_state = int(tcfg.get('mamba_d_state', 64))
    ssm_conv = int(tcfg.get('mamba_d_conv', 4))
    ssm_inner = int(tcfg.get('mamba_expand', 2)) * hidden
    ssm_group = int(tcfg.get('mamba_ngroups', 1))
    ssm_head_dim = int(tcfg.get('mamba_headdim', 64))
    ssm_dt_rank = int(tcfg.get('mamba_dt_rank', tcfg.get('time_step_rank', -1)))
    if ssm_dt_rank < 0:
        ssm_dt_rank = ssm_inner  # default

    # Hybrid layers
    hybrid_ids = tcfg.get('hybrid_layer_ids', [])
    block_types = tcfg.get('layers_block_type', [])
    if not hybrid_ids and block_types:
        hybrid_ids = [i for i, t in enumerate(block_types) if t == 'hybrid']
    
    n_ffn = int(tcfg.get('intermediate_size', tcfg.get('ffn_hidden_size', hidden * 4)))

    print(f"  Model: {cfg.get('_name_or_path', os.path.basename(model_dir))}")
    print(f"  Architecture: zamba2, Layers: {n_layers}, Hidden: {hidden}")
    print(f"  Heads: {n_heads}, KV: {n_kv}, Vocab: {vocab}")
    print(f"  SSM: state={ssm_state}, conv={ssm_conv}, inner={ssm_inner}")
    print(f"  Hybrid layers: {hybrid_ids}")
    print(f"  Output: {output_path}")
    print()

    writer = GGUFWriter(output_path, 'zamba2')

    # Metadata
    writer.add_block_count(n_layers)
    writer.add_context_length(max_seq)
    writer.add_embedding_length(hidden)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv)
    writer.add_layer_norm_rms_eps(norm_eps)
    writer.add_vocab_size(vocab)
    writer.add_rope_dimension_count(head_dim)
    writer.add_rope_freq_base(rope_base)
    writer.add_key_length(head_dim)
    writer.add_uint32('rope.use_mem_rope', 1 if tcfg.get('use_mem_rope', True) else 0)
    writer.add_feed_forward_length(n_ffn)
    writer.add_ssm_conv_kernel(ssm_conv)
    writer.add_ssm_inner_size(ssm_inner)
    writer.add_ssm_state_size(ssm_state)
    writer.add_ssm_time_step_rank(ssm_dt_rank)
    writer.add_ssm_group_count(ssm_group)
    writer.add_file_type(1)

    t0 = time.time()
    total_tensors = 0

    def add_t(name, tensor):
        """Write weights in the 1bit loader's convention:
        - 2D matrices: flat data [input, output] row-major (= checkpoint
          transpose) with raw_shape = original (out, in) so the GGUF ne is
          (input, output) — the loader's read_tensor_transposed treats
          ne[0] as the input dim and transposes to [output, input].
        - ssm_conv1d.weight: the engine indexes the flat as [d_conv, conv_dim]
          (k-major); write (d_conv, 1, conv_dim)."""
        nonlocal total_tensors
        t = to_f16(tensor)
        if t.ndim == 2:
            writer.add_tensor(name, np.ascontiguousarray(t.T), raw_shape=(t.shape[0], t.shape[1]))
        elif t.ndim == 3 and name.endswith('ssm_conv1d.weight'):
            writer.add_tensor(name, np.ascontiguousarray(t.transpose(2, 0, 1)))  # (d_conv, 1, conv_dim)
        else:
            writer.add_tensor(name, t)
        total_tensors += 1
        if total_tensors % 50 == 0:
            print(f"    {total_tensors} tensors ({time.time()-t0:.0f}s)...")

    def add_t_embd(tensor):
        """token_embd in llama.cpp/engine convention: data stored [d_model, vocab]
        (numpy (d_model, vocab), ne[0]=d_model); the engine's loader transposes
        when ne[0]==d_model. The gguf writer stores numpy bytes as-is with
        ne=reversed(raw_shape), so pass the transposed array with the original
        (vocab, d_model) raw_shape."""
        nonlocal total_tensors
        f16 = np.ascontiguousarray(to_f16(tensor).T)
        writer.add_tensor('token_embd.weight', f16, raw_shape=(tensor.shape[0], tensor.shape[1]))
        total_tensors += 1

    # Shared transformer blocks: Zamba ties weights cyclically (num_mem_blocks=2,
    # ABAB pattern). Only the first `num_mem_blocks` hybrid layers physically
    # store shared_transformer tensors; every later hybrid layer must duplicate
    # the block and fold its per-layer gate_up LoRA adapter
    # (W_eff = W_shared + B @ A, A=adapter.0 rank->hidden, B=adapter.1 2*inter->rank).
    n_mem_blocks = int(tcfg.get('num_mem_blocks', 2))
    hybrid_pos = {li: p for p, li in enumerate(hybrid_ids)}
    shared_cache = {}   # block_idx -> dict(gguf_name -> np array, adapters: {p: (A, B)})

    def load_shared_block(li):
        """Load the shared transformer block physically stored at hybrid layer li."""
        out = {}
        for st_suf, gguf_suf in HYBRID_ATTN_MAP.items():
            st_name = f"{vl_prefix}model.layers.{li}.{st_suf}"
            if st_name in wm and gguf_suf is not None:
                out[gguf_suf] = load_tensor(model_dir, wm[st_name], st_name)
        # Norms follow the engine/ref convention: post_attention_norm.weight =
        # concat/attention input norm (HF input_layernorm), ffn_norm.weight =
        # pre-FFN norm (HF pre_ff_layernorm). NOTE: the loader/ref apply
        # post_attn_norm to the concat and ffn_norm before the FFN.
        st_name = f"{vl_prefix}model.layers.{li}.shared_transformer.input_layernorm.weight"
        if st_name in wm:
            out['post_attention_norm.weight'] = load_tensor(model_dir, wm[st_name], st_name)
        st_name = f"{vl_prefix}model.layers.{li}.shared_transformer.pre_ff_layernorm.weight"
        if st_name in wm:
            out['ffn_norm.weight'] = load_tensor(model_dir, wm[st_name], st_name)
        gu_name = f"{vl_prefix}model.layers.{li}.shared_transformer.feed_forward.gate_up_proj.weight"
        if gu_name in wm:
            out['gate_up'] = load_tensor(model_dir, wm[gu_name], gu_name).astype(np.float32)
        out['adapters'] = {}
        for idx in range(len(hybrid_ids)):
            a0 = f"{vl_prefix}model.layers.{li}.shared_transformer.feed_forward.gate_up_proj_adapter_list.{idx}.0.weight"
            a1 = f"{vl_prefix}model.layers.{li}.shared_transformer.feed_forward.gate_up_proj_adapter_list.{idx}.1.weight"
            if a0 in wm and a1 in wm:
                out['adapters'][idx] = (
                    load_tensor(model_dir, wm[a0], a0).astype(np.float32),
                    load_tensor(model_dir, wm[a1], a1).astype(np.float32))
        return out

    def write_shared(li):
        """Write the shared block for hybrid layer li, duplicating from the
        physical owner and folding the per-layer gate_up adapter."""
        p = hybrid_pos[li]
        block_idx = p % n_mem_blocks
        if block_idx not in shared_cache:
            shared_cache[block_idx] = load_shared_block(li)
        blk = shared_cache[block_idx]
        for suf, t in blk.items():
            if suf in ('gate_up', 'adapters'):
                continue
            add_t(f"blk.{li}.{suf}", t)
        gu = blk['gate_up']
        if p in blk['adapters']:
            A, B = blk['adapters'][p]
            gu = gu + B @ A
        gate, up = np.split(gu, 2, axis=0)
        add_t(f"blk.{li}.ffn_gate.weight", gate)
        add_t(f"blk.{li}.ffn_up.weight", up)

    # Check if VL model (language_model prefix)
    vl_prefix = ''
    test_key = 'model.layers.0.input_layernorm.weight'
    if test_key not in wm:
        test_key_vl = f'language_model.{test_key}'
        if test_key_vl in wm:
            vl_prefix = 'language_model.'
            print(f"  Detected VL model (prefix: '{vl_prefix}')")

    # Global tensors
    print("  Global tensors...")
    for st_name, gguf_name in GLOBAL_MAP.items():
        full_name = f"{vl_prefix}{st_name}"
        if full_name in wm:
            t = load_tensor(model_dir, wm[full_name], full_name)
        elif st_name in wm:
            t = load_tensor(model_dir, wm[st_name], st_name)
        else:
            continue
        if gguf_name == 'token_embd.weight':
            add_t_embd(t)
        else:
            add_t(gguf_name, t)

    # Per-layer
    print(f"  Layers (0..{n_layers-1})...")
    for li in range(n_layers):
        is_hybrid = li in hybrid_ids

        if is_hybrid:
            # Hybrid layer: SSM tensors with mamba_decoder prefix + attention + FFN
            for st_suf, gguf_suf in HYBRID_MAMBA_MAP.items():
                st_name = f"{vl_prefix}model.layers.{li}.{st_suf}"
                if st_name in wm:
                    t = load_tensor(model_dir, wm[st_name], st_name)
                    add_t(f"blk.{li}.{gguf_suf}", t)

            # Shared transformer block (duplicated + adapter-folded)
            write_shared(li)

            # Linear mixing projection (ssm_mix)
            lin_name = f"{vl_prefix}model.layers.{li}.linear.weight"
            if lin_name in wm:
                t = load_tensor(model_dir, wm[lin_name], lin_name)
                add_t(f"blk.{li}.ssm_mix.weight", t)

        else:
            # Pure SSM layer
            for st_suf, gguf_suf in SSM_MAP.items():
                st_name = f"{vl_prefix}model.layers.{li}.{st_suf}"
                if st_name in wm:
                    t = load_tensor(model_dir, wm[st_name], st_name)
                    add_t(f"blk.{li}.{gguf_suf}", t)

    # Write
    elapsed = time.time() - t0
    print(f"\n  Writing {total_tensors} tensors ({elapsed:.0f}s)...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size = os.path.getsize(output_path)
    print(f"  ✅ Done: {output_path} ({size/1e9:.2f} GB)")
    print(f"  Total tensors: {total_tensors}")


if __name__ == '__main__':
    main()
