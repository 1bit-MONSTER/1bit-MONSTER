#!/usr/bin/env python3
"""qwen3_to_onebp.py — convert a Qwen3-family HF text model to 1BP (Q4NX).

Mirrors the loader expectations of GenericBackend::load_1bp
(src/backend_generic.cpp) and the NPU engine loader:
  - 2-D tensors: Q4NX tiles (32x256, gs=32, bf16 scales), stored exactly as
    the HF safetensors layout ([out, in] row-major — no transposes)
  - 1-D tensors (norms, Qwen3 per-head QK-norm): raw F32, ndim=1
  - names: blk.N.attn_q/attn_k/attn_v/attn_output/ffn_gate/ffn_up/ffn_down,
    blk.N.attn_norm/ffn_norm, blk.N.attn_q_norm/attn_k_norm,
    token_embd.weight, output_norm.weight, output.weight
  - header: arch=ONEBP_DEEPSEEK2(20) like the original Mage-VL conversion,
    has_q_norm/has_k_norm=1, rope_theta in fixed point, bos/eos/tensor_count

Handles the Mage-VL nesting (model.language_model.*) and plain Qwen3
(model.*). Usage:
  python3 qwen3_to_onebp.py /path/to/checkpoint models/out.1bp
"""
import argparse, glob, json, os, struct, sys, time
import numpy as np

ONEBP_MAGIC = 0x00504231
ONEBP_Q4NX = 0
ONEBP_DEEPSEEK2 = 20  # Qwen3-style with QK-norm (matches original Mage-VL file)

TR, TC = 32, 256  # tile_rows, tile_cols
GS = 32


def f32b(v):
    """float32 -> upper 16 bits (BF16 for Q4NX scales)."""
    return np.float32(v).view(np.uint32) >> 16


def quant_tile_q4nx(data):
    """One Q4NX tile: bf16 scales/zps + packed 4-bit, layout matches the
    engine loader's dequant_tile (scales, zps, then low-nibble=even cols)."""
    r, c = data.shape
    grps = TC // GS
    padded = np.zeros((TR, TC), dtype=np.float32)
    padded[:r, :c] = data
    grouped = padded.reshape(TR, grps, GS)
    mn = grouped.min(axis=2)
    mx = grouped.max(axis=2)
    rng = mx - mn
    flat = rng < 1e-10
    scale = np.where(flat, 1.0, rng / 15.0).astype(np.float32)
    zp = np.where(flat, 0.0, mn).astype(np.float32)
    inv = 1.0 / scale
    qi = np.clip(np.round((grouped - zp[:, :, None]) * inv[:, :, None]), 0, 15).astype(np.uint8)
    qi = qi.reshape(TR, TC)
    pk = (qi[:, 1::2] << 4) | qi[:, 0::2]
    return f32b(scale).astype(np.uint16).tobytes() + \
           f32b(zp).astype(np.uint16).tobytes() + pk.tobytes()


def load_shard_tensors(paths):
    tensors = {}
    for fn in paths:
        with open(fn, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            hdr = json.loads(f.read(n))
            data = f.read()
        for name, e in hdr.items():
            if name == '__metadata__':
                continue
            o0, o1 = e['data_offsets']
            raw = data[o0:o1]
            if e['dtype'] == 'BF16':
                u16 = np.frombuffer(raw, dtype='<u2').reshape(e['shape'])
                arr = (u16.astype(np.uint32) << 16).view(np.float32)
            elif e['dtype'] == 'F32':
                arr = np.frombuffer(raw, dtype='<f4').reshape(e['shape']).copy()
            elif e['dtype'] == 'F16':
                arr = np.frombuffer(raw, dtype='<f2').reshape(e['shape']).astype(np.float32)
            else:
                raise ValueError(f'unhandled dtype {e["dtype"]} for {name}')
            tensors[name] = arr
    return tensors


def map_name(hf_name):
    """HF Qwen3 / Mage-VL text tensor name -> 1BP canonical name (or None)."""
    n = hf_name
    if n.startswith('model.language_model.'):
        n = n[len('model.language_model.'):]
    if n.startswith('model.visual.') or n.startswith('visual.'):
        return None
    if n.startswith('layers.'):
        n = n[len('layers.'):]
        parts = n.split('.', 1)
        blk = f'blk.{parts[0]}.'
        rest = parts[1]
        table = {
            'input_layernorm.weight': 'attn_norm.weight',
            'post_attention_layernorm.weight': 'ffn_norm.weight',
            'self_attn.q_proj.weight': 'attn_q.weight',
            'self_attn.q_norm.weight': 'attn_q_norm.weight',
            'self_attn.k_proj.weight': 'attn_k.weight',
            'self_attn.k_norm.weight': 'attn_k_norm.weight',
            'self_attn.v_proj.weight': 'attn_v.weight',
            'self_attn.o_proj.weight': 'attn_output.weight',
            'mlp.gate_proj.weight': 'ffn_gate.weight',
            'mlp.up_proj.weight': 'ffn_up.weight',
            'mlp.down_proj.weight': 'ffn_down.weight',
        }
        if rest not in table:
            return None
        return blk + table[rest]
    if n == 'embed_tokens.weight':
        return 'token_embd.weight'
    if n == 'norm.weight':
        return 'output_norm.weight'
    if n == 'lm_head.weight':
        return 'output.weight'
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input', help='checkpoint dir with model-*.safetensors')
    ap.add_argument('out')
    ap.add_argument('--arch', type=int, default=ONEBP_DEEPSEEK2)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.input, 'model-*.safetensors')))
    if not paths:
        print(f'error: no model-*.safetensors in {args.input}', file=sys.stderr)
        sys.exit(1)
    t0 = time.time()
    print(f'loading {len(paths)} shards...')
    t = load_shard_tensors(paths)
    print(f'loaded {len(t)} tensors in {time.time()-t0:.0f}s')

    cfg = json.load(open(os.path.join(args.input, 'config.json')))
    tc = cfg.get('text_config') or {}
    cfg = {**cfg, **tc}
    H = cfg['hidden_size']; L = cfg['num_hidden_layers']
    NH = cfg['num_attention_heads']; NKV = cfg.get('num_key_value_heads', NH)
    HD = cfg.get('head_dim', H // NH)
    FF = cfg['intermediate_size']
    V = cfg['vocab_size']
    max_seq = cfg.get('max_position_embeddings', 131072)
    rope = float(cfg.get('rope_theta', 1000000.0))
    bos = cfg.get('bos_token_id', 0); eos = cfg.get('eos_token_id', 0)

    # ── Collect mapped tensors ──
    entries = []  # (name, ndim, dims, data)
    for name, arr in t.items():
        m = map_name(name)
        if not m:
            continue
        entries.append((m, arr))
    entries.sort(key=lambda e: e[0])
    if not entries:
        print('error: no text-model tensors mapped', file=sys.stderr)
        sys.exit(1)
    have = {n for n, _ in entries}
    expect = {'token_embd.weight', 'output_norm.weight', 'output.weight',
              'blk.0.attn_q.weight', 'blk.0.attn_q_norm.weight',
              'blk.0.ffn_gate.weight'}
    for e in sorted(expect - have):
        print(f'  WARN missing {e}')

    # ── Header (256 B, layout per include/onebp_format.h) ──
    hdr = struct.pack('<5I', ONEBP_MAGIC, 1, args.arch, ONEBP_Q4NX, 0)
    hdr += struct.pack('<8i', H, L, NH, NKV, HD, FF, V, max_seq)
    hdr += struct.pack('<10I', TR, TC, GS, 1, 1, 0, int(rope * 1000), bos, eos,
                       len(entries))
    hdr += struct.pack('<14I', *([0] * 14))
    hdr += struct.pack('<6I', *([0] * 6))  # reserved[0..5]
    hdr += b'\x00' * (192 - len(hdr))
    tag = os.path.basename(args.out).encode()[:63]
    hdr += tag + b'\x00' * (64 - len(tag))
    assert len(hdr) == 256

    # ── Index + data ──
    idx = b''
    off = 0
    total = 0
    sizes2 = []
    for name, arr in entries:
        nd = arr.ndim
        dims = list(arr.shape)
        if nd == 1:
            sz = dims[0] * 4
        else:
            rows, cols = dims
            ntr = (rows + TR - 1) // TR
            ntc = (cols + TC - 1) // TC
            sz = ntr * ntc * (TR * (TC // GS) * 4 + TR * TC // 2)
        sizes2.append(sz)
        nb = name.encode()
        idx += struct.pack('<I', len(nb)) + nb + b'\x00'
        idx += struct.pack('<I', nd) + struct.pack(f'<{nd}I', *dims)
        idx += struct.pack('<2Q', off, sz)
        off += sz
    with open(args.out, 'wb') as f:
        f.write(hdr)
        f.write(idx)
        # second pass: quantized data
        for (name, arr), sz in zip(entries, sizes2):
            if arr.ndim == 1:
                f.write(np.ascontiguousarray(arr, dtype='<f4').tobytes())
            else:
                rows, cols = arr.shape
                for r0 in range(0, rows, TR):
                    for c0 in range(0, cols, TC):
                        f.write(quant_tile_q4nx(arr[r0:r0+TR, c0:c0+TC]))
            total += sz
    print(f'wrote {args.out}: {len(entries)} tensors, {total/1e6:.0f} MB '
          f'({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
