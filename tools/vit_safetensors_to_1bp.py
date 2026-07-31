#!/usr/bin/env python3
"""vit_safetensors_to_1bp.py — convert a Mage-ViT vision tower to 1BP.

The vision_server (issue #1244) loads vision towers via
mage_vit_load_weights_1bp, which expects:
  - tensor names with the "visual." prefix (HF Mage-ViT convention)
  - ViT dims in OnebpHeader.reserved[0..5] as u32 (H, L, NH, FF, 0, PS)
  - patch_embedding flattened to ndim=2 [hidden, 3*PS*PS]
  - F32 raw data (no tiles), name+NUL index entries
  - merger tensors (ln_q + mlp.0 + mlp.2) when present

Accepts either the standalone Mage-ViT tower (bare tensor names) or the
Mage-VL checkpoint (model.visual.* names, possibly sharded — the merger
lives on the Mage-VL side):

  python3 vit_safetensors_to_1bp.py /path/to/Mage-VL/*.safetensors out.1bp
  (optional: --hidden --layers --heads --ff --patch to override config.json)
"""
import argparse, glob, json, struct, sys
import numpy as np

# Tensor suffixes that belong to the ViT tower (as stored by HF, WITHOUT the
# model.visual. / visual. prefix). Everything else in the checkpoint is
# skipped (text-model weights, the standalone Mage-ViT head.* projector, ...).
_LAYER_SUFFIXES = (
    'layer_norm1.weight', 'layer_norm1.bias',
    'layer_norm2.weight', 'layer_norm2.bias',
    'self_attn.qkv.weight', 'self_attn.qkv.bias',
    'self_attn.proj.weight', 'self_attn.proj.bias',
    'mlp.fc1.weight', 'mlp.fc1.bias',
    'mlp.fc2.weight', 'mlp.fc2.bias',
)
_TOWER_SUFFIXES = {
    'embeddings.patch_embedding.weight',
    'layernorm_pre.weight', 'layernorm_pre.bias',
    'layernorm_post.weight', 'layernorm_post.bias',
    'merger.ln_q.weight', 'merger.ln_q.bias',
    'merger.mlp.0.weight', 'merger.mlp.0.bias',
    'merger.mlp.2.weight', 'merger.mlp.2.bias',
}


def is_tower_tensor(core_name):
    """True if core_name (no model./visual. prefix) is part of the ViT tower."""
    if core_name in _TOWER_SUFFIXES:
        return True
    for il in range(64):  # generous layer cap; dims come from config anyway
        p = f'encoder.layers.{il}.'
        if core_name.startswith(p) and core_name[len(p):] in _LAYER_SUFFIXES:
            return True
    return False


def load_safetensors(paths):
    tensors = {}
    for path in paths:
        with open(path, 'rb') as f:
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
            if name in tensors:
                raise ValueError(f'duplicate tensor {name} across shards')
            tensors[name] = arr
    return tensors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('safetensors', nargs='+',
                    help='one or more .safetensors files (globs ok)')
    ap.add_argument('out')
    ap.add_argument('--hidden', type=int, default=0)
    ap.add_argument('--layers', type=int, default=0)
    ap.add_argument('--heads', type=int, default=0)
    ap.add_argument('--ff', type=int, default=0)
    ap.add_argument('--patch', type=int, default=0)
    args = ap.parse_args()

    paths = []
    for pat in args.safetensors:
        paths.extend(sorted(glob.glob(pat)))
    if not paths:
        print(f'error: no .safetensors files matched {args.safetensors}', file=sys.stderr)
        sys.exit(1)
    t = load_safetensors(paths)

    # config.json sits next to the first shard (HF convention).
    cfg = {}
    import os
    cfg_path = os.path.join(os.path.dirname(paths[0]), 'config.json')
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        vc = cfg.get('vision_config') or {}
        cfg = {**cfg, **vc}  # vision_config overrides top-level keys
    H  = args.hidden or cfg.get('hidden_size', 1024)
    L  = args.layers or cfg.get('num_hidden_layers', 24)
    NH = args.heads  or cfg.get('num_attention_heads', 16)
    FF = args.ff     or cfg.get('intermediate_size', 4096)
    PS = args.patch  or cfg.get('patch_size', 16)
    rope_theta = float(cfg.get('rope_theta', 10000.0))

    # Collect tower tensors: strip model. -> visual. -> core name.
    out = []  # (name, ndim, dims, f32 bytes)
    for name in t:
        core = name
        if core.startswith('model.'):
            core = core[len('model.'):]
        if core.startswith('visual.'):
            core = core[len('visual.'):]
        if not is_tower_tensor(core):
            continue
        arr = np.ascontiguousarray(t[name], dtype='<f4')
        if core == 'embeddings.patch_embedding.weight':
            arr = arr.reshape(H, 3 * PS * PS)
        out.append(('visual.' + core, arr.ndim, list(arr.shape), arr.tobytes()))

    if not out:
        print('error: no ViT tower tensors found in the given shards '
              '(looked for model.visual.* / embeddings.patch_embedding.*)',
              file=sys.stderr)
        sys.exit(1)
    have = {n for n, _, _, _ in out}
    for suf in _TOWER_SUFFIXES:
        if 'visual.' + suf not in have:
            print(f'  WARN missing {suf}')

    # ── header (v1, arch=ONEBP_VISION(2); ViT dims in reserved[0..5]) ──
    hdr = struct.pack('<5I', 0x00504231, 1, 2, 0, 0)
    hdr += struct.pack('<8i', H, L, NH, NH, H // NH, FF, 1, 4096)
    hdr += struct.pack('<10I', 32, 256, 32, 0, 0, 0,
                       int(rope_theta * 1000), 0, 0, len(out))
    hdr += struct.pack('<14I', *([0] * 14))
    hdr += struct.pack('<6I', H, L, NH, FF, 0, PS)
    hdr += b'\x00' * (192 - len(hdr))  # pad reserved[6..43]
    tag = b'Mage-ViT'
    hdr += tag + b'\x00' * (64 - len(tag))  # model_tag[64]
    assert len(hdr) == 256

    idx = b''
    off = 0
    for name, nd, dims, data in out:
        nb = name.encode()
        idx += struct.pack('<I', len(nb)) + nb + b'\x00'
        idx += struct.pack('<I', nd) + struct.pack(f'<{nd}I', *dims)
        idx += struct.pack('<2Q', off, len(data))
        off += len(data)

    with open(args.out, 'wb') as f:
        f.write(hdr); f.write(idx)
        for _, _, _, data in out:
            f.write(data)
    print(f'wrote {args.out}: H={H} L={L} NH={NH} FF={FF} PS={PS} '
          f'{len(out)} tensors, {off/1e6:.1f} MB')


if __name__ == '__main__':
    main()
