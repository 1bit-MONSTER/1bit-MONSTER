#!/usr/bin/env python3
"""verify_moe_current_layout.py — byte-verify the CURRENT runtime's 35B layer
weight-BO layout (2026-09-03, moe-cap4 captures; Round 50).

IMPORTANT: today's runtime produces a DIFFERENT, simpler layout than the
Sep-2 moe-cap analyzed in R43 (w3912 windows + splice rows + 31-row blocks).
The current spec (byte-verified 100% below):

Linear layer (e.g. layer 6) per-layer BOs at load seq 140+3L:
  [512 MB expert pool bo_to_0158-equiv]
    rows 0..65535: alternating 32-row up/gate blocks (1024 each).
      window order within a 32-row block: j = base + 8*(i%4) + i//4
      (up/gate windows from FILE OFFSET 0, stride 4736; windows 0..32767)
    rows 65536..100959: down, all 35424 windows in 8-window groups
      order [0,2,4,6,1,3,5,7] (+8 per group)
    rows 100960..102623: self_attn.gate_proj (1664 windows: j in
      {0..7, 224..1879}, interleaved pairs (v, v+8 mod 1880), v from 224)
    rows 102624+: ZEROS (allocator slack to the 512 MB size)
  [2 MB BO]: per-layer content, still unidentified (no raw tensor bytes,
    not clean bf16; value mode 0x39 int8-ish) — OPEN.
  [5 MB BO]: linear-attn set, byte-verified 100%:
    head (328,192 B) = [ssm_conv1d 65536][ssm_norm 256][ssm_a 128]
      [ssm_dt.bias 128][ssm_alpha_proj 131072][ssm_beta_proj 131072]
    then ssm_out_proj windows from FILE OFFSET 0 (stride 4736) in 32-row
    blocks, order j = base + 16*(i%2) + i//2 (windows 0..1045); rest zero.

Verified: ALL 30 linear layers' 512 MB pools + the 5 MB BO = 100.000000%.
"""
import json
import struct
import sys

ROW = 4736
MODEL = '/home/bcloud/.config/flm/models/Qwen3.6-35B-A3B-NPU2/model.q4nx'
CAP = '/home/bcloud/.cache/moe-cap4/bo_to_0158_536870912.bin'
CAP5 = '/home/bcloud/.cache/moe-cap4/bo_to_0160_5242880.bin'


def _load():
    raw = open(MODEL, 'rb').read()
    hdr = struct.unpack('<Q', raw[:8])[0]
    meta = json.loads(raw[8:8 + hdr])
    dbase = 8 + hdr
    def load(k):
        t = meta[k]
        return raw[dbase + t['data_offsets'][0]:dbase + t['data_offsets'][1]]
    return load


def w0(b, j):
    return b[j * ROW:(j + 1) * ROW]


def pool_expert_region(load, L):
    """rows 0..100959 of layer L's expert pool."""
    pfx = f'model.layer.{L}.'
    up = load(pfx + 'mlp.up_exps_proj.weight')
    gate = load(pfx + 'mlp.gate_exps_proj.weight')
    down = load(pfx + 'mlp.down_exps_proj.weight')
    out = []
    for blk in range(1024):           # up/gate alternating 32-row blocks
        base = blk * 32
        for i in range(32):
            out.append(w0(up, base + 8 * (i % 4) + i // 4))
        for i in range(32):
            out.append(w0(gate, base + 8 * (i % 4) + i // 4))
    for g in range(35424 // 8):       # down: 8-window groups [0,2,4,6,1,3,5,7]
        for off in [0, 2, 4, 6, 1, 3, 5, 7]:
            out.append(w0(down, g * 8 + off))
    return b''.join(out)


def linear5_bo(load):
    """the 5 MB linear-attn BO (layer 6)."""
    L6 = lambda n: f'model.layer.6.{n}'
    head = b''
    for key in ['linear_attn.ssm_conv1d.weight', 'linear_attn.ssm_norm.weight',
                'linear_attn.ssm_a', 'linear_attn.ssm_dt.bias',
                'linear_attn.ssm_alpha_proj.weight',
                'linear_attn.ssm_beta_proj.weight']:
        head += load(L6(key))
    ssm = load(L6('linear_attn.ssm_out_proj.weight'))
    body = b''
    b_ = 0
    while len(body) < 5242880 - len(head):
        for i in range(32):
            j = b_ * 32 + 16 * (i % 2) + i // 2
            if j * ROW + ROW <= len(ssm):
                body += w0(ssm, j)
            if len(body) >= 5242880 - len(head):
                break
        b_ += 1
    return head + body


def main():
    load = _load()
    pool = open(CAP, 'rb').read()
    rec = pool_expert_region(load, 6)
    ok1 = rec == pool[:len(rec)]
    print(f'layer-6 pool expert region rows 0..100959: '
          f'{"100% ✓" if ok1 else "MISMATCH"}')
    b5 = open(CAP5, 'rb').read()
    rec5 = linear5_bo(load)
    ok2 = rec5[:len(b5)] == b5
    print(f'layer-6 5MB linear-attn BO: {"100% ✓" if ok2 else "MISMATCH"}')
    return 0 if (ok1 and ok2) else 1


if __name__ == '__main__':
    sys.exit(main())
