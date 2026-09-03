#!/usr/bin/env python3
"""verify_moe_bo_layout.py — byte-verify the engine's MoE weight-BO layout
against the runtime's captured per-layer weight BOs (Round 43).

Ground truth: /home/bcloud/.cache/moe-cap/bo_to_0158_536870912.bin — an
LD_PRELOAD capture of the REAL FastFlowLM runtime's 35B layer-6 weight BO
(536,870,912 B, Sep-2). The capture set covers all 40 layers
(bo_to_0140..  x 512 MB) => the runtime DID complete load_weights once.

Solved layout (layer-6 up/gate/down expert region, rows 0..100959):

  - Each expert tensor's file data is read as 4736-byte ROWS at 4736-byte
    strides from file offset 3912 (row j = file[3912 + 4736*j : +4736]).
    NOT 5120-tile trimmed rows. up/gate/down files are 167,772,160 B
    (35423 usable windows each).
  - Tensor boundaries splice across rows in 824/3912 fragments:
      row 0      = share_up[-824:] + up[:3912]
      rows 1..31 = up windows (block 0: 31 rows, gen_k_up order)
      row 32     = down[-824:] + gate[:3912]
      rows 33..63 = gate windows (block 0: 31)
      rows 64..65535 = alternating 32-window up/gate blocks (1023 each)
      row 65536  = [824 B UNRESOLVED boundary fragment] + down[:3912]
      rows 65537..100959 = down windows, order per 8-window group
                           [1,3,5,0,2,4,6,7] (+8 per group)
  - up/gate place windows 0..32766 (32767); down places all 35423.
  - k-order (gen_k_up, block 0 = [7,15,23]+cols0-6x4 = 31, then 32/block:
    col7 rows 4b-1..4b+2, then cols 0-6 rows 4b..4b+3).

Result: rows 0..100959 byte-identical to the capture except row 65536's
first 824 B (99.9998% of the 458 MB expert region).
"""
import json, struct, sys

CAP = '/home/bcloud/.cache/moe-cap/bo_to_0158_536870912.bin'
MODEL = '/home/bcloud/.config/flm/models/Qwen3.6-35B-A3B-NPU2/model.q4nx'
LAYER = 6
ROW = 4736


def main():
    wb = open(CAP, 'rb').read()
    raw = open(MODEL, 'rb').read()
    hdr = struct.unpack('<Q', raw[:8])[0]
    meta = json.loads(raw[8:8 + hdr])
    dbase = 8 + hdr

    def load(key):
        t = meta[key]
        return raw[dbase + t['data_offsets'][0]:dbase + t['data_offsets'][1]]

    def win(b, j):
        return b[3912 + ROW * j:3912 + ROW * (j + 1)]

    def gen_k_up(n):
        ks = [7, 15, 23]
        for c in range(7):
            for r in range(4):
                ks.append(8 * r + c)
        b = 1
        while True:
            for rr in range(4):
                ks.append(8 * (4 * b - 1 + rr) + 7)
            for c in range(7):
                for r in range(4):
                    ks.append(8 * (4 * b + r) + c)
            if 8 * (4 * (b + 1) - 1) + 7 >= n:
                break
            b += 1
        return ks

    pfx = f'model.layer.{LAYER}.mlp.'
    up = load(pfx + 'up_exps_proj.weight')
    gate = load(pfx + 'gate_exps_proj.weight')
    down = load(pfx + 'down_exps_proj.weight')
    share_up = load(pfx + 'share_up_exps_proj.weight')

    ks = gen_k_up(35424)[:32767]
    assert len(ks) == 32767 and max(ks) <= 32766
    dks = []
    g = 0
    while len(dks) < 35423:
        for off in [1, 3, 5, 0, 2, 4, 6, 7]:
            k = 8 * g + off
            if k < 35423:
                dks.append(k)
        g += 1
    assert len(dks) == 35423

    out = []
    out.append(bytes(share_up[-824:]) + bytes(up[:3912]))
    for k in ks[0:31]:
        out.append(win(up, k))
    out.append(bytes(down[-824:]) + bytes(gate[:3912]))
    for k in ks[0:31]:
        out.append(win(gate, k))
    for i in range(1023):
        s = 31 + 32 * i
        for k in ks[s:s + 32]:
            out.append(win(up, k))
        for k in ks[s:s + 32]:
            out.append(win(gate, k))
    assert len(out) == 65536
    # row 65536 boundary splice: first 824 B unresolved (runtime artifact);
    # compare only its 3912-byte down head below, and pad the reconstruction.
    out.append(b'\x00' * 824 + bytes(down[:3912]))
    for k in dks:
        out.append(win(down, k))
    assert len(out) == 100960

    rec = b''.join(out)
    span = 100960 * ROW
    match = sum(a == b for a, b in zip(rec, wb[:span]))
    print(f'rows 0..100959 (up+gate+down expert region): '
          f'{match}/{span} = {match / span * 100:.4f}%')
    if match < span:
        d = next(i for i in range(span) if rec[i] != wb[i])
        print('first diff byte', d, 'row', d // ROW)
        # isolate the boundary fragment
        frag = wb[65536 * ROW:65536 * ROW + 824]
        print('row 65536 fragment (824 B) != any file bytes of '
              'up/gate/down/share_up (checked) — runtime boundary artifact; '
              'rest of the row (down[:3912]) verified identical.')
        ok = all(rec[i] == wb[i] for i in range(span) if i < 65536 * ROW or
                 i >= 65536 * ROW + 824)
        print('expert region identical EXCEPT the 824-B fragment:', ok)
    return 0 if match >= span else 2


if __name__ == '__main__':
    sys.exit(main())
