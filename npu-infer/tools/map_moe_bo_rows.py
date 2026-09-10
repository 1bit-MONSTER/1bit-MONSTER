#!/usr/bin/env python3
"""map_moe_bo_rows.py — map rows of a captured 35B layer weight BO to their
source tensors (Round 43/44 tooling). Usage:

    map_moe_bo_rows.py <bo_file> <layer> [row_start row_end]

Prints, per contiguous run of rows, the source tensor + window convention.

Solved conventions (layer-6, capture bo_to_0158):
  - expert tensors + self_attn.gate_proj: 4736-B windows at 4736-B stride
    from file offset 3912 (tag @w3912)
  - 4736-windows from offset 0 (@w0) and 5120-tile trims (@tile) checked
  - Rows 0..100959 (up/gate/down expert region) + 100960..102623
    (self_attn.gate_proj) map; qkv_proj / ssm_out_proj / share_* /
    moe_router / norms / ssm_* use other in-BO forms (still open).
"""
import json
import struct
import sys
from collections import OrderedDict

ROW = 4736
MODEL = '/home/bcloud/.config/flm/models/Qwen3.6-35B-A3B-NPU2/model.q4nx'


def main():
    cap = sys.argv[1] if len(sys.argv) > 1 else \
        '/home/bcloud/.cache/moe-cap/bo_to_0158_536870912.bin'
    layer = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    r0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    r1 = int(sys.argv[4]) if len(sys.argv) > 4 else 0x7fffffff

    wb = open(cap, 'rb').read()
    raw = open(MODEL, 'rb').read()
    hdr = struct.unpack('<Q', raw[:8])[0]
    meta = json.loads(raw[8:8 + hdr])
    dbase = 8 + hdr

    def load(key):
        t = meta[key]
        return raw[dbase + t['data_offsets'][0]:dbase + t['data_offsets'][1]]

    def tname(k):
        return k.replace(f'model.layer.{layer}.', '').replace('.weight', '')

    tensors = [k for k in meta if f'model.layer.{layer}.' in k]
    sig = {}

    def add(tag, b, stride, off):
        for j in range((len(b) - off) // stride):
            w = b[off + stride * j:off + stride * j + ROW]
            if len(w) < ROW:
                break
            sig.setdefault((w[0:12], w[ROW - 12:ROW]), []).append((tag, j))

    for k in tensors:
        b = load(k)
        nm = tname(k)
        if len(b) > 2 * 1048576:
            add(nm + '@w3912', b, ROW, 3912)
            if len(b) % 8704 == 0:
                for j in range(len(b) // 8704):
                    w = b[j * 8704:j * 8704 + 5120]
                    if len(w) == 5120:
                        sig.setdefault((w[0:12], w[4096:4108]), []).append(
                            (nm + '@trim87', j))
        else:
            add(nm + '@w0', b, ROW, 0)
        for j in range(len(b) // 5120):
            w = b[j * 5120:j * 5120 + ROW]
            if len(w) < ROW:
                break
            sig.setdefault((w[0:12], w[ROW - 12:ROW]), []).append((nm + '@tile', j))

    runs = OrderedDict()
    for r in range(r0, min(r1, len(wb) // ROW)):
        seg = wb[r * ROW:(r + 1) * ROW]
        hits = sig.get((seg[0:12], seg[ROW - 12:ROW]), [])
        tag = '?' if len(hits) != 1 else hits[0][0]
        j = -1 if tag == '?' else hits[0][1]
        # contiguous-run bookkeeping per tensor tag
        if tag in runs and runs[tag][-1][1] == r - 1:
            prev = runs[tag][-1]
            runs[tag][-1] = (prev[0], r, min(prev[2], j), max(prev[3], j))
        else:
            runs.setdefault(tag, []).append((r, r, j, j))

    total = {}
    for tag, lst in runs.items():
        n = sum(b - a + 1 for a, b, _, _ in lst)
        total[tag] = n
        jlo = min(x[2] for x in lst)
        jhi = max(x[3] for x in lst)
        print(f'{tag:30s} n={n:6d} runs={len(lst):3d} '
              f'rows {lst[0][0]}..{lst[-1][1]}  j {jlo}..{jhi}')
    print(f'--- rows {r0}..{min(r1, len(wb) // ROW) - 1}: '
          f'mapped {sum(v for k, v in total.items() if k != "?")} / '
          f'{sum(total.values())}')


if __name__ == '__main__':
    main()
