import numpy as np, glob, json, struct

D = '/home/bcloud/.cache/moe-cap'
MODEL = '/home/bcloud/.config/flm/models/Qwen3.6-35B-A3B-NPU2/model.q4nx'

w0 = np.frombuffer(open(glob.glob(f'{D}/bo_to_*_536870912.bin')[0], 'rb').read(), dtype=np.uint8)
data = open(MODEL, 'rb').read()
hdr = struct.unpack('<Q', data[:8])[0]
meta = json.loads(data[8:8 + hdr])
dbase = 8 + hdr

LAYER = 6
ROW = 4736


def region(key):
    t = meta[key]
    return dbase + t['data_offsets'][0], dbase + t['data_offsets'][1]


def load(key):
    lo, hi = region(key)
    return np.frombuffer(data, dtype=np.uint8, count=hi - lo, offset=lo)


def gen_k_up(n_tiles):
    """up/gate order: col7 rows 0-2, cols 0-6 rows 0-3; then per 4-row block:
    col7 rows 4b-1..4b+2, cols 0-6 rows 4b..4b+3"""
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
        if 8 * (4 * (b + 1) - 1) + 7 >= n_tiles:
            break
        b += 1
    return ks


def gen_k_down(n_tiles):
    """down order: cols [1,3,5,0,2,4,6,7], rows 0-3 per block."""
    ks = []
    b = 0
    while True:
        for c in [1, 3, 5, 0, 2, 4, 6, 7]:
            for r in range(4):
                k = 8 * (4 * b + r) + c
                if k < n_tiles:
                    ks.append(k)
        if 8 * (4 * (b + 1)) + 7 >= n_tiles:
            break
        b += 1
    return ks


def emit_row(out, pos, src, off):
    seg = src[off:off + ROW]
    if len(seg) < ROW:
        return None
    out[pos * ROW:(pos + 1) * ROW] = seg
    return pos + 1


up = load(f'model.layer.{LAYER}.mlp.up_exps_proj.weight')
gate = load(f'model.layer.{LAYER}.mlp.gate_exps_proj.weight')
down = load(f'model.layer.{LAYER}.mlp.down_exps_proj.weight')
share_up = load(f'model.layer.{LAYER}.mlp.share_up_exps_proj.weight')

n_tiles = len(up) // 5120
ks_up = gen_k_up(n_tiles)
ks_down = gen_k_down(n_tiles)
print('up ks:', len(ks_up), 'down ks:', len(ks_down))

out = np.zeros(len(w0), dtype=np.uint8)
pos = 0
# row 0: share_up tail (824) + up head (3912)
out[0:ROW] = np.concatenate([share_up[1114112 - 824:1114112], up[0:3912]])
pos = 1

# the alternating up/gate blocks (the block 0 = 31 ks, the blocks 1+ = 32)
bi = 0
oi = 0
while oi < len(ks_up):
    block = ks_up[0:31] if bi == 0 else ks_up[31 + (bi - 1) * 32:31 + bi * 32]
    if not block:
        break
    for k in block:
        pos = emit_row(out, pos, up, 3912 + ROW * k)
        if pos is None: break
    if pos is None: break
    if bi == 0:
        out[pos * ROW:(pos + 1) * ROW] = np.concatenate([down[167772160 - 824:167772160], gate[0:3912]])
        pos += 1
    for k in block:
        pos = emit_row(out, pos, gate, 3912 + ROW * k)
        if pos is None: break
    if pos is None: break
    bi += 1
    oi += len(block)

print('up+gate rows:', pos)
# the down: cross (the gate tail? no - the row 65536) then the k-rows
# row 65536: the boundary — check what it is (the gate's tail + the down's head?)
# the down's k-rows at 65537+
for k in ks_down:
    pos = emit_row(out, pos, down, 3912 + ROW * k)
    if pos is None:
        print('down ran out at row', pos)
        break

match = np.sum(out[:pos * ROW] == w0[:pos * ROW])
print(f'match: {match}/{pos*ROW} = {match/(pos*ROW)*100:.2f}%')
if match < pos * ROW:
    d = np.where(out[:pos * ROW] != w0[:pos * ROW])[0]
    print('first mismatch at row', d[0] // ROW, 'byte', d[0])
