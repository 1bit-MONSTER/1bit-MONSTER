#!/usr/bin/env python3
# synth_embed.py — synthesize model.embed_tokens.weight for Qwen3.5-4B from the
# tied lm_head (tie_word_embeddings=true) and build a modified model.q4nx that
# FLM's qwen3_5vl class can load, so the native dequant can be byte-verified
# against FLM's reference boot token.
#
# embed_tokens.weight = lm_head.weight = [vocab=248320, H=2560] BF16.
# lm_head is stored as Q8_0 (8704-B tiles): [0:512] 256 bf16 scales (group-major
# g*32+lr), [512:8704] 8192 signed int8 (row-major). value = int8 * scale.
import struct, json, sys, os
import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else '/home/bcloud/.config/flm/models/Qwen3.5-4B-NPU2/model.q4nx'
DST = sys.argv[2] if len(sys.argv) > 2 else '/tmp/qwen35_4b_embed.q4nx'

H = 2560
VOCAB = 248320
TILE_ROWS, TILE_COLS = 32, 256
Q8_ROW = 8704

def bf16_to_f32(u):
    return (u.astype(np.uint32) << 16).view(np.float32)

def f32_to_bf16(f):
    b = f.astype(np.float32).view(np.uint32)
    return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)

f = open(SRC, 'rb')
hdr = f.read(8)
hsz = struct.unpack('<Q', hdr)[0]
df = 8 + hsz
manifest = json.loads(f.read(hsz))

lm = manifest['lm_head.weight']
lo = lm['data_offsets'][0]
n_tiles = lm['shape'][0] * lm['shape'][1]   # 7760 * 10 = 77600
n_tile_cols = H // TILE_COLS                 # 10
n_tile_rows = n_tiles // n_tile_cols         # 7760

# data end (append point)
max_end = max(v['data_offsets'][1] for v in manifest.values() if 'data_offsets' in v)
embed_size = VOCAB * H * 2  # bf16
print(f'[synth] lm_head tiles={n_tiles} ({n_tile_rows}x{n_tile_cols}), embed {VOCAB}x{H} bf16 = {embed_size/1e9:.2f} GB, append @ {max_end}')

# stream the dequantized bf16 embed to a temp file
tmp = '/tmp/synth_embed_bf16.bin'
fo = open(tmp, 'wb')
f.seek(df + lo)
BATCH = 64  # tiles per chunk
out_rows = n_tile_rows * TILE_ROWS
# write column-major chunks? no — row-major [vocab, H]. Each tile writes 32 rows x 256 cols
# into out[tr*32 + lr, tc*256 + c]. We stream row blocks.
# Process tiles in tile-row order so we can write contiguous row bands.
buf = np.zeros((TILE_ROWS, H), np.float32)  # one tile-row band = 32 x 2560
for tr in range(n_tile_rows):
    buf.fill(0.0)
    for tc in range(n_tile_cols):
        raw = f.read(Q8_ROW)
        rd = np.frombuffer(raw, np.uint8)
        scales = bf16_to_f32(rd[0:512].view('<u2'))   # 256, group-major
        vals = rd[512:Q8_ROW].view(np.int8).reshape(TILE_ROWS, TILE_COLS)
        for lr in range(TILE_ROWS):
            for c in range(TILE_COLS):
                g = c // 32
                buf[lr, tc*TILE_COLS + c] = float(vals[lr, c]) * float(scales[g*32+lr])
    fo.write(f32_to_bf16(buf.reshape(-1)).tobytes())
fo.close()
print(f'[synth] embed bf16 written to {tmp}')

# build the new manifest + file
new_manifest = dict(manifest)
new_manifest['model.embed_tokens.weight'] = {
    'dtype': 'BF16', 'shape': [VOCAB, H],
    'data_offsets': [max_end, max_end + embed_size],
}
new_js = json.dumps(new_manifest, separators=(',', ':')).encode()
new_hsz = len(new_js)

# original data section
f.seek(df)
data = f.read(max_end)
f.close()

out = open(DST, 'wb')
out.write(struct.pack('<Q', new_hsz))
out.write(new_js)
out.write(data)
with open(tmp, 'rb') as fi:
    while True:
        chunk = fi.read(1 << 26)
        if not chunk: break
        out.write(chunk)
out.close()
print(f'[synth] wrote {DST} ({os.path.getsize(DST)/1e9:.2f} GB)')
os.remove(tmp)
