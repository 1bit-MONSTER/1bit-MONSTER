#!/usr/bin/env python3
"""q4nx_tile_dequant.py — reference dequant for a raw Q4NX "I8" tile (5120 B).

This is the INVERSE of the validated packer `fastflowlm_analysis/q4nx_assemble.py`
(_pack_q4nx_tensor), which is byte-identical to FastFlowLM's vendored converter
and NPU-verified (94.3 t/s decode on the real NPU, q4nx-owning-standard.md).

The within-tile permutation (#2102) is:
  - scales: bytes [0:512],   bf16, group-major index (g*32 + lr), g = col//32, lr = row
  - mins:   bytes [512:1024], bf16, same layout
  - nibbles: bytes [1024:5120], packed int4, lane-swizzled:
        lane = row // 16
        byte_idx = (row % 16) // 2
        nib   = (row % 16) % 2        # 0 = low nibble (even row), 1 = high
        byte  = lane*2048 + col*8 + byte_idx
  - dequant: W[row][col] = scale[g*32+row] * q + min[g*32+row],  g = col//32

Round-trip self-test (no NPU): pack random d/m/qw -> unpack -> must equal d*q+m.
"""
import numpy as np

ROW_BLOCK = 32
COL_BLOCK = 256
TILE_BYTES = 5120


def _f32_to_bf16_bits(a):
    u = a.astype(np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + 0x7FFF
    return ((u + r) >> 16).astype(np.uint16)


def pack_tile(d, m, qw):
    """Mirror of q4nx_assemble._pack_q4nx_tensor for a single 32x256 tile."""
    # d, m: f32 [32, 8]; qw: f32 [32, 256] (0..15)
    dbf = _f32_to_bf16_bits(d.reshape(32, 8).T.reshape(-1))   # (g,lr) order
    mbf = _f32_to_bf16_bits(m.reshape(32, 8).T.reshape(-1))
    out = np.zeros(TILE_BYTES, np.uint8)
    out[0:512] = dbf.view(np.uint8)
    out[512:1024] = mbf.view(np.uint8)
    packed = np.zeros(32 * 256 // 2, np.uint8)
    Qc = qw.astype(np.int16)
    for lr in range(ROW_BLOCK):
        lane = lr // 16
        lane_row = lr % 16
        byte_idx = lane_row // 2
        nib = lane_row % 2
        for col in range(COL_BLOCK):
            v = int(Qc[lr, col]) & 0x0F
            off = lane * (COL_BLOCK * 16 // 2) + col * (16 // 2) + byte_idx
            if nib == 0:
                packed[off] = (packed[off] & 0xF0) | v
            else:
                packed[off] = (packed[off] & 0x0F) | ((v << 4) & 0xF0)
    out[1024:5120] = packed
    return out


def unpack_tile(tile):
    """Raw 5120-byte tile -> W[32,256] f32 (natural order)."""
    t = np.frombuffer(tile, dtype=np.uint8)
    scales = (t[0:512].view('<u2').astype(np.uint32) << 16).view(np.float32)
    mins = (t[512:1024].view('<u2').astype(np.uint32) << 16).view(np.float32)
    packed = t[1024:5120]
    W = np.zeros((32, 256), np.float32)
    for lr in range(ROW_BLOCK):
        lane = lr // 16
        lane_row = lr % 16
        byte_idx = lane_row // 2
        nib = lane_row % 2
        for g in range(8):
            idx = g * 32 + lr
            s = scales[idx]
            mn = mins[idx]
            base = g * 32
            for c in range(32):
                col = base + c
                off = lane * 2048 + col * 8 + byte_idx
                q = (packed[off] >> (4 * nib)) & 0x0F
                W[lr, col] = s * float(q) + mn
    return W


def round_trip():
    rng = np.random.default_rng(0)
    # realistic-ish: scales ~0.01, mins ~ -0.05, qw 0..15
    d = (0.005 + rng.random((32, 8)) * 0.02).astype(np.float32)
    m = (rng.random((32, 8)) * -0.1).astype(np.float32)
    qw = rng.integers(0, 16, (32, 256)).astype(np.float32)
    tile = pack_tile(d, m, qw)
    W = unpack_tile(tile.tobytes())
    # expected: W[lr][col] = bf16(d)[lr][col//32] * qw + bf16(m)[lr][col//32]
    # (scale/min are stored as bf16, so compare against the bf16-rounded values)
    g = np.arange(256) // 32
    d_bf = (_f32_to_bf16_bits(d).astype(np.uint32) << 16).view(np.float32)
    m_bf = (_f32_to_bf16_bits(m).astype(np.uint32) << 16).view(np.float32)
    W_gt = d_bf[:, g] * qw + m_bf[:, g]
    maxdiff = float(np.abs(W - W_gt).max())
    exact = int(np.count_nonzero(W == W_gt))
    print(f"round-trip: max abs diff = {maxdiff:.6f}, exact positions = {exact}/8192")
    return maxdiff == 0.0 and exact == 8192


if __name__ == "__main__":
    ok = round_trip()
    # also decode a real tile if a path is given
    import sys
    if len(sys.argv) > 1:
        raw = open(sys.argv[1], "rb").read()
        n = len(raw) // TILE_BYTES
        print(f"{sys.argv[1]}: {n} tile(s), decoding tile 0")
        W = unpack_tile(raw[:TILE_BYTES])
        print("W[0,:8] =", np.round(W[0, :8], 5))
        print("W[1,:8] =", np.round(W[1, :8], 5))
    print("round-trip OK" if ok else "round-trip FAILED")
