#!/usr/bin/env python3
"""M1: TileFuse-style weight pre-tiler (research proof).

Layout per TileFuse (arXiv 2606.11357 §4.2):
  tile = k x n weights, row-major INT4 codes (2/byte) -> BF16 scales ->
  INT8 zero-points (group size gs, per-column groups), duplicated zp so each
  tile payload is 128 B divisible.
  Tiles are then arranged INTERLEAVED COLUMN-MAJOR: tiles assigned to the
  same AIE output-column slot are memory-contiguous (32K-dim support).

This is a NEW output format — the existing q4nx pack is lossy (nibble-
truncated Q8_0 codes; opaque dequant) and was verified non-faithful
(corr 0.01-0.09 vs GGUF). Round-trip test included.

Usage:
  python3 tilefuse_prep.py in.npy out.tf   # prep a k x n f32 matrix
  python3 tilefuse_prep.py --check out.tf  # round-trip dequant + stats
"""
import sys, struct
import numpy as np

TILE_K, TILE_N, GROUP = 128, 64, 128  # TileFuse defaults (k x n tile, gs=128)


def pack_tile(w: np.ndarray) -> bytes:
    """w: TILE_K x TILE_N f32 -> codes(int4, row-major) + scales(bf16) + zps(int8).
    Group size GROUP along k; per (group, n-column) scale/zp (64 scales + 64 zps
    for 128x64 with gs=128)."""
    assert w.shape == (TILE_K, TILE_N)
    codes = np.zeros((TILE_K, TILE_N), dtype=np.uint8)
    scales = np.zeros((TILE_N,), dtype=np.float32)
    zps = np.zeros((TILE_N,), dtype=np.int8)
    for c in range(TILE_N):
        col = w[:, c]
        # per-channel min-max int4
        lo, hi = col.min(), col.max()
        scale = (hi - lo) / 15.0
        if scale == 0:
            scale = abs(lo) if lo != 0 else 1.0  # constant column: (0-zp)*scale = lo
        q = np.clip(np.round((col - lo) / scale), 0, 15).astype(np.uint8)
        codes[:, c] = q
        scales[c] = scale
        zps[c] = int(np.clip(round(-lo / scale), -128, 127))  # code-domain zp
    # pack nibbles row-major, adjacent COLUMNS per byte: byte(r, c) holds
    # cols 2c (lo) and 2c+1 (hi) of row r (standard INT4 layout)
    flat = codes.reshape(-1)
    packed = (flat[0::2] & 0x0F) | ((flat[1::2] & 0x0F) << 4)
    out = bytearray(packed.tobytes())
    # scales as bf16 (truncate f32 mantissa)
    for s in scales:
        out += struct.pack('<H', (struct.unpack('<I', struct.pack('<f', s))[0] >> 16) & 0xFFFF)
    # zps duplicated x2 for 128B alignment (TileFuse: zp dup for DMA)
    zps_dup = np.repeat(zps, 2).astype(np.int8)
    out += zps_dup.tobytes()
    return bytes(out)


def pre_tile_interleaved(w: np.ndarray) -> bytes:
    """Full matrix -> interleaved column-major tile stream.
    w: K x N f32. AIE has 8 output-column slots: tile-column slot s owns
    output tiles at columns s, s+8, ... (round-robin). Tiles for the same
    slot are contiguous in memory."""
    K, N = w.shape
    kt, nt = (K + TILE_K - 1) // TILE_K, (N + TILE_N - 1) // TILE_N
    wpad = np.zeros((kt * TILE_K, nt * TILE_N), dtype=np.float32)
    wpad[:K, :N] = w
    out = bytearray()
    for s in range(8):                       # AIE column slot
        for ti in range(s, nt, 8):           # tile columns for this slot
            for tj in range(kt):             # tile rows
                tile = wpad[tj*TILE_K:(tj+1)*TILE_K, ti*TILE_N:(ti+1)*TILE_N]
                out += pack_tile(tile)
    return bytes(out)


def dequant_tile(blob: bytes) -> np.ndarray:
    """Inverse of pack_tile -> f32 (for the round-trip test)."""
    n_codes = TILE_K * TILE_N // 2
    packed = np.frombuffer(blob[:n_codes], dtype=np.uint8)
    codes = np.zeros((TILE_K, TILE_N), dtype=np.uint8).reshape(-1)
    codes[0::2] = packed & 0x0F
    codes[1::2] = (packed >> 4) & 0x0F
    codes = codes.reshape(TILE_K, TILE_N)
    n_scales = TILE_N
    scales = np.array([bf16_to_f32(struct.unpack('<H', blob[n_codes+2*i:n_codes+2*i+2])[0])
                       for i in range(n_scales)])
    zps = np.frombuffer(blob[n_codes + 2*n_scales: n_codes + 2*n_scales + 2*TILE_N],
                        dtype=np.int8)[0::2].astype(np.float32)
    w = np.zeros((TILE_K, TILE_N), dtype=np.float32)
    for c in range(TILE_N):
        w[:, c] = (codes[:, c].astype(np.float32) - zps[c]) * scales[c]
    return w


def bf16_to_f32(h: int) -> float:
    return struct.unpack('<f', struct.pack('<I', h << 16))[0]


def check_roundtrip(path: str, k: int, n: int):
    blob = open(path, 'rb').read()
    per_tile = TILE_K * TILE_N // 2 + TILE_N * 2 + 2 * TILE_N
    assert len(blob) % per_tile == 0, f"blob {len(blob)} not tile-multiple of {per_tile}"
    # reconstruct in interleaved order (8 slots round-robin over nt tile-cols)
    kt, nt = (k + TILE_K - 1) // TILE_K, (n + TILE_N - 1) // TILE_N
    w = np.zeros((kt*TILE_K, nt*TILE_N), dtype=np.float32)
    off = 0
    for s in range(8):
        for ti in range(s, nt, 8):
            for tj in range(kt):
                w[tj*TILE_K:(tj+1)*TILE_K, ti*TILE_N:(ti+1)*TILE_N] = dequant_tile(blob[off:off+per_tile])
                off += per_tile
    return w[:k, :n]


def main():
    if sys.argv[1] == "--check":
        w = check_roundtrip(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
        ref = np.load(sys.argv[5]) if len(sys.argv) > 5 else None
        print(f"decoded {w.shape}: std={w.std():.5f} max|.|={np.abs(w).max():.4f}")
        if ref is not None:
            err = np.abs(w - ref).max()
            # int4 physics: max error <= column scale (q-rounding + zp-rounding)
            max_scale = max((ref[:, c].max() - ref[:, c].min()) / 15.0 for c in range(ref.shape[1]))
            print(f"vs ref {ref.shape}: max|err|={err:.6f} (int4 ceiling={max_scale:.6f})")
            print("ROUND-TRIP", "PASS" if err <= max_scale + 1e-6 else "FAIL")
        return
    w = np.load(sys.argv[1])
    open(sys.argv[2], 'wb').write(pre_tile_interleaved(w))
    print(f"pre-tiled {w.shape} -> {sys.argv[2]} ({TILE_K}x{TILE_N} tiles, gs={GROUP})")


if __name__ == "__main__":
    main()
