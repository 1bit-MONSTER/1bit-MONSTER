#!/usr/bin/env python3
"""Golden for Qwen3.5-4B layer-0 GDN, compared against /tmp/l0_*.bin dumps.

Replicates the native engine's dequant + GDN math in float32 and diffs each
intermediate against what npu_engine_qwen3_5_4b dumped under NPU_DUMP_L0.
"""
import json, mmap, struct
import numpy as np

MODEL = "/home/bcloud/.config/flm/models/Qwen3.5-4B-NPU2/model.q4nx"
H = 2560
NV, NK, HDK, HDV = 32, 16, 128, 128
CONV_K = 4
KD = NK * HDK          # 2048
VD = NV * HDV          # 4096
CONV_DIM = KD * 2 + VD  # 8192
EPS = 1e-6
TILE_COLS, TILE_ROWS = 256, 32


class Q4nx:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        hsz = struct.unpack("<Q", self.mm[:8])[0]
        raw = self.mm[8:8 + hsz].decode("utf-8", "replace")
        self.hdr = json.loads(raw[: raw.rindex("}") + 1])
        self.df = 8 + hsz

    def raw(self, key):
        v = self.hdr[key]
        o = v["data_offsets"]
        return v, self.mm[self.df + o[0]: self.df + o[1]]


def bf16(b):
    u = np.frombuffer(b, dtype="<u2").astype(np.uint32) << 16
    return u.view(np.float32).astype(np.float64)


def dequant_4736(data, i8_rows, in_features):
    """I8 4736-byte tile -> [out_rows, out_cols] f32, matching dequant_i8_4736_to_float mode 0."""
    ROW_BYTES = 4736
    n_tile_cols = in_features // TILE_COLS
    n_tile_rows = i8_rows // n_tile_cols
    out_rows = n_tile_rows * TILE_ROWS
    out_cols = n_tile_cols * TILE_COLS
    out = np.zeros((out_rows, out_cols), dtype=np.float64)
    buf = np.frombuffer(data, dtype=np.uint8)
    for ir in range(i8_rows):
        rd = buf[ir * ROW_BYTES:(ir + 1) * ROW_BYTES]
        tr = ir // n_tile_cols
        tc = ir % n_tile_cols
        packed = rd[:4096]
        scales = rd[4096:4352].view(np.int8).astype(np.float64)   # 256 signed int8
        mins = rd[4352:4608].view(np.int8).astype(np.float64)     # 256 signed int8
        rsc = bf16(rd[4608:4672].tobytes())                        # 32 bf16
        rmn = bf16(rd[4672:4736].tobytes())                        # 32 bf16
        for lr in range(TILE_ROWS):
            lane = lr // 16
            lane_row = lr % 16
            byte_idx = lane_row // 2
            nib_sel = lr % 2
            lane_data = packed[lane * (TILE_COLS * 8):(lane + 1) * (TILE_COLS * 8)]
            rs, rm = rsc[lr], rmn[lr]
            for col in range(TILE_COLS):
                bv = int(lane_data[col * 8 + byte_idx])
                q = (bv & 0x0F) if nib_sel == 0 else ((bv >> 4) & 0x0F)
                s = scales[col]
                m = mins[col]
                out[tr * TILE_ROWS + lr, tc * TILE_COLS + col] = q * s * rs + m * rs + rm
    return out


def dequant_q8_0(data, i8_rows, in_features):
    """Q8_0 8704-byte row -> [out_rows, out_cols] f32, matching dequant_q8_0_to_float_ex."""
    ROW_BYTES = 8704
    n_tile_cols = in_features // TILE_COLS
    n_tile_rows = i8_rows // n_tile_cols
    out_rows = n_tile_rows * TILE_ROWS
    out_cols = n_tile_cols * TILE_COLS
    out = np.zeros((out_rows, out_cols), dtype=np.float64)
    buf = np.frombuffer(data, dtype=np.uint8)
    for ir in range(i8_rows):
        rd = buf[ir * ROW_BYTES:(ir + 1) * ROW_BYTES]
        tr = ir // n_tile_cols
        tc = ir % n_tile_cols
        scales = bf16(rd[:512].tobytes())                          # 256 bf16
        values = rd[512:8704].view(np.int8).astype(np.float64)     # 8192 int8
        for lr in range(TILE_ROWS):
            for col in range(TILE_COLS):
                g = col // 32
                s = scales[g * 32 + lr]
                out[tr * TILE_ROWS + lr, tc * TILE_COLS + col] = values[lr * TILE_COLS + col] * s
    return out


def silu(x):
    return x / (1.0 + np.exp(-x))


def softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


def load_f32(path, n):
    return np.fromfile(path, dtype=np.float32, count=n)


def diff(name, golden, dump_path):
    try:
        d = np.fromfile(dump_path, dtype=np.float32)
    except FileNotFoundError:
        print(f"  {name:12s} DUMP MISSING")
        return
    if d.size != golden.size:
        print(f"  {name:12s} SIZE MISMATCH golden={golden.size} dump={d.size}")
        return
    g = golden.astype(np.float32)
    err = np.abs(g - d)
    rel = err / (np.abs(d) + 1e-6)
    corr = np.corrcoef(g, d)[0, 1] if np.std(g) > 1e-12 and np.std(d) > 1e-12 else float('nan')
    print(f"  {name:12s} max_abs={err.max():.4e} mean_abs={err.mean():.4e} "
          f"rel>1%={(rel > 0.01).sum()} corr={corr:.5f}")
    if err.max() < 1e-2:
        print(f"  {name:12s} MATCH")
    else:
        print(f"  {name:12s} DIVERGES")


def main():
    m = Q4nx(MODEL)
    p = "model.layers.0.linear_attn."
    x = load_f32("/tmp/l0_x.bin", H).astype(np.float64)

    # qkv_proj [8192, 2560] via 4736
    v, b = m.raw(p + "qkv_proj.weight")
    i8_rows = v["shape"][0] * v["shape"][1]
    qkv_w = dequant_4736(b, i8_rows, H)          # [8192, 2560]
    qkv = qkv_w @ x                              # [8192]
    diff("qkv_raw", qkv, "/tmp/l0_fqo_raw.bin")

    # conv1d (first token: only the last tap matters, but run full causal form)
    v, b = m.raw(p + "ssm_conv1d.weight")
    cw = bf16(b).reshape(4, 8192)      # [4, 8192] kk-major
    # first token: state [0,0,0,qkv] -> conv_out = silu(qkv * cw[3])
    conv_out = silu(qkv * cw[3])
    diff("conv", conv_out, "/tmp/l0_conv.bin")

    # split q/k/v + repeat x2 + l2norm (q NOT pre-scaled; scale goes on core)
    q = conv_out[:KD]
    k = conv_out[KD:2 * KD]
    vv = conv_out[2 * KD:]
    qq = np.empty((NV, HDK)); kk = np.empty((NV, HDK))
    for h in range(NV):
        qq[h] = q[(h // 2) * HDK:(h // 2 + 1) * HDK]
        kk[h] = k[(h // 2) * HDK:(h // 2 + 1) * HDK]
        qq[h] = qq[h] / np.sqrt((qq[h] ** 2).sum() + EPS)
        kk[h] = kk[h] / np.sqrt((kk[h] ** 2).sum() + EPS)
    diff("qq", qq.reshape(-1), "/tmp/l0_qq.bin")
    diff("kk", kk.reshape(-1), "/tmp/l0_kk.bin")

    # alpha/beta via Q8_0 [32, 2560]
    for name, path in (("alpha", p + "ssm_alpha_proj.weight"), ("beta", p + "ssm_beta_proj.weight")):
        v, b = m.raw(path)
        ir = v["shape"][0] * v["shape"][1]
        w = dequant_q8_0(b, ir, H)               # [32, 2560]
        proj = w @ x                             # [32]
        diff(name + "_proj", proj, "/tmp/l0_" + ("g" if name == "alpha" else "b") + ".bin")

    # g / beta (ssm_a already negated -A; used directly)
    v, b = m.raw(p + "ssm_a")
    ssm_a = np.frombuffer(b, dtype=np.float32).astype(np.float64)
    v, b = m.raw(p + "ssm_dt.bias")
    dt_bias = np.frombuffer(b, dtype=np.float32).astype(np.float64)
    v, b = m.raw(p + "ssm_alpha_proj.weight")
    aw = dequant_q8_0(b, v["shape"][0] * v["shape"][1], H)
    v, b = m.raw(p + "ssm_beta_proj.weight")
    bw = dequant_q8_0(b, v["shape"][0] * v["shape"][1], H)
    a = aw @ x
    bb = bw @ x
    g = ssm_a * softplus(a + dt_bias)
    beta = 1.0 / (1.0 + np.exp(-bb))
    diff("g", g, "/tmp/l0_g.bin")
    diff("beta", beta, "/tmp/l0_b.bin")

    # z-gate [4096, 2560] via 4736
    v, b = m.raw("model.layers.0.self_attn.gate_proj.weight")
    ir = v["shape"][0] * v["shape"][1]
    zw = dequant_4736(b, ir, H)
    z = zw @ x
    diff("z", z, "/tmp/l0_z.bin")

    # delta-rule recurrence (state zero-init on first token)
    state = np.zeros((NV, HDK, HDV))
    eg = np.exp(g)
    state = state * eg[:, None, None]
    kv_mem = (state * kk[:, :, None]).sum(axis=1)          # [32,128]
    delta = (vv.reshape(NV, HDV) - kv_mem) * beta[:, None]
    state = state + kk[:, :, None] * delta[:, None, :]
    core = (state * qq[:, :, None]).sum(axis=1) / np.sqrt(HDK)
    diff("rawcore", core.reshape(-1), "/tmp/l0_rawcore.bin")

    # gated RMSNorm
    nw_v, nw_b = m.raw(p + "ssm_norm.weight")
    nw = bf16(nw_b)  # [128]
    var = (core * core).mean(-1, keepdims=True)
    normed = core * (1.0 / np.sqrt(var + EPS)) * nw
    normed = normed * silu(z.reshape(NV, HDV))
    print(f"  normed[0,:4] = {normed[0,:4]}")

    # dump golden core for a byte-level diff
    np.save("/tmp/golden_rawcore.npy", core.astype(np.float32))
    np.save("/tmp/golden_qkv.npy", qkv.astype(np.float32))
    print("wrote /tmp/golden_rawcore.npy /tmp/golden_qkv.npy")


if __name__ == "__main__":
    main()
