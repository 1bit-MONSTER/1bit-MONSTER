#!/usr/bin/env python3
"""Verify layer-0 GDN conv + recurrence against the engine's conv_state dump."""
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


def bf16(b):
    u = np.frombuffer(b, dtype="<u2").astype(np.uint32) << 16
    return u.view(np.float32).astype(np.float64)


def silu(x):
    return x / (1.0 + np.exp(-x))


def diff(name, g, dump_path):
    d = np.fromfile(dump_path, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32).reshape(-1)
    if d.size != g.size:
        print(f"  {name:12s} SIZE g={g.size} d={d.size}")
        return
    err = np.abs(g - d)
    corr = np.corrcoef(g, d)[0, 1] if np.std(g) > 1e-12 and np.std(d) > 1e-12 else float('nan')
    print(f"  {name:12s} max_abs={err.max():.4e} corr={corr:.5f}" + (" MATCH" if corr > 0.999 else ""))


def main():
    m = None
    f = open(MODEL, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    hsz = struct.unpack("<Q", mm[:8])[0]
    raw = mm[8:8 + hsz].decode("utf-8", "replace")
    hdr = json.loads(raw[: raw.rindex("}") + 1])
    df = 8 + hsz
    v = hdr["model.layers.0.linear_attn.ssm_conv1d.weight"]
    o = v["data_offsets"]
    cw_flat = bf16(mm[df + o[0]:df + o[1]])   # 32768
    cw_kk = cw_flat.reshape(4, 8192)          # [kk][cc]

    conv_state = np.fromfile("/tmp/l0_convstate.bin", dtype=np.float32)  # [4][8192] = 32768
    print("conv_state shape:", conv_state.shape)
    if conv_state.size != 32768:
        print("conv_state dump missing/incomplete — waiting?")
        return
    cs = conv_state.reshape(4, 8192)          # [kk][cc]
    conv_out = np.zeros(8192)
    for cc in range(8192):
        s = sum(cs[kk, cc] * cw_kk[kk, cc] for kk in range(4))
        conv_out[cc] = silu(s)
    diff("conv", conv_out, "/tmp/l0_conv.bin")

    # qq/kk: split + repeat + l2norm from conv_out
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

    # rawcore from qq/kk + g/beta (dump l0_g/l0_b)
    g = np.fromfile("/tmp/l0_g.bin", dtype=np.float32).astype(np.float64)
    beta = np.fromfile("/tmp/l0_b.bin", dtype=np.float32).astype(np.float64)
    state = np.zeros((NV, HDK, HDV))
    eg = np.exp(g)
    state = state * eg[:, None, None]
    kv_mem = (state * kk[:, :, None]).sum(axis=1)
    delta = (vv.reshape(NV, HDV) - kv_mem) * beta[:, None]
    state = state + kk[:, :, None] * delta[:, None, :]
    core = (state * qq[:, :, None]).sum(axis=1) / np.sqrt(HDK)
    diff("rawcore", core.reshape(-1), "/tmp/l0_rawcore.bin")


if __name__ == "__main__":
    main()
