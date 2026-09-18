#!/usr/bin/env python3
"""Full-model single-token golden for Qwen3.5-4B — bisects the first diverging layer.

Reproduces token 16 through all 32 layers in float64 (dequantized weights, no
int8 activation quantization) and diffs the post-layer hidden state against the
engine's NPU_DUMP_HIDDEN dump. The first layer whose hidden state diverges far
beyond int8-noise (corr << 0.999) is the structural bug.
"""
import json, mmap, struct
import numpy as np

MODEL = "/home/bcloud/.config/flm/models/Qwen3.5-4B-NPU2/model.q4nx"
H = 2560
IM = 9216
NV, NK, HDK, HDV = 32, 16, 128, 128
KD = NK * HDK          # 2048
VD = NV * HDV          # 4096
CONV_K = 4
CONV_DIM = KD * 2 + VD  # 8192
NH, NKV, HD = 16, 4, 256
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
    ROW_BYTES = 4736
    n_tile_cols = in_features // TILE_COLS
    n_tile_rows = i8_rows // n_tile_cols
    out_rows = n_tile_rows * TILE_ROWS
    out_cols = n_tile_cols * TILE_COLS
    out = np.zeros((out_rows, out_cols), dtype=np.float64)
    buf = np.frombuffer(data, dtype=np.uint8)
    for ir in range(i8_rows):
        rd = buf[ir * ROW_BYTES:(ir + 1) * ROW_BYTES]
        tr, tc = ir // n_tile_cols, ir % n_tile_cols
        packed = rd[:4096]
        scales = rd[4096:4352].view(np.int8).astype(np.float64)
        mins = rd[4352:4608].view(np.int8).astype(np.float64)
        rsc = bf16(rd[4608:4672].tobytes())
        rmn = bf16(rd[4672:4736].tobytes())
        for lr in range(TILE_ROWS):
            lane = lr // 16
            lane_row = lr % 16
            byte_idx = lane_row // 2
            nib = lr % 2
            lane_data = packed[lane * (TILE_COLS * 8):(lane + 1) * (TILE_COLS * 8)]
            rs, rm = rsc[lr], rmn[lr]
            for col in range(TILE_COLS):
                bv = int(lane_data[col * 8 + byte_idx])
                q = (bv & 0x0F) if nib == 0 else ((bv >> 4) & 0x0F)
                out[tr * TILE_ROWS + lr, tc * TILE_COLS + col] = q * scales[col] * rs + mins[col] * rs + rm
    return out


def dequant_q8_0(data, i8_rows, in_features):
    ROW_BYTES = 8704
    n_tile_cols = in_features // TILE_COLS
    n_tile_rows = i8_rows // n_tile_cols
    out_rows = n_tile_rows * TILE_ROWS
    out_cols = n_tile_cols * TILE_COLS
    out = np.zeros((out_rows, out_cols), dtype=np.float64)
    buf = np.frombuffer(data, dtype=np.uint8)
    for ir in range(i8_rows):
        rd = buf[ir * ROW_BYTES:(ir + 1) * ROW_BYTES]
        tr, tc = ir // n_tile_cols, ir % n_tile_cols
        scales = bf16(rd[:512].tobytes())
        values = rd[512:8704].view(np.int8).astype(np.float64)
        for lr in range(TILE_ROWS):
            for col in range(TILE_COLS):
                out[tr * TILE_ROWS + lr, tc * TILE_COLS + col] = values[lr * TILE_COLS + col] * scales[(col // 32) * 32 + lr]
    return out


def load_4736(m, key, out_f, in_f):
    v, b = m.raw(key)
    ir = v["shape"][0] * v["shape"][1]
    return dequant_4736(b, ir, in_f)  # [out_f, in_f]


def silu(x):
    return x / (1.0 + np.exp(-x))


def softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


def rmsnorm(x, w):
    return x / np.sqrt((x ** 2).mean() + EPS) * w


def main():
    m = Q4nx(MODEL)
    layer_types = m.hdr.get("layer_types") or json.loads(open(MODEL.replace("model.q4nx", "config.json")).read()).get("layer_types")
    # derive from config.json
    import json as _j
    cfg = _j.load(open("/home/bcloud/.config/flm/models/Qwen3.5-4B-NPU2/config.json"))
    layer_types = cfg["layer_types"]

    # token 16 embedding (verified byte-correct)
    v, b = m.raw("lm_head.weight")
    lm = dequant_q8_0(b, v["shape"][0] * v["shape"][1], H)  # [248320, 2560]
    emb = lm[16].copy()

    # per-layer weights
    in_n = np.array([bf16(m.raw(f"model.layers.{l}.input_layernorm.weight")[1]) for l in range(32)])
    pa_n = np.array([bf16(m.raw(f"model.layers.{l}.post_attention_layernorm.weight")[1]) for l in range(32)])
    fin_v = bf16(m.raw("model.norm.weight")[1])

    # final hidden = token 16 through all layers
    h = emb.copy()

    engine_hidden = np.fromfile("/tmp/native_hidden.bin", dtype=np.float32)
    npt = 16  # ids_short prompt length
    per_layer = npt * H
    n_layers_dumped = engine_hidden.size // per_layer
    print(f"engine hidden dump: {engine_hidden.shape} = {n_layers_dumped} layers x {npt} tokens")
    have_dump = n_layers_dumped >= 1
    def engine_h(l):
        return engine_hidden[l * per_layer:(l * per_layer) + H]  # first token

    for l in range(32):
        kind = layer_types[l]
        # input norm
        xn = rmsnorm(h, in_n[l])
        if kind == "linear_attention":
            p = f"model.layers.{l}.linear_attn."
            qkv_w = load_4736(m, p + "qkv_proj.weight", 8192, H)
            out_w = load_4736(m, p + "ssm_out_proj.weight", H, VD)
            z_w = load_4736(m, f"model.layers.{l}.self_attn.gate_proj.weight", VD, H)
            # alpha/beta (Q8_0, [32, H])
            av, ab = m.raw(p + "ssm_alpha_proj.weight")
            aw = dequant_q8_0(ab, av["shape"][0] * av["shape"][1], H)
            bv, bb = m.raw(p + "ssm_beta_proj.weight")
            bw = dequant_q8_0(bb, bv["shape"][0] * bv["shape"][1], H)
            cv, cb = m.raw(p + "ssm_conv1d.weight")
            cw = bf16(cb).reshape(4, 8192)
            sv, sb = m.raw(p + "ssm_a")
            ssm_a = np.frombuffer(sb, dtype=np.float32).astype(np.float64)
            dv, db = m.raw(p + "ssm_dt.bias")
            dt_bias = np.frombuffer(db, dtype=np.float32).astype(np.float64)
            nv, nb = m.raw(p + "ssm_norm.weight")
            nw = bf16(nb)

            qkv = qkv_w @ xn                      # [8192]
            # conv1d (first token: only tap 3)
            conv = silu(qkv * cw[3])
            q, k, vv = conv[:KD], conv[KD:2 * KD], conv[2 * KD:]
            qq = np.empty((NV, HDK)); kk = np.empty((NV, HDK))
            for hh in range(NV):
                qq[hh] = q[(hh // 2) * HDK:(hh // 2 + 1) * HDK]
                kk[hh] = k[(hh // 2) * HDK:(hh // 2 + 1) * HDK]
                qq[hh] = qq[hh] / np.sqrt((qq[hh] ** 2).sum() + EPS)
                kk[hh] = kk[hh] / np.sqrt((kk[hh] ** 2).sum() + EPS)
            a = aw @ xn; bproj = bw @ xn
            g = ssm_a * softplus(a + dt_bias)
            beta = 1.0 / (1.0 + np.exp(-bproj))
            state = np.zeros((NV, HDK, HDV))
            state = state * np.exp(g)[:, None, None]
            kv_mem = (state * kk[:, :, None]).sum(axis=1)
            delta = (vv.reshape(NV, HDV) - kv_mem) * beta[:, None]
            state = state + kk[:, :, None] * delta[:, None, :]
            core = (state * qq[:, :, None]).sum(axis=1) / np.sqrt(HDK)
            var = (core ** 2).mean(-1, keepdims=True)
            core = core * (1.0 / np.sqrt(var + EPS)) * nw
            z = z_w @ xn
            core = core * silu(z.reshape(NV, HDV))
            at = core.reshape(-1)
            oo = out_w @ at
        else:
            # full attention
            p = f"model.layers.{l}.self_attn."
            q_w = load_4736(m, p + "q_proj.weight", 2 * NH * HD, H)   # q+gate (interleaved per-head)
            k_w = load_4736(m, p + "k_proj.weight", NKV * HD, H)
            v_w = load_4736(m, p + "v_proj.weight", NKV * HD, H)
            o_w = load_4736(m, p + "o_proj.weight", H, NH * HD)
            qn = bf16(m.raw(p + "q_norm.weight")[1])
            kn = bf16(m.raw(p + "k_norm.weight")[1])
            qg = q_w @ xn                          # [8192] interleaved: [q0,g0,q1,g1,...]
            q = np.empty((NH, HD)); gate = np.empty((NH, HD))
            for hh in range(NH):
                q[hh] = qg[hh * 2 * HD:hh * 2 * HD + HD]
                gate[hh] = qg[hh * 2 * HD + HD:hh * 2 * HD + 2 * HD]
            k = (k_w @ xn).reshape(NKV, HD)
            vv = (v_w @ xn).reshape(NKV, HD)
            for hh in range(NH):
                q[hh] = q[hh] / np.sqrt((q[hh] ** 2).mean() + EPS) * qn
            for kvh in range(NKV):
                k[kvh] = k[kvh] / np.sqrt((k[kvh] ** 2).mean() + EPS) * kn
            # pos=0: RoPE identity; causal attn over 1 position = softmax=1 -> v
            at = np.empty((NH, HD))
            for hh in range(NH):
                at[hh] = vv[hh // (NH // NKV)] * (1.0 / (1.0 + np.exp(-gate[hh])))
            oo = o_w @ at.reshape(-1)
            if l == 3:
                print(f"  [l3] q std={q.std():.4f} gate std={gate.std():.4f} k std={k.std():.4f} v std={vv.std():.4f}")
                print(f"  [l3] at std={at.std():.4f} oo std={oo.std():.4f} oo max={np.abs(oo).max():.4f}")
                print(f"  [l3] qn[0:4]={qn[:4]} kn[0:4]={kn[:4]}")
        h = h + oo
        # FFN (SwiGLU)
        pmlp = f"model.layers.{l}.mlp."
        g_w = load_4736(m, pmlp + "gate_proj.weight", IM, H)
        u_w = load_4736(m, pmlp + "up_proj.weight", IM, H)
        d_w = load_4736(m, pmlp + "down_proj.weight", H, IM)
        xf = rmsnorm(h, pa_n[l])
        su = silu(g_w @ xf) * (u_w @ xf)
        dw = d_w @ su
        h = h + dw

        if have_dump and (l + 1) <= n_layers_dumped:
            e = engine_h(l)
            g32 = h.astype(np.float32)
            corr = np.corrcoef(g32, e)[0, 1]
            mad = np.abs(g32 - e).max()
            flag = "" if corr > 0.99 else "  <-- DIVERGES"
            print(f"layer {l:2d} ({kind[:7]:7s}): corr={corr:.5f} max_abs={mad:.4e}{flag}")

    # final logits
    fin = rmsnorm(h, fin_v)
    logits = lm @ fin
    top = np.argsort(logits)[::-1][:5]
    print("golden top-5:", [(int(t), float(logits[t])) for t in top])
    print("golden token16 logit:", float(logits[16]), " token50:", float(logits[50]))


if __name__ == "__main__":
    main()
