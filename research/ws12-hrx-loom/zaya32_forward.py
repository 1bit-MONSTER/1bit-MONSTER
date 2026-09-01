#!/usr/bin/env python3
"""32-token zaya forward — exact numpy port of the hrx-v2 fork's src/models/zaya.cpp
graph (both CCA attention AND MoE in every layer). Used as an independent CPU
reference: validate vs dump32(zaya-f32.gguf, ngl=0), then run with Q4NX-
dequantized (twin) weights to measure the true Q4NX-HRX execution gap.

Usage:
  zaya32_forward.py f32_model.gguf        -> reference logits (float weights)
  zaya32_forward.py q4nx_model.gguf shape_model.gguf  -> twin logits (dequant)
"""
import sys
import numpy as np

sys.path.insert(0, "/home/bcloud/hrx-ws/hrx-v2-src/gguf-py")
from gguf import GGUFReader

H, nq, nkv, hd = 2048, 8, 2, 128
qd, kd = nq * hd, nkv * hd
qkv = qd + kd
gc = qkv // (nq + nkv)          # 128 channels per group (qkv = 10 groups)
nrot = hd // 2
gqa = nq // nkv
n_ff, n_exp, n_exp_t = 2048, 16, 17
NC = 40
rope_base = 5e6
eps_norm = 1e-5
import os
NT = int(os.environ.get('ZAYA_NT', '32'))

TILE_ROWS, TILE_COLS = 32, 256
_R = np.arange(TILE_ROWS)
_LANE = _R // 16
_BYTE_IDX = (_R % 16) // 2
_NIB = _R % 2
_COLS = np.arange(TILE_COLS)


def _bf16_arr(b):
    u16 = b.reshape(b.shape[:-1] + (-1, 2)).astype(np.uint32)
    u16 = (u16[..., 0] | (u16[..., 1] << 8)) << 16
    return u16.view(np.float32)


def dequant_tensor(raw, rows, cols):
    n_tc = cols // TILE_COLS
    n_tr = rows // TILE_ROWS
    multi = raw.ndim == 3
    n_ex = raw.shape[0] if multi else 1
    tiles = raw.shape[1] if multi else raw.shape[0]
    view = raw.reshape(n_ex, tiles, 5120)
    scale = _bf16_arr(view[..., 0:512]).reshape(n_ex, tiles, TILE_ROWS, 8)
    zp = _bf16_arr(view[..., 512:1024]).reshape(n_ex, tiles, TILE_ROWS, 8)
    packed = view[..., 1024:5120].reshape(n_ex, tiles, 2, TILE_COLS, 8)
    byte = np.stack([packed[:, :, _LANE[r], :, _BYTE_IDX[r]]
                     for r in range(TILE_ROWS)], axis=2)
    q = np.where(_NIB[None, None, :, None] == 0, byte & 0x0F, (byte >> 4) & 0x0F)
    q32 = q.astype(np.int32)
    val = np.where(q32 < 8, q32, q32 - 16).astype(np.float32)
    g = _COLS // 32
    scale = scale[:, :, :, g]
    zp = zp[:, :, :, g]
    np.nan_to_num(scale, copy=False)
    np.nan_to_num(zp, copy=False)
    scale[np.abs(scale) > 100.0] = 0.0
    zp[np.abs(zp) > 100.0] = 0.0
    vals = val * scale + zp
    out = np.zeros((n_ex, rows, cols), np.float32)
    for e in range(n_ex):
        for ti in range(tiles):
            tr, tc = divmod(ti, n_tc)
            out[e, tr * TILE_ROWS:(tr + 1) * TILE_ROWS,
                   tc * TILE_COLS:(tc + 1) * TILE_COLS] = vals[e, ti]
    return out if multi else out[0]


class M:
    def __init__(self, path, shape_model=None):
        r = GGUFReader(path)
        self.r = r
        self.shapes = {}
        if shape_model:
            rs = GGUFReader(shape_model)
            self.shapes = {t.name: tuple(int(x) for x in t.shape) for t in rs.tensors}
        self.idx = {t.name: t for t in r.tensors}

    def gt(self, name):
        t = self.idx[name]
        if t.tensor_type == 42:
            shape = self.shapes[name]  # file dims [ne0=k, ne1=n, ne2=e]
            cols, rows = shape[0], shape[1]
            # dequant_tensor returns numpy order (experts, n, k) / (n, k),
            # matching the F32 t.data layout
            return dequant_tensor(t.data, rows, cols)
        # F32: t.data is already numpy-order (reversed file dims)
        return np.ascontiguousarray(t.data.astype(np.float32))


def rmsnorm(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps_norm) * w


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * x * (1.0 + 0.044715 * x * x)))


def silu(x):
    return x / (1.0 + np.exp(-x))


def dbg(name, arr, l=0):
    np.save(f'/tmp/np_{name}_{l}.npy', np.ascontiguousarray(arr.astype(np.float32)))


def main():
    args = sys.argv[1:]
    if len(args) < 1:
        print(__doc__)
        return 1
    path = args[0]
    shape_model = args[1] if len(args) > 1 else None
    m = M(path, shape_model)

    emb = m.gt('token_embd.weight')
    isc = m.gt('input_hidden_states_scale.weight')
    ib = m.gt('input_hidden_states_scale.bias')
    out_norm_w = m.gt('output_norm.weight')

    toks = np.full(NT, 2, np.int64)
    h = (emb[toks] + ib) * isc            # [NT, H]

    conv_state = {l: np.zeros((2, qkv), np.float32) for l in range(NC)}
    prev_hs = {l: np.zeros(H, np.float32) for l in range(NC)}
    kv_cache = {l: (np.zeros((0, kd), np.float32), np.zeros((0, kd), np.float32))
                for l in range(NC)}
    prev_router = None

    for l in range(NC):
        residual = h
        cur = rmsnorm(residual, m.gt(f'blk.{l}.attn_norm.weight'))
        dbg('input_norm', cur, l)

        # ---------- CCA attention ----------
        wq = m.gt(f'blk.{l}.attn_q.weight')
        wk = m.gt(f'blk.{l}.attn_k.weight')
        wv1 = m.gt(f'blk.{l}.cca_val_proj1.weight')
        wv2 = m.gt(f'blk.{l}.cca_val_proj2.weight')
        wo = m.gt(f'blk.{l}.attn_output.weight')
        cdw = m.gt(f'blk.{l}.ssm_conv1d.weight')
        cdb = m.gt(f'blk.{l}.ssm_conv1d.bias')
        cgw = m.gt(f'blk.{l}.cca_conv_grp.weight')
        cgb = m.gt(f'blk.{l}.cca_conv_grp.bias')
        ks = m.gt(f'blk.{l}.cca_k_scale.weight')

        Qraw = wq @ cur.T                 # [qd, NT]
        Kraw = wk @ cur.T                 # [kd, NT]
        dbg('Qraw', Qraw, l)
        dbg('Kraw', Kraw, l)
        sqk = np.concatenate([Qraw, Kraw], axis=0)   # [qkv, NT]

        # ssm_conv over [conv_state(2); sqk(NT)] -> NT+1 outputs
        cs = conv_state[l]
        out_conv = np.zeros((qkv, NT + 1), np.float32)
        out_conv[:, 0] = cdw[:, 0] * cs[0] + cdw[:, 1] * cs[1] + cdb
        for t in range(1, NT + 1):
            prev = cs[1] if t == 1 else sqk[:, t - 2]
            curv = sqk[:, t - 1]
            out_conv[:, t] = cdw[:, 0] * prev + cdw[:, 1] * curv + cdb
        conv_state[l] = np.stack([sqk[:, NT - 2], sqk[:, NT - 1]])
        dw0 = out_conv[:, :NT].copy()
        dw1 = out_conv[:, 1:NT + 1].copy()
        dbg('QK_dw', out_conv.T, l)   # ggml: [33, 1280]

        # grouped conv (groups = n_head + n_head_kv = 10)
        gout = np.zeros_like(sqk)
        for t in range(NT):
            for oc in range(qkv):
                grp = oc // gc
                base = grp * gc
                cwr = cgw[oc]
                gout[oc, t] = np.sum(cwr[:, 0] * dw0[base:base + gc, t] +
                                     cwr[:, 1] * dw1[base:base + gc, t]) + cgb[oc]
        sqk = gout
        dbg('QK_grp', sqk.T, l)       # ggml: [32, 1280]

        # qk_means (Q side: GQA repeat; K side: mean over gqa)
        qkm_q = np.zeros_like(Qraw)
        qkm_k = np.zeros((qkv, NT), np.float32)
        for hh in range(nq):
            kvh = hh // gqa
            qkm_q[hh * hd:(hh + 1) * hd] = 0.5 * Qraw[hh * hd:(hh + 1) * hd] + \
                0.5 * Kraw[kvh * hd:(kvh + 1) * hd]
        for khv in range(nkv):
            sm = np.mean(Qraw[khv * gqa * hd:(khv * gqa + gqa) * hd].reshape(gqa, hd, NT),
                         axis=0)
            qkm_k[qd + khv * hd:qd + (khv + 1) * hd] = 0.5 * sm + 0.5 * Kraw[khv * hd:(khv + 1) * hd]
        sqk[:qd] += qkm_q
        sqk[qd:] += qkm_k[qd:]
        dbg('qk_mean_q', qkm_q.reshape(nq, hd, NT).transpose(1, 0, 2), l)
        dbg('qk_mean_k', qkm_k[qd:].reshape(nkv, hd, NT).transpose(1, 0, 2), l)

        # l2 normalize each head then scale sqrt(hd); K heads by cca_k_scale
        shd = np.sqrt(hd)
        for hh in range(nq):
            s = np.sum(sqk[hh * hd:(hh + 1) * hd] ** 2, axis=0)
            sqk[hh * hd:(hh + 1) * hd] *= (shd / (np.sqrt(s) + 1e-12))[None, :]
        for khv in range(nkv):
            s = np.sum(sqk[qd + khv * hd:qd + (khv + 1) * hd] ** 2, axis=0)
            sqk[qd + khv * hd:qd + (khv + 1) * hd] *= (shd * ks[khv] / (np.sqrt(s) + 1e-12))[None, :]

        # neox rope, positions 0..NT-1, first nrot dims per head
        # (pair (i, i+nrot//2) computed from ORIGINAL values — no in-place
        #  corruption of the partner)
        for t in range(NT):
            rc = np.zeros(nrot, np.float32)
            rs_ = np.zeros(nrot, np.float32)
            for i in range(nrot // 2):
                th = t * rope_base ** (-2.0 * i / nrot)
                rc[i] = np.cos(th)
                rs_[i] = np.sin(th)
                rc[nrot // 2 + i] = rc[i]
                rs_[nrot // 2 + i] = rs_[i]
            for hh in range(nq + nkv):
                base = hh * hd if hh < nq else qd + (hh - nq) * hd
                for i in range(nrot // 2):
                    x0 = sqk[base + i, t]
                    x1 = sqk[base + i + nrot // 2, t]
                    sqk[base + i, t] = x0 * rc[i] - x1 * rs_[i]
                    sqk[base + i + nrot // 2, t] = x0 * rs_[i] + x1 * rc[i]
        dbg('Qcur_pre_rope', sqk[:qd].reshape(nq, hd, NT).transpose(1, 0, 2), l)
        dbg('Kcur_pre_rope', sqk[qd:].reshape(nkv, hd, NT).transpose(1, 0, 2), l)
        if l == 0:
            print('DBG h1t10 after l2:', sqk[128, 10], flush=True)
        qo = sqk[:qd].copy()
        ko = sqk[qd:].copy()

        # V: V1 = wv1@cur; V2 = wv2@[prev_hs; cur[:-1]]
        hs_d = np.zeros((H, NT), np.float32)
        hs_d[:, 1:] = cur.T[:, :-1]
        hs_d[:, 0] = prev_hs[l]
        prev_hs[l] = cur[NT - 1].copy()
        v_del = wv2 @ hs_d
        vo = np.concatenate([wv1 @ cur.T, v_del], axis=0)   # [kd, NT]

        K, V = kv_cache[l]
        K = np.vstack([K, ko.T])
        V = np.vstack([V, vo.T])
        kv_cache[l] = (K, V)

        scale = 1.0 / np.sqrt(hd)
        ao = np.zeros((qd, NT))
        for t in range(NT):
            seq = t + 1
            for hh in range(nq):
                kvh = hh // gqa
                qh = qo[hh * hd:(hh + 1) * hd, t]
                scores = (K[:seq, kvh * hd:(kvh + 1) * hd] @ qh) * scale
                mx = scores.max()
                p = np.exp(scores - mx)
                p /= p.sum()
                ao[hh * hd:(hh + 1) * hd, t] = V[:seq, kvh * hd:(kvh + 1) * hd].T @ p
        attn_out = wo @ ao                 # [H, NT]
        dbg('attn_out', attn_out, l)
        dbg('Qcur', qo.reshape(nq, hd, NT).transpose(1, 0, 2), l)
        dbg('Kcur', ko.reshape(nkv, hd, NT).transpose(1, 0, 2), l)
        dbg('Vcur', np.concatenate([wv1 @ cur.T, v_del], axis=0), l)

        # ---- post-attention residual scale ----
        hs_w = m.gt(f'blk.{l}.res_scale_hs.weight')
        hs_b = m.gt(f'blk.{l}.res_scale_hs.bias')
        rs_w = m.gt(f'blk.{l}.res_scale_res.weight')
        rs_b = m.gt(f'blk.{l}.res_scale_res.bias')
        residual = (attn_out.T + hs_b) * hs_w + (residual + rs_b) * rs_w
        dbg('residual_post_attn', residual, l)

        cur = rmsnorm(residual, m.gt(f'blk.{l}.post_attn_norm.weight'))

        # ---------- MoE ----------
        gdw = m.gt(f'blk.{l}.ffn_gate_inp.weight')
        gdb = m.gt(f'blk.{l}.ffn_gate_inp.bias')
        rfn = m.gt(f'blk.{l}.ffn_norm.weight')
        rf1 = m.gt(f'blk.{l}.ffn_gate.weight')
        rf1b = m.gt(f'blk.{l}.ffn_gate.bias')
        rf2 = m.gt(f'blk.{l}.zaya_router_mlp2.weight')
        rf2b = m.gt(f'blk.{l}.zaya_router_mlp2.bias')
        rout = m.gt(f'blk.{l}.zaya_router_mlp4.weight')
        bb = m.gt(f'blk.{l}.zaya_router_biases.weight')
        eda_name = f'blk.{l}.zaya_router_eda.weight'
        eda = m.gt(eda_name) if eda_name in m.idx else None
        gu = m.gt(f'blk.{l}.ffn_gate_up_exps.weight')
        dn = m.gt(f'blk.{l}.ffn_down_exps.weight')

        rs = gdw @ cur.T + gdb[:, None]     # [rtr_h, NT]
        if prev_router is not None and eda is not None:
            rs = rs + prev_router * eda[:, None]
        dbg('router_down', rs, l)
        prev_router = rs.copy()
        dbg('router_norm', rsn if False else rmsnorm(rs.T, rfn).T, l)
        rsn = rmsnorm(rs.T, rfn).T
        rs = gelu_tanh(rf1 @ rsn + rf1b[:, None])
        rs = gelu_tanh(rf2 @ rs + rf2b[:, None])
        logits = rout @ rs                  # [17, NT]
        dbg('router_logits', logits, l)
        mx = logits.max(0, keepdims=True)
        pr = np.exp(logits - mx)
        pr /= pr.sum(0, keepdims=True)
        gate_probs = pr[:n_exp] + bb[:n_exp, None]
        es = np.argmax(gate_probs, axis=0)

        out = np.zeros((H, NT))
        for t in range(NT):
            e = int(es[t])
            g = gu[e, :n_ff] @ cur[t]
            u = gu[e, n_ff:] @ cur[t]
            g2 = silu(g) * u
            o = dn[e] @ g2
            out[:, t] = o * pr[e, t]        # weights = UNBIASED probs
        moe_out = out.T                      # [NT, H]

        # ---- post-MLP residual scale ----
        hm_w = m.gt(f'blk.{l}.res_scale_hs_mlp.weight')
        hm_b = m.gt(f'blk.{l}.res_scale_hs_mlp.bias')
        rm_w = m.gt(f'blk.{l}.res_scale_res_mlp.weight')
        rm_b = m.gt(f'blk.{l}.res_scale_res_mlp.bias')
        h = (moe_out + hm_b) * hm_w + (residual + rm_b) * rm_w
        dbg('layer_out', h, l)
        dbg('moe_out', moe_out, l)

    cur = rmsnorm(h, out_norm_w)
    dbg('result_norm', cur, -1)
    logits = emb @ cur.T                    # [vocab, NT]
    last = logits[:, -1]
    top = np.argsort(-last)[:5]
    print(f"top5: {top.tolist()}  (nt={NT})", file=sys.stderr)
    np.save('/tmp/zaya32_numpy_logits.npy', last.astype(np.float32))
    return 0


if __name__ == "__main__":
    sys.exit(main())
