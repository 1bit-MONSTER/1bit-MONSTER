#!/usr/bin/env python3
"""3-token prefill forward — port of engine flow with n_tokens=3, verifying the
conv_state/prev_hs recurrence inside the batch. Compares against ZAYA_DUMP files."""
import numpy as np, glob, os
from gguf import GGUFReader
WEIGHT_MOE = os.environ.get('WEIGHT_MOE', '1') == '1'

H, nq, nkv, hd = 2048, 8, 2, 128
qd, kd, qkv = nq*hd, nkv*hd, nq*hd + nkv*hd
gc = qkv // (nq + nkv)
nrot = hd // 2
gqa = nq // nkv
n_groups = nq + nkv
n_ff, n_exp, n_exp_t, rtr_h = 2048, 16, 17, 256
NC = 40
rope_base = 5e6
eps_norm = 1e-5

def rmsnorm(x, w, eps=eps_norm):
    return x / np.sqrt((x*x).mean(-1, keepdims=True) + eps) * w

def gelu_tanh(x):
    return 0.5*x*(1.0 + np.tanh(0.79788456*x*(1.0 + 0.044715*x*x)))

r = GGUFReader('/home/bcloud/zaya-f32.gguf')
tensors = {t.name: t.data.astype(np.float32) for t in r.tensors}
def gt(name):
    return tensors[name]

emb = gt('token_embd.weight')
isc = gt('input_hidden_states_scale.weight'); ib = gt('input_hidden_states_scale.bias')
out_norm = gt('output_norm.weight')

toks = np.array([2, 2202])
nt = len(toks)
h = (emb[toks] + ib) * isc   # [nt, H]
residual = np.zeros((nt, H), np.float32)
has_res = False
prev_router = None
kv_cache = {}
conv_state = {}
vrec = {}

def cmp(name, arr, il):
    pats = sorted(glob.glob(f'/tmp/zcpu_*_{name}-{il}.bin'))
    if not pats: return
    d = np.fromfile(pats[0], dtype=np.float32)
    arr = np.ascontiguousarray(arr)
    arr = arr.ravel()
    if d.shape != arr.shape:
        print(f"  {name}-{il}: SHAPE dump={d.shape} ref={arr.shape}")
        return
    c = np.corrcoef(d, arr)[0,1]
    md = np.abs(d-arr).max()
    flag = "OK " if c > 0.99999 and md < 1e-3 else ("~" if c > 0.999 else "FAIL")
    print(f"  {name}-{il}: corr={c:.6f} maxdiff={md:.3e} {flag}")

def cca_prep(l, q, k, v_cur, pos):
    cdw = gt(f'blk.{l}.ssm_conv1d.weight'); cdb = gt(f'blk.{l}.ssm_conv1d.bias')
    cgw = gt(f'blk.{l}.cca_conv_grp.weight'); cgb = gt(f'blk.{l}.cca_conv_grp.bias')
    ks = gt(f'blk.{l}.cca_k_scale.weight')
    cs = conv_state.setdefault(l, np.zeros((2, qkv), np.float32))
    sqk = np.concatenate([q, k])   # [nt, qkv]
    # 2-tap depthwise conv over time with state: per token t
    # dw(t) = cdw0*s(t-2) + cdw1*s(t-1) + cdb  (state holds [s(t-2), s(t-1)])
    dw = np.zeros_like(sqk)
    for t in range(nt):
        s0 = cs[0].copy(); s1 = cs[1].copy()
        cur_v = sqk[t]
        dw[t] = cdw[:,0]*s0 + cdw[:,1]*s1 + cdb
        cs[0] = s1; cs[1] = cur_v
    # grouped conv per time step
    out = np.zeros_like(sqk)
    for t in range(nt):
        for oc in range(qkv):
            grp = oc // gc; base = grp * gc
            cwr = cgw[oc]
            out[t, oc] = np.sum(cwr[:,0]*dw[t, base:base+gc] + cwr[:,1]*dw[t, base:base+gc+1][:gc] if False else cwr[:,0]*dw[t, base:base+gc] + cwr[:,1]*dw[t, base:base+gc])
    # hmm need dw(t-1) for the second tap: grouped conv over dw0/dw1 which are the
    # two conv taps. engine: a = sum_j cgw[oc, j, 0]*dw0[base+j] + cgw[oc, j, 1]*dw1[base+j]
    # dw0 = dw at t, dw1 = dw at t+1. Our dw rows are the per-token outputs; the
    # grouped conv mixes tap0 (token t) and tap1 (token t+1)?? Actually engine cca_prep
    # computes dw0/dw1 from the conv state for ONE token (pos), then grouped conv.
    # For batch prefill, llama's ssm_conv produces per-token outputs; the grouped
    # conv uses QK (all tokens) with kernel width 2 over the time axis? No --
    # ggml_conv_1d_grouped operates over the sequence axis with kernel width d_conv.
    # Check: conv_input = [conv_state(2); QKraw_t(nt)] -> ssm_conv (2-tap) -> [nt+1, qkv]
    # then grouped conv 1d over time with kernel 2 -> [nt, qkv] where out[t] =
    # sum_j cgw[oc,j,0]*in[t] + cgw[oc,j,1]*in[t+1].
    # engine for single token: dw0 = w0*s0 + w1*s1 + b (uses state), dw1 = w0*s1 + w1*cur + b
    # which matches ssm_conv outputs for the LAST token of the batch.
    return None

def attn_layer(l, cur, pos):
    wq = gt(f'blk.{l}.attn_q.weight'); wk = gt(f'blk.{l}.attn_k.weight')
    wv1 = gt(f'blk.{l}.cca_val_proj1.weight'); wv2 = gt(f'blk.{l}.cca_val_proj2.weight')
    wo = gt(f'blk.{l}.attn_output.weight')
    q = wq @ cur.T   # [qd, nt]
    k = wk @ cur.T
    v_cur = wv1 @ cur.T
    v_del = wv2 @ cur.T
    # conv + qk_means + l2 + rope for all tokens, with state across batch
    sqk = np.concatenate([q, k], axis=0)  # [qkv, nt]
    cdw = gt(f'blk.{l}.ssm_conv1d.weight'); cdb = gt(f'blk.{l}.ssm_conv1d.bias')
    cgw = gt(f'blk.{l}.cca_conv_grp.weight'); cgb = gt(f'blk.{l}.cca_conv_grp.bias')
    ks = gt(f'blk.{l}.cca_k_scale.weight')
    cs = conv_state.setdefault(l, np.zeros((2, qkv), np.float32))
    # ssm_conv(2-tap) over [state(2); sqk(nt)] -> nt+1 outputs:
    #   out[0] = w0*s0 + w1*s1 + b            (both taps on state)
    #   out[t] = w0*sqk[t-2] + w1*sqk[t-1] + b for t=1..nt  (last: prev token + cur)
    # dw0[t] = out[t], dw1[t] = out[t+1]  (tap pair per token, matches engine cca_prep)
    out_conv = np.zeros((qkv, nt+1))
    s0 = cs[0].copy(); s1 = cs[1].copy()
    out_conv[:, 0] = cdw[:,0]*s0 + cdw[:,1]*s1 + cdb
    for t in range(1, nt+1):
        prev = s1 if t == 1 else sqk[:, t-2]
        curv = sqk[:, t-1]
        out_conv[:, t] = cdw[:,0]*prev + cdw[:,1]*curv + cdb
    cs[0] = sqk[:, nt-2] if nt >= 2 else s1
    cs[1] = sqk[:, nt-1]
    dw0 = out_conv[:, :nt].copy()
    dw1 = out_conv[:, 1:nt+1].copy()
    # grouped conv: out[oc, t] = sum_j cgw[oc,j,0]*dw0[base+j,t] + cgw[oc,j,1]*dw1[base+j,t]
    gout = np.zeros_like(sqk)
    for t in range(nt):
        for oc in range(qkv):
            grp = oc // gc; base = grp * gc
            cwr = cgw[oc]
            gout[oc, t] = np.sum(cwr[:,0]*dw0[base:base+gc, t] + cwr[:,1]*dw1[base:base+gc, t]) + cgb[oc]
    sqk = gout
    # qk_means
    for hh in range(nq):
        kv = hh // gqa
        sqk[hh*hd:(hh+1)*hd] += 0.5*q[hh*hd:(hh+1)*hd] + 0.5*k[kv*hd:(kv+1)*hd]
    for khv in range(nkv):
        sm = np.mean(q[khv*gqa*hd:(khv*gqa+gqa)*hd].reshape(gqa, hd, nt), axis=0)
        sqk[qd+khv*hd:qd+(khv+1)*hd] += 0.5*sm + 0.5*k[khv*hd:(khv+1)*hd]
    # l2
    shd = np.sqrt(hd)
    for hh in range(nq):
        s = np.sum(sqk[hh*hd:(hh+1)*hd]**2, axis=0)
        sqk[hh*hd:(hh+1)*hd] *= (shd/(np.sqrt(s)+1e-12))[None, :]
    for khv in range(nkv):
        s = np.sum(sqk[qd+khv*hd:qd+(khv+1)*hd]**2, axis=0)
        sqk[qd+khv*hd:qd+(khv+1)*hd] *= (shd*ks[khv]/(np.sqrt(s)+1e-12))[None, :]
    # rope per token (positions 0..nt-1)
    sqk_rope = sqk.copy()
    for t in range(nt):
        rc = np.zeros(nrot, np.float32); rs = np.zeros(nrot, np.float32)
        for i in range(nrot//2):
            th = t * rope_base ** (-2.0*i/nrot)
            rc[i] = np.cos(th); rs[i] = np.sin(th)
            rc[nrot//2+i] = rc[i]; rs[nrot//2+i] = rs[i]
        for hh in range(nq + nkv):
            base = hh*hd if hh < nq else qd + (hh-nq)*hd
            for dd in range(nrot):
                d2 = dd + nrot//2 if dd < nrot//2 else dd - nrot//2
                xv = sqk_rope[base+dd, t]; xw = sqk_rope[base+d2, t]
                rh = -xw if dd < nrot//2 else xw
                sqk[base+dd, t] = xv*rc[dd] + rh*rs[dd]
    qo = sqk[:qd].copy(); ko = sqk[qd:].copy()
    # llama graph: V2 = wv2 @ hs_d, hs_d[t] = cur[t-1] (prev token's input), hs_d[0] = prev state (0)
    hs_d = np.zeros_like(cur.T)   # [H, nt]
    hs_d[:, 1:] = cur.T[:, :-1]
    v_del = wv2 @ hs_d            # [kd/2, nt]
    vo = np.concatenate([v_cur, v_del], axis=0)  # [kd, nt]
    # kv cache per token
    for t in range(nt):
        K, V = kv_cache.get(l, (np.zeros((0, kd), np.float32), np.zeros((0, kd), np.float32)))
        K = np.vstack([K, ko[:, t][None, :]]); V = np.vstack([V, vo[:, t][None, :]])
        kv_cache[l] = (K, V)
    # attention per token
    scale = 1.0/np.sqrt(hd)
    ao = np.zeros((qd, nt))
    K, V = kv_cache[l]
    for t in range(nt):
        seq = t + 1
        for hh in range(nq):
            kvh = hh // gqa
            qh = qo[hh*hd:(hh+1)*hd, t]
            scores = (K[:seq, kvh*hd:(kvh+1)*hd] @ qh) * scale
            mx = scores.max(); p = np.exp(scores - mx); p /= p.sum()
            ao[hh*hd:(hh+1)*hd, t] = V[:seq, kvh*hd:(kvh+1)*hd].T @ p
    attn_out = wo @ ao  # [H, nt]
    return attn_out.T

def moe_layer(l, cur):
    gdw = gt(f'blk.{l}.ffn_gate_inp.weight'); gdb = gt(f'blk.{l}.ffn_gate_inp.bias')
    rfn = gt(f'blk.{l}.ffn_norm.weight')
    rf1 = gt(f'blk.{l}.ffn_gate.weight'); rf1b = gt(f'blk.{l}.ffn_gate.bias')
    rf2 = gt(f'blk.{l}.zaya_router_mlp2.weight'); rf2b = gt(f'blk.{l}.zaya_router_mlp2.bias')
    rout = gt(f'blk.{l}.zaya_router_mlp4.weight')
    bb = gt(f'blk.{l}.zaya_router_biases.weight')
    eda = gt(f'blk.{l}.zaya_router_eda.weight')
    gu = gt(f'blk.{l}.ffn_gate_up_exps.weight'); dn = gt(f'blk.{l}.ffn_down_exps.weight')
    global prev_router
    rs = gdw @ cur.T + gdb[:, None]   # [rtr_h, nt]
    if l != 1 and prev_router is not None:
        rs = rs + prev_router * eda[:, None]
    prev_router = rs.copy()
    rsn = rmsnorm(rs.T, rfn).T
    rs = gelu_tanh(rf1 @ rsn + rf1b[:, None])
    rs = gelu_tanh(rf2 @ rs + rf2b[:, None])
    logits = rout @ rs               # [17, nt]
    mx = logits.max(0, keepdims=True); pr = np.exp(logits - mx); pr /= pr.sum(0, keepdims=True)
    exp_probs = pr[:n_exp] + bb[:n_exp, None]
    es = np.argmax(exp_probs, axis=0)
    out = np.zeros((H, nt))
    for t in range(nt):
        e = int(es[t])
        gate = gu[e, :n_ff]; up = gu[e, n_ff:]
        g = gate @ cur[t]; u = up @ cur[t]
        g2 = g / (1.0 + np.exp(-g)) * u
        o = dn[e] @ g2
        if WEIGHT_MOE: o = o * float(exp_probs[e, t])
        out[:, t] = o
    return out.T, es

for l in range(NC):
    if l % 2 == 0:
        hs = gt(f'blk.{l}.res_scale_hs.weight'); hb = gt(f'blk.{l}.res_scale_hs.bias')
        rs = gt(f'blk.{l}.res_scale_res.weight'); rb = gt(f'blk.{l}.res_scale_res.bias')
    else:
        hs = gt(f'blk.{l}.res_scale_hs_mlp.weight'); hb = gt(f'blk.{l}.res_scale_hs_mlp.bias')
        rs = gt(f'blk.{l}.res_scale_res_mlp.weight'); rb = gt(f'blk.{l}.res_scale_res_mlp.bias')
    if h.ndim == 2 and h.shape[0] == H: h = h.T   # -> [nt, H]
    tmp = (h + hb[None, :]) * hs[None, :]
    if has_res:
        residual = tmp + (residual + rb[None, :]) * rs[None, :]
    else:
        residual = tmp; has_res = True
    cur = rmsnorm(residual, gt(f'blk.{l}.attn_norm.weight'))
    cmp('residual', residual, l)
    cmp('input_norm', cur, l)
    if l % 2 == 0:
        h = attn_layer(l, cur, 0)
    else:
        h, es = moe_layer(l, cur)
        print(f"  [L{l}] experts={es.tolist()}")

tmp = h + residual
cur = rmsnorm(tmp, out_norm)
cmp('result_norm', cur, -1)
logits = emb @ cur.T   # [vocab, nt]
np.save('/tmp/zaya_prefill3_logits.npy', logits[:, -1].astype(np.float32))
top = np.argsort(-logits[:, -1])[:5]
print("ref top5:", top.tolist())
