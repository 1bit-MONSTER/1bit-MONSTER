#!/usr/bin/env python3
"""Zaya 8B numpy forward — CORRECT HF structure: every layer runs CCA attention
AND MoE with the HF residual chain:
  residual = h
  h = input_layernorm(residual)
  attn = CCA(h)
  residual = (attn+hs_b)*hs_s + (residual+res_b)*res_s      (post_attention)
  h = post_attention_layernorm(residual)
  moe = MoE(h)  (prev_router EDA)
  h = (moe+hs_b)*hs_s + (residual+res_b)*res_s              (post_mlp)
final: logits = emb @ norm(h)
"""
import numpy as np, os
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
np.save('/tmp/np_h0.npy', h[-1].astype(np.float32))
prev_router = None
kv_cache = {}
conv_state = {}

def cca_layer(l, cur, pos):
    """cur: [nt, H] -> attn_out [nt, H] (per-token, batch prefill)"""
    wq = gt(f'blk.{l}.attn_q.weight').reshape(qd, H); wk = gt(f'blk.{l}.attn_k.weight').reshape(kd, H)
    wv1 = gt(f'blk.{l}.cca_val_proj1.weight').reshape(kd//2, H); wv2 = gt(f'blk.{l}.cca_val_proj2.weight').reshape(kd//2, H)
    wo = gt(f'blk.{l}.attn_output.weight').reshape(H, qd)
    cdw = gt(f'blk.{l}.ssm_conv1d.weight').reshape(qkv, 2); cdb = gt(f'blk.{l}.ssm_conv1d.bias')
    cgw = gt(f'blk.{l}.cca_conv_grp.weight').reshape(qkv, gc, 2); cgb = gt(f'blk.{l}.cca_conv_grp.bias')
    ks = gt(f'blk.{l}.cca_k_scale.weight')
    q = (wq @ cur.T)   # [qd, nt]
    k = (wk @ cur.T)
    sqk = np.concatenate([q, k], axis=0)  # [qkv, nt]
    # depthwise conv (2-tap) with state: out_conv[t] = w0*in[t-2] + w1*in[t-1] + b
    # in = [state(2); sqk(nt)] -> out length nt+1
    cs = conv_state.setdefault(l, np.zeros((2, qkv), np.float32))
    s0, s1 = cs[0].copy(), cs[1].copy()
    out_conv = np.zeros((qkv, nt+1))
    out_conv[:, 0] = cdw[:,0]*s0 + cdw[:,1]*s1 + cdb
    for t in range(1, nt+1):
        prev = s1 if t == 1 else sqk[:, t-2]
        curv = sqk[:, t-1]
        out_conv[:, t] = cdw[:,0]*prev + cdw[:,1]*curv + cdb
    cs[0] = sqk[:, nt-2] if nt >= 2 else s1
    cs[1] = sqk[:, nt-1]
    # grouped conv (2-tap over the conv outputs)
    gout = np.zeros((qkv, nt))
    for t in range(nt):
        for oc in range(qkv):
            base = (oc//gc)*gc
            cwr = cgw[oc]
            gout[oc, t] = np.sum(cwr[:,0]*out_conv[base:base+gc, t] + cwr[:,1]*out_conv[base:base+gc, t+1]) + cgb[oc]
    # qk means
    qkm_q = np.zeros_like(q); qkm_k = np.zeros_like(k)
    for hh in range(nq):
        kv = hh // gqa
        qkm_q[hh*hd:(hh+1)*hd] = 0.5*q[hh*hd:(hh+1)*hd] + 0.5*k[kv*hd:(kv+1)*hd]
    for khv in range(nkv):
        sm = np.mean(q[khv*gqa*hd:(khv*gqa+gqa)*hd].reshape(gqa, hd, nt), axis=0)
        qkm_k[khv*hd:(khv+1)*hd] = 0.5*sm + 0.5*k[khv*hd:(khv+1)*hd]
    Qcur = gout[:qd] + qkm_q
    Kcur = gout[qd:] + qkm_k
    # L2
    shd = np.sqrt(hd)
    for hh in range(nq):
        s = np.sum(Qcur[hh*hd:(hh+1)*hd]**2, axis=0)
        Qcur[hh*hd:(hh+1)*hd] *= (shd/(np.sqrt(s)+1e-12))[None,:]
    for khv in range(nkv):
        s = np.sum(Kcur[khv*hd:(khv+1)*hd]**2, axis=0)
        Kcur[khv*hd:(khv+1)*hd] *= (shd*ks[khv]/(np.sqrt(s)+1e-12))[None,:]
    # rope per token
    Qr = np.zeros_like(Qcur); Kr = np.zeros_like(Kcur)
    for t in range(nt):
        rc = np.zeros(nrot, np.float32); rs = np.zeros(nrot, np.float32)
        for i in range(nrot//2):
            th = t * rope_base ** (-2.0*i/nrot)
            rc[i] = np.cos(th); rs[i] = np.sin(th)
            rc[nrot//2+i] = rc[i]; rs[nrot//2+i] = rs[i]
        for src, dst in [(Qcur, Qr), (Kcur, Kr)]:
            s2 = src.copy()
            nheads = src.shape[0]//hd
            for hh in range(nheads):
                base = hh*hd
                for dd in range(nrot):
                    d2 = dd + nrot//2 if dd < nrot//2 else dd - nrot//2
                    xv = s2[base+dd, t]; xw = s2[base+d2, t]
                    rh = -xw if dd < nrot//2 else xw
                    dst[base+dd, t] = xv*rc[dd] + rh*rs[dd]
    # V: v_cur + delayed (prev token's delayed proj); first token delayed=0
    v_cur = (wv1 @ cur.T)
    v_del = (wv2 @ cur.T)
    V = np.zeros((kd, nt))
    V[:kd//2] = v_cur
    V[kd//2:, 0] = 0
    V[kd//2:, 1:] = v_del[:, :-1]
    # attention per token with kv cache
    scale = 1.0/np.sqrt(hd)
    ao = np.zeros((qd, nt))
    K, Vc = kv_cache.get(l, (np.zeros((0, kd), np.float32), np.zeros((0, kd), np.float32)))
    for t in range(nt):
        K = np.vstack([K, Kr[:, t][None, :]])
        Vc = np.vstack([Vc, V[:, t][None, :]])
        for hh in range(nq):
            kvh = hh // gqa
            qh = Qr[hh*hd:(hh+1)*hd, t]
            scores = (K[:t+1, kvh*hd:(kvh+1)*hd] @ qh) * scale
            mx = scores.max(); p = np.exp(scores-mx); p /= p.sum()
            ao[hh*hd:(hh+1)*hd, t] = Vc[:t+1, kvh*hd:(kvh+1)*hd].T @ p
    kv_cache[l] = (K, Vc)
    return (wo @ ao).T   # [nt, H]

def moe_layer(l, cur):
    gdw = gt(f'blk.{l}.ffn_gate_inp.weight').reshape(rtr_h, H); gdb = gt(f'blk.{l}.ffn_gate_inp.bias')
    rfn = gt(f'blk.{l}.ffn_norm.weight')
    rf1 = gt(f'blk.{l}.ffn_gate.weight').reshape(rtr_h, rtr_h); rf1b = gt(f'blk.{l}.ffn_gate.bias')
    rf2 = gt(f'blk.{l}.zaya_router_mlp2.weight').reshape(rtr_h, rtr_h); rf2b = gt(f'blk.{l}.zaya_router_mlp2.bias')
    rout = gt(f'blk.{l}.zaya_router_mlp4.weight').reshape(n_exp_t, rtr_h)
    bb = gt(f'blk.{l}.zaya_router_biases.weight')
    gu = gt(f'blk.{l}.ffn_gate_up_exps.weight').reshape(n_exp, 2*n_ff, H)
    dn = gt(f'blk.{l}.ffn_down_exps.weight').reshape(n_exp, H, n_ff)
    global prev_router
    rs = gdw @ cur.T + gdb[:, None]
    if prev_router is not None and f'blk.{l}.zaya_router_eda.weight' in tensors:
        eda = gt(f'blk.{l}.zaya_router_eda.weight')
        rs = rs + prev_router * eda[:, None]
    prev_router = rs.copy()
    rs = rmsnorm(rs.T, rfn).T
    rs = gelu_tanh(rf1 @ rs + rf1b[:, None])
    rs = gelu_tanh(rf2 @ rs + rf2b[:, None])
    logits = rout @ rs
    mx = logits.max(0, keepdims=True); pr = np.exp(logits-mx); pr /= pr.sum(0, keepdims=True)
    exp_probs = pr[:n_exp] + bb[:n_exp, None]
    es = np.argmax(exp_probs, axis=0)
    out = np.zeros((H, nt))
    for t in range(nt):
        e = int(es[t])
        gate = gu[e, :n_ff]; up = gu[e, n_ff:]
        g = gate @ cur[t]; u = up @ cur[t]
        g2 = g/(1.0+np.exp(-g))*u
        o = dn[e] @ g2
        if WEIGHT_MOE: o = o * float(pr[e, t])   # HF: weight by UNBIASED prob
        out[:, t] = o
    return out.T, es

for l in range(NC):
    residual = h.copy()
    cur = rmsnorm(residual, gt(f'blk.{l}.attn_norm.weight'))
    attn = cca_layer(l, cur, 0)
    residual = (attn + gt(f'blk.{l}.res_scale_hs.bias')) * gt(f'blk.{l}.res_scale_hs.weight') \
             + (residual + gt(f'blk.{l}.res_scale_res.bias')) * gt(f'blk.{l}.res_scale_res.weight')
    cur = rmsnorm(residual, gt(f'blk.{l}.post_attn_norm.weight'))
    moe, es = moe_layer(l, cur)
    h = (moe + gt(f'blk.{l}.res_scale_hs_mlp.bias')) * gt(f'blk.{l}.res_scale_hs_mlp.weight') \
      + (residual + gt(f'blk.{l}.res_scale_res_mlp.bias')) * gt(f'blk.{l}.res_scale_res_mlp.weight')
    if l < 3:
        np.save(f'/tmp/np_L{l}.npy', h[-1].astype(np.float32))
        print(f"L{l}: experts={es.tolist()} h_rms={np.sqrt((h**2).mean()):.4f}")

cur = rmsnorm(h, out_norm)
logits = emb @ cur.T   # [vocab, nt]
np.save('/tmp/zaya2_logits.npy', logits[:, -1].astype(np.float32))
top = np.argsort(-logits[:, -1])[:5]
print("ref top5:", top.tolist())
