#!/usr/bin/env python3
"""Full Zaya 1-8B forward in numpy — 1:1 port of engine zaya_decode.cpp +
zaya_cca_attn_cpu.h + zaya_moe_cpu.h, reading F32 weights from the GGUF.
Compares against llama.cpp ZAYA_DUMP files (/tmp/zcpu_*.bin)."""
import sys, os, glob, numpy as np
from gguf import GGUFReader
WEIGHT_MOE = os.environ.get('WEIGHT_MOE', '1') == '1'

H, nq, nkv, hd = 2048, 8, 2, 128
qd, kd, qkv = nq*hd, nkv*hd, nq*hd + nkv*hd
gc = qkv // (nq + nkv)   # 128
nrot = hd // 2            # 64
gqa = nq // nkv           # 4
n_groups = nq + nkv       # 10
n_ff, n_exp, n_exp_t, rtr_h = 2048, 16, 17, 256
NC = 40
rope_base = 5e6
eps_norm = 1e-5

def rmsnorm(x, w, eps=eps_norm):
    return x / np.sqrt((x*x).mean() + eps) * w

def gelu_tanh(x):
    return 0.5*x*(1.0 + np.tanh(0.79788456*x*(1.0 + 0.044715*x*x)))

r = GGUFReader('/home/bcloud/zaya-f32.gguf')
tensors = {}
for t in r.tensors:
    tensors[t.name] = t.data.astype(np.float32)

def gt(name):
    return tensors[name]

# ---- layer 0-39 loop (engine forward) ----
emb = gt(f'token_embd.weight')          # [vocab, H]
isc = gt(f'input_hidden_states_scale.weight')
ib  = gt(f'input_hidden_states_scale.bias')
out_norm = gt(f'output_norm.weight')

tok = 5631  # llama_tokenize("hi") -> [5631]
h = (emb[tok] + ib) * isc
residual = np.zeros(H, np.float32)
has_res = False
prev_router = None
kv_cache = {}  # layer -> (K, V) arrays of rows
conv_state = {}  # layer -> [2, qkv]
vrec = {}        # layer -> [kd/2]

def cmp(name, arr, il):
    # find the dump file for this tensor
    pats = glob.glob(f'/tmp/zcpu_*_{name}-{il}.bin')
    if not pats: return
    d = np.fromfile(pats[0], dtype=np.float32)
    arr = np.ascontiguousarray(arr).ravel()
    if d.shape != arr.shape:
        print(f"  {name}-{il}: SHAPE mismatch dump={d.shape} ref={arr.shape}")
        return
    c = np.corrcoef(d, arr)[0,1]
    md = np.abs(d-arr).max()
    flag = "OK " if c > 0.99999 and md < 1e-3 else ("~" if c > 0.999 else "FAIL")
    print(f"  {name}-{il}: corr={c:.6f} maxdiff={md:.3e} {flag}")

def cca_prep(l, q, k, v_cur, pos):
    """Engine cca_prep — conv_qk + qk_means + L2 + RoPE."""
    w = {}
    cdw = gt(f'blk.{l}.ssm_conv1d.weight')   # [qkv, 2] data[c*2+tap]
    cdb = gt(f'blk.{l}.ssm_conv1d.bias')     # [qkv]
    cgw = gt(f'blk.{l}.cca_conv_grp.weight') # [qkv, gc, 2] data[oc*(gc*2)+j*2+tap]
    cgb = gt(f'blk.{l}.cca_conv_grp.bias')   # [qkv]
    ks  = gt(f'blk.{l}.cca_k_scale.weight')  # [nkv]
    cs  = conv_state.setdefault(l, np.zeros((2, qkv), np.float32))

    sqk = np.concatenate([q, k])            # [qkv]
    s0 = cs[0].copy(); s1 = cs[1].copy(); cur = sqk.copy()
    dw0 = cdw[:,0]*s0 + cdw[:,1]*s1 + cdb
    dw1 = cdw[:,0]*s1 + cdw[:,1]*cur + cdb
    cs[0] = s1; cs[1] = cur
    # grouped conv
    out = np.zeros(qkv, np.float32)
    for oc in range(qkv):
        grp = oc // gc; base = grp * gc
        cwr = cgw[oc]   # [gc, 2]
        a = float(np.sum(cwr[:,0]*dw0[base:base+gc] + cwr[:,1]*dw1[base:base+gc]))
        sqk[oc] = a + cgb[oc]
    # qk_means
    for hh in range(nq):
        kv = hh // gqa
        sqk[hh*hd:(hh+1)*hd] += 0.5*q[hh*hd:(hh+1)*hd] + 0.5*k[kv*hd:(kv+1)*hd]
    for khv in range(nkv):
        sm = np.mean([q[(khv*gqa+g)*hd:(khv*gqa+g+1)*hd] for g in range(gqa)], axis=0)
        sqk[qd+khv*hd:qd+(khv+1)*hd] += 0.5*sm + 0.5*k[khv*hd:(khv+1)*hd]
    # L2
    shd = np.sqrt(hd)
    for hh in range(nq):
        s = np.sum(sqk[hh*hd:(hh+1)*hd]**2)
        iv = shd / (np.sqrt(s) + 1e-12)
        sqk[hh*hd:(hh+1)*hd] *= iv
    for khv in range(nkv):
        s = np.sum(sqk[qd+khv*hd:qd+(khv+1)*hd]**2)
        iv = shd * ks[khv] / (np.sqrt(s) + 1e-12)
        sqk[qd+khv*hd:qd+(khv+1)*hd] *= iv
    # RoPE (partial, first nrot dims, half-rotation pairing)
    rc = np.zeros(nrot, np.float32); rs = np.zeros(nrot, np.float32)
    for i in range(nrot//2):
        th = pos * rope_base ** (-2.0*i/nrot)
        rc[i] = np.cos(th); rs[i] = np.sin(th)
        rc[nrot//2+i] = rc[i]; rs[nrot//2+i] = rs[i]
    for hh in range(nq + nkv):
        base = hh*hd if hh < nq else qd + (hh-nq)*hd
        for dd in range(nrot):
            d2 = dd + nrot//2 if dd < nrot//2 else dd - nrot//2
            xv = sqk[base+dd]; xw = sqk[base+d2]
            rh = -xw if dd < nrot//2 else xw
            sqk[base+dd] = xv*rc[dd] + rh*rs[dd]
    q_out = sqk[:qd].copy(); k_out = sqk[qd:].copy()
    return q_out, k_out

def attn_layer(l, cur, pos):
    wq = gt(f'blk.{l}.attn_q.weight')     # [qd, H]
    wk = gt(f'blk.{l}.attn_k.weight')     # [kd, H]
    wv1 = gt(f'blk.{l}.cca_val_proj1.weight')  # [kd/2, H]
    wv2 = gt(f'blk.{l}.cca_val_proj2.weight')  # [kd/2, H]
    wo = gt(f'blk.{l}.attn_output.weight')     # [H, qd]
    q = wq @ cur; k = wk @ cur
    v_cur = wv1 @ cur
    # engine: v_del = wv2 @ cur (current normed hs); v_out 2nd half = vrec (prev token)
    v_del = wv2 @ cur
    qo, ko = cca_prep(l, q, k, v_cur, pos)
    # V assembly
    vr = vrec.setdefault(l, np.zeros(kd//2, np.float32))
    vo = np.concatenate([v_cur, vr])
    vr[:] = v_del
    # KV cache append
    K, V = kv_cache.get(l, (np.zeros((0, kd), np.float32), np.zeros((0, kd), np.float32)))
    K = np.vstack([K, ko[None, :]]); V = np.vstack([V, vo[None, :]])
    kv_cache[l] = (K, V)
    seq = K.shape[0]
    # GQA attention
    scale = 1.0/np.sqrt(hd)
    ao = np.zeros(qd, np.float32)
    for hh in range(nq):
        kvh = hh // gqa
        qh = qo[hh*hd:(hh+1)*hd]
        scores = (K[:, kvh*hd:(kvh+1)*hd] @ qh) * scale
        mx = scores.max()
        p = np.exp(scores - mx); p /= p.sum()
        ao[hh*hd:(hh+1)*hd] = V[:, kvh*hd:(kvh+1)*hd].T @ p
    attn_out = wo @ ao
    return attn_out

def moe_layer(l, cur):
    gdw = gt(f'blk.{l}.ffn_gate_inp.weight')   # [rtr_h, H]
    gdb = gt(f'blk.{l}.ffn_gate_inp.bias')
    rfn = gt(f'blk.{l}.ffn_norm.weight')       # [rtr_h]
    rf1 = gt(f'blk.{l}.ffn_gate.weight')       # [rtr_h, rtr_h]
    rf1b = gt(f'blk.{l}.ffn_gate.bias')
    rf2 = gt(f'blk.{l}.zaya_router_mlp2.weight')
    rf2b = gt(f'blk.{l}.zaya_router_mlp2.bias')
    rout = gt(f'blk.{l}.zaya_router_mlp4.weight')  # [17, rtr_h]
    bb = gt(f'blk.{l}.zaya_router_biases.weight')  # [17]
    eda = gt(f'blk.{l}.zaya_router_eda.weight')  # [rtr_h]
    gu = gt(f'blk.{l}.ffn_gate_up_exps.weight')  # [n_exp, 2*n_ff, H]
    dn = gt(f'blk.{l}.ffn_down_exps.weight')     # [n_exp, H, n_ff]
    global prev_router
    rs = gdw @ cur + gdb
    if l != 1 and prev_router is not None and eda is not None:
        rs = rs + prev_router * eda
    prev_router = rs.copy()  # stored BEFORE norm
    rs = rmsnorm(rs, rfn)
    rs = gelu_tanh(rf1 @ rs + rf1b)
    rs = gelu_tanh(rf2 @ rs + rf2b)
    logits = rout @ rs
    probs = np.exp(logits - logits.max()); probs /= probs.sum()
    exp_probs = probs[:n_exp] + bb[:n_exp]
    e = int(np.argmax(exp_probs))
    # expert ffn
    gate = gu[e, :n_ff]; up = gu[e, n_ff:]
    g = gate @ cur; u = up @ cur
    g2 = g / (1.0 + np.exp(-g)) * u   # silu(gate)*up
    out = dn[e] @ g2                   # [H, n_ff] @ [n_ff]
    if WEIGHT_MOE:
        out = out * exp_probs[e]       # HF semantics: weight by selected expert prob
    return out, e

for l in range(NC):
    if l % 2 == 0:
        hs = gt(f'blk.{l}.res_scale_hs.weight'); hb = gt(f'blk.{l}.res_scale_hs.bias')
        rs = gt(f'blk.{l}.res_scale_res.weight'); rb = gt(f'blk.{l}.res_scale_res.bias')
    else:
        hs = gt(f'blk.{l}.res_scale_hs_mlp.weight'); hb = gt(f'blk.{l}.res_scale_hs_mlp.bias')
        rs = gt(f'blk.{l}.res_scale_res_mlp.weight'); rb = gt(f'blk.{l}.res_scale_res_mlp.bias')
    tmp = (h + hb) * hs
    if has_res:
        residual = tmp + (residual + rb) * rs
    else:
        residual = tmp; has_res = True
    cur = rmsnorm(residual, gt(f'blk.{l}.attn_norm.weight'))
    cmp('residual', residual, l)
    cmp('input_norm', cur, l)
    if l % 2 == 0:
        cur_out = attn_layer(l, cur, 0)
        cmp('Qraw', gt(f'blk.{l}.attn_q.weight') @ cur, l)
        cmp('Kraw', gt(f'blk.{l}.attn_k.weight') @ cur, l)
    else:
        cur_out, e = moe_layer(l, cur)
        print(f"  [L{l}] expert={e}")
    h = cur_out

# final
tmp = h + residual
cur = rmsnorm(tmp, out_norm)
cmp('result_norm', cur, -1)
logits = emb @ cur
np.save('/tmp/zaya_ref_logits.npy', logits.astype(np.float32))
top = np.argsort(-logits)[:5]
print("ref top5:", top.tolist())
