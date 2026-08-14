#!/usr/bin/env python3
# e2e_numpy_ref.py — authoritative numpy reference for families whose
# llama.cpp oracle is unreliable (internlm2: near-tie Q8; minicpm: llama.cpp
# ignores scale_emb/scale_depth/logits_scaling). Verifies the ENGINE's logit
# dump matches a faithful forward of the model's own semantics bit-for-bit.
# usage: e2e_numpy_ref.py <fixture_dir> <family> [engine_logits_dump]
import sys, os
import numpy as np
from safetensors import safe_open
from llama_cpp import Llama

dir, family = sys.argv[1], sys.argv[2]
f = safe_open(f"{dir}/model.safetensors", "pt")
keys = set(f.keys())

def silu(x): return x * (1/(1+np.exp(-x)))
def rms(x, w, eps=1e-5): return x * (1.0/np.sqrt((x*x).mean(-1,keepdims=True)+eps)) * w

def forward(ids):
    if family == "internlm2":
        H, NH, NKV, HD, IM, L = 2048, 16, 8, 128, 8192, 24
        theta, ROT = 1e6, 128
    elif family == "gptneox":
        H, NH, NKV, HD, IM, L = 512, 8, 8, 64, 2048, 6
        theta, ROT = 10000.0, 16  # rotary_pct 0.25
    else:  # minicpm
        H, NH, NKV, HD, IM, L = 2304, 36, 36, 64, 5760, 40
        theta, ROT = 10000.0, 64
    def rope(qk, pos, nh):
        d = ROT
        freqs = 1.0/(theta ** (np.arange(0, d, 2)/d))
        t = pos*freqs; c = np.cos(t); s = np.sin(t)
        r = qk.reshape(-1, nh, HD); o = r.copy()
        for h in range(nh):
            x = r[:, h]
            half = np.concatenate([-x[:, d//2:d], x[:, :d//2]], axis=1)
            o[:, h, :d] = x[:, :d] * np.concatenate([c,c]) + half * np.concatenate([s,s])
        return o.reshape(-1, nh*HD)
    if family == "gptneox":
        import math as _m
        erf = np.vectorize(_m.erf)
        def gelu_erf(x): return 0.5*x*(1+erf(x/1.41421356237))
        def ln(x, w, b, eps=1e-5):
            m = x.mean(-1, keepdims=True); v = ((x-m)**2).mean(-1, keepdims=True)
            return (x-m)/np.sqrt(v+eps)*w + b
        emb = np.array(f.get_slice("gpt_neox.embed_in.weight")[:].tolist(), dtype=np.float64)
        fn = np.array(f.get_slice("gpt_neox.final_layer_norm.weight")[:].tolist(), dtype=np.float64)
        fnb = np.array(f.get_slice("gpt_neox.final_layer_norm.bias")[:].tolist(), dtype=np.float64)
        lm = np.array(f.get_slice("embed_out.weight")[:].tolist(), dtype=np.float64)  # untied
        x = emb[ids]
        K = np.zeros((L, len(ids), NKV*HD)); V = np.zeros((L, len(ids), NKV*HD))
        for l in range(L):
            nw1 = np.array(f.get_slice(f"gpt_neox.layers.{l}.input_layernorm.weight")[:].tolist(), dtype=np.float64)
            nb1 = np.array(f.get_slice(f"gpt_neox.layers.{l}.input_layernorm.bias")[:].tolist(), dtype=np.float64)
            qkv = np.array(f.get_slice(f"gpt_neox.layers.{l}.attention.query_key_value.weight")[:].tolist(), dtype=np.float64)
            qkvb = np.array(f.get_slice(f"gpt_neox.layers.{l}.attention.query_key_value.bias")[:].tolist(), dtype=np.float64)
            wo = np.array(f.get_slice(f"gpt_neox.layers.{l}.attention.dense.weight")[:].tolist(), dtype=np.float64)
            wob = np.array(f.get_slice(f"gpt_neox.layers.{l}.attention.dense.bias")[:].tolist(), dtype=np.float64)
            w1 = np.array(f.get_slice(f"gpt_neox.layers.{l}.mlp.dense_h_to_4h.weight")[:].tolist(), dtype=np.float64)
            w1b = np.array(f.get_slice(f"gpt_neox.layers.{l}.mlp.dense_h_to_4h.bias")[:].tolist(), dtype=np.float64)
            w3 = np.array(f.get_slice(f"gpt_neox.layers.{l}.mlp.dense_4h_to_h.weight")[:].tolist(), dtype=np.float64)
            w3b = np.array(f.get_slice(f"gpt_neox.layers.{l}.mlp.dense_4h_to_h.bias")[:].tolist(), dtype=np.float64)
            xn = ln(x, nw1, nb1)
            qkv_out = xn @ qkv.T + qkvb
            q = np.stack([qkv_out[:, (h*3+0)*HD:(h*3+1)*HD] for h in range(NH)], axis=1).reshape(len(ids), NH*HD)
            k = np.stack([qkv_out[:, (h*3+1)*HD:(h*3+2)*HD] for h in range(NH)], axis=1).reshape(len(ids), NKV*HD)
            v = np.stack([qkv_out[:, (h*3+2)*HD:(h*3+3)*HD] for h in range(NH)], axis=1).reshape(len(ids), NKV*HD)
            qr = np.stack([rope(q[i:i+1], i, NH) for i in range(len(ids))]).reshape(len(ids), NH*HD)
            kr = np.stack([rope(k[i:i+1], i, NKV) for i in range(len(ids))]).reshape(len(ids), NKV*HD)
            K[l] = kr; V[l] = v
            att_out = np.zeros((len(ids), NH*HD))
            for t in range(len(ids)):
                qq = qr[t].reshape(NH, HD)
                for h in range(NH):
                    kvh = h//(NH//NKV)
                    kk = K[l, :t+1, kvh*HD:(kvh+1)*HD]; vv = V[l, :t+1, kvh*HD:(kvh+1)*HD]
                    sc = qq[h] @ kk.T / np.sqrt(HD)
                    sc = np.exp(sc - sc.max()); sc /= sc.sum()
                    att_out[t, h*HD:(h+1)*HD] = sc @ vv
            x = x + (att_out @ wo.T + wob)
            g = xn @ w1.T + w1b
            x = x + (gelu_erf(g) @ w3.T + w3b)  # parallel: same normed input
        xf = ln(x, fn, fnb)
        return xf @ lm.T
    emb = np.array(f.get_slice("model.tok_embeddings.weight" if family=="internlm2" else "model.embed_tokens.weight")[:].tolist(), dtype=np.float64)
    if family == "internlm2":
        fn = np.array(f.get_slice("model.norm.weight")[:].tolist(), dtype=np.float64)
        lm = np.array(f.get_slice("output.weight")[:].tolist(), dtype=np.float64)
        sres, sscale_emb, sscale_logits = 1.0, 1.0, 1.0
    else:
        fn = np.array(f.get_slice("model.norm.weight")[:].tolist(), dtype=np.float64)
        lm = np.array(f.get_slice("lm_head.weight")[:].tolist(), dtype=np.float64) if "lm_head.weight" in keys else emb  # MiniCPM ties embed
        sres = 1.4/np.sqrt(L); sscale_emb = 12.0; sscale_logits = 9.0
    x = emb[ids] * sscale_emb
    K = np.zeros((L, len(ids), NKV*HD)); V = np.zeros((L, len(ids), NKV*HD))
    for l in range(L):
        nw1 = np.array(f.get_slice(f"model.layers.{l}.{'attention_norm' if family=='internlm2' else 'input_layernorm'}.weight")[:].tolist(), dtype=np.float64)
        nw2 = np.array(f.get_slice(f"model.layers.{l}.{'ffn_norm' if family=='internlm2' else 'post_attention_layernorm'}.weight")[:].tolist(), dtype=np.float64)
        if family == "internlm2":
            wqkv = np.array(f.get_slice(f"model.layers.{l}.attention.wqkv.weight")[:].tolist(), dtype=np.float64)
            wo = np.array(f.get_slice(f"model.layers.{l}.attention.wo.weight")[:].tolist(), dtype=np.float64)
            w1 = np.array(f.get_slice(f"model.layers.{l}.feed_forward.w1.weight")[:].tolist(), dtype=np.float64)
            w2 = np.array(f.get_slice(f"model.layers.{l}.feed_forward.w2.weight")[:].tolist(), dtype=np.float64)
            w3 = np.array(f.get_slice(f"model.layers.{l}.feed_forward.w3.weight")[:].tolist(), dtype=np.float64)
            xn = rms(x, nw1)
            qkv = xn @ wqkv.T
            q = np.stack([qkv[:, (g*4+qh)*HD:(g*4+qh+1)*HD] for g in range(NKV) for qh in range(2)], axis=1).reshape(len(ids), NH*HD)
            k = np.stack([qkv[:, (g*4+2)*HD:(g*4+3)*HD] for g in range(NKV)], axis=1).reshape(len(ids), NKV*HD)
            v = np.stack([qkv[:, (g*4+3)*HD:(g*4+4)*HD] for g in range(NKV)], axis=1).reshape(len(ids), NKV*HD)
            qr = np.stack([rope(q[i:i+1], i, NH) for i in range(len(ids))]).reshape(len(ids), NH*HD)
            kr = np.stack([rope(k[i:i+1], i, NKV) for i in range(len(ids))]).reshape(len(ids), NKV*HD)
            K[l] = kr; V[l] = v
            att_out = np.zeros((len(ids), NH*HD))
            for t in range(len(ids)):
                qq = qr[t].reshape(NH, HD)
                for h in range(NH):
                    kvh = h//2
                    kk = K[l, :t+1, kvh*HD:(kvh+1)*HD]; vv = V[l, :t+1, kvh*HD:(kvh+1)*HD]
                    sc = qq[h] @ kk.T / np.sqrt(HD)
                    sc = np.exp(sc - sc.max()); sc /= sc.sum()
                    att_out[t, h*HD:(h+1)*HD] = sc @ vv
            x = x + (att_out @ wo.T) * sres
            xf = rms(x, nw2)
            x = x + (silu(xf @ w1.T) * (xf @ w3.T)) @ w2.T * sres
        else:
            wq = np.array(f.get_slice(f"model.layers.{l}.self_attn.q_proj.weight")[:].tolist(), dtype=np.float64)
            wk = np.array(f.get_slice(f"model.layers.{l}.self_attn.k_proj.weight")[:].tolist(), dtype=np.float64)
            wv = np.array(f.get_slice(f"model.layers.{l}.self_attn.v_proj.weight")[:].tolist(), dtype=np.float64)
            wo = np.array(f.get_slice(f"model.layers.{l}.self_attn.o_proj.weight")[:].tolist(), dtype=np.float64)
            w1 = np.array(f.get_slice(f"model.layers.{l}.mlp.gate_proj.weight")[:].tolist(), dtype=np.float64)
            w2 = np.array(f.get_slice(f"model.layers.{l}.mlp.up_proj.weight")[:].tolist(), dtype=np.float64)
            w3 = np.array(f.get_slice(f"model.layers.{l}.mlp.down_proj.weight")[:].tolist(), dtype=np.float64)
            xn = rms(x, nw1)
            q = xn @ wq.T; k = xn @ wk.T; v = xn @ wv.T
            qr = np.stack([rope(q[i:i+1], i, NH) for i in range(len(ids))]).reshape(len(ids), NH*HD)
            kr = np.stack([rope(k[i:i+1], i, NKV) for i in range(len(ids))]).reshape(len(ids), NKV*HD)
            K[l] = kr; V[l] = v
            att_out = np.zeros((len(ids), NH*HD))
            for t in range(len(ids)):
                qq = qr[t].reshape(NH, HD)
                for h in range(NH):
                    kvh = h//(NH//NKV)
                    kk = K[l, :t+1, kvh*HD:(kvh+1)*HD]; vv = V[l, :t+1, kvh*HD:(kvh+1)*HD]
                    sc = qq[h] @ kk.T / np.sqrt(HD)
                    sc = np.exp(sc - sc.max()); sc /= sc.sum()
                    att_out[t, h*HD:(h+1)*HD] = sc @ vv
            x = x + (att_out @ wo.T) * sres
            xf = rms(x, nw2)
            x = x + (silu(xf @ w1.T) * (xf @ w2.T)) @ w3.T * sres
    xf = rms(x, fn)
    return (xf @ lm.T) / sscale_logits

llm = Llama(model_path=f"{dir}/oracle-q8.gguf", n_ctx=512, n_gpu_layers=0, verbose=False)
ids = llm.tokenize("The capital of France is".encode(), add_bos=False)
ref = forward(ids)
top_ref = np.argsort(ref[-1])[::-1][:8]

# engine dump — regenerate ids for THIS model, run the engine, capture logits
import subprocess, tempfile
with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
    tf.write(" ".join(map(str, ids)))
    ids_path = tf.name
dump = sys.argv[3] if len(sys.argv) > 3 else "/tmp/onebit_ref_logits.txt"
os.environ["E2E_FULL_LOGITS"] = dump
subprocess.run(["/tmp/e2e_seq", dir, ids_path, "1"], capture_output=True, text=True, timeout=600)
os.remove(ids_path)
eng = {}
for line in open(dump):
    p = line.split()
    if len(p) == 2: eng[int(p[0])] = float(p[1])
top_eng = sorted(eng.items(), key=lambda kv: -kv[1])[:8]

ok = all(a == b for (a, _), b in zip(top_eng, top_ref))
print(f"{family}: engine-top8 {'MATCHES' if ok else 'DIFFERS'} numpy reference")
print("  ref  :", [int(x) for x in top_ref])
print("  engine:", [x for x, _ in top_eng])
sys.exit(0 if ok else 1)
