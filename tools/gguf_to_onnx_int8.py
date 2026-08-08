#!/usr/bin/env python3
"""GGUF → INT8 QDQ ONNX converter (microslop-style).

Reads a Q8_0 GGUF, dequantizes, re-quantizes linear weights to per-channel
INT8 with zero point 0 (the ORT QDQ convention the loader expects), keeps
embeddings/norms as-is, and writes an ONNX with DequantizeLinear nodes.

Usage: gguf_to_onnx_int8.py model.gguf outdir/
"""
import sys, os
import numpy as np
import gguf
import onnx
from onnx import helper, TensorProto, numpy_helper

GGUF_QTYPES = {1: 'F32', 0: 'F16', 2: 'Q8_0', 3: 'Q4_0', 7: 'Q4_K', 8: 'Q5_K', 9: 'Q6_K'}

def dequant_tensor(t):
    """Dequantize a gguf tensor to float32 (supports F32/F16/Q8_0 — our sources)."""
    tt = int(t.tensor_type)
    data = np.asarray(t.data)
    shape = tuple(int(x) for x in t.shape[::-1])  # gguf shape is reversed
    n = int(np.prod(shape))
    Q = gguf.GGMLQuantizationType
    if tt == int(Q.F16):
        return data.astype(np.float32).reshape(shape)
    if tt == int(Q.F32):
        return data.reshape(shape)
    if tt == int(Q.Q8_0):  # block 32, fp16 scale + 32 int8
        raw = np.asarray(t.data).ravel().view(np.uint8)
        block = 34
        nb = n // 32
        out = np.empty(n, dtype=np.float32)
        for b in range(nb):
            s = np.frombuffer(raw[b*block:b*block+2].tobytes(), dtype=np.float16)[0]
            q = raw[b*block+2:(b+1)*block].astype(np.int8).astype(np.float32)  # SIGNED!
            out[b*32:(b+1)*32] = q * float(s)
        return out.reshape(shape)
    raise SystemExit(f"unsupported gguf qtype {tt}")

def int8_quant(w):
    """Per-row (output-channel) INT8 quant, zp=0, axis=0 convention."""
    R, C = w.shape
    scale = np.abs(w).max(axis=1, keepdims=True) / 127.0
    scale = np.maximum(scale, 1e-10)
    wq = np.clip(np.round(w / scale), -128, 127).astype(np.int8)
    return wq, scale.astype(np.float32)

def main():
    gguf_path, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    r = gguf.GGUFReader(gguf_path)
    tensors = {t.name: t for t in r.tensors}
    kv = r.fields
    arch = bytes(kv["general.architecture"].parts[kv["general.architecture"].data[0]]).decode()
    def gk(name, default=None):
        f = kv.get(f"{arch}.{name}") or kv.get(name)
        if f is None: return default
        return f.parts[f.data[0]].tolist()[0] if len(f.data) == 1 else [p.tolist()[0] for p in f.parts]

    n_layer = int(gk("block_count"))
    hs = int(gk("embedding_length"))
    is_ = int(gk("feed_forward_length"))
    nh = int(gk("attention.head_count"))
    nkv = int(gk("attention.head_count_kv"))
    rope = float(gk("rope.freq_base", 10000.0))
    eps = float(gk("attention.layer_norm_rms_epsilon", 1e-5))

    def get(name):
        return dequant_tensor(tensors[name]) if name in tensors else None

    # Embedding: keep F16 (matches real exports; exercises the loader F16 path)
    emb = get("token_embd.weight")
    if emb is None: raise SystemExit("no token_embd.weight")
    vocab = int(gk("vocab_size") or emb.shape[0])
    print(f"[conv] {arch}: {n_layer} layers hs={hs} is={is_} heads={nh} nkv={nkv} rope={rope} vocab={vocab}")

    inits, nodes = [], []
    def add_tensor(name, arr, dtype=None):
        t = numpy_helper.from_array(arr.astype(dtype) if dtype else arr, name)
        inits.append(t)

    add_tensor("model.embed_tokens.weight", emb.astype(np.float16))
    add_tensor("model.norm.weight", get("output_norm.weight").astype(np.float32))

    # Untied LM head (llama-family): keep FP16 — full precision on the final
    # projection, routed via out_model->lm_head_dev (existing fp16_gemv path).
    lm = get("output.weight")
    if lm is not None:
        add_tensor("lm_head.weight", lm.astype(np.float16))
        print(f"[conv] lm_head: {lm.shape} (fp16, untied)")

    linear_map = {
        "attn_q.weight":  "self_attn.q_proj.weight",
        "attn_k.weight":  "self_attn.k_proj.weight",
        "attn_v.weight":  "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight":  "mlp.up_proj.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
    }
    for l in range(n_layer):
        add_tensor(f"model.layers.{l}.input_layernorm.weight",
                   get(f"blk.{l}.attn_norm.weight").astype(np.float32))
        add_tensor(f"model.layers.{l}.post_attention_layernorm.weight",
                   get(f"blk.{l}.ffn_norm.weight").astype(np.float32))
        # Qwen3 q/k RMSNorms (per-head, before RoPE)
        qn = get(f"blk.{l}.attn_q_norm.weight")
        kn = get(f"blk.{l}.attn_k_norm.weight")
        if qn is not None:
            add_tensor(f"model.layers.{l}.self_attn.q_norm.weight", qn.astype(np.float32))
        if kn is not None:
            add_tensor(f"model.layers.{l}.self_attn.k_norm.weight", kn.astype(np.float32))
        for gguf_name, onnx_name in linear_map.items():
            w = get(f"blk.{l}.{gguf_name}")
            if w is None: raise SystemExit(f"missing {gguf_name} layer {l}")
            # gguf stores [N, K] row-major already (output-first) — ONNX wants
            # the same [out, in] layout. No transpose.
            wq, sc = int8_quant(w)
            full = f"model.layers.{l}.{onnx_name}"
            add_tensor(full, wq)
            add_tensor(full + "_scale", sc.reshape(-1))
            nodes.append(helper.make_node("DequantizeLinear", [full, full + "_scale"],
                                          [full + "_dq"], name=f"dq_l{l}_{onnx_name}"))

    graph = helper.make_graph(nodes, "g", [], [], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, os.path.join(outdir, "model_int8.onnx"))
    print(f"[conv] wrote {os.path.join(outdir, 'model_int8.onnx')}: {len(inits)} initializers, {len(nodes)} DQ nodes")

    # head_dim: q_proj rows / num_heads (qwen3 small models use 128, not hs/nh)
    hd = int(tensors["blk.0.attn_q.weight"].shape[-1]) // nh if nh else 0
    cfg = {"model_type": arch, "hidden_size": hs, "intermediate_size": is_,
           "num_attention_heads": nh, "num_key_value_heads": nkv, "head_dim": hd,
           "max_position_embeddings": int(gk("context_length", 4096)),
           "rms_norm_eps": eps, "rope_theta": rope, "vocab_size": vocab,
           "tie_word_embeddings": 1}
    import json
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print("[conv] wrote config.json")

if __name__ == "__main__":
    main()
