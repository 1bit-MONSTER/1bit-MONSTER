#!/usr/bin/env python3
"""gguf_to_onnx.py — export a dense Qwen3-style GGUF to a fp16 ONNX graph for
the VitisAI EP (Strix Halo NPU). Pinned format per issue #1468 probe:

  - fp16 MatMul with weights BAKED as initializers (runtime-weight inputs are
    rejected by the EP's custom op)
  - fp16 boundary tensors (fp32/bf16/int8 boundaries rejected at runtime)
  - static shapes only: KV cache is a fixed [1, NKV, MAX, HD] buffer per layer
    with a scalar `pos` input; attention masks positions > pos to -inf.
    (The EP compiles shape-locked overlays; dynamic dims are unsupported.)

Graph I/O (per layer i):
  in:  input_ids i64 [1,1], pos i64 [1], past_k{i} f16 [1,NKV,MAX,HD],
       past_v{i} f16 [1,NKV,MAX,HD]
  out: logits f32 [1,V], present_k{i}/present_v{i} f16 [1,NKV,MAX,HD]

Usage:
  gguf_to_onnx.py model.gguf out.onnx [--layers N] [--max-seq S] [--vocab N]
                  [--rope neox|norm] [--freq-base F]

--layers/--vocab truncate for bring-up (small first NPU compiles). Full model
is the default. Requires the `gguf` and `onnx` pip packages.
"""
import sys
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from gguf import GGUFReader, dequantize

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
I64 = TensorProto.INT64
NP = {F16: np.float16, F32: np.float32, I64: np.int64}

def to_f32(ten):
    """Dequantize a GGUF tensor to flat float32 (any dtype)."""
    if ten.tensor_type <= 1:  # F32 or F16
        dt = np.float32 if ten.tensor_type == 0 else np.float16
        return np.frombuffer(ten.data, dtype=dt).astype(np.float32)
    return dequantize(ten.data, ten.tensor_type).astype(np.float32).reshape(-1)

def gf(rd, field, default):
    v = rd.fields.get(field)
    if v is None or len(v.parts) < 4:
        return default
    val = v.parts[3]
    try:
        if hasattr(val, '__len__') and len(val) > 0:
            return int(val[0])
        return int(val)
    except Exception:
        return default

def gs(rd, field, default):
    v = rd.fields.get(field)
    if v is None or len(v.parts) < 5:
        return default
    try:
        return bytes(v.parts[4]).decode().strip()
    except Exception:
        return default

def init(name, arr, dtype):
    return numpy_helper.from_array(np.asarray(arr).astype(NP[dtype]), name=name)

class G:
    """Tiny ONNX graph builder."""
    def __init__(self):
        self.n, self.i, self.c = [], [], 0
    def uniq(self, p):
        self.c += 1
        return f"{p}_{self.c}"
    def add(self, op, ins, outs, **kw):
        self.n.append(helper.make_node(op, ins, outs, name=self.uniq("n"), **kw))
        return outs[0] if len(outs) == 1 else outs
    def rmsnorm(self, x, w, eps, dim):
        """RMSNorm in fp16 composed ops. The EP fuses both the fp32 composed
        form and the native RMSNormalization op into its rmsnorm1pass
        superkernel, which hangs the NPU (ERT_CMD_STATE_TIMEOUT); the fp16
        composition doesn't match the fusion pattern and stays on CPU."""
        x2 = self.add("Mul", [x, x], [self.uniq("x2")])
        m = self.add("ReduceMean", [x2, self.axm1], [self.uniq("m")], keepdims=1)
        v = self.add("Add", [m, eps], [self.uniq("v")])
        r = self.add("Pow", [v, self.half], [self.uniq("r")])
        n = self.add("Mul", [self.add("Mul", [x, r], [self.uniq("n0")]), w], [self.uniq("n")])
        return n
    def rope(self, x, cos_t, sin_t, pos, hdim, mode, reshape_to):
        """RoPE on x [.., hdim] (last dim). Tables [MAX, hdim/2] f16. pos i64 [1].
        reshape_to: explicit [.., hdim] target for the neox interleave (the EP's
        shape inferrer cannot resolve 0/-1 reshape data)."""
        even = self.add("Slice", [x, self.s0, self.big, self.last, self.st2], [self.uniq("e")])
        odd = self.add("Slice", [x, self.s1, self.big, self.last, self.st2], [self.uniq("o")])
        c = self.add("Unsqueeze", [self.add("Gather", [cos_t, pos], [self.uniq("c")]), self.ax1], [self.uniq("cu")])
        s = self.add("Unsqueeze", [self.add("Gather", [sin_t, pos], [self.uniq("s")]), self.ax1], [self.uniq("su")])
        e2 = self.add("Sub", [self.add("Mul", [even, c], [self.uniq("ec")]),
                              self.add("Mul", [odd, s], [self.uniq("os")])], [self.uniq("e2")])
        o2 = self.add("Add", [self.add("Mul", [even, s], [self.uniq("es")]),
                              self.add("Mul", [odd, c], [self.uniq("oc")])], [self.uniq("o2")])
        if mode == "neox":
            e2u = self.add("Unsqueeze", [e2, self.last], [self.uniq("e2u")])  # [.., half, 1]
            o2u = self.add("Unsqueeze", [o2, self.last], [self.uniq("o2u")])
            st = self.add("Concat", [e2u, o2u], [self.uniq("st")], axis=-1)   # [.., half, 2]
            rn = self.uniq("rshape")
            self.i.append(init(rn, np.array(reshape_to, np.int64), I64))
            return self.add("Reshape", [st, rn], [self.uniq("rot")])
        return self.add("Concat", [e2, o2], [self.uniq("rot")], axis=-1)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--max-seq", type=int, default=256)
    ap.add_argument("--vocab", type=int, default=0)
    ap.add_argument("--rope", choices=["neox", "norm"], default="neox")
    ap.add_argument("--freq-base", type=float, default=0)
    ap.add_argument("--no-rope", action="store_true", help="isolation: skip RoPE")
    ap.add_argument("--no-rmsnorm", action="store_true", help="isolation: skip RMSNorm")
    ap.add_argument("--no-attn", action="store_true", help="isolation: skip attention math")
    ap.add_argument("--no-qknorm", action="store_true", help="isolation: skip QK norms")
    args = ap.parse_args()

    print(f"Reading {args.model}...")
    rd = GGUFReader(args.model)
    arch = gs(rd, "general.architecture", "qwen3")
    P = arch + "."
    by_name = {t.name: t for t in rd.tensors}
    n_layer = gf(rd, P + "block_count", 0)
    H = gf(rd, P + "embedding_length", 0)
    NH = gf(rd, P + "attention.head_count", 0)
    NKV = gf(rd, P + "attention.head_count_kv", 0)
    HD = gf(rd, P + "attention.key_length", 0) or (H // NH)
    FF = gf(rd, P + "feed_forward_length", 0)
    V = gf(rd, P + "vocab_size", 0)
    if V <= 0 and "token_embd.weight" in by_name:
        V = int(by_name["token_embd.weight"].shape[1])  # [H, V] layout
    eps = gf(rd, P + "attention.layer_norm_rms_epsilon", 1e-6)
    if not (1e-9 < eps < 1.0):
        eps = 1e-6
    freq_base = args.freq_base or gf(rd, P + "rope.freq_base", 10000)
    rot_dim = gf(rd, P + "rope.dimension_count", HD)
    if rot_dim <= 0 or rot_dim > HD:
        rot_dim = HD
    if args.layers:
        n_layer = min(n_layer, args.layers)
    if args.vocab:
        V = min(V, args.vocab)
    has_qnorm = "blk.0.attn_q_norm.weight" in by_name
    print(f"  arch={arch} layers={n_layer} H={H} NH={NH} NKV={NKV} HD={HD} FF={FF} "
          f"V={V} eps={eps} freq_base={freq_base} rot_dim={rot_dim} max_seq={args.max_seq}")

    g = G()
    n, inits = g.n, g.i
    def i(name, arr, dt):
        inits.append(init(name, arr, dt))

    # ── shared constants ──
    g.half = "c_half"; i("c_half", np.array([-0.5], np.float16), F16)
    g.s0 = "c_s0";   i("c_s0", np.array([0], np.int64), I64)
    g.s1 = "c_s1";   i("c_s1", np.array([1], np.int64), I64)
    g.big = "c_big"; i("c_big", np.array([np.iinfo(np.int32).max], np.int64), I64)
    g.last = "c_last"; i("c_last", np.array([-1], np.int64), I64)
    g.st2 = "c_st2"; i("c_st2", np.array([2], np.int64), I64)
    g.ax1 = "c_ax1"; i("c_ax1", np.array([1], np.int64), I64)
    g.axm1 = "c_axm1"; i("c_axm1", np.array([-1], np.int64), I64)
    g.ax2 = "c_ax2"; i("c_ax2", np.array([2], np.int64), I64)
    g.one = "c_one"; i("c_one", np.array([1], np.int64), I64)
    g.zero = "c_zero"; i("c_zero", np.array([0], np.int64), I64)

    i("c_scale", np.array([1.0 / np.sqrt(HD)], np.float32), F32)
    i("c_negbig", np.array([-1e30], np.float32), F32)
    i("c_eps", np.array([eps], np.float16), F16)
    i("c_rs1", np.array([1, NH * HD], np.int64), I64)       # attn out [1, NH*HD]
    i("c_rstack_q", np.array([1, NH, 1, HD], np.int64), I64)
    i("c_rstack_k", np.array([1, NKV, 1, HD], np.int64), I64)
    i("c_rsq", np.array([1, NH, 1, HD], np.int64), I64)     # q [1,NH,1,HD]
    i("c_rsk", np.array([1, NKV, 1, HD], np.int64), I64)    # k/v [1,NKV,1,HD]
    i("c_rq5", np.array([1, NH // NKV, NKV, 1, HD], np.int64), I64)
    i("c_rm1", np.array([1, 1, 1, args.max_seq], np.int64), I64)
    i("c_rlg", np.array([1, V], np.int64), I64)
    i("c_range", np.arange(args.max_seq, dtype=np.int64), I64)
    half = rot_dim // 2
    freqs = np.array([1.0 / (freq_base ** (2.0 * j / rot_dim)) for j in range(half)], np.float32)
    ang = np.arange(args.max_seq, dtype=np.float32).reshape(-1, 1) * freqs.reshape(1, -1)
    i("cos_table", np.cos(ang).astype(np.float16), F16)
    i("sin_table", np.sin(ang).astype(np.float16), F16)

    # ── I/O ──
    inputs = [helper.make_tensor_value_info("input_ids", I64, [1, 1]),
              helper.make_tensor_value_info("pos", I64, [1])]
    outputs = [helper.make_tensor_value_info("logits", F32, [1, V])]
    for k in range(n_layer):
        inputs += [helper.make_tensor_value_info(f"past_k{k}", F16, [1, NKV, args.max_seq, HD]),
                   helper.make_tensor_value_info(f"past_v{k}", F16, [1, NKV, args.max_seq, HD])]
        outputs += [helper.make_tensor_value_info(f"present_k{k}", F16, [1, NKV, args.max_seq, HD]),
                    helper.make_tensor_value_info(f"present_v{k}", F16, [1, NKV, args.max_seq, HD])]

    def w16(name, out_n, vocab_cut=0):
        """GGUF [in,out] weight -> fp16 [in,out] initializer (qwen3 layout).
        vocab_cut: keep only the first V output columns (--vocab truncation)."""
        ten = by_name[name]
        arr = to_f32(ten).reshape(ten.shape)
        if vocab_cut:
            arr = arr[:, :vocab_cut]
        inits.append(init(out_n, arr.astype(np.float16), F16))

    def w16t(name, out_n, vocab_cut=0):
        """GGUF [H,V] -> fp16 [V,H] (transposed) initializer."""
        ten = by_name[name]
        arr = to_f32(ten).reshape(ten.shape)
        if vocab_cut:
            arr = arr[:, :vocab_cut]
        inits.append(init(out_n, arr.T.astype(np.float16), F16))

    def w16cat(names, out_n, axis=0):
        arrs = [to_f32(by_name[nm]).reshape(by_name[nm].shape) for nm in names]
        inits.append(init(out_n, np.concatenate(arrs, axis=axis).astype(np.float16), F16))

    def w32(name, out_n):
        ten = by_name[name]
        inits.append(init(out_n, to_f32(ten).reshape(ten.shape), F16))

    w16t("token_embd.weight", "emb", V)  # transposed: Gather needs rows=vocab
    if int(by_name["token_embd.weight"].shape[0]) != H:
        sys.exit(f"token_embd [{list(by_name['token_embd.weight'].shape)}] is not [H={H}, V] "
                 f"— this exporter assumes the qwen3 [in,out] GGUF layout")

    # causal mask: -1e30 where j > pos
    pos_ge = g.add("Less", ["pos", "c_range"], [g.uniq("m")])          # [MAX] pos < j
    mask = g.add("Mul", [g.add("Cast", [pos_ge], [g.uniq("mf")], to=F32), "c_negbig"], [g.uniq("mv")])
    mask = g.add("Reshape", [mask, "c_rm1"], [g.uniq("mr")])

    x = g.add("Gather", ["emb", "input_ids"], [g.uniq("xh")])          # [1,1,H] f16

    for k in range(n_layer):
        L = f"blk.{k}."
        w32(L + "attn_norm.weight", f"an{k}")
        if f"blk.0.attn_qkv.weight" in by_name:
            w16(L + "attn_qkv.weight", f"qkv{k}")
        else:
            w16cat([L + "attn_q.weight", L + "attn_k.weight", L + "attn_v.weight"], f"qkv{k}", axis=1)
        w16(L + "attn_output.weight", f"wo{k}")
        w16(L + "ffn_gate.weight", f"wg{k}")
        w16(L + "ffn_up.weight", f"wu{k}")
        w16(L + "ffn_down.weight", f"wd{k}")
        w32(L + "ffn_norm.weight", f"fn{k}")
        if has_qnorm:
            w32(L + "attn_q_norm.weight", f"qn{k}")
            w32(L + "attn_k_norm.weight", f"kn{k}")

        # attention
        if args.no_rmsnorm:
            xn = x
        else:
            xn = g.rmsnorm(x, f"an{k}", "c_eps", H)
        qkv = g.add("MatMul", [xn, f"qkv{k}"], [g.uniq("qkv")])        # [1,1,3H]
        i(f"c_split{k}", np.array([NH * HD, NKV * HD, NKV * HD], np.int64), I64)
        qt, kt, vt = g.add("Split", [qkv, f"c_split{k}"], [g.uniq("q"), g.uniq("k"), g.uniq("v")], axis=-1)
        q = g.add("Reshape", [qt, "c_rsq"], [g.uniq("q4")])            # [1,NH,1,HD]
        kx = g.add("Reshape", [kt, "c_rsk"], [g.uniq("k4")])
        vx = g.add("Reshape", [vt, "c_rsk"], [g.uniq("v4")])
        if not args.no_rmsnorm and not args.no_qknorm:
            q = g.rmsnorm(q, f"qn{k}", "c_eps", HD)
            kx = g.rmsnorm(kx, f"kn{k}", "c_eps", HD)
        if args.no_rope:
            q = g.add("Identity", [q], [g.uniq("qid")])
            kx = g.add("Identity", [kx], [g.uniq("kid")])
        else:
            q = g.rope(q, "cos_table", "sin_table", "pos", HD, args.rope, [1, NH, 1, HD])
            kx = g.rope(kx, "cos_table", "sin_table", "pos", HD, args.rope, [1, NKV, 1, HD])

        # kv buffer update: concat(past[:,:,:pos], new, past[:,:,pos+1:])
        p1 = g.add("Add", ["pos", "c_one"], [g.uniq("p1")])
        pk = g.add("Concat", [
            g.add("Slice", [f"past_k{k}", "c_zero", "pos", "c_ax2"], [g.uniq("pkl")]),
            kx,
            g.add("Slice", [f"past_k{k}", p1, "c_big", "c_ax2"], [g.uniq("pkr")])],
            [g.uniq("pk")], axis=2)
        pv = g.add("Concat", [
            g.add("Slice", [f"past_v{k}", "c_zero", "pos", "c_ax2"], [g.uniq("pvl")]),
            vx,
            g.add("Slice", [f"past_v{k}", p1, "c_big", "c_ax2"], [g.uniq("pvr")])],
            [g.uniq("pv")], axis=2)

        if args.no_attn:
            qf = g.add("Reshape", [q, "c_rs1"], [g.uniq("qf")])        # [1,NH*HD]
            o = g.add("MatMul", [qf, f"wo{k}"], [g.uniq("o")])         # [1,H] (garbage math)
            x = g.add("Add", [x, o], [g.uniq("x1")])
            xn = x
            gg = g.add("MatMul", [xn, f"wg{k}"], [g.uniq("gg")])
            uu = g.add("MatMul", [xn, f"wu{k}"], [g.uniq("uu")])
            gu = g.add("Mul", [g.add("Sigmoid", [gg], [g.uniq("sg")]), uu], [g.uniq("gu")])
            d = g.add("MatMul", [gu, f"wd{k}"], [g.uniq("d")])
            x = g.add("Add", [x, d], [g.uniq("x2")])
            continue
        # GQA attention
        q5 = g.add("Reshape", [q, "c_rq5"], [g.uniq("q5")])            # [1,NH//NKV,NKV,1,HD]
        k5 = g.add("Unsqueeze", [pk, "c_ax1"], [g.uniq("k5")])         # [1,1,NKV,MAX,HD]
        v5 = g.add("Unsqueeze", [pv, "c_ax1"], [g.uniq("v5")])
        kt5 = g.add("Transpose", [k5], [g.uniq("kt5")], perm=[0, 1, 2, 4, 3])
        sc = g.add("MatMul", [q5, kt5], [g.uniq("sc")])                # [1,NH//NKV,NKV,1,MAX]
        sc = g.add("Cast", [sc], [g.uniq("scf")], to=F32)  # keep grouped [1,NH//NKV,NKV,1,MAX]
        sc = g.add("Mul", [sc, "c_scale"], [g.uniq("scs")])
        sc = g.add("Add", [sc, mask], [g.uniq("scm")])
        pr = g.add("Softmax", [sc], [g.uniq("pr")], axis=-1)
        pr = g.add("Cast", [pr], [g.uniq("prf")], to=F16)
        at = g.add("MatMul", [pr, v5], [g.uniq("at")])                 # [1,NH//NKV,NKV,1,HD]
        at = g.add("Reshape", [at, "c_rs1"], [g.uniq("at2")])          # [1,H]
        o = g.add("MatMul", [at, f"wo{k}"], [g.uniq("o")])             # [1,H]
        x = g.add("Add", [x, o], [g.uniq("x1")])                       # [1,1,H]+[1,H] -> [1,1,H]

        # MLP
        xn = g.rmsnorm(x, f"fn{k}", "c_eps", H)
        gg = g.add("MatMul", [xn, f"wg{k}"], [g.uniq("gg")])
        uu = g.add("MatMul", [xn, f"wu{k}"], [g.uniq("uu")])
        gu = g.add("Mul", [g.add("Sigmoid", [gg], [g.uniq("sg")]), uu], [g.uniq("gu")])
        d = g.add("MatMul", [gu, f"wd{k}"], [g.uniq("d")])
        x = g.add("Add", [x, d], [g.uniq("x2")])

        # outputs: present buffers
        g.add("Identity", [pk], [f"present_k{k}"])
        g.add("Identity", [pv], [f"present_v{k}"])

    # final norm + lm_head
    w32("output_norm.weight", "fnorm")
    w16("output.weight", "outw", V)
    xf = g.rmsnorm(x, "fnorm", "c_eps", H)
    lg = g.add("MatMul", [xf, "outw"], [g.uniq("lg")])                 # [1,1,V]
    lg = g.add("Reshape", [lg, "c_rlg"], [g.uniq("lg2")])
    lg = g.add("Cast", [lg], ["logits"], to=F32)

    graph = helper.make_graph(n, "qwen3", inputs, outputs, initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.shape_inference.infer_shapes(model, strict_mode=False, check_type=True)
    onnx.save(model, args.out)
    # dims sidecar for the backend (this ORT build returns empty input shapes)
    import json, os
    side = os.path.splitext(args.out)[0] + ".dims.json"
    with open(side, "w") as f:
        json.dump({"hidden": H, "n_layers": n_layer, "n_kv_heads": NKV, "head_dim": HD,
                   "max_seq": args.max_seq, "vocab": V}, f, indent=1)
    print(f"wrote {args.out}: {len(n)} nodes, {len(inits)} initializers, "
          f"{n_layer} layers, max_seq={args.max_seq} (dims: {side})")


if __name__ == "__main__":
    main()
