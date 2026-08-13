# tools/safetensors_to_onnx_int8.mojo — Mojo 1.0 twin of
# safetensors_to_onnx_int8.py (fold P2.2). HF safetensors (bf16) -> INT8
# QDQ ONNX for bitnet_decode. Byte-identical output, same CLI:
#   safetensors_to_onnx_int8 <source.safetensors|dir> <outdir>
# Build:
#   mojo build tools/safetensors_to_onnx_int8.mojo -o build/safetensors_to_onnx_int8
#
# Hand-rolled protobuf writer (the onnx package is a Python dependency the
# fold removes). Model: ir_version=13, graph "g" with DQ nodes + int8
# initializers, opset ("", 13). External-data path (>2GB estimate) mirrors
# onnx.save's convert_model_to_external_data: tensors with raw data size
# >= 1024 B (incl. ~33 B python-object overhead, like sys.getsizeof) go
# external with location/offset/length entries, the rest stay inline; the
# .data file holds them sequentially in initializer order. The .py's
# HF_REPO config fetch is not ported — a local config.json is required.

from jsonx import (
    STensor,
    cfg_float,
    cfg_int,
    elem_f32,
    parse_cfg_object,
    parse_safetensors,
    parse_string,
    read_file_bytes,
    skip_value,
    skip_ws,
    str_to_int,
)
from std.collections import Dict, List
from std.math import round
from std.memory import bitcast, alloc, Layout
from std.os import listdir, makedirs
from std.os.path import exists, isdir
from std.sys import argv

comptime TENSOR_INT8: Int = 3

comptime BIG_THRESHOLD: Int = 1900000000  # 2 GB protobuf cap, like the .py


def comptime_big_threshold() -> Int:
    return BIG_THRESHOLD
comptime TENSOR_FLOAT: Int = 1
comptime TENSOR_FLOAT16: Int = 10


# ── protobuf wire writer ────────────────────────────────────────────────

def pb_varint(mut buf: List[UInt8], v: Int):
    var u = UInt64(v)
    while u >= 128:
        buf.append(UInt8((u & 0x7F) | 0x80))
        u >>= 7
    buf.append(UInt8(u))


def pb_tag(mut buf: List[UInt8], field: Int, wire: Int):
    pb_varint(buf, (field << 3) | wire)


def pb_varint_field(mut buf: List[UInt8], field: Int, v: Int):
    pb_tag(buf, field, 0)
    pb_varint(buf, v)


def pb_str(mut buf: List[UInt8], field: Int, s: String):
    pb_tag(buf, field, 2)
    pb_varint(buf, s.byte_length())
    var sb = s.as_bytes()
    for i in range(sb.__len__()):
        buf.append(sb[i])


def pb_bytes(mut buf: List[UInt8], field: Int, src: List[UInt8]):
    pb_tag(buf, field, 2)
    pb_varint(buf, src.__len__())
    for i in range(src.__len__()):
        buf.append(src[i])


def pb_len(mut buf: List[UInt8], field: Int, payload: List[UInt8]):
    pb_tag(buf, field, 2)
    pb_varint(buf, payload.__len__())
    for i in range(payload.__len__()):
        buf.append(payload[i])


# ── INT8 QDQ building blocks ────────────────────────────────────────────

struct Init:
    var name: String
    var dims: List[Int]
    var data_type: Int
    var raw: List[UInt8]  # raw_data bytes
    var external: Bool  # external-data path (big models)

    def __init__(
        out self,
        name: String,
        var dims: List[Int],
        data_type: Int,
        var raw: List[UInt8],
        external: Bool,
    ):
        self.name = name
        self.dims = dims^
        self.data_type = data_type
        self.raw = raw^
        self.external = external


struct Node:
    var inputs: List[String]
    var outputs: List[String]
    var name: String
    var op_type: String

    def __init__(
        out self, var inputs: List[String], var outputs: List[String], name: String, op_type: String
    ):
        self.inputs = inputs^
        self.outputs = outputs^
        self.name = name
        self.op_type = op_type


def f32_bytes(mut out: List[UInt8], v: Float32):
    var u = bitcast[DType.uint32, 1](v)
    out.append(UInt8(u & 0xFF))
    out.append(UInt8((u >> 8) & 0xFF))
    out.append(UInt8((u >> 16) & 0xFF))
    out.append(UInt8((u >> 24) & 0xFF))


def f16_bytes(mut out: List[UInt8], v: Float32):
    var u = bitcast[DType.uint16, 1](Float16(v))
    out.append(UInt8(u & 0xFF))
    out.append(UInt8((u >> 8) & 0xFF))


def make_init(name: String, var dims: List[Int], data_type: Int, var raw: List[UInt8], big: Bool) -> Init:
    # external when the raw data (with ~33 B python-object overhead, like
    # sys.getsizeof) is >= the 1024 B threshold onnx uses
    var external = big and raw.__len__() + 33 >= 1024
    return Init(name, dims^, data_type, raw^, external)


def elem_size(dtype: Int) -> Int:
    if dtype == 1 or dtype == 7 or dtype == 8:
        return 4
    if dtype == 9 or dtype == 10:
        return 8
    if dtype == 3 or dtype == 4:
        return 1
    return 2


def int8_quant(
    b: Span[Byte, _],
    off: Int,
    dtype: Int,
    rows: Int,
    cols: Int,
    mut wq: List[UInt8],
    mut scales: List[UInt8],
):
    # per-row scale = max|w|/127 clamped to 1e-10 (numpy-exact:
    # np.abs(w).max(axis=1)/127.0, np.maximum(scale, 1e-10));
    # wq = np.clip(np.round(w/scale), -128, 127) — ties-to-even.
    var esize = elem_size(dtype)
    for r in range(rows):
        var mx: Float32 = 0.0
        var row_off = off + r * cols * esize
        for c in range(cols):
            var x = elem_f32(b, row_off, dtype, c)
            var ax = x if x >= Float32(0.0) else -x
            if ax > mx:
                mx = ax
        var scale = mx / Float32(127.0)
        if scale < Float32(1e-10):
            scale = Float32(1e-10)
        f32_bytes(scales, scale)
        for c in range(cols):
            var x = elem_f32(b, row_off, dtype, c)
            var t = round(x / scale)
            if t < Float32(-128.0):
                t = Float32(-128.0)
            elif t > Float32(127.0):
                t = Float32(127.0)
            wq.append(UInt8(Int(t) & 0xFF))


def raw_f32(t: STensor, shards: List[Tuple[Pointer[UInt8, MutUntrackedOrigin], Int]]) -> List[UInt8]:
    var sb = shard_span(t, shards)
    var out = List[UInt8]()
    var n = 1
    for d in t.dims:
        n *= d
    for i in range(n):
        f32_bytes(out, elem_f32(sb, t.off, t.dtype, i))
    return out^


def raw_f16(t: STensor, shards: List[Tuple[Pointer[UInt8, MutUntrackedOrigin], Int]]) -> List[UInt8]:
    var sb = shard_span(t, shards)
    var out = List[UInt8]()
    var n = 1
    for d in t.dims:
        n *= d
    for i in range(n):
        f16_bytes(out, elem_f32(sb, t.off, t.dtype, i))
    return out^


def dims_of(t: STensor) -> List[Int]:
    var out = List[Int]()
    for i in range(t.dims.__len__()):
        out.append(t.dims[i])
    return out^


# ── serialization ───────────────────────────────────────────────────────

def serialize_model(
    inits: List[Init],
    nodes: List[Node],
    offs: List[Int],
) -> List[UInt8]:
    # ModelProto: ir_version=13 (field 1), graph (field 7), opset (field 8).
    # GraphProto: node=1, name=2, initializer=5. NodeProto: input=1,
    # output=2, name=3, op_type=4. TensorProto: dims=1, data_type=2,
    # name=8, raw_data=9, external_data=13, data_location=14.
    var g = List[UInt8]()
    for i in range(nodes.__len__()):
        var nb = List[UInt8]()
        for k in range(nodes[i].inputs.__len__()):
            pb_str(nb, 1, nodes[i].inputs[k])
        for k in range(nodes[i].outputs.__len__()):
            pb_str(nb, 2, nodes[i].outputs[k])
        pb_str(nb, 3, nodes[i].name)
        pb_str(nb, 4, nodes[i].op_type)
        pb_len(g, 1, nb)
    pb_str(g, 2, "g")
    for i in range(inits.__len__()):
        var tb = List[UInt8]()
        for d in inits[i].dims:
            pb_varint_field(tb, 1, d)
        pb_varint_field(tb, 2, inits[i].data_type)
        pb_str(tb, 8, inits[i].name)
        if inits[i].external:
            var e = List[UInt8]()
            pb_str(e, 1, "location")
            pb_str(e, 2, "model_int8.onnx.data")
            pb_len(tb, 13, e)
            e.clear()
            pb_str(e, 1, "offset")
            pb_str(e, 2, String(offs[i]))
            pb_len(tb, 13, e)
            e.clear()
            pb_str(e, 1, "length")
            pb_str(e, 2, String(inits[i].raw.__len__()))
            pb_len(tb, 13, e)
            pb_varint_field(tb, 14, 1)  # data_location = EXTERNAL
        else:
            pb_bytes(tb, 9, inits[i].raw)
        pb_len(g, 5, tb)
    var m = List[UInt8]()
    pb_varint_field(m, 1, 13)
    pb_len(m, 7, g)
    var ops = List[UInt8]()
    pb_str(ops, 1, "")
    pb_varint_field(ops, 2, 13)
    pb_len(m, 8, ops)
    return m^


def write_file_bytes(path: String, var data: List[UInt8]) raises:
    var n = data.__len__()
    var alloc_obj = data.unsafe_take_allocation()
    var ptr = alloc_obj^.unsafe_leak()
    var span = Span[UInt8, MutUntrackedOrigin](unsafe_ptr=ptr, length=n)
    var f = open(path, "w")
    f.write_bytes(span)
    f.close()
    ptr.unsafe_free()


def shard_span(
    t: STensor, shards: List[Tuple[Pointer[UInt8, MutUntrackedOrigin], Int]]
) -> Span[UInt8, MutUntrackedOrigin]:
    return Span[UInt8, MutUntrackedOrigin](
        unsafe_ptr=shards[t.shard][0], length=shards[t.shard][1]
    )


# ── config.json output (json.dump(indent=2) byte-parity) ────────────────

def jstr(s: String) -> String:
    var out = String()
    out += '"'
    var sb = s.as_bytes()
    for i in range(sb.__len__()):
        out += chr(Int(sb[i]))
    out += '"'
    return out


def cfg_str_v(cfg: Dict[String, String], key: String, default: String) raises -> String:
    var v = cfg.get(key)
    if not v:
        return default
    var raw = v.value()
    if raw.startswith('"'):
        var rb = raw.as_bytes()
        var q = 0
        return parse_string(rb, q)
    return raw


def write_config(
    cfg: Dict[String, String],
    outdir: String,
    hs: Int,
    is_: Int,
    nh: Int,
    nkv: Int,
    max_seq: Int,
    eps: Float64,
    rope: Float64,
    vs: Int,
    tie: Int,
) raises:
    var hd = cfg_int(cfg, "head_dim", hs // nh)
    var body = String()
    body += '{\n'
    body += '  "model_type": ' + jstr(cfg_str_v(cfg, "model_type", "qwen2")) + ',\n'
    body += '  "hidden_size": ' + String(hs) + ',\n'
    body += '  "intermediate_size": ' + String(is_) + ',\n'
    body += '  "num_attention_heads": ' + String(nh) + ',\n'
    body += '  "num_key_value_heads": ' + String(nkv) + ',\n'
    body += '  "head_dim": ' + String(hd) + ',\n'
    body += '  "max_position_embeddings": ' + String(max_seq) + ',\n'
    body += '  "rms_norm_eps": ' + String(eps) + ',\n'
    body += '  "rope_theta": ' + String(rope) + ',\n'
    body += '  "vocab_size": ' + String(vs) + ',\n'
    body += '  "tie_word_embeddings": ' + String(tie) + '\n'
    body += '}'
    var f = open(outdir + "/config.json", "w")
    f.write_string(body)
    f.close()


def get_tensor(by_name: Dict[String, STensor], name: String) raises -> STensor:
    var t = by_name.get(name)
    if not t:
        raise Error("tensor " + name + " not found")
    return t.value().copy()


# ── main ────────────────────────────────────────────────────────────────

def main() raises:
    var args = argv()
    if len(args) != 3:
        print("usage: safetensors_to_onnx_int8 <source.safetensors|dir> <outdir>")
        return
    var st_path = String(args[1])
    var outdir = String(args[2])
    makedirs(outdir)

    # config: local config.json next to the source (the .py's HF_REPO
    # urllib fetch is not ported)
    var base_dir = st_path
    if not isdir(st_path):
        var slash = -1
        var pb = st_path.as_bytes()
        for i in range(pb.__len__()):
            if pb[i] == 47:
                slash = i
        if slash >= 0:
            base_dir = String(unsafe_from_utf8=st_path[byte=0:slash].as_bytes())
    var cfg = Dict[String, String]()
    if exists(base_dir + "/config.json"):
        var cf = open(base_dir + "/config.json", "r")
        var cs = cf.read()
        cf.close()
        var cb = cs.as_bytes()
        var cp = 0
        parse_cfg_object(cb, cp, cfg)
    else:
        raise Error("config.json not found next to the source (the .py's HF_REPO fetch is not ported)")

    # load shards (dir needs model.safetensors.index.json, like the .py)
    var shards = List[Tuple[Pointer[UInt8, MutUntrackedOrigin], Int]]()
    var by_name = Dict[String, STensor]()
    var files = List[String]()
    if isdir(st_path):
        var idx = st_path + "/model.safetensors.index.json"
        if not exists(idx):
            raise Error("directory source needs model.safetensors.index.json")
        var wf = open(idx, "r")
        var ws = wf.read()
        wf.close()
        var wb = ws.as_bytes()
        var wp = 0
        var seen = Dict[String, Bool]()
        skip_ws(wb, wp)
        wp += 1  # {
        skip_ws(wb, wp)
        while True:
            var key = parse_string(wb, wp)
            skip_ws(wb, wp)
            wp += 1  # :
            skip_ws(wb, wp)
            if key == "weight_map":
                wp += 1  # {
                skip_ws(wb, wp)
                if wb[wp] != 125:
                    while True:
                        _ = parse_string(wb, wp)
                        skip_ws(wb, wp)
                        wp += 1  # :
                        skip_ws(wb, wp)
                        var fname = parse_string(wb, wp)
                        if not seen.__contains__(fname):
                            seen[fname] = True
                            files.append(st_path + "/" + fname)
                        skip_ws(wb, wp)
                        if wb[wp] == 44:
                            wp += 1
                            skip_ws(wb, wp)
                        else:
                            wp += 1
                            break
            else:
                skip_value(wb, wp)
            skip_ws(wb, wp)
            if wb[wp] == 44:
                wp += 1
                skip_ws(wb, wp)
            else:
                break
    else:
        files.append(st_path)
    for f in files:
        var rd = read_file_bytes(f)
        var ptr = rd[0]
        var sz = rd[1]
        var sb = Span[UInt8, MutUntrackedOrigin](unsafe_ptr=ptr, length=sz)
        var sts = List[STensor]()
        parse_safetensors(sb, shards.__len__(), sts)
        shards.append((ptr, sz))
        for i in range(sts.__len__()):
            # the .py's single-file path crashes on __metadata__ (no
            # 'shape'); the dir path guards it — skip here (superset fix)
            if sts[i].name == "__metadata__":
                continue
            if not by_name.__contains__(sts[i].name):
                var nm = sts[i].name
                var dims_copy = List[Int]()
                for d in sts[i].dims:
                    dims_copy.append(d)
                by_name[nm] = STensor(
                    nm, sts[i].shard, sts[i].off, sts[i].size, sts[i].dtype,
                    sts[i].ndim, dims_copy^,
                )

    # layer count from tensor names, never the config
    var max_layer = -1
    for e in by_name.items():
        var n = e.key
        if ".layers." in n:
            var dot = n.find(".layers.")
            var rest = String(unsafe_from_utf8=n[byte=dot + 8 :].as_bytes())
            var d2 = rest.find(".")
            var li = str_to_int(String(unsafe_from_utf8=rest[byte=0:d2].as_bytes()))
            if li > max_layer:
                max_layer = li
    if max_layer < 0:
        raise Error("no .layers. tensors found")
    var L = max_layer + 1

    var hs = cfg_int(cfg, "hidden_size", 0)
    var is_ = cfg_int(cfg, "intermediate_size", 0)
    var nh = cfg_int(cfg, "num_attention_heads", 0)
    var nkv = cfg_int(cfg, "num_key_value_heads", 0)
    var max_seq = cfg_int(cfg, "max_position_embeddings", 0)
    var eps = cfg_float(cfg, "rms_norm_eps", 1e-6)
    var rope = cfg_float(cfg, "rope_theta", 10000.0)
    var vs = cfg_int(cfg, "vocab_size", 0)
    print("[st]", L, "layers hs=", hs, "is=", is_, "heads=", nh, "nkv=", nkv)

    var inits = List[Init]()
    var nodes = List[Node]()

    # 2 GB protobuf-cap estimate (element counts, like the .py)
    var emb = get_tensor(by_name, "model.embed_tokens.weight")
    var wb = 0
    for e in by_name.items():
        var n = e.key
        if "layers" in n and ".weight" in n and "norm" not in n:
            var cnt = 1
            for d in e.value.dims:
                cnt *= d
            wb += cnt
    var eb = emb.dims[0] * emb.dims[1] * 2
    var big = (wb + eb + 8 * 1024 * 1024) > comptime_big_threshold()

    var emb_dims = dims_of(emb)
    if big:
        var wq = List[UInt8]()
        var sc = List[UInt8]()
        int8_quant(
            shard_span(emb, shards), emb.off, emb.dtype, emb.dims[0], emb.dims[1], wq, sc
        )
        inits.append(make_init("model.embed_tokens.weight", emb_dims^, TENSOR_INT8, wq^, big))
        var sd = List[Int]()
        sd.append(emb.dims[0])
        inits.append(make_init("model.embed_tokens.weight_scale", sd^, TENSOR_FLOAT, sc^, big))
        var ins = List[String]()
        ins.append("model.embed_tokens.weight")
        ins.append("model.embed_tokens.weight_scale")
        var outs = List[String]()
        outs.append("model.embed_tokens.weight_dq")
        nodes.append(Node(ins^, outs^, "dq_embed", "DequantizeLinear"))
    else:
        inits.append(
            make_init("model.embed_tokens.weight", emb_dims^, TENSOR_FLOAT16, raw_f16(emb, shards), big)
        )

    inits.append(
        make_init("model.norm.weight", dims_of(get_tensor(by_name, "model.norm.weight")), TENSOR_FLOAT, raw_f32(get_tensor(by_name, "model.norm.weight"), shards), big)
    )

    var linears = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]
    for l in range(L):
        var p = "model.layers." + String(l) + "."
        for nm in ["input_layernorm.weight", "post_attention_layernorm.weight"]:
            var t = get_tensor(by_name, p + nm)
            inits.append(make_init(p + nm, dims_of(t), TENSOR_FLOAT, raw_f32(t, shards), big))
        for b in ["self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias"]:
            if by_name.__contains__(p + b):
                var t = get_tensor(by_name, p + b)
                inits.append(make_init(p + b, dims_of(t), TENSOR_FLOAT, raw_f32(t, shards), big))
        for ln in linears:
            var t = get_tensor(by_name, p + ln)
            var wq = List[UInt8]()
            var sc = List[UInt8]()
            int8_quant(shard_span(t, shards), t.off, t.dtype, t.dims[0], t.dims[1], wq, sc)
            inits.append(make_init(p + ln, dims_of(t), TENSOR_INT8, wq^, big))
            var sdd = List[Int]()
            sdd.append(t.dims[0])
            inits.append(make_init(p + ln + "_scale", sdd^, TENSOR_FLOAT, sc^, big))
            var ins = List[String]()
            ins.append(p + ln)
            ins.append(p + ln + "_scale")
            var outs = List[String]()
            outs.append(p + ln + "_dq")
            nodes.append(Node(ins^, outs^, "dq_l" + String(l) + "_" + ln, "DequantizeLinear"))

    var tie = 1
    if by_name.__contains__("lm_head.weight"):
        var t = get_tensor(by_name, "lm_head.weight")
        inits.append(make_init("lm_head.weight", dims_of(t), TENSOR_FLOAT16, raw_f16(t, shards), big))
        print("[st] lm_head:", t.dims[0], "x", t.dims[1], "(fp16, untied)")
        tie = 0

    # ── write (big: external tensors to the sibling .data file) ──
    var offs = List[Int]()
    if big:
        var off = 0
        var data_file = List[UInt8]()
        for i in range(inits.__len__()):
            if inits[i].external:
                offs.append(off)
                off += inits[i].raw.__len__()
                for k in range(inits[i].raw.__len__()):
                    data_file.append(inits[i].raw[k])
            else:
                offs.append(-1)
        write_file_bytes(outdir + "/model_int8.onnx.data", data_file^)
    write_file_bytes(outdir + "/model_int8.onnx", serialize_model(inits, nodes, offs))
    for sh in shards:
        sh[0].unsafe_free()
    print(
        "[st] wrote", outdir + "/model_int8.onnx:",
        inits.__len__(), "initializers,", nodes.__len__(), "DQ nodes",
    )
    write_config(cfg, outdir, hs, is_, nh, nkv, max_seq, eps, rope, vs, tie)
    print("[st] wrote config.json")
