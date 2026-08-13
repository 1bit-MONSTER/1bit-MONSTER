# tools/qwen3_to_onebp.mojo — Mojo 1.0 twin of qwen3_to_onebp.py (fold P2.2).
# Qwen3-family HF text model (safetensors) -> 1BP Q4NX. Byte-identical
# output, same CLI:
#   qwen3_to_onebp input_dir out.1bp [--arch N]
# Build:
#   mojo build tools/qwen3_to_onebp.mojo -o build/qwen3_to_onebp
#
# Mirrors the loader expectations of GenericBackend::load_1bp
# (src/backend_generic.cpp): 2-D tensors as Q4NX tiles (32x256, gs=32, bf16
# scales) in the HF safetensors layout ([out, in] row-major, no transposes);
# 1-D tensors raw F32. Difference vs the Python original: the shard glob is
# model*.safetensors (the .py's model-*.safetensors missed single-shard
# repos like Qwen's own model.safetensors).

from jsonx import (
    STensor,
    cfg_float,
    cfg_int,
    elem_f32,
    parse_cfg_object,
    parse_safetensors,
    read_file_bytes,
    str_to_int,
)
from std.collections import Dict, List

from std.memory import bitcast
from std.os import listdir
from std.math import round
from std.sys import argv

comptime ONEBP_MAGIC: Int = 0x00504231
comptime ONEBP_Q4NX: Int = 0
comptime ONEBP_DEEPSEEK2: Int = 20  # Qwen3-style with QK-norm (matches Mage-VL)

comptime TR: Int = 32  # tile_rows
comptime TC: Int = 256  # tile_cols
comptime GS: Int = 32


struct Entry(
    ImplicitlyCopyable,
):
    var name: String
    var shard: Int
    var off: Int  # absolute byte offset into the shard string
    var size: Int
    var dtype: Int  # 0=bf16 1=f32 2=f16
    var rows: Int
    var cols: Int  # -1 for 1-D, -2 for unmappable ndim

    def __init__(
        out self,
        name: String,
        shard: Int,
        off: Int,
        size: Int,
        dtype: Int,
        rows: Int,
        cols: Int,
    ):
        self.name = name
        self.shard = shard
        self.off = off
        self.size = size
        self.dtype = dtype
        self.rows = rows
        self.cols = cols


def b_u16(mut buf: List[UInt8], v: Int):
    buf.append(UInt8(v & 0xFF))
    buf.append(UInt8((v >> 8) & 0xFF))


def b_u32(mut buf: List[UInt8], v: Int):
    buf.append(UInt8(v & 0xFF))
    buf.append(UInt8((v >> 8) & 0xFF))
    buf.append(UInt8((v >> 16) & 0xFF))
    buf.append(UInt8((v >> 24) & 0xFF))


def b_u64(mut buf: List[UInt8], v: Int):
    for k in range(8):
        buf.append(UInt8((v >> (8 * k)) & 0xFF))


def b_str(mut buf: List[UInt8], s: String):
    var sb = s.as_bytes()
    for i in range(sb.__len__()):
        buf.append(sb[i])


# ── name mapping ────────────────────────────────────────────────────────

def map_name(hf_name: String) -> Optional[String]:
    # HF Qwen3 / Mage-VL text tensor name -> 1BP canonical name (or None).
    # Prefix lengths are comptime: len("model.language_model.") = 21,
    # len("layers.") = 7, len("model.") = 6.
    var n = hf_name
    if n.startswith("model.language_model."):
        var lm = String(unsafe_from_utf8=n[byte=21:].as_bytes())
        n = lm
    if n.startswith("model.visual.") or n.startswith("visual."):
        return None
    if n.startswith("model."):
        # plain Qwen3 (model.layers.*, model.embed_tokens.weight): the .py
        # only handled pre-stripped keys; strip the prefix so real repos work
        var md = String(unsafe_from_utf8=n[byte=6:].as_bytes())
        n = md
    if n.startswith("layers."):
        var ly = String(unsafe_from_utf8=n[byte=7:].as_bytes())
        n = ly
        var dot = n.find(".")
        if dot < 0:
            return None
        var blk = String("blk.") + String(unsafe_from_utf8=n[byte=0:dot].as_bytes()) + "."
        var rest = String(unsafe_from_utf8=n[byte=dot + 1 :].as_bytes())
        if rest == "input_layernorm.weight":
            return blk + "attn_norm.weight"
        if rest == "post_attention_layernorm.weight":
            return blk + "ffn_norm.weight"
        if rest == "self_attn.q_proj.weight":
            return blk + "attn_q.weight"
        if rest == "self_attn.q_norm.weight":
            return blk + "attn_q_norm.weight"
        if rest == "self_attn.k_proj.weight":
            return blk + "attn_k.weight"
        if rest == "self_attn.k_norm.weight":
            return blk + "attn_k_norm.weight"
        if rest == "self_attn.v_proj.weight":
            return blk + "attn_v.weight"
        if rest == "self_attn.o_proj.weight":
            return blk + "attn_output.weight"
        if rest == "mlp.gate_proj.weight":
            return blk + "ffn_gate.weight"
        if rest == "mlp.up_proj.weight":
            return blk + "ffn_up.weight"
        if rest == "mlp.down_proj.weight":
            return blk + "ffn_down.weight"
        return None
    if n == "embed_tokens.weight":
        return "token_embd.weight"
    if n == "norm.weight":
        return "output_norm.weight"
    if n == "lm_head.weight":
        return "output.weight"
    return None


def sort_entries(mut entries: List[Entry]):
    # insertion sort by name (tensor count is small)
    for i in range(1, entries.__len__()):
        var v = entries[i]
        var j = i - 1
        while j >= 0 and entries[j].name > v.name:
            entries[j + 1] = entries[j]
            j -= 1
        entries[j + 1] = v


def sort_strings(mut names: List[String]):
    for i in range(1, names.__len__()):
        var v = names[i]
        var j = i - 1
        while j >= 0 and names[j] > v:
            names[j + 1] = names[j]
            j -= 1
        names[j + 1] = v


# ── Q4NX quantization ───────────────────────────────────────────────────

def quant_qi(x: Float32, scale: Float32, zp: Float32) -> Int:
    # round ties-to-even (matches np.round), clip to [0, 15]
    var t = round((x - zp) * (Float32(1.0) / scale))
    if t < Float32(0.0):
        return 0
    if t > Float32(15.0):
        return 15
    return Int(t)


def append_tile_2d(
    mut buf: List[UInt8],
    b: Span[Byte, _],
    off: Int,
    dtype: Int,
    rows: Int,
    cols: Int,
):
    # Q4NX tile stream, byte-identical to quant_tile_q4nx in the .py:
    # for each 32x256 tile: per 32-wide group over all 32 rows, bf16
    # scale/zp (flat group -> 1.0/0.0; padding cells count as 0.0 in
    # min/max), then 4-bit codes packed low-nibble = even columns.
    # Byte order per tile: [scales (32 rows x 8 groups, u16)][zps same]
    # [codes (32 rows x 128 bytes)].
    var grps = TC // GS
    var row_max = (rows + TR - 1) // TR
    var col_max = (cols + TC - 1) // TC
    var esize = 2 if dtype != 1 else 4
    for r0 in range(row_max):
        for c0 in range(col_max):
            # .py semantics: min/max are PER ROW (grouped.min(axis=2) over
            # (TR, grps, GS)); each (row, group) has its own scale/zp.
            var scale = List[Float32]()
            var zp = List[Float32]()
            for r in range(TR):
                var rrow = r0 * TR + r
                var row_off = off + rrow * cols * esize
                for g in range(grps):
                    var gcol = c0 * TC + g * GS
                    # first-element init; padding cells (x=0.0) participate
                    # exactly like numpy's padded array
                    var mn: Float32 = 0.0
                    var mx: Float32 = 0.0
                    var have = False
                    for c in range(GS):
                        var ccol = gcol + c
                        var x: Float32
                        if rrow < rows and ccol < cols:
                            x = elem_f32(b, row_off, dtype, ccol)
                        else:
                            x = Float32(0.0)
                        if not have:
                            mn = x
                            mx = x
                            have = True
                        elif x < mn:
                            mn = x
                        elif x > mx:
                            mx = x
                    var rng = mx - mn
                    var sc: Float32
                    var z: Float32
                    if rng < Float32(1e-10):
                        sc = Float32(1.0)
                        z = Float32(0.0)
                    else:
                        sc = rng / Float32(15.0)
                        z = mn
                    # quantization uses the FULL-precision f32 scale/zp (like
                    # the .py); only the file copy is truncated to bf16
                    scale.append(sc)
                    zp.append(z)
            # scales block: 32 rows x 8 groups
            for i in range(scale.__len__()):
                b_u16(buf, Int(bitcast[DType.uint32, 1](scale[i]) >> 16))
            # zps block
            for i in range(zp.__len__()):
                b_u16(buf, Int(bitcast[DType.uint32, 1](zp[i]) >> 16))
            # codes block
            for r in range(TR):
                var rrow = r0 * TR + r
                var row_off = off + rrow * cols * esize
                for g in range(grps):
                    var gcol = c0 * TC + g * GS
                    for k in range(GS // 2):
                        var ccol = gcol + 2 * k
                        var xe: Float32
                        var xo: Float32
                        if rrow < rows and ccol < cols:
                            xe = elem_f32(b, row_off, dtype, ccol)
                        else:
                            xe = Float32(0.0)
                        if rrow < rows and ccol + 1 < cols:
                            xo = elem_f32(b, row_off, dtype, ccol + 1)
                        else:
                            xo = Float32(0.0)
                        var qe = quant_qi(xe, scale[r * grps + g], zp[r * grps + g])
                        var qo = quant_qi(xo, scale[r * grps + g], zp[r * grps + g])
                        buf.append(UInt8((qo << 4) | qe))


# ── main ────────────────────────────────────────────────────────────────

def basename(p: String) -> String:
    var b = p.as_bytes()
    var slash = -1
    for i in range(b.__len__()):
        if b[i] == 47:  # /
            slash = i
    if slash < 0:
        return p
    return String(unsafe_from_utf8=p[byte=slash + 1 :].as_bytes())


def main() raises:
    var args = argv()
    var input_dir = String()
    var out_path = String()
    var arch = ONEBP_DEEPSEEK2
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--arch":
            if i + 1 >= len(args):
                raise Error("--arch needs a value")
            arch = str_to_int(String(args[i + 1]))
            i += 2
        elif input_dir == "":
            input_dir = a
            i += 1
        elif out_path == "":
            out_path = a
            i += 1
        else:
            raise Error("unexpected argument " + a)
    if input_dir == "" or out_path == "":
        print("usage: qwen3_to_onebp input_dir out.1bp [--arch N]")
        return

    # shards: model*.safetensors (superset of the .py's model-* glob so
    # single-shard repos like model.safetensors work too)
    var names = List[String]()
    for n in listdir(input_dir):
        if n.startswith("model") and n.endswith(".safetensors"):
            names.append(n)
    sort_strings(names)
    if names.__len__() == 0:
        raise Error("error: no model*.safetensors in " + input_dir)

    print("loading", names.__len__(), "shards...")
    # shards are binary — read raw bytes (String read() validates UTF-8)
    var shards = List[Tuple[Pointer[UInt8, MutUntrackedOrigin], Int]]()
    var tensors = Dict[String, Entry]()
    for n in names:
        var fp = input_dir + "/" + n
        var rd = read_file_bytes(fp)
        var shard_ptr = rd[0]
        var shard_sz = rd[1]
        var sb = Span[UInt8, MutUntrackedOrigin](
            unsafe_ptr=shard_ptr, length=shard_sz
        )
        var sts = List[STensor]()
        parse_safetensors(sb, shards.__len__(), sts)
        for i in range(sts.__len__()):
            var nm = sts[i].name
            var rows: Int
            var cols: Int
            if sts[i].ndim == 1:
                rows = sts[i].dims[0]
                cols = -1
            elif sts[i].ndim == 2:
                rows = sts[i].dims[0]
                cols = sts[i].dims[1]
            else:
                # ndim > 2 never maps; marker errors if it somehow does
                rows = -2
                cols = -2
            tensors[nm] = Entry(
                nm, sts[i].shard, sts[i].off, sts[i].size, sts[i].dtype, rows, cols
            )
        shards.append((shard_ptr, shard_sz))
    print("loaded", tensors.__len__(), "tensors")

    # config (text_config overrides top-level, like the .py's merge)
    var cfg = Dict[String, String]()
    var cf = open(input_dir + "/config.json", "r")
    var cs = cf.read()
    cf.close()
    var cb = cs.as_bytes()
    var cp = 0
    parse_cfg_object(cb, cp, cfg)
    var tc = cfg.get("text_config")
    if tc:
        var tcv = tc.value()
        var tcb = tcv.as_bytes()
        var tcp = 0
        parse_cfg_object(tcb, tcp, cfg)

    var H = cfg_int(cfg, "hidden_size", 0)
    var L = cfg_int(cfg, "num_hidden_layers", 0)
    var NH = cfg_int(cfg, "num_attention_heads", 0)
    var NKV = cfg_int(cfg, "num_key_value_heads", NH)
    var HD = cfg_int(cfg, "head_dim", H // NH)
    var FF = cfg_int(cfg, "intermediate_size", 0)
    var V = cfg_int(cfg, "vocab_size", 0)
    var max_seq = cfg_int(cfg, "max_position_embeddings", 131072)
    var rope = cfg_float(cfg, "rope_theta", 1000000.0)
    var bos = cfg_int(cfg, "bos_token_id", 0)
    var eos = cfg_int(cfg, "eos_token_id", 0)

    # map + sort
    var entries = List[Entry]()
    for e in tensors.items():
        var m = map_name(e.key)
        if not m:
            continue
        if e.value.rows == -2:
            raise Error("unexpected ndim for " + e.key)
        entries.append(
            Entry(
                m.value(),
                e.value.shard,
                e.value.off,
                e.value.size,
                e.value.dtype,
                e.value.rows,
                e.value.cols,
            )
        )
    sort_entries(entries)
    if entries.__len__() == 0:
        raise Error("error: no text-model tensors mapped")
    var have = Dict[String, Bool]()
    for e in entries:
        have[e.name] = True
    for want in [
        "token_embd.weight",
        "output_norm.weight",
        "output.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_q_norm.weight",
        "blk.0.ffn_gate.weight",
    ]:
        if not have.__contains__(want):
            print("  WARN missing " + want)

    # ── header (256 B, layout per include/onebp_format.h) ──
    var hdr = List[UInt8](capacity=256)
    b_u32(hdr, ONEBP_MAGIC)
    b_u32(hdr, 1)  # version
    b_u32(hdr, arch)
    b_u32(hdr, ONEBP_Q4NX)
    b_u32(hdr, 0)
    b_u32(hdr, H)
    b_u32(hdr, L)
    b_u32(hdr, NH)
    b_u32(hdr, NKV)
    b_u32(hdr, HD)
    b_u32(hdr, FF)
    b_u32(hdr, V)
    b_u32(hdr, max_seq)
    b_u32(hdr, TR)
    b_u32(hdr, TC)
    b_u32(hdr, GS)
    b_u32(hdr, 1)
    b_u32(hdr, 1)
    b_u32(hdr, 0)
    b_u32(hdr, Int(rope * 1000.0))
    b_u32(hdr, bos)
    b_u32(hdr, eos)
    b_u32(hdr, entries.__len__())
    for _ in range(14):
        b_u32(hdr, 0)
    for _ in range(6):
        b_u32(hdr, 0)
    while hdr.__len__() < 192:
        hdr.append(0)
    var tag = basename(out_path)
    var tb = tag.as_bytes()
    for k in range(64):
        if k < tb.__len__():
            hdr.append(tb[k])
        else:
            hdr.append(0)
    if hdr.__len__() != 256:
        raise Error("internal: header size != 256")

    # ── index + data sizes ──
    var idx = List[UInt8]()
    var sizes = List[Int]()
    var off = 0
    var total = 0
    for e in entries:
        var nd = 1 if e.cols < 0 else 2
        var sz: Int
        if nd == 1:
            sz = e.rows * 4
        else:
            var ntr = (e.rows + TR - 1) // TR
            var ntc = (e.cols + TC - 1) // TC
            sz = ntr * ntc * (TR * (TC // GS) * 4 + TR * TC // 2)
        sizes.append(sz)
        b_u32(idx, e.name.byte_length())
        b_str(idx, e.name)
        idx.append(0)
        b_u32(idx, nd)
        b_u32(idx, e.rows)
        if nd == 2:
            b_u32(idx, e.cols)
        b_u64(idx, off)
        b_u64(idx, sz)
        off += sz
        total += sz

    # ── data ──
    var buf = List[UInt8](capacity=256 + idx.__len__() + total)
    for i in range(hdr.__len__()):
        buf.append(hdr[i])
    for i in range(idx.__len__()):
        buf.append(idx[i])
    for e in entries:
        var sb = Span[UInt8, MutUntrackedOrigin](
            unsafe_ptr=shards[e.shard][0], length=shards[e.shard][1]
        )
        if e.cols < 0:
            for k in range(e.rows):
                b_u32(buf, Int(bitcast[DType.uint32, 1](elem_f32(sb, e.off, e.dtype, k))))
        else:
            append_tile_2d(buf, sb, e.off, e.dtype, e.rows, e.cols)

    var n = buf.__len__()
    var data = buf.unsafe_take_allocation()
    var ptr = data^.unsafe_leak()
    var span = Span[UInt8, MutUntrackedOrigin](unsafe_ptr=ptr, length=n)
    var g = open(out_path, "w")
    g.write_bytes(span)
    g.close()
    ptr.unsafe_free()
    for sh in shards:
        sh[0].unsafe_free()
    print(
        "wrote",
        out_path,
        ":",
        entries.__len__(),
        "tensors,",
        total // 1000000,
        "MB",
    )
