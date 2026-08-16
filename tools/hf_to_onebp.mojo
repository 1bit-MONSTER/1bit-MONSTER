# tools/hf_to_onebp.mojo — Mojo 1.0 twin of hf_to_onebp.py (fold P2.2).
# HuggingFace safetensors -> 1BP, Moonshot/Kimi + general transformers.
# Byte-identical output, same CLI:
#   hf_to_onebp -i input_dir -o out.1bp [--arch kimi_k3|moonlight|kimi_vl|kimi]
#                [--quant Q4NX|I8|TQ1|TQ2|F16|F32|MXFP4] [-q]
# Build:
#   mojo build tools/hf_to_onebp.mojo -o build/hf_to_onebp
#
# Notes vs the .py (kept for parity): the .py's quant_tile dispatch has no
# I8/TQ1 case — those labels write raw F32 tiles (replicated); the TQ1
# tiled_size formula then mismatches, so a TQ1 file's index is wrong in
# both versions. map_tensor_name applies only the FIRST matching
# replacement (so real repos keep names like blk.0.self_attn.q_proj.weight
# — replicated exactly). --repo (huggingface_hub download) is not ported:
# download first, pass -i.

from jsonx import (
    STensor,
    cfg_first_list_str,
    cfg_float,
    cfg_int,
    cfg_str,
    elem_f32,
    parse_cfg_object,
    parse_safetensors,
    parse_string,
    read_file_bytes,
    skip_value,
    skip_ws,
)
from std.collections import Dict, List
from std.math import ceil, log2, pow, round
from std.memory import alloc, bitcast, Layout
from std.os import listdir
from std.os.path import exists, isdir, isfile
from std.sys import argv

comptime ONEBP_MAGIC: Int = 0x00504231  # "1BP\0"
comptime ONEBP_VERSION: Int = 1
comptime ONEBP_Q4NX: Int = 0
comptime ONEBP_I8: Int = 1
comptime ONEBP_TQ1: Int = 2
comptime ONEBP_TQ2: Int = 3
comptime ONEBP_F16: Int = 4
comptime ONEBP_F32: Int = 5
comptime ONEBP_MXFP4: Int = 6

comptime ONEBP_DENSE: Int = 0
comptime ONEBP_KIMI_K3: Int = 50
comptime ONEBP_MOONLIGHT: Int = 51
comptime ONEBP_KIMI_VL: Int = 52

comptime TR: Int = 32
comptime TC: Int = 256
comptime GS: Int = 32

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


# ── quantization ────────────────────────────────────────────────────────

def esize(dtype: Int) -> Int:
    # bytes per element for the tile row offsets
    if dtype == 1 or dtype == 7 or dtype == 8:  # f32 i32 u32
        return 4
    if dtype == 9 or dtype == 10:  # i64 u64
        return 8
    if dtype == 3 or dtype == 4:  # i8 u8
        return 1
    return 2  # bf16 f16 i16 u16


def b_u32_hi16(mut buf: List[UInt8], v: Float32):
    var u = f32b16(v)
    buf.append(UInt8(u & 0xFF))
    buf.append(UInt8((u >> 8) & 0xFF))


def f32b16(v: Float32) -> UInt32:
    # float32 -> upper 16 bits (BF16 truncation)
    return bitcast[DType.uint32, 1](v) >> 16


def quant_qi_q4nx(x: Float32, scale: Float32, zp: Float32) -> Int:
    var t = round((x - zp) * (Float32(1.0) / scale))
    if t < Float32(0.0):
        return 0
    if t > Float32(15.0):
        return 15
    return Int(t)


def tile_q4nx(
    mut buf: List[UInt8],
    b: Span[Byte, _],
    off: Int,
    dtype: Int,
    rows: Int,
    cols: Int,
):
    # Q4NX with the .py's TWO-STAGE flat handling (differs from the qwen3
    # converter): rng < 1e-10 -> scale=0,zp=0; then scale < 1e-10 -> 1.0/0.0.
    var grps = TC // GS
    var row_max = (rows + TR - 1) // TR
    var col_max = (cols + TC - 1) // TC
    var esize = esize(dtype)
    for r0 in range(row_max):
        for c0 in range(col_max):
            # .py semantics: min/max are PER ROW (grouped.min(axis=2) over
            # (32 rows, 8 groups, 32 cols)) — every row has its own
            # scale/zp, unlike the qwen3 converter's group-wide stats.
            var scale = List[Float32]()
            var zp = List[Float32]()
            for r in range(TR):
                var rrow = r0 * TR + r
                var row_off = off + rrow * cols * esize
                for g in range(grps):
                    var gcol = c0 * TC + g * GS
                    # first-element init: padding cells (x=0.0) participate
                    # exactly like numpy's padded array — no phantom zero
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
                    var sc: Float32 = rng / Float32(15.0)
                    var z: Float32 = mn
                    if rng < Float32(1e-10):
                        sc = Float32(0.0)
                        z = Float32(0.0)
                    if sc < Float32(1e-10):
                        sc = Float32(1.0)
                        z = Float32(0.0)
                    scale.append(sc)
                    zp.append(z)
            for i in range(scale.__len__()):
                b_u32_hi16(buf, scale[i])
            for i in range(zp.__len__()):
                b_u32_hi16(buf, zp[i])
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
                        var qe = quant_qi_q4nx(xe, scale[r * grps + g], zp[r * grps + g])
                        var qo = quant_qi_q4nx(xo, scale[r * grps + g], zp[r * grps + g])
                        buf.append(UInt8((qo << 4) | qe))


def tile_tq2(
    mut buf: List[UInt8],
    b: Span[Byte, _],
    off: Int,
    dtype: Int,
    rows: Int,
    cols: Int,
):
    # TQ2 symmetric ternary: scale = max|abs| per group (1.0 if < 1e-10),
    # 2-bit codes (signed+1: -1->0, 0->1, 1->2), 4 per byte.
    var grps = TC // GS
    var row_max = (rows + TR - 1) // TR
    var col_max = (cols + TC - 1) // TC
    var esize = esize(dtype)
    for r0 in range(row_max):
        for c0 in range(col_max):
            # .py semantics: max|abs| is PER ROW (grouped.max(axis=2))
            var scale = List[Float32]()
            for r in range(TR):
                var rrow = r0 * TR + r
                var row_off = off + rrow * cols * esize
                for g in range(grps):
                    var gcol = c0 * TC + g * GS
                    var mx: Float32 = 0.0
                    var have = False
                    for c in range(GS):
                        var ccol = gcol + c
                        var x: Float32
                        if rrow < rows and ccol < cols:
                            x = elem_f32(b, row_off, dtype, ccol)
                        else:
                            x = Float32(0.0)
                        var ax = x if x >= Float32(0.0) else -x
                        if not have:
                            mx = ax
                            have = True
                        elif ax > mx:
                            mx = ax
                    var sc = mx
                    if mx < Float32(1e-10):
                        sc = Float32(1.0)
                    scale.append(sc)
            for i in range(scale.__len__()):
                b_u32_hi16(buf, scale[i])
            for r in range(TR):
                var rrow = r0 * TR + r
                var row_off = off + rrow * cols * esize
                for g in range(grps):
                    var gcol = c0 * TC + g * GS
                    var inv = Float32(1.0) / scale[r * grps + g]
                    for k in range(GS // 4):
                        var code: Int = 0
                        for j in range(4):
                            var ccol = gcol + 4 * k + j
                            var x: Float32
                            if rrow < rows and ccol < cols:
                                x = elem_f32(b, row_off, dtype, ccol)
                            else:
                                x = Float32(0.0)
                            var s = round(x * inv)
                            if s < Float32(-1.0):
                                s = Float32(-1.0)
                            elif s > Float32(1.0):
                                s = Float32(1.0)
                            code |= (Int(s) + 1) << (2 * j)
                        buf.append(UInt8(code))


def quant_mxfp4_row(
    mut nibbles: List[UInt8],
    mut scales: List[UInt8],
    b: Span[Byte, _],
    off: Int,
    dtype: Int,
    rrow: Int,
    cols: Int,
    c0: Int,
):
    # One row's worth of MXFP4: per 32-wide group, E8M0 scale byte + 16
    # packed nibble bytes (nearest LUT value, ties to the first entry).
    var esize = esize(dtype)
    var row_off = off + rrow * cols * esize
    for g in range(TC // GS):
        var gcol = c0 * TC + g * GS
        var max_abs: Float32 = 0.0
        for k in range(GS):
            var ccol = gcol + k
            var x: Float32 = elem_f32(b, row_off, dtype, ccol) if ccol < cols else Float32(0.0)
            var ax = x if x >= Float32(0.0) else -x
            if ax > max_abs:
                max_abs = ax
        var s: Int
        if max_abs < Float32(1e-10):
            s = 0
            for _ in range(GS // 2):
                nibbles.append(0)
        else:
            var target = Float64(max_abs) / 6.0
            var lg = log2(target)
            s = Int(ceil(lg)) + 127
            if s < 0:
                s = 0
            elif s > 255:
                s = 255
            var scale_val = Float32(pow(2.0, Float64(s - 127)))
            var packed = List[UInt8]()
            for k in range(GS // 2):
                var ccol = gcol + 2 * k
                var xe: Float32 = elem_f32(b, row_off, dtype, ccol) if ccol < cols else Float32(0.0)
                var xo: Float32 = elem_f32(b, row_off, dtype, ccol + 1) if ccol + 1 < cols else Float32(0.0)
                var ie = mxfp4_index(xe / scale_val)
                var io = mxfp4_index(xo / scale_val)
                packed.append(UInt8((io << 4) | ie))
            for v in packed:
                nibbles.append(v)
        scales.append(UInt8(s))


def mxfp4_lut_val(j: Int) -> Float32:
    # MXFP4 E2M1 lookup; order matters (argmin ties go to the first entry)
    if j == 1:
        return Float32(0.5)
    if j == 2:
        return Float32(1.0)
    if j == 3:
        return Float32(1.5)
    if j == 4:
        return Float32(2.0)
    if j == 5:
        return Float32(3.0)
    if j == 6:
        return Float32(4.0)
    if j == 7:
        return Float32(6.0)
    if j == 8:
        return Float32(-0.0)
    if j == 9:
        return Float32(-0.5)
    if j == 10:
        return Float32(-1.0)
    if j == 11:
        return Float32(-1.5)
    if j == 12:
        return Float32(-2.0)
    if j == 13:
        return Float32(-3.0)
    if j == 14:
        return Float32(-4.0)
    if j == 15:
        return Float32(-6.0)
    return Float32(0.0)


def mxfp4_index(x: Float32) -> Int:
    # argmin |LUT - x| over all 16 entries, first wins ties
    # (numpy argmin semantics — negative LUT entries win for x < 0)
    var best = 0
    var best_d = x if x >= Float32(0.0) else -x  # |LUT[0] - x| = |x|
    for j in range(1, 16):
        var d = mxfp4_lut_val(j) - x
        if d < Float32(0.0):
            d = -d
        if d < best_d:
            best_d = d
            best = j
    return best


def tile_mxfp4(
    mut buf: List[UInt8],
    b: Span[Byte, _],
    off: Int,
    dtype: Int,
    rows: Int,
    cols: Int,
):
    var row_max = (rows + TR - 1) // TR
    var col_max = (cols + TC - 1) // TC
    var buf_scales = List[UInt8]()
    for r0 in range(row_max):
        for c0 in range(col_max):
            # nibbles for all 32 rows, then scales for all 32 rows
            # (the .py concatenates per row: nibbles+scales — so we must
            # interleave per row instead)
            for r in range(TR):
                var rrow = r0 * TR + r
                if rrow >= rows:
                    # padded row: zeros -> zero nibbles + zero scales
                    for _ in range(TC // GS):
                        for _ in range(GS // 2):
                            buf.append(0)
                    for _ in range(TC // GS):
                        buf.append(0)
                    continue
                quant_mxfp4_row(buf, buf_scales, b, off, dtype, rrow, cols, c0)
                for v in buf_scales:
                    buf.append(v)
                buf_scales.clear()


def tile_f16(
    mut buf: List[UInt8],
    b: Span[Byte, _],
    off: Int,
    dtype: Int,
    rows: Int,
    cols: Int,
):
    var row_max = (rows + TR - 1) // TR
    var col_max = (cols + TC - 1) // TC
    var esize = esize(dtype)
    for r0 in range(row_max):
        for c0 in range(col_max):
            for r in range(TR):
                var rrow = r0 * TR + r
                var row_off = off + rrow * cols * esize
                for c in range(TC):
                    var ccol = c0 * TC + c
                    var x: Float32
                    if rrow < rows and ccol < cols:
                        x = elem_f32(b, row_off, dtype, ccol)
                    else:
                        x = Float32(0.0)
                    var f = Float16(x)
                    var u = bitcast[DType.uint16, 1](f)
                    buf.append(UInt8(u & 0xFF))
                    buf.append(UInt8((u >> 8) & 0xFF))


def tile_f32(
    mut buf: List[UInt8],
    b: Span[Byte, _],
    off: Int,
    dtype: Int,
    rows: Int,
    cols: Int,
):
    # .py semantics: the F32 (else) branch writes the RAW unpadded tile
    # (rh x cw f32) and the caller zero-fills to the full tile size — NOT
    # an internally padded tile (unlike F16).
    var row_max = (rows + TR - 1) // TR
    var col_max = (cols + TC - 1) // TC
    var esize = esize(dtype)
    var full = TR * TC * 4
    for r0 in range(row_max):
        for c0 in range(col_max):
            var rh = rows - r0 * TR
            if rh > TR:
                rh = TR
            var cw = cols - c0 * TC
            if cw > TC:
                cw = TC
            var written = 0
            for r in range(rh):
                var rrow = r0 * TR + r
                var row_off = off + rrow * cols * esize
                for c in range(cw):
                    var ccol = c0 * TC + c
                    b_u32(buf, Int(bitcast[DType.uint32, 1](elem_f32(b, row_off, dtype, ccol))))
                    written += 4
            while written < full:
                buf.append(0)
                written += 1


def tiled_size(rows: Int, cols: Int, quant: Int) -> Int:
    var ntr = (rows + TR - 1) // TR
    var ntc = (cols + TC - 1) // TC
    var tile_b: Int
    if quant == ONEBP_Q4NX:
        tile_b = TR * (TC // GS) * 4 + TR * TC // 2
    elif quant == ONEBP_TQ2:
        tile_b = TR * (TC // GS) * 2 + TR * TC // 4
    elif quant == ONEBP_TQ1:
        var tq1_grps = (TC + 4) // 5
        tile_b = TR * tq1_grps * 2 + TR * tq1_grps
    elif quant == ONEBP_MXFP4:
        var n_grps = TC // GS
        tile_b = TR * (n_grps * 16 + n_grps * 1)
    elif quant == ONEBP_F16:
        tile_b = TR * TC * 2
    else:  # F32 default (also what I8/TQ1 actually write)
        tile_b = TR * TC * 4
    return ntr * ntc * tile_b


def quant_code(q: String) -> Int:
    if q == "Q4NX":
        return ONEBP_Q4NX
    if q == "I8":
        return ONEBP_I8
    if q == "TQ1":
        return ONEBP_TQ1
    if q == "TQ2":
        return ONEBP_TQ2
    if q == "F16":
        return ONEBP_F16
    if q == "F32":
        return ONEBP_F32
    if q == "MXFP4":
        return ONEBP_MXFP4
    return ONEBP_Q4NX


def put_u32(mut hdr: List[UInt8], off: Int, v: Int):
    for k in range(4):
        hdr[off + k] = UInt8((v >> (8 * k)) & 0xFF)


def arch_code(a: String) -> Int:
    if a == "kimi_k3" or a == "kimi":
        return ONEBP_KIMI_K3
    if a == "moonlight":
        return ONEBP_MOONLIGHT
    if a == "kimi_vl":
        return ONEBP_KIMI_VL
    return ONEBP_DENSE


# ── name mapping (first matching replacement wins — .py parity) ─────────

def map_tensor_name(hf_name: String) -> Tuple[String, Bool]:
    if "grad" in hf_name or "adam" in hf_name or "momentum" in hf_name or "optimizer" in hf_name:
        return (hf_name, True)
    var name = hf_name
    # the .py applies only the FIRST replacement that matches
    if "model.layers." in name:
        name = name.replace("model.layers.", "blk.")
    elif "model." in name:
        name = name.replace("model.", "")
    elif "self_attn." in name:
        name = name.replace("self_attn.", "attn.")
    elif "mlp." in name:
        name = name.replace("mlp.", "ffn.")
    elif "input_layernorm" in name:
        name = name.replace("input_layernorm", "rms_attn_w")
    elif "post_attention_layernorm" in name:
        name = name.replace("post_attention_layernorm", "rms_ffn_w")
    elif "q_proj.weight" in name:
        name = name.replace("q_proj.weight", "attn_q.weight")
    elif "k_proj.weight" in name:
        name = name.replace("k_proj.weight", "attn_k.weight")
    elif "v_proj.weight" in name:
        name = name.replace("v_proj.weight", "attn_v.weight")
    elif "o_proj.weight" in name:
        name = name.replace("o_proj.weight", "attn_o.weight")
    elif "gate_proj.weight" in name:
        name = name.replace("gate_proj.weight", "ffn_gate.weight")
    elif "up_proj.weight" in name:
        name = name.replace("up_proj.weight", "ffn_up.weight")
    elif "down_proj.weight" in name:
        name = name.replace("down_proj.weight", "ffn_down.weight")
    elif "embed_tokens.weight" in name:
        name = name.replace("embed_tokens.weight", "token_embd.weight")
    elif "norm.weight" in name:
        name = name.replace("norm.weight", "output_norm.weight")
    elif "lm_head.weight" in name:
        name = name.replace("lm_head.weight", "output.weight")
    var skip = not (
        ".weight" in hf_name
        or "_proj" in hf_name
        or "embed" in hf_name
        or "norm" in hf_name
        or "lm_head" in hf_name
    )
    return (name, skip)


# ── architecture detection (.py precedence replicated exactly) ──────────

def tolower(s: String) -> String:
    var out = String()
    var sb = s.as_bytes()
    for i in range(sb.__len__()):
        var c = sb[i]
        if c >= 65 and c <= 90:  # A-Z
            out += chr(Int(c) + 32)
        else:
            out += chr(Int(c))
    return out


def detect_architecture(cfg: Dict[String, String], names: List[String], n_params: Int) raises -> String:
    var model_type = cfg_str(cfg, "model_type", "unknown")
    var arch_str = cfg_first_list_str(cfg, "architectures", "")
    var has_kda = False
    var has_mla = False
    var has_moe = False
    var has_moonvit = False
    var has_attnres = False
    for n in names:
        var low = tolower(n)
        if "kda" in low:
            has_kda = True
        if "mla" in low or "kv_a" in low:
            has_mla = True
        if "expert" in low or "moe" in low or "router" in low:
            has_moe = True
        if "moonvit" in low or "moon_vit" in low:
            has_moonvit = True
        if "attnres" in low or "attn_res" in low:
            has_attnres = True
    if has_kda and has_attnres and n_params > 1000000000000:
        return "kimi_k3"
    # (has_mla and has_moe and "moonlight" in mt) or ("kimi" in mt and not has_kda)
    if (has_mla and has_moe and "moonlight" in model_type) or ("kimi" in model_type and not has_kda):
        return "moonlight"
    if has_moonvit:
        return "kimi_vl"
    if "kimi" in model_type or "kimi" in arch_str:
        if "vl" in model_type:
            return "kimi_vl"
        return "kimi_k3" if "k3" in model_type else "moonlight"
    return model_type


def build_header(
    cfg: Dict[String, String],
    tensors: List[STensor],
    arch_str: String,
    quant: Int,
    n_entries: Int,
) raises -> List[UInt8]:
    # 256 B, fixed offsets (must match OnebpHeader in include/onebp_format.h)
    var hdr = List[UInt8]()
    for _ in range(256):
        hdr.append(0)
    var hs = cfg_int(cfg, "hidden_size", cfg_int(cfg, "d_model", 0))
    var nl = cfg_int(cfg, "num_hidden_layers", cfg_int(cfg, "num_layers", 0))
    var nh = cfg_int(cfg, "num_attention_heads", cfg_int(cfg, "num_heads", 0))
    var nkv = cfg_int(cfg, "num_key_value_heads", cfg_int(cfg, "num_kv_heads", nh))
    var hd = cfg_int(cfg, "head_dim", 0)
    var im = cfg_int(cfg, "intermediate_size", 0)
    var vs = cfg_int(cfg, "vocab_size", 0)
    var msl = cfg_int(cfg, "max_position_embeddings", cfg_int(cfg, "max_seq_len", 2048))
    if hd == 0 and nh > 0 and hs > 0:
        hd = hs // nh
    if hs == 0:
        for i in range(tensors.__len__()):
            var nm = tensors[i].name
            if ("embed_tokens" in nm or "embedding" in nm) and tensors[i].ndim >= 2:
                hs = tensors[i].dims[tensors[i].dims.__len__() - 1]
                vs = tensors[i].dims[tensors[i].dims.__len__() - 2]
                break
    if nh == 0:
        for i in range(tensors.__len__()):
            var nm = tensors[i].name
            if "q_proj" in nm and "weight" in nm and tensors[i].ndim >= 2:
                var q_dim = tensors[i].dims[0]
                var kname = nm.replace("q", "k")
                for j in range(tensors.__len__()):
                    if "k_proj" in kname and tensors[j].ndim >= 2:
                        var kv_dim = tensors[j].dims[0]
                        if hs > 0 and q_dim > 0:
                            nh = q_dim // (q_dim // (kv_dim // (nkv if nkv > 0 else 1)) if hs > 0 else 1)
                        break
                break

    put_u32(hdr, 0, ONEBP_MAGIC)
    put_u32(hdr, 4, ONEBP_VERSION)
    put_u32(hdr, 8, arch_code(arch_str))
    put_u32(hdr, 12, quant)
    put_u32(hdr, 16, 0)  # scale_type
    put_u32(hdr, 20, hs)
    put_u32(hdr, 24, nl)
    put_u32(hdr, 28, nh if nh > 0 else 1)
    put_u32(hdr, 32, nkv if nkv > 0 else nh)
    put_u32(hdr, 36, hd if hd > 0 else (hs // nh if nh > 0 else 128))
    put_u32(hdr, 40, im)
    put_u32(hdr, 44, vs)
    put_u32(hdr, 48, msl)
    put_u32(hdr, 52, TR)
    put_u32(hdr, 56, TC)
    put_u32(hdr, 60, GS)
    var has_q = 1 if cfg_int(cfg, "q_norm", cfg_int(cfg, "has_q_norm", 0)) != 0 else 0
    var has_k = 1 if cfg_int(cfg, "k_norm", cfg_int(cfg, "has_k_norm", 0)) != 0 else 0
    var has_bias = 1 if cfg_int(cfg, "bias", cfg_int(cfg, "has_bias", 0)) != 0 else 0
    put_u32(hdr, 64, has_q)
    put_u32(hdr, 68, has_k)
    put_u32(hdr, 72, has_bias)
    var rope = cfg_float(cfg, "rope_theta", cfg_float(cfg, "rope.theta", 10000.0))
    put_u32(hdr, 76, Int(rope * 1000.0))
    put_u32(hdr, 80, cfg_int(cfg, "bos_token_id", 1))
    put_u32(hdr, 84, cfg_int(cfg, "eos_token_id", 2))
    put_u32(hdr, 88, n_entries)
    if arch_str == "kimi_k3" or arch_str == "moonlight" or arch_str == "kimi_vl":
        var n_exp = cfg_int(cfg, "num_experts", cfg_int(cfg, "moe_num_experts", 256))
        var n_used = cfg_int(cfg, "num_experts_per_tok", cfg_int(cfg, "top_k", 8))
        var n_ff_exp = cfg_int(cfg, "intermediate_size", im)
        var n_ff_shexp = cfg_int(cfg, "shared_expert_intermediate_size", n_ff_exp)
        put_u32(hdr, 92, n_exp)
        put_u32(hdr, 96, n_used)
        put_u32(hdr, 100, n_ff_exp)
        put_u32(hdr, 104, n_ff_shexp)
        if arch_str == "kimi_k3":
            put_u32(hdr, 148, cfg_int(cfg, "num_kda_layers", 69))
            put_u32(hdr, 152, cfg_int(cfg, "num_mla_layers", 24))
            put_u32(hdr, 156, cfg_int(cfg, "kda_latent_dim", 3584))
    if arch_str == "kimi_vl":
        put_u32(hdr, 160, cfg_int(cfg, "vision_hidden_size", 1024))
    var tag = cfg_str(cfg, "_name_or_path", cfg_str(cfg, "model_type", arch_str))
    var tb = tag.as_bytes()
    for k in range(64):
        if k < tb.__len__():
            hdr[192 + k] = tb[k]
    return hdr^


def sort_strings(mut names: List[String]):
    for i in range(1, names.__len__()):
        var v = names[i]
        var j = i - 1
        while j >= 0 and names[j] > v:
            names[j + 1] = names[j]
            j -= 1
        names[j + 1] = v


# ── main ────────────────────────────────────────────────────────────────

def main() raises:
    var args = argv()
    var input_path = String()
    var output_path = String()
    var arch = Optional[String]()
    var quant = String("Q4NX")
    var quiet = False
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "-i" or a == "--input":
            if i + 1 >= len(args):
                raise Error(a + " needs a value")
            input_path = String(args[i + 1])
            i += 2
        elif a == "--repo":
            raise Error(
                "--repo (huggingface_hub download) is not ported to the Mojo twin; "
                + "download the repo first and pass -i"
            )
        elif a == "-o" or a == "--output":
            if i + 1 >= len(args):
                raise Error(a + " needs a value")
            output_path = String(args[i + 1])
            i += 2
        elif a == "--arch":
            if i + 1 >= len(args):
                raise Error("--arch needs a value")
            arch = String(args[i + 1])
            i += 2
        elif a == "--quant":
            if i + 1 >= len(args):
                raise Error("--quant needs a value")
            quant = String(args[i + 1])
            i += 2
        elif a == "-q" or a == "--quiet":
            quiet = True
            i += 1
        else:
            raise Error("unexpected argument " + a)
    if output_path == "":
        raise Error("the following arguments are required: -o/--output")
    if input_path == "":
        raise Error("one of -i/--input or --repo is required")

    # config
    var cfg = Dict[String, String]()
    if exists(input_path + "/config.json"):
        var cf = open(input_path + "/config.json", "r")
        var cs = cf.read()
        cf.close()
        var cb = cs.as_bytes()
        var cp = 0
        parse_cfg_object(cb, cp, cfg)

    # load safetensors (file or dir)
    var shards = List[Tuple[Pointer[UInt8, MutUntrackedOrigin], Int]]()
    var sts_all = List[STensor]()
    var files = List[String]()
    if isfile(input_path):
        files.append(input_path)
    elif isdir(input_path):
        var idx = input_path + "/model.safetensors.index.json"
        if exists(idx):
            # index: only the shards referenced by the weight map
            var wf = open(idx, "r")
            var ws = wf.read()
            wf.close()
            var wb = ws.as_bytes()
            var wp = 0
            # walk the top-level object, collect "filename" strings under
            # "weight_map" into a set (order-preserving)
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
                                files.append(input_path + "/" + fname)
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
            for n in listdir(input_path):
                if n.endswith(".safetensors"):
                    files.append(input_path + "/" + n)
            sort_strings(files)
    else:
        raise Error("Not found: " + input_path)

    var n_params = 0
    for f in files:
        var rd = read_file_bytes(f)
        var ptr = rd[0]
        var sz = rd[1]
        var sb = Span[UInt8, MutUntrackedOrigin](unsafe_ptr=ptr, length=sz)
        parse_safetensors(sb, shards.__len__(), sts_all)
        shards.append((ptr, sz))
    for i in range(sts_all.__len__()):
        var cnt = 1
        for d in sts_all[i].dims:
            cnt *= d
        n_params += cnt

    if not quiet:
        print("Loaded", sts_all.__len__(), "tensors (", n_params, " params )")

    # architecture
    var names = List[String]()
    for i in range(sts_all.__len__()):
        names.append(sts_all[i].name)
    var arch_str: String
    if arch:
        arch_str = arch.value()
    else:
        arch_str = detect_architecture(cfg, names, n_params)
        if not quiet:
            print("Detected architecture:", arch_str)
    var quant_code_v = quant_code(quant)

    # map + index
    var entries = List[Tuple[String, Int, Int, Int, Int, Int]]()  # name, shard, off, rows, cols, dtype
    for i in range(sts_all.__len__()):
        var mapped = map_tensor_name(sts_all[i].name)
        var skip = mapped[1]
        if skip:
            continue
        if sts_all[i].ndim < 2:
            continue
        entries.append(
            (mapped[0], sts_all[i].shard, sts_all[i].off, sts_all[i].dims[sts_all[i].dims.__len__() - 2], sts_all[i].dims[sts_all[i].dims.__len__() - 1], sts_all[i].dtype)
        )
    # no sort: the .py writes in safetensors load order

    var total = 0
    var sizes = List[Int]()
    for e in entries:
        var tsize = tiled_size(e[3], e[4], quant_code_v)
        sizes.append(tsize)
        total += tsize
    if not quiet:
        print("Tensors to convert:", entries.__len__(), "Data size:", total // (1024 * 1024), "MB")

    # header + index
    var hdr = build_header(cfg, sts_all, arch_str, quant_code_v, entries.__len__())
    var idx = List[UInt8]()
    # Offsets are DATA-RELATIVE: NpuOnebpModel adds the parsed index size
    # (file_offset += data_start). The .py starts them at 256, so its
    # outputs read 256 bytes past every tensor and overrun the file end;
    # the correct base is 0 (as in the qwen3 converter). This is the only
    # byte-level deviation from the .py.
    var off = 0
    for i in range(entries.__len__()):
        var nm = entries[i][0]
        var nb = nm.as_bytes()
        var name_len = nb.__len__() if nb.__len__() < 63 else 63
        b_u32(idx, name_len)
        for k in range(name_len):
            idx.append(nb[k])
        idx.append(0)
        b_u32(idx, 2)  # ndim
        b_u32(idx, entries[i][3])
        b_u32(idx, entries[i][4])
        b_u64(idx, off)
        b_u64(idx, sizes[i])
        off += sizes[i]

    # write header + index, then stream tiles per tensor
    var out = List[UInt8](capacity=256 + idx.__len__())
    for k in range(256):
        out.append(hdr[k])
    for k in range(idx.__len__()):
        out.append(idx[k])
    var n = out.__len__()
    var alloc_obj = out.unsafe_take_allocation()
    var ptr = alloc_obj^.unsafe_leak()
    var span = Span[UInt8, MutUntrackedOrigin](unsafe_ptr=ptr, length=n)
    var f = open(output_path, "w")
    f.write_bytes(span)
    ptr.unsafe_free()

    for i in range(entries.__len__()):
        var e = entries[i]
        var sb = Span[UInt8, MutUntrackedOrigin](
            unsafe_ptr=shards[e[1]][0], length=shards[e[1]][1]
        )
        var tbuf = List[UInt8](capacity=sizes[i])
        if quant_code_v == ONEBP_Q4NX:
            tile_q4nx(tbuf, sb, e[2], e[5], e[3], e[4])
        elif quant_code_v == ONEBP_TQ2:
            tile_tq2(tbuf, sb, e[2], e[5], e[3], e[4])
        elif quant_code_v == ONEBP_MXFP4:
            tile_mxfp4(tbuf, sb, e[2], e[5], e[3], e[4])
        elif quant_code_v == ONEBP_F16:
            tile_f16(tbuf, sb, e[2], e[5], e[3], e[4])
        else:
            # F32 (also what the .py writes for I8/TQ1 labels)
            tile_f32(tbuf, sb, e[2], e[5], e[3], e[4])
        var tn = tbuf.__len__()
        var talloc = tbuf.unsafe_take_allocation()
        var tptr = talloc^.unsafe_leak()
        var tsp = Span[UInt8, MutUntrackedOrigin](unsafe_ptr=tptr, length=tn)
        f.write_bytes(tsp)
        tptr.unsafe_free()
    f.close()
    for sh in shards:
        sh[0].unsafe_free()
    if not quiet:
        print("wrote", output_path, ":", entries.__len__(), "tensors,", total // (1024 * 1024), "MB")


