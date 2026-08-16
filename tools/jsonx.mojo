# tools/jsonx.mojo — hand-rolled JSON scanner + raw file IO shared by the
# Mojo converter twins (fold P2.2). No std.json in Mojo 1.0; the payloads
# are fixed-shape, so this exposes token-level primitives: strings (with
# \uXXXX + surrogate pairs), ints, floats, arrays-of-ints, and syntactic
# skip for everything else. Pattern from the Adrenalin rewrite.
#
# Parsers work on raw byte spans (Span[Byte, _]) — not String — because
# safetensors shards are binary (String construction validates UTF-8 and
# would reject them). String slices are only built for JSON text, which is
# valid UTF-8 by construction.
#
# Cursor convention: mut p advances past each value; callers must call
# skip_ws before reading src[p] (skip_ws raises at EOF, so a well-formed
# document never reads OOB).

from std.io import FileDescriptor
from std.memory import alloc, bitcast, Layout
from std.os.path import getsize


def read_file_bytes(
    path: String,
) raises -> Tuple[Pointer[UInt8, MutUntrackedOrigin], Int]:
    # Raw (possibly binary) file read: FileHandle.read() returns a String
    # which validates UTF-8, so binary shards go through libc directly.
    var sz = getsize(path)
    var alloc_obj = alloc[UInt8](Layout[UInt8](count=sz))
    var ptr = alloc_obj^.unsafe_leak()
    var f = open(path, "r")
    var fd = FileDescriptor(f)
    var got: Int = 0
    while got < sz:
        var sub = Span[mut=True, Byte, MutUntrackedOrigin](
            unsafe_ptr=ptr.unsafe_offset(got), length=sz - got
        )
        var n = fd.read_bytes(sub)
        if n <= 0:
            f.close()
            ptr.unsafe_free()
            raise Error("short read: " + path)
        got += n
    f.close()
    return (ptr, sz)


def is_ws(c: Byte) -> Bool:
    return c == 32 or c == 9 or c == 10 or c == 13  # space \t \n \r


def skip_ws(src: Span[Byte, _], mut p: Int) raises:
    while p < src.__len__() and is_ws(src[p]):
        p += 1
    if p >= src.__len__():
        raise Error("unexpected end of JSON")


def parse_hex4(src: Span[Byte, _], p: Int) -> Int:
    var v = 0
    for k in range(4):
        var c = Int(src[p + k])
        v = v * 16
        if c >= 48 and c <= 57:  # 0-9
            v += c - 48
        elif c >= 97 and c <= 102:  # a-f
            v += c - 87
        elif c >= 65 and c <= 70:  # A-F
            v += c - 55
        else:
            v = -1
            break
    return v


def parse_string(src: Span[Byte, _], mut p: Int) raises -> String:
    # p points at the opening quote; returns the unescaped string.
    var n = src.__len__()
    p += 1
    var out = String()
    var run = p
    while p < n:
        var c = src[p]
        if c == 34:  # "
            if p > run:
                out += String(unsafe_from_utf8=src[run:p])
            p += 1
            return out
        if c == 92:  # \
            if p > run:
                out += String(unsafe_from_utf8=src[run:p])
            p += 1
            if p >= n:
                break
            var e = src[p]
            p += 1
            if e == 34 or e == 92 or e == 47:  # \" \\ \/
                out += chr(Int(e))
            elif e == 98:  # \b
                out += chr(8)
            elif e == 102:  # \f
                out += chr(12)
            elif e == 110:  # \n
                out += chr(10)
            elif e == 114:  # \r
                out += chr(13)
            elif e == 116:  # \t
                out += chr(9)
            elif e == 117:  # \uXXXX (surrogate pairs combined)
                if p + 4 > n:
                    break
                var cp = parse_hex4(src, p)
                p += 4
                if (
                    cp >= 0xD800
                    and cp <= 0xDBFF
                    and p + 6 <= n
                    and src[p] == 92
                    and src[p + 1] == 117
                ):
                    var lo = parse_hex4(src, p + 2)
                    if lo < 0xDC00 or lo > 0xDFFF:
                        raise Error("bad surrogate pair in \\u escape")
                    p += 6
                    cp = 0x10000 + (cp - 0xD800) * 0x400 + (lo - 0xDC00)
                if cp >= 0xD800 and cp <= 0xDFFF:
                    raise Error("unpaired surrogate in \\u escape")
                out += chr(cp)
            else:
                raise Error("bad escape in JSON string")
            run = p
        else:
            p += 1
    raise Error("unterminated JSON string")


def parse_int(src: Span[Byte, _], mut p: Int) -> Int:
    var neg = False
    if src[p] == 45:  # -
        neg = True
        p += 1
    var v = 0
    while p < src.__len__() and src[p] >= 48 and src[p] <= 57:
        v = v * 10 + (Int(src[p]) - 48)
        p += 1
    return -v if neg else v


def parse_float(src: Span[Byte, _], mut p: Int) -> Float64:
    var neg = False
    if src[p] == 45:  # -
        neg = True
        p += 1
    var vi = 0
    while p < src.__len__() and src[p] >= 48 and src[p] <= 57:
        vi = vi * 10 + (Int(src[p]) - 48)
        p += 1
    var v = Float64(vi)
    if p < src.__len__() and src[p] == 46:  # .
        p += 1
        var frac: Float64 = 0.0
        var scale: Float64 = 0.1
        while p < src.__len__() and src[p] >= 48 and src[p] <= 57:
            frac += Float64(Int(src[p]) - 48) * scale
            scale *= 0.1
            p += 1
        v += frac
    if p < src.__len__() and (src[p] == 101 or src[p] == 69):  # e E
        p += 1
        var eneg = False
        if src[p] == 43:  # +
            p += 1
        elif src[p] == 45:  # -
            eneg = True
            p += 1
        var ei = 0
        while p < src.__len__() and src[p] >= 48 and src[p] <= 57:
            ei = ei * 10 + (Int(src[p]) - 48)
            p += 1
        var m: Float64 = 1.0
        for _ in range(ei):
            m *= 10.0
        v = v / m if eneg else v * m
    return -v if neg else v


def skip_value(src: Span[Byte, _], mut p: Int) raises:
    skip_ws(src, p)
    var c = src[p]
    if c == 123:  # {
        p += 1
        skip_ws(src, p)
        if src[p] == 125:
            p += 1
            return
        while True:
            _ = parse_string(src, p)
            skip_ws(src, p)
            p += 1  # :
            skip_value(src, p)
            skip_ws(src, p)
            if src[p] == 44:  # ,
                p += 1
                skip_ws(src, p)
            elif src[p] == 125:
                p += 1
                return
            else:
                raise Error("bad object in skip_value")
    elif c == 91:  # [
        p += 1
        skip_ws(src, p)
        if src[p] == 93:
            p += 1
            return
        while True:
            skip_value(src, p)
            skip_ws(src, p)
            if src[p] == 44:
                p += 1
                skip_ws(src, p)
            elif src[p] == 93:
                p += 1
                return
            else:
                raise Error("bad array in skip_value")
    elif c == 34:  # "
        _ = parse_string(src, p)
    else:
        # number / true / false / null
        while (
            p < src.__len__()
            and not is_ws(src[p])
            and src[p] != 44
            and src[p] != 93
            and src[p] != 125
        ):
            p += 1


def raw_value(src: Span[Byte, _], mut p: Int) raises -> String:
    # The raw JSON text of the value at p (leading ws skipped).
    skip_ws(src, p)
    var start = p
    skip_value(src, p)
    return String(unsafe_from_utf8=src[start:p])


def parse_array_of_ints(src: Span[Byte, _], mut p: Int) raises -> List[Int]:
    var out = List[Int]()
    p += 1  # [
    skip_ws(src, p)
    if src[p] == 93:
        p += 1
        return out^
    while True:
        out.append(parse_int(src, p))
        skip_ws(src, p)
        if src[p] == 44:  # ,
            p += 1
            skip_ws(src, p)
        elif src[p] == 93:
            p += 1
            return out^
        else:
            raise Error("bad int array")


def str_to_int(s: String) -> Int:
    var sb = s.as_bytes()
    var q = 0
    return parse_int(sb, q)


def str_to_float(s: String) -> Float64:
    var sb = s.as_bytes()
    var q = 0
    return parse_float(sb, q)


# ── safetensors (shared by the qwen3 + hf converter twins) ─────────────
# dtype codes: 0=bf16 1=f32 2=f16 3=i8 4=u8 5=i16 6=u16 7=i32 8=u32
# 9=i64 10=u64

struct STensor(Copyable):
    var name: String
    var shard: Int
    var off: Int  # absolute byte offset into the shard buffer
    var size: Int
    var dtype: Int
    var ndim: Int
    var dims: List[Int]

    def __init__(
        out self,
        name: String,
        shard: Int,
        off: Int,
        size: Int,
        dtype: Int,
        ndim: Int,
        var dims: List[Int],
    ):
        self.name = name
        self.shard = shard
        self.off = off
        self.size = size
        self.dtype = dtype
        self.ndim = ndim
        self.dims = dims^

    def __copyinit__(out self, other: Self):
        # explicit copy (List member is not implicitly copyable) so
        # Dict[String, STensor].get works
        self.name = other.name
        self.shard = other.shard
        self.off = other.off
        self.size = other.size
        self.dtype = other.dtype
        self.ndim = other.ndim
        self.dims = other.dims.copy()


def dtype_code(dt: String, tensor_name: String) raises -> Int:
    # returns the dtype code or raises with the .py-compatible message
    if dt == "BF16":
        return 0
    if dt == "F32":
        return 1
    if dt == "F16":
        return 2
    if dt == "I8":
        return 3
    if dt == "U8":
        return 4
    if dt == "I16":
        return 5
    if dt == "U16":
        return 6
    if dt == "I32":
        return 7
    if dt == "U32":
        return 8
    if dt == "I64":
        return 9
    if dt == "U64":
        return 10
    raise Error("unhandled dtype " + dt + " for " + tensor_name)


def u64le_at(src: Span[Byte, _], i: Int) -> Int:
    var v = 0
    for k in range(8):
        v |= Int(src[i + k]) << (8 * k)
    return v


def parse_safetensors(
    src: Span[Byte, _],
    shard_idx: Int,
    mut out: List[STensor],
) raises:
    # Parse one safetensors shard (u64 header length, JSON header, data
    # blob). Every tensor is appended; callers filter/quantize as needed.
    var hlen = u64le_at(src, 0)
    var data_start = 8 + hlen
    var p = 8
    skip_ws(src, p)
    if src[p] != 123:  # {
        raise Error("bad safetensors header")
    p += 1
    skip_ws(src, p)
    if src[p] == 125:
        return
    while True:
        var name = parse_string(src, p)
        skip_ws(src, p)
        p += 1  # :
        skip_ws(src, p)
        if name == "__metadata__":
            skip_value(src, p)
        else:
            p += 1  # {
            var dtype = -1
            var shape = List[Int]()
            var o0 = 0
            var o1 = 0
            skip_ws(src, p)
            if src[p] != 125:
                while True:
                    var k = parse_string(src, p)
                    skip_ws(src, p)
                    p += 1  # :
                    skip_ws(src, p)
                    if k == "dtype":
                        var dt = parse_string(src, p)
                        dtype = dtype_code(dt, name)
                    elif k == "shape":
                        shape = parse_array_of_ints(src, p)
                    elif k == "data_offsets":
                        var ds = parse_array_of_ints(src, p)
                        o0 = ds[0]
                        o1 = ds[1]
                    else:
                        skip_value(src, p)
                    skip_ws(src, p)
                    if src[p] == 44:  # ,
                        p += 1
                        skip_ws(src, p)
                    elif src[p] == 125:
                        p += 1
                        break
                    else:
                        raise Error("bad tensor entry " + name)
            if dtype < 0:
                raise Error("missing dtype for " + name)
            out.append(
                STensor(name, shard_idx, data_start + o0, o1 - o0, dtype, shape.__len__(), shape^)
            )
        skip_ws(src, p)
        if src[p] == 44:  # ,
            p += 1
            skip_ws(src, p)
        elif src[p] == 125:
            p += 1
            return
        else:
            raise Error("bad safetensors object")


def elem_f32(src: Span[Byte, _], off: Int, dtype: Int, i: Int) -> Float32:
    # Element i of a tensor as float32 (numpy astype(f32) semantics:
    # exact for bf16/f16/f32, widening for ints, RN rounding for i64/u64
    # beyond 2^24).
    if dtype == 0:  # BF16
        var u = UInt32(src[off + 2 * i]) | (UInt32(src[off + 2 * i + 1]) << 8)
        return bitcast[DType.float32, 1](u << 16)
    if dtype == 1:  # F32
        var u = (
            UInt32(src[off + 4 * i])
            | (UInt32(src[off + 4 * i + 1]) << 8)
            | (UInt32(src[off + 4 * i + 2]) << 16)
            | (UInt32(src[off + 4 * i + 3]) << 24)
        )
        return bitcast[DType.float32, 1](u)
    if dtype == 2:  # F16
        var u = UInt16(src[off + 2 * i]) | (UInt16(src[off + 2 * i + 1]) << 8)
        return Float32(bitcast[DType.float16, 1](u))
    if dtype == 3:  # I8
        var b = Int(src[off + i])
        return Float32(b - 256 if b >= 128 else b)
    if dtype == 4:  # U8
        return Float32(Int(src[off + i]))
    if dtype == 5:  # I16
        var u = Int(src[off + 2 * i]) | (Int(src[off + 2 * i + 1]) << 8)
        return Float32(u - 65536 if u >= 32768 else u)
    if dtype == 6:  # U16
        var u = Int(src[off + 2 * i]) | (Int(src[off + 2 * i + 1]) << 8)
        return Float32(u)
    if dtype == 7:  # I32
        var u = (
            Int(src[off + 4 * i])
            | (Int(src[off + 4 * i + 1]) << 8)
            | (Int(src[off + 4 * i + 2]) << 16)
            | (Int(src[off + 4 * i + 3]) << 24)
        )
        return Float32(u - 4294967296 if u >= 2147483648 else u)
    if dtype == 8:  # U32
        var u = (
            Int(src[off + 4 * i])
            | (Int(src[off + 4 * i + 1]) << 8)
            | (Int(src[off + 4 * i + 2]) << 16)
            | (Int(src[off + 4 * i + 3]) << 24)
        )
        return Float32(u)
    if dtype == 9:  # I64
        var u = Int(src[off + 8 * i]) | (Int(src[off + 8 * i + 1]) << 8) | (Int(src[off + 8 * i + 2]) << 16) | (
            Int(src[off + 8 * i + 3]) << 24
        ) | (Int(src[off + 8 * i + 4]) << 32) | (Int(src[off + 8 * i + 5]) << 40) | (
            Int(src[off + 8 * i + 6]) << 48
        ) | (Int(src[off + 8 * i + 7]) << 56)
        return Float32(u)
    # U64
    var u = Int(src[off + 8 * i]) | (Int(src[off + 8 * i + 1]) << 8) | (Int(src[off + 8 * i + 2]) << 16) | (
        Int(src[off + 8 * i + 3]) << 24
    ) | (Int(src[off + 8 * i + 4]) << 32) | (Int(src[off + 8 * i + 5]) << 40) | (
        Int(src[off + 8 * i + 6]) << 48
    ) | (Int(src[off + 8 * i + 7]) << 56)
    return Float32(u)


# ── config.json (raw-value dict; text_config merge by re-invoking) ─────

def parse_cfg_object(b: Span[Byte, _], mut p: Int, mut cfg: Dict[String, String]) raises:
    # Walk an object, storing raw JSON value text per key.
    skip_ws(b, p)
    p += 1  # {
    skip_ws(b, p)
    if b[p] == 125:
        p += 1
        return
    while True:
        var key = parse_string(b, p)
        skip_ws(b, p)
        p += 1  # :
        skip_ws(b, p)
        cfg[key] = raw_value(b, p)
        skip_ws(b, p)
        if b[p] == 44:  # ,
            p += 1
            skip_ws(b, p)
        elif b[p] == 125:
            p += 1
            return
        else:
            raise Error("bad config object")


def cfg_int(cfg: Dict[String, String], key: String, default: Int) -> Int:
    var v = cfg.get(key)
    if not v:
        return default
    var raw = v.value()
    if raw == "true" or raw == "True":
        return 1
    if raw == "false" or raw == "False":
        return 0
    return str_to_int(raw)


def cfg_float(cfg: Dict[String, String], key: String, default: Float64) -> Float64:
    var v = cfg.get(key)
    if not v:
        return default
    return str_to_float(v.value())


def cfg_str(cfg: Dict[String, String], key: String, default: String) raises -> String:
    # String value: raw JSON text is quoted; strip the quotes.
    var v = cfg.get(key)
    if not v:
        return default
    var raw = v.value()
    if raw.startswith("\""):
        var rb = raw.as_bytes()
        var q = 0
        return parse_string(rb, q)
    return raw


def cfg_first_list_str(cfg: Dict[String, String], key: String, default: String) raises -> String:
    # config.get("architectures", [""])[0] — first element of a JSON array.
    var v = cfg.get(key)
    if not v:
        return default
    var raw = v.value()
    var rb = raw.as_bytes()
    var q = 0
    skip_ws(rb, q)
    if q >= rb.__len__() or rb[q] != 91:  # [
        return default
    q += 1
    skip_ws(rb, q)
    if q >= rb.__len__() or rb[q] != 34:  # "
        return default
    return parse_string(rb, q)
