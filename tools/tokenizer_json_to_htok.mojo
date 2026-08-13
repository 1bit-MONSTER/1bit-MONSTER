# tools/tokenizer_json_to_htok.mojo — Mojo 1.0 twin of tokenizer_json_to_htok.py
# (fold P2.2: Python tooling -> Mojo, same CLI contract, byte-identical output).
#
# Convert a HuggingFace tokenizer.json to the .htok v2 binary read by
# rcpp_tokenizer_load (src/tokenizer.cpp). The Python original is deleted;
# this is the through-line converter (convert_model.py safetensors route).
#
#   layout: "HTOK" u32 version(2) u32 vocab_size u32 num_merges u32 bos u32 eos
#           vocab[].: u16 len + bytes (id = position; empty bytes = unused id)
#           merges[]: u32 a u32 b u32 merged (rank = insertion order)
#           u32 num_special + u32 special_ids[]
#
# Usage:
#   tokenizer_json_to_htok tokenizer.json out.htok
# Build:
#   mojo build tools/tokenizer_json_to_htok.mojo -o build/tokenizer_json_to_htok

from std.collections import Dict, List
from std.sys import argv


def b_u16(mut buf: List[UInt8], v: Int):
    buf.append(UInt8(v & 0xFF))
    buf.append(UInt8((v >> 8) & 0xFF))


def b_u32(mut buf: List[UInt8], v: Int):
    buf.append(UInt8(v & 0xFF))
    buf.append(UInt8((v >> 8) & 0xFF))
    buf.append(UInt8((v >> 16) & 0xFF))
    buf.append(UInt8((v >> 24) & 0xFF))


def b_str(mut buf: List[UInt8], s: String):
    var sb = s.as_bytes()
    for i in range(sb.__len__()):
        buf.append(sb[i])


# ── JSON scanner (tokenizer.json is fixed-shape: we only materialize the
#    vocab dict, the merges array and added_tokens; everything else is
#    skipped syntactically). ─────────────────────────────────────────────

def is_ws(c: Byte) -> Bool:
    return c == 32 or c == 9 or c == 10 or c == 13  # space \t \n \r


def skip_ws(b: Span[Byte, _], mut p: Int) raises:
    while p < b.__len__() and is_ws(b[p]):
        p += 1
    if p >= b.__len__():
        raise Error("unexpected end of JSON")


def parse_hex4(b: Span[Byte, _], p: Int) -> Int:
    var v = 0
    for k in range(4):
        var c = Int(b[p + k])
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


def parse_string(s: String, b: Span[Byte, _], mut p: Int) raises -> String:
    # p points at the opening quote; returns the unescaped string.
    var n = b.__len__()
    p += 1
    var out = String()
    var run = p
    while p < n:
        var c = b[p]
        if c == 34:  # "
            if p > run:
                out += s[byte=run:p]
            p += 1
            return out
        if c == 92:  # \
            if p > run:
                out += s[byte=run:p]
            p += 1
            if p >= n:
                break
            var e = b[p]
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
                var cp = parse_hex4(b, p)
                p += 4
                if (
                    cp >= 0xD800
                    and cp <= 0xDBFF
                    and p + 6 <= n
                    and b[p] == 92
                    and b[p + 1] == 117
                ):
                    var lo = parse_hex4(b, p + 2)
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


def parse_int(b: Span[Byte, _], mut p: Int) -> Int:
    var neg = False
    if b[p] == 45:  # -
        neg = True
        p += 1
    var v = 0
    while p < b.__len__() and b[p] >= 48 and b[p] <= 57:
        v = v * 10 + (Int(b[p]) - 48)
        p += 1
    return -v if neg else v


def skip_value(s: String, b: Span[Byte, _], mut p: Int) raises:
    skip_ws(b, p)
    var c = b[p]
    if c == 123:  # {
        p += 1
        skip_ws(b, p)
        if b[p] == 125:
            p += 1
            return
        while True:
            _ = parse_string(s, b, p)
            skip_ws(b, p)
            p += 1  # :
            skip_value(s, b, p)
            skip_ws(b, p)
            if b[p] == 44:  # ,
                p += 1
                skip_ws(b, p)
            elif b[p] == 125:
                p += 1
                return
            else:
                raise Error("bad object in skip_value")
    elif c == 91:  # [
        p += 1
        skip_ws(b, p)
        if b[p] == 93:
            p += 1
            return
        while True:
            skip_value(s, b, p)
            skip_ws(b, p)
            if b[p] == 44:
                p += 1
                skip_ws(b, p)
            elif b[p] == 93:
                p += 1
                return
            else:
                raise Error("bad array in skip_value")
    elif c == 34:  # "
        _ = parse_string(s, b, p)
    else:
        # number / true / false / null
        while p < b.__len__() and not is_ws(b[p]) and b[p] != 44 and b[p] != 93 and b[p] != 125:
            p += 1


def parse_vocab(s: String, b: Span[Byte, _], mut p: Int, mut vocab: Dict[String, Int]) raises:
    p += 1  # {
    skip_ws(b, p)
    if b[p] == 125:
        p += 1
        return
    while True:
        var key = parse_string(s, b, p)
        skip_ws(b, p)
        p += 1  # :
        skip_ws(b, p)
        var id = parse_int(b, p)
        vocab[key] = id
        skip_ws(b, p)
        if b[p] == 44:  # ,
            p += 1
            skip_ws(b, p)
        elif b[p] == 125:
            p += 1
            return
        else:
            raise Error("bad vocab entry")


def split_pair(v: String) raises -> Tuple[String, String]:
    # Python str.split() semantics (whitespace runs) on a "a b" merge entry.
    var vb = v.as_bytes()
    var n = vb.__len__()
    var i = 0
    while i < n and is_ws(vb[i]):
        i += 1
    var a_start = i
    while i < n and not is_ws(vb[i]):
        i += 1
    var a = String(unsafe_from_utf8=v[byte=a_start:i].as_bytes())
    while i < n and is_ws(vb[i]):
        i += 1
    var b_start = i
    while i < n and not is_ws(vb[i]):
        i += 1
    var b = String(unsafe_from_utf8=v[byte=b_start:i].as_bytes())
    var j = i
    while j < n and is_ws(vb[j]):
        j += 1
    if j < n:
        raise Error("merge entry has more than 2 components: '" + v + "'")
    return (a, b)


def parse_merges(
    s: String,
    b: Span[Byte, _],
    mut p: Int,
    mut merges: List[Tuple[String, String]],
) raises:
    p += 1  # [
    skip_ws(b, p)
    if b[p] == 93:
        p += 1
        return
    while True:
        skip_ws(b, p)
        if b[p] == 34:  # string form "a b"
            merges.append(split_pair(parse_string(s, b, p)))
        elif b[p] == 91:  # list form ["a", "b"] (Qwen3+)
            p += 1
            skip_ws(b, p)
            var a = parse_string(s, b, p)
            skip_ws(b, p)
            if b[p] == 44:  # ,
                p += 1
                skip_ws(b, p)
            var bb = parse_string(s, b, p)
            skip_ws(b, p)
            while b[p] == 44:  # Python keeps only [0] and [1]
                p += 1
                skip_ws(b, p)
                skip_value(s, b, p)
                skip_ws(b, p)
            if b[p] != 93:
                raise Error("bad merge list")
            p += 1
            merges.append((a, bb))
        else:
            raise Error("bad merge entry")
        skip_ws(b, p)
        if b[p] == 44:  # ,
            p += 1
        elif b[p] == 93:
            p += 1
            return
        else:
            raise Error("bad merges array")


def parse_added(
    s: String,
    b: Span[Byte, _],
    mut p: Int,
    mut added: Dict[String, Int],
) raises:
    p += 1  # [
    skip_ws(b, p)
    if b[p] == 93:
        p += 1
        return
    while True:
        skip_ws(b, p)
        p += 1  # {
        var content = String()
        var id = -1
        skip_ws(b, p)
        if b[p] != 125:
            while True:
                var key = parse_string(s, b, p)
                skip_ws(b, p)
                p += 1  # :
                skip_ws(b, p)
                if key == "content":
                    content = parse_string(s, b, p)
                elif key == "id":
                    id = parse_int(b, p)
                else:
                    skip_value(s, b, p)
                skip_ws(b, p)
                if b[p] == 44:  # ,
                    p += 1
                    skip_ws(b, p)
                elif b[p] == 125:
                    p += 1
                    break
                else:
                    raise Error("bad added_tokens entry")
        if id >= 0:
            added[content] = id
        skip_ws(b, p)
        if b[p] == 44:
            p += 1
        elif b[p] == 93:
            p += 1
            return
        else:
            raise Error("bad added_tokens array")


def parse_model(
    s: String,
    b: Span[Byte, _],
    mut p: Int,
    mut vocab: Dict[String, Int],
    mut merges: List[Tuple[String, String]],
) raises:
    p += 1  # {
    skip_ws(b, p)
    if b[p] == 125:
        return
    while True:
        var key = parse_string(s, b, p)
        skip_ws(b, p)
        p += 1  # :
        skip_ws(b, p)
        if key == "vocab":
            parse_vocab(s, b, p, vocab)
        elif key == "merges":
            parse_merges(s, b, p, merges)
        else:
            skip_value(s, b, p)
        skip_ws(b, p)
        if b[p] == 44:  # ,
            p += 1
            skip_ws(b, p)
        elif b[p] == 125:
            p += 1
            return
        else:
            raise Error("bad model entry")


def parse_tokenizer(
    s: String,
    b: Span[Byte, _],
    mut p: Int,
    mut vocab: Dict[String, Int],
    mut merges: List[Tuple[String, String]],
    mut added: Dict[String, Int],
) raises:
    skip_ws(b, p)
    if b[p] != 123:  # {
        raise Error("not a JSON object")
    p += 1
    skip_ws(b, p)
    if b[p] == 125:
        return
    while True:
        var key = parse_string(s, b, p)
        skip_ws(b, p)
        if b[p] != 58:  # :
            raise Error("expected ':' after key")
        p += 1
        skip_ws(b, p)
        if key == "model":
            parse_model(s, b, p, vocab, merges)
        elif key == "added_tokens":
            parse_added(s, b, p, added)
        else:
            skip_value(s, b, p)
        skip_ws(b, p)
        if b[p] == 44:  # ,
            p += 1
            skip_ws(b, p)
        elif b[p] == 125:
            p += 1
            return
        else:
            raise Error("bad top-level separator")


def main() raises:
    var args = argv()
    if len(args) != 3:
        print("usage: tokenizer_json_to_htok tokenizer.json out.htok")
        return
    var in_path = String(args[1])
    var out_path = String(args[2])

    var f = open(in_path, "r")
    var s = f.read()
    f.close()
    var b = s.as_bytes()

    var vocab = Dict[String, Int](capacity=300000)
    var merges = List[Tuple[String, String]]()
    var added = Dict[String, Int](capacity=512)
    var p = 0
    parse_tokenizer(s, b, p, vocab, merges, added)

    # id -> string, size = max id + 1; added_tokens override vocab entries
    # (same dict-overwrite order as the Python original).
    var max_id = -1
    for e in vocab.items():
        if e.value > max_id:
            max_id = e.value
    for e in added.items():
        if e.value > max_id:
            max_id = e.value
    var vocab_size = max_id + 1
    var id_to_tok = List[String]()
    id_to_tok.resize(vocab_size, "")
    for e in vocab.items():
        id_to_tok[e.value] = e.key
    for e in added.items():
        id_to_tok[e.value] = e.key

    # merges -> (a, b, merged) triples, ids resolved via vocab then added.
    var triples = List[Tuple[Int, Int, Int]]()
    for m in merges:
        var a = m[0]
        var bb = m[1]
        var merged = a + bb
        var ia = vocab.get(a)
        if not ia:
            ia = added.get(a)
        if not ia:
            raise Error("error: merge component '" + a + "' not in vocab")
        var ib = vocab.get(bb)
        if not ib:
            ib = added.get(bb)
        if not ib:
            raise Error("error: merge component '" + bb + "' not in vocab")
        var im = vocab.get(merged)
        if not im:
            im = added.get(merged)
        if not im:
            raise Error("error: merge component '" + merged + "' not in vocab")
        triples.append((ia.value(), ib.value(), im.value()))

    # bos/eos: same defaults as the Python original.
    var bos = 0
    var bv = vocab.get("<|endoftext|>")
    if not bv:
        bv = added.get("<|endoftext|>")
    if bv:
        bos = bv.value()
    var eos = bos
    var ev = vocab.get("<|im_end|>")
    if ev:
        eos = ev.value()
    else:
        var ae = added.get("<|im_end|>")
        if ae:
            eos = ae.value()

    # special_ids = all added-token ids, ascending (small n: insertion sort).
    var special_ids = List[Int]()
    for e in added.items():
        special_ids.append(e.value)
    for i in range(1, special_ids.__len__()):
        var v = special_ids[i]
        var j = i - 1
        while j >= 0 and special_ids[j] > v:
            special_ids[j + 1] = special_ids[j]
            j -= 1
        special_ids[j + 1] = v

    var buf = List[UInt8](capacity=4000000)
    b_str(buf, "HTOK")
    b_u32(buf, 2)
    b_u32(buf, vocab_size)
    b_u32(buf, triples.__len__())
    b_u32(buf, bos)
    b_u32(buf, eos)
    for t in id_to_tok:
        b_u16(buf, t.byte_length())
        b_str(buf, t)
    for tp in triples:
        b_u32(buf, tp[0])
        b_u32(buf, tp[1])
        b_u32(buf, tp[2])
    b_u32(buf, special_ids.__len__())
    for sid in special_ids:
        b_u32(buf, sid)

    var n = buf.__len__()
    var data = buf.unsafe_take_allocation()
    var ptr = data^.unsafe_leak()
    var span = Span[UInt8, MutUntrackedOrigin](unsafe_ptr=ptr, length=n)
    var g = open(out_path, "w")
    g.write_bytes(span)
    g.close()
    ptr.unsafe_free()
    print(
        "wrote",
        out_path,
        ": vocab=",
        vocab_size,
        " merges=",
        triples.__len__(),
        " specials=",
        special_ids.__len__(),
        " bos=",
        bos,
        " eos=",
        eos,
    )
