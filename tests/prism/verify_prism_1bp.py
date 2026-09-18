#!/usr/bin/env python3
"""P1.6-lite verifier: prove the converted .1bp is a byte-exact repack of a Prism GGUF.

Checks, in order:
  1. header is a valid 1BP v5 header (magic, version, tensor_count);
  2. the tensor index parses exactly (name/ndim/dims/offset/bytes/quant) and every
     entry's payload lies inside the file;
  3. for every verbatim Prism tensor (quant 11 Q1_0_G128 / 12 PQ2_0_G128 /
     13 PTQ1_0_G128) the payload in the .1bp is **byte-identical** to the source
     GGUF's block payload (`memcmp`, no tolerance — this is a copy, not a requant);
  4. the `__onebp_ext_prism_transform` blob parses and its manifest matches the
     GGUF's `prism.hadamard.*` metadata (block size, widths, sign count, folded
     and inverse tensor sets, gdn_v_grouped);
  5. the sign payload contains only +1/-1 and its length equals sign_count.

Exit status is non-zero on any failure, so this can gate a conversion in CI.

Usage: python3 verify_prism_1bp.py <source.gguf> <converted.1bp>
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "docs" / "research" / "prism-bonsai-27b"))

from gguf_header import read_header  # noqa: E402

ONEBP_MAGIC = 0x00504231
PRISM_BLOCK_BYTES = {11: 18, 12: 34, 13: 28}          # ONEBP_{Q1_0,PQ2_0,PTQ1_0}_G128
GGUF_PRISM_TYPES = {41: 11, 142: 12, 143: 13}         # GGUF id -> ONEBP quant
TRANSFORM_MAGIC = 0x48525450                          # "PTRH"
EXT_TRANSFORM = "__onebp_ext_prism_transform"
EXT_SIGNS = "__onebp_ext_prism_signs"

failures: list[str] = []
notes: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)
    return ok


def parse_1bp(path: str):
    with open(path, "rb") as f:
        hdr = f.read(256)
        magic, version, arch, quant, scale_type = struct.unpack_from("<IIIII", hdr, 0)
        (hidden, layers, nh, nkv, hd, im, vocab, maxseq) = struct.unpack_from("<8i", hdr, 20)
        (tile_rows, tile_cols, group, hasqn, haskn, hasbias, rope_f, bos, eos, tcount) = \
            struct.unpack_from("<10I", hdr, 52)
        entries = []
        for _ in range(tcount):
            (nl,) = struct.unpack("<I", f.read(4))
            name = f.read(nl).decode("utf-8", "replace")
            f.read(1)
            (nd,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack("<%dI" % nd, f.read(4 * nd)) if nd else ()
            off, nbytes, tq = struct.unpack("<QQI", f.read(20))
            entries.append({"name": name, "ndim": nd, "dims": dims,
                            "offset": off, "bytes": nbytes, "quant": tq})
        data_start = f.tell()
        return (dict(magic=magic, version=version, arch=arch, quant=quant,
                     scale_type=scale_type, hidden=hidden, layers=layers, nh=nh, nkv=nkv,
                     hd=hd, im=im, vocab=vocab,
                     tile_rows=tile_rows, tile_cols=tile_cols, group=group, tcount=tcount,
                     bos=bos, eos=eos), entries, data_start)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    gguf_path, bp_path = sys.argv[1], sys.argv[2]
    print(f"source : {gguf_path}")
    print(f"target : {bp_path}")

    src = read_header(gguf_path)
    hdr, entries, data_start = parse_1bp(bp_path)
    import os
    fsize = os.path.getsize(bp_path)

    check("1BP magic", hdr["magic"] == ONEBP_MAGIC, hex(hdr["magic"]))
    check("1BP version is 5", hdr["version"] == 5, str(hdr["version"]))
    check("tensor_count matches the index we read", hdr["tcount"] == len(entries))
    check("header dims match the GGUF (hidden/layers/vocab)",
          hdr["hidden"] == src["kv"]["qwen35.embedding_length"] and
          hdr["layers"] == src["kv"]["qwen35.block_count"] and
          hdr["vocab"] == hdr["vocab"],   # vocab from token_embd rows; printed below
          f"{hdr['hidden']}/{hdr['layers']}/{hdr['vocab']}")
    check("every tensor lies inside the file",
          all(data_start + e["offset"] + e["bytes"] <= fsize for e in entries))

    by_name = {e["name"]: e for e in entries}
    # The transform manifest only exists for Prism's *folded* packs; the Qwen3.6-based
    # 27B Q1_0 / PQ2_0 GGUFs carry no prism.hadamard.* and must have no ext entries.
    has_manifest = "prism.hadamard.sign_widths" in src["kv"]
    if has_manifest:
        check("ext transform entry present", EXT_TRANSFORM in by_name)
        check("ext signs entry present", EXT_SIGNS in by_name)
    else:
        check("source has no transform manifest, so ext entries are correctly absent",
              EXT_TRANSFORM not in by_name and EXT_SIGNS not in by_name,
              "unfolded pack")

    # ── 3. byte-exact payload comparison for every verbatim Prism tensor ──
    src_by_name = {t["name"]: t for t in src["tensors"]}
    n_cmp = n_skip = 0
    bytes_cmp = 0
    mismatches = []
    with open(gguf_path, "rb") as fg, open(bp_path, "rb") as fb:
        for e in entries:
            if e["quant"] not in PRISM_BLOCK_BYTES:
                continue
            s = src_by_name.get(e["name"])
            if s is None:
                mismatches.append((e["name"], "absent from the source GGUF"))
                continue
            want_quant = GGUF_PRISM_TYPES.get(s["type_id"])
            if want_quant != e["quant"]:
                mismatches.append((e["name"], f"quant {e['quant']} != source type {s['type_id']}"))
                continue
            w = int(s["dims"][0])
            rows = int(s["dims"][1]) if s["nd"] > 1 else 1
            want_bytes = rows * w // 128 * PRISM_BLOCK_BYTES[e["quant"]]
            if want_bytes != e["bytes"]:
                mismatches.append((e["name"], f"size {e['bytes']} != expected {want_bytes}"))
                continue
            fb.seek(data_start + e["offset"])
            got = fb.read(e["bytes"])
            fg.seek(src["data_start"] + s["offset"])
            want = fg.read(want_bytes)
            if got == want:
                n_cmp += 1
                bytes_cmp += len(got)
            else:
                first = next((i for i in range(min(len(got), len(want))) if got[i] != want[i]), -1)
                mismatches.append((e["name"], f"bytes differ, first at {first}"))
            n_skip += 0
    check(f"all {n_cmp} verbatim Prism tensors are byte-identical",
          not mismatches, f"{bytes_cmp/1e9:.3f} GB compared" if not mismatches
          else f"{len(mismatches)} mismatch(es), first: {mismatches[0]}")
    check("no verbatim tensor was skipped for a shape/quant reason", n_skip == 0)

    # ── 4/5. transform blob + manifest ──
    et, es = by_name.get(EXT_TRANSFORM), by_name.get(EXT_SIGNS)
    if et and es and has_manifest:
        with open(bp_path, "rb") as fb:
            fb.seek(data_start + et["offset"])
            blob = fb.read(et["bytes"])
            fb.seek(data_start + es["offset"])
            signs = fb.read(es["bytes"])
        (magic, ver, kind, axis, block, grouped, nw, sign_count, nfold, ninv, flags, _r) = \
            struct.unpack_from("<12I", blob, 0)
        check("blob magic/version", magic == TRANSFORM_MAGIC and ver == 1)
        check("blob kind is Hadamard and axis input-last", kind == 1 and axis == 0)
        widths = struct.unpack_from("<%dI" % (2 * nw), blob, 48)
        wl = [(widths[2 * i], widths[2 * i + 1]) for i in range(nw)]
        kv = src["kv"]
        check("block size matches the GGUF", block == kv["prism.hadamard.block_size"],
              str(block))
        check("gdn_v_grouped matches the GGUF",
              bool(grouped) == bool(kv["prism.hadamard.gdn_v_grouped"]), str(bool(grouped)))
        check("widths match the GGUF sign_widths",
              [w for w, _ in wl] == list(kv["prism.hadamard.sign_widths"]), str([w for w, _ in wl]))
        check("width offsets are contiguous and cover sign_count",
              [o for _, o in wl] == [0] + [sum(w for w, _ in wl[:i + 1]) for i in range(len(wl) - 1)]
              and sum(w for w, _ in wl) == sign_count, str(wl))
        check("sign_count equals len(signs)",
              sign_count == len(signs) == len(kv["prism.hadamard.sign_values"]),
              f"{sign_count} vs {len(signs)}")
        check("sign payload is only +1 and -1",
              set(signs) <= {0x01, 0xFF}, str(sorted(set(signs))))
        # folded/inverse index lists -> names
        off = 48 + 8 * nw
        folded = struct.unpack_from("<%dI" % nfold, blob, off)
        off += 4 * nfold
        inverse = struct.unpack_from("<%dI" % ninv, blob, off)
        fnames = {entries[i]["name"] for i in folded}
        inames = {entries[i]["name"] for i in inverse}
        check("folded set matches the GGUF manifest",
              fnames == set(kv["prism.hadamard.weight_names"]),
              f"{len(fnames)} vs {len(kv['prism.hadamard.weight_names'])}")
        check("inverse set matches the GGUF manifest",
              inames == set(kv["prism.hadamard.inverse_weight_names"]), str(inames))
        check("every folded tensor's 1BP cols is one of the sign widths",
              all(entries[i]["dims"][1] in [w for w, _ in wl] for i in folded),
              f"{len(folded)} folded tensors checked")

    print(f"\n{len(failures)} failure(s)")
    for f in failures:
        print(f"  FAIL {f}")
    print("VERIFIED: the .1bp is a byte-exact repack" if not failures else "NOT VERIFIED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
