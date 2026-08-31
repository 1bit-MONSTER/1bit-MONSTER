#!/usr/bin/env python3
"""annotate_hrx_embedding_quants.py — per-entry token-embedding quant for the
`*-HRX` registry entries (issues #1943/#1955).

The HRX runtime fail-closes at GET_ROWS for non-K-quant token embeddings
(q5_0/q8_0/IQ2XXS/Q4_K_S verified), so each `*-HRX` entry is only genuinely
useful if its GGUF checkpoint has a K-quant `token_embd.weight`. This tool
reads JUST the GGUF header (HTTP range request, ~first tens of KB — no
multi-GB download) and prints each entry's embedding dtype + a serve-verdict.

Usage:
    python3 third_party/lemonade/tools/annotate_hrx_embedding_quants.py
        [--limit N]          # only first N entries (quick sanity)
        [--write]            # write the annotation into server_models.json
                              # (adds "hrx_token_embd" + "hrx_serve" + "hrx_embd_w"
                              #  per entry)

Requires: gguf package (pip install gguf) or the bundled GGUF reader.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

REGISTRY = Path(__file__).resolve().parent.parent / "src" / "cpp" / "resources" / "server_models.json"
GGUF_FTYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "IQ1_M", 25: "BF16",
}

# HRX GET_ROWS serve set — gate is PROVABLE from llama.cpp PR #27218
# (ggml-hrx dispatch-qwen-preamble.cpp, match_qwen_token_embedding):
#   weight->type == Q4_K
#   is_supported_hidden_size(hidden_size)  -> hidden_size == 2048
#   is_supported_vocabulary_count(vocab)   -> vocab <= 262144
#   is_supported_token_count(token_count)  -> token_count <= 2048
# Anything else (q6_K/q5_x/q8_0/Q4_0/Q4_1/F16/MXFP4, or q4_K at w != 2048)
# fails closed at the scheduler: "unsupported HRX node 0: GET_ROWS".
# Empirically confirmed on strixhalo (2026-08-30): Qwen3-30B-A3B-Instruct-2507
# (q4_K w=2048) decodes 36.31 tok/s; MiniCPM5-1B (q4_K w=1536) and
# MiniCPM4-8B (q4_K w=4096) both fail at GET_ROWS.
K_QUANTS = {"Q4_K"}
HRX_GET_ROWS_HIDDEN = 2048


_HF_FILE_CACHE: dict[str, list[str]] = {}


def hf_url(checkpoint: str) -> str | None:
    """checkpoint field: 'owner/repo:file-or-quant-tag' → raw resolve URL.

    The tag may be a literal filename (e.g. 'Foo-Q4_K_M.gguf') or a bare
    quant tag (e.g. 'Q4_0') that must be resolved to a real filename via the
    HF API (the registry uses bare tags like 'Q4_0'/'Q1_0').
    """
    if ":" not in checkpoint:
        return None
    repo, _, tag = checkpoint.rpartition(":")
    if tag.endswith(".gguf"):
        return f"https://huggingface.co/{repo}/resolve/main/{tag}"
    # bare quant tag → find the matching file in the repo
    if repo not in _HF_FILE_CACHE:
        try:
            req = urllib.request.Request(
                f"https://huggingface.co/api/models/{repo}",
                headers={"User-Agent": "1bit-hrx-annotate/1.0"},
            )
            with urllib.request.urlopen(req, timeout=25) as r:
                data = json.load(r)
            _HF_FILE_CACHE[repo] = [s.get("rfilename", "")
                                    for s in data.get("siblings", [])]
        except Exception:  # noqa: BLE001
            _HF_FILE_CACHE[repo] = []
    for fn in _HF_FILE_CACHE[repo]:
        if fn.endswith(".gguf") and tag in fn:
            return f"https://huggingface.co/{repo}/resolve/main/{fn}"
    return None


def fetch_header(url: str, size: int = 1048576) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{size - 1}"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return r.read()


def read_gguf_ftype(data: bytes) -> tuple[str | None, int]:
    """Minimal GGUF metadata walk: return (token_embd.weight dtype, n_kv)."""
    try:
        if data[:4] != b"GGUF":
            return None, 0
        import struct
        off = 4
        (ver,) = struct.unpack_from("<I", data, off); off += 4
        (n_tensors,) = struct.unpack_from("<Q", data, off); off += 8
        (n_kv,) = struct.unpack_from("<Q", data, off); off += 8

        def read_str(off):
            if off + 8 > len(data):
                raise IndexError("read_str OOB")
            (n,) = struct.unpack_from("<Q", data, off)
            off += 8
            if n > len(data) - off:
                raise IndexError(f"read_str length {n} exceeds buffer")
            return data[off:off + n].decode("utf-8", "replace"), off + n

        for _ in range(n_kv):
            key, off = read_str(off)
            (typ,) = struct.unpack_from("<I", data, off); off += 4
            if typ == 0:  # uint8
                off += 1
            elif typ == 1:  # int8
                off += 1
            elif typ == 2:  # uint16
                off += 2
            elif typ == 3:  # int16
                off += 2
            elif typ == 4:  # uint32
                off += 4
            elif typ == 5:  # int32
                off += 4
            elif typ == 6:  # float32
                off += 4
            elif typ == 7:  # bool
                off += 1
            elif typ == 8:  # string
                _, off = read_str(off)
            elif typ == 9:  # array
                (atype,) = struct.unpack_from("<I", data, off); off += 4
                (alen,) = struct.unpack_from("<Q", data, off); off += 8
                for _ in range(alen):
                    if atype == 8:
                        _, off = read_str(off)
                    elif atype in (0, 1, 7):
                        off += 1
                    elif atype in (2, 3):
                        off += 2
                    elif atype in (4, 5, 6):
                        off += 4
                    elif atype in (10, 11):
                        off += 8
                    elif atype == 12:
                        off += 8
                    else:
                        return None, n_kv
            elif typ == 10:  # uint64
                off += 8
            elif typ == 11:  # int64
                off += 8
            elif typ == 12:  # float64
                off += 8
            else:
                return None, n_kv
        # now tensors: name(string), n_dims(u32), dims(u64 x n_dims),
        # ftype(u32), offset_to_data(u64) — NO alignment padding between
        # tensor-info fields (gguf GGUFReader._get_tensor_info_field).
        for _ in range(n_tensors):
            name, off = read_str(off)
            (n_dims,) = struct.unpack_from("<I", data, off); off += 4
            dims = struct.unpack_from(f"<{n_dims}Q", data, off)
            off += 8 * n_dims
            (ftype,) = struct.unpack_from("<I", data, off); off += 4
            off += 8  # offset_to_data
            if name == "token_embd.weight":
                return GGUF_FTYPE_NAMES.get(ftype, f"UNK({ftype})"), dims, n_kv
        return None, (), n_kv
    except (struct.error, UnicodeDecodeError, IndexError, OverflowError):
        return None, (), 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    registry = json.loads(REGISTRY.read_text())
    hrx = [(k, v) for k, v in registry.items() if k.endswith("-HRX")]
    if args.limit:
        hrx = hrx[: args.limit]

    verdicts = {}
    print(f"{'entry':45s} {'token_embd':10s} {'serve-on-HRX':12s}")
    print("-" * 75)
    for name, entry in hrx:
        cp = entry.get("checkpoint", "")
        url = hf_url(cp)
        if not url:
            verdicts[name] = ("?", "no-url")
            print(f"{name:45s} {'?':10s} {'no-url':12s}")
            continue
        try:
            # GGUF metadata (kv + tensor names) can be several MB when the
            # tokenizer arrays are large — fetch in growing chunks until the
            # walk finds token_embd.weight or we give up at 64 MB.
            ftype, dims = None, ()
            for size in (1 << 20, 4 << 20, 16 << 20, 64 << 20):
                data = fetch_header(url, size)
                if data[:4] != b"GGUF":
                    break
                ftype, dims, _ = read_gguf_ftype(data)
                if ftype:
                    break
            if not ftype:
                verdicts[name] = ("?", "metadata-too-large")
                print(f"{name:45s} {'?':10s} {'metadata-too-large':12s}")
                continue
            # Issue #1944 (shape refinement): the GET_ROWS ceiling is NOT
            # purely quant-gated — PR #27218's qwen_token_embedding dispatch
            # hard-codes hidden_size == 2048. So q4_K is necessary but not
            # sufficient: q4_K at w != 2048 fails closed at GET_ROWS
            # (MiniCPM5-1B w=1536, MiniCPM4-8B w=4096 — both measured on
            # strixhalo 2026-08-30). Verified end-to-end on strixhalo:
            #   30B-A3B-Instruct-2507  (q4_K w=2048) — 36.31 tok/s
            #   Coder-30B-A3B-Instruct (q4_K w=2048) — 71.16 tok/s
            #   GLM-4.7-Flash          (q4_K w=2048) — GET_ROWS gate OK,
            #     then fails at RMS_NORM after the embedding (non-Qwen
            #     graph; the HRX bundle only ships the Qwen3-MoE corpus)
            #   Qwen3-Next-80B-A3B     (q4_K w=2048) — load-time abort on
            #     SCALE (cross-layer attention, also outside the corpus)
            # So the full serve condition is q4_K AND w==2048 AND a
            # Qwen3-MoE-shaped graph. GGUF dims are reversed, so the
            # embedding width is the MIN of the two leading dims.
            embd_w = int(min(dims[:2])) if len(dims) >= 2 else int(dims[0] or 0)
            if ftype in K_QUANTS and embd_w == HRX_GET_ROWS_HIDDEN:
                if name in ("Qwen3-30B-A3B-Instruct-2507-HRX", "Qwen3-Coder-30B-A3B-Instruct-HRX"):
                    serve = "YES"   # empirically verified (36.31 / 71.16 tok/s)
                elif name == "GLM-4.7-Flash-HRX":
                    serve = "NO (GET_ROWS gate ok, but RMS_NORM after embd unsupported — non-Qwen graph)"
                elif name == "Qwen3-Next-80B-A3B-Instruct-HRX":
                    serve = "NO (load-time abort: SCALE / cross-layer attention unsupported — non-Qwen3-MoE graph)"
                else:
                    serve = "SUSPECT (q4_K w=2048, GET_ROWS gate ok, unverified e2e)"
            elif ftype in K_QUANTS:
                serve = "NO (GET_ROWS: hidden_size != 2048)"
            else:
                serve = "NO (GET_ROWS)"
            verdicts[name] = (ftype, serve, embd_w)
            print(f"{name:45s} {ftype:10s} {serve:24s} w={embd_w}")
        except Exception as exc:  # noqa: BLE001
            verdicts[name] = ("ERR", str(exc)[:40], 0)
            print(f"{name:45s} {'ERR':10s} {str(exc)[:40]:12s}")

    n_yes = sum(1 for v in verdicts.values() if v[1] == "YES")
    n_susp = sum(1 for v in verdicts.values() if v[1] and v[1].startswith("SUSPECT"))
    n_no = sum(1 for v in verdicts.values() if v[1] and v[1].startswith("NO"))
    print(f"\n{len(verdicts)} entries: {n_yes} verified-serve, {n_susp} SUSPECT (q4_K w=2048 unverified e2e), "
          f"{n_no} fail-closed, {len(verdicts) - n_yes - n_susp - n_no} unknown/error")

    if args.write:
        for name, (ftype, serve, embd_w) in verdicts.items():
            if name in registry and ftype not in ("?", "ERR"):
                registry[name]["hrx_token_embd"] = ftype
                registry[name]["hrx_serve"] = serve
                if embd_w:
                    registry[name]["hrx_embd_w"] = embd_w
        # Match the registry's original formatting exactly (indent=4, no
        # sort_keys) so the diff is surgical — only the new keys per
        # entry are added, nothing re-ordered or re-indented.
        REGISTRY.write_text(json.dumps(registry, indent=4) + "\n")
        print(f"wrote annotations to {REGISTRY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
