#!/usr/bin/env python3
"""Which deployed models have an xclbin for every slot the legacy path asks for.

Why this exists: the open half of issue #2329. The instruction-file half of that
census was done, but the xclbin half was withdrawn by its author as untrustworthy
— the first attempt ignored the two things that make xp() non-obvious, so it
demanded `final_i8_G_*` from models that correctly use GU and concluded that
Gemma3 and Llama-3.1 "cannot load at all". A census that says that is worse than
no census, so this one takes both from the source:

  shapes   npu-infer/include/model_config.h:262-279
             QKV  K=H          N=NH*HD + 2*NKV*HD
             O    K=NH*HD      N=H
             G,U  K=H          N=IM          when gu_split = (IM*2 > 14336)
             GU   K=H          N=IM*2        otherwise
             D    K=IM         N=H

  lookup   engine/npu/src/npu_engine_universal.cpp:829-844, in order
             1. final_i8_<t>_<model_tag>.xclbin
             2. the same with leading underscore-token groups stripped off the
                tag, repeatedly (qwen3_4b -> 4b)
             3. final_i8_<t>_K<K>_N<N>.xclbin   <- returned even when ABSENT

  tag      ...:562-584 — basename without extension; if that is literally
           "model", the parent directory name instead; lowercased; '-' '.' '\\'
           become '_'; one trailing suffix stripped from
           {_npu2, _instruct, _it, _it_npu2}.

Point 3 is why this reports rather than asserts: xp() hands back a path it never
checked, and the failure surfaces later as the context init failing. A slot is
only "present" here if a candidate file actually exists.

Scope, stated so nobody reads more into the table than it says: this models the
LEGACY per-op path (init_i8). It does not model --use-flm-xclbin (mm.xclbin), the
i4 / GUSILU_i4 path, or the instruction half (ip()), which was censused in #2329
separately. Shapes come from each model's own config.json, not from the engine's
tile-metadata derivation; that was validated against the one case where the
engine's numbers are recorded (Qwen3-4B: H=2560 NH=32 NKV=8 HD=128 IM=9728,
which reproduces its final_i8_G_K2560_N9728.xclbin and its qwen3_4b tag).

Reports only — not wired into run_all.sh. Missing artifacts are a known existing
condition, and a report that fails CI would be the same over-reach this file's
own docstring warns about.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SUFFIXES = ("_npu2", "_instruct", "_it", "_it_npu2")
GU_SPLIT_LIMIT = 14336


def model_tag(model_path: Path) -> str:
    """The engine's cfg.model_tag for a model path (…:562-584)."""
    tag = model_path.stem                      # basename without extension
    if tag == "model":                          # generic name -> parent dir
        tag = model_path.parent.name
    tag = tag.lower().replace("-", "_").replace(".", "_").replace("\\", "_")
    for sfx in SUFFIXES:
        if tag.endswith(sfx) and len(tag) > len(sfx):
            tag = tag[: -len(sfx)]
    return tag


def xp_candidates(t: str, k: int, n: int, tag: str, xd: Path) -> list[tuple[str, Path]]:
    """(kind, path) in the order xp() tries them. The last is the
    dimension-keyed name, which xp() returns whether or not it exists."""
    out: list[tuple[str, Path]] = []
    tg = tag
    while True:
        out.append(("tag", xd / f"final_i8_{t}_{tg}.xclbin"))
        u = tg.find("_")
        if u == -1 or u == len(tg) - 1:
            break
        tg = tg[u + 1:]
    out.append(("dim", xd / f"final_i8_{t}_K{k}_N{n}.xclbin"))
    return out


def resolve(t: str, k: int, n: int, tag: str, xd: Path):
    cands = xp_candidates(t, k, n, tag, xd)
    for kind, path in cands:
        if path.exists():
            return kind, path
    return "MISSING", cands[-1][1]


def dims(cfg: dict) -> tuple[dict, list[str]]:
    """The five dims the shape formulas need, plus what was missing."""
    missing: list[str] = []
    h = cfg.get("hidden_size")
    nh = cfg.get("num_attention_heads")
    nkv = cfg.get("num_key_value_heads", nh)
    hd = cfg.get("head_dim") or (h // nh if h and nh else None)
    im = cfg.get("intermediate_size") or cfg.get("moe_intermediate_size")
    for name, val in (("hidden_size", h), ("num_attention_heads", nh),
                      ("num_key_value_heads", nkv), ("head_dim", hd),
                      ("intermediate_size", im)):
        if not val:
            missing.append(name)
    return {"H": h, "NH": nh, "NKV": nkv, "HD": hd, "IM": im}, missing


def slots(d: dict) -> list[tuple[str, int, int]]:
    h, nh, nkv, hd, im = d["H"], d["NH"], d["NKV"], d["HD"], d["IM"]
    out = [("QKV", h, nh * hd + 2 * nkv * hd), ("O", nh * hd, h)]
    if im * 2 > GU_SPLIT_LIMIT:
        out += [("G", h, im), ("U", h, im)]
    else:
        out += [("GU", h, im * 2)]
    out.append(("D", im, h))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / ".config/flm/models"),
                    help="directory of deployed model dirs")
    ap.add_argument("--xclbins", default=None,
                    help="xclbin directory (default: <repo>/engine/npu/xclbins)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    xd = Path(args.xclbins) if args.xclbins else repo / "engine" / "npu" / "xclbins"
    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"npu_xclbin_census: no model root at {root}")
    if not xd.is_dir():
        sys.exit(f"npu_xclbin_census: no xclbin directory at {xd}")

    models, missing_total, underivable = [], 0, []
    for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
        cfgp = mdir / "config.json"
        if not cfgp.is_file():
            underivable.append((mdir.name, ["config.json missing"]))
            continue
        cfg = json.loads(cfgp.read_text())
        d, missing = dims(cfg)
        if missing:
            underivable.append((mdir.name, missing))
            continue
        tag = model_tag(mdir / "model.q4nx")
        rows = []
        for t, k, n in slots(d):
            kind, path = resolve(t, k, n, tag, xd)
            if kind == "MISSING":
                missing_total += 1
            rows.append({"slot": t, "k": k, "n": n, "kind": kind,
                         "path": path.name})
        models.append({"model": mdir.name, "tag": tag, "dims": d, "slots": rows})

    if args.json:
        print(json.dumps({"xclbins": str(xd), "models": models,
                          "underivable": underivable}, indent=1))
        return 0

    print(f"xclbin dir : {xd}  ({len(list(xd.glob('*.xclbin')))} xclbins)")
    print(f"model root : {root}\n")
    for m in models:
        miss = [s for s in m["slots"] if s["kind"] == "MISSING"]
        d = m["dims"]
        flag = "OK " if not miss else f"{len(miss)} MISSING"
        print(f"{m['model']:28} tag={m['tag']:22} H={d['H']:<5} NH={d['NH']:<3} "
              f"NKV={d['NKV']:<3} HD={d['HD']:<4} IM={d['IM']:<6} "
              f"gu_split={'Y' if d['IM'] * 2 > GU_SPLIT_LIMIT else 'n'}  {flag}")
        for s in m["slots"]:
            via = f"via {s['kind']}" if s["kind"] != "MISSING" else "MISSING"
            print(f"    {s['slot']:3} K={s['k']:<6} N={s['n']:<6} {via:9} "
                  f"{s['path']}")
    if underivable:
        print("\nunderivable from config.json (needs the dims by hand):")
        for name, miss in underivable:
            print(f"    {name:28} missing {', '.join(miss)}")

    total_slots = sum(len(m["slots"]) for m in models)
    print(f"\n{len(models)} models, {total_slots} slots checked "
          f"(4 where GU is fused, 5 where it is split) — {missing_total} slot(s) "
          f"with no xclbin under any candidate name")
    return 0


if __name__ == "__main__":
    sys.exit(main())
