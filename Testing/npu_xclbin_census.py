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

Two modes:

  --pairing   audits the COMMITTED set: every final_i8_*.xclbin should ship its
              insts_i8_*.txt, because init_i8() falls back to the runtime generator
              when the .txt is absent and that generator emits single-core-row
              instructions which, against a multi-row (v27) xclbin, silently
              compute the WRONG result (the engine's own warning, :859). Findings
              are split into RISK — the engine names that slot as a literal, so it
              will really take the fallback — and a note for artifacts no engine
              slot token matches (another loader, or not this engine's).
  (default)   the deployed-model census, BOTH halves: for every model and slot it
              resolves the xclbin (xp()) and the instruction file (ip()) and
              reports them separately, because their failure modes differ in kind.
              No xclbin STOPS the engine (init returns false, the caller prints
              FAIL <slot>); no .txt does NOT — init_i8() drops to the runtime
              generator, which against a multi-row (v27) xclbin computes the wrong
              answer. "xclbin present, insts missing" is therefore the dangerous
              row, and the summary counts it on its own.

Reports only — not wired into run_all.sh. Missing artifacts are a known existing
condition, and a report that fails CI would be the same over-reach this file's
own docstring warns about. (Its classifier IS exercised: run_all.sh invokes
Testing/npu_xclbin_census_selfcheck.sh, which drives this file against a synthetic
fixture and never asserts anything about the real artifact set.)
"""
from __future__ import annotations

import argparse
import json
import re
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


def candidates(t: str, k: int, n: int, tag: str, xd: Path,
               stem: str = "final_i8_", ext: str = ".xclbin") -> list[tuple[str, Path]]:
    """(kind, path) in the order the engine tries them. xp() uses the defaults;
    ip() uses stem='insts_i8_', ext='.txt'. The last candidate is the
    dimension-keyed name, which the engine returns whether or not it exists."""
    out: list[tuple[str, Path]] = []
    tg = tag
    while True:
        out.append(("tag", xd / f"{stem}{t}_{tg}{ext}"))
        u = tg.find("_")
        if u == -1 or u == len(tg) - 1:
            break
        tg = tg[u + 1:]
    out.append(("dim", xd / f"{stem}{t}_K{k}_N{n}{ext}"))
    return out


def resolve(t: str, k: int, n: int, tag: str, xd: Path):
    cands = candidates(t, k, n, tag, xd)
    for kind, path in cands:
        if path.exists():
            return kind, path
    return "MISSING", cands[-1][1]


def resolve_insts(t: str, k: int, n: int, tag: str, xd: Path):
    """The ip() half — the same walk over insts_i8_*.txt.

    Worth reporting separately from the xclbin because the two failure modes
    differ in kind: no xclbin stops the engine (init returns false and the caller
    prints FAIL <slot>), while no .txt does NOT stop it — init_i8() drops to the
    runtime generator, which against a multi-row (v27) xclbin computes the wrong
    answer. An xclbin WITHOUT its insts file is therefore the dangerous row, and
    it is the one nothing reports."""
    cands = candidates(t, k, n, tag, xd, stem="insts_i8_", ext=".txt")
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


# A slot reaches a context in one of two ways, and the two sets differ:
#   * as the slot argument of a context init — init_i8(cd,"D", …), xpm(t,"MOE"), …
#   * inside a literal path the engine builds — D"/final_i8_KV_v.xclbin"
# npu_engine_hybrid.cpp and npu_engine_cb.cpp hand whole paths to .init(), so the
# first pattern alone cannot see their slots; a caller that passes a variable needs
# the second. Both are "the engine names that slot as a literal".
_SLOT_ARGUMENT = re.compile(r'\(\s*[A-Za-z_][A-Za-z0-9_]*\s*,\s*"([A-Z][A-Z0-9_]*)"')
_PATH_LITERAL = re.compile(r'final_i8_([A-Z0-9_]+)')
_ENGINE_SOURCE_GLOBS = ("*.c", "*.cc", "*.cpp", "*.h", "*.hpp")


def _strip_c_comments(text: str) -> str:
    """Drop // and /* */ spans, so a name that lives only in prose is not a token.

    engine/npu/tests/check_xclbin_provenance.py records "grep for final_i8_ATTN/
    insts_i8_ATTN over engine/ = 0" in a comment; a scan that read comments would
    take that sentence for a loader and print ATTN as a RISK row — a wrong-answer
    verdict about the one committed artifact nothing loads."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _prefixes(run: str) -> set[str]:
    """Every underscore-prefix of a name run: QKV_K2048_N2560 -> {QKV, QKV_K2048, …}.

    Sound by construction — each one is a prefix of a path the engine builds, so the
    engine really does ask for it (xp() walks exactly these prefixes)."""
    parts = [p for p in run.rstrip("_").split("_") if p]
    return {"_".join(parts[:i]) for i in range(1, len(parts) + 1)}


def engine_slot_tokens(repo: Path) -> set[str]:
    """Every slot name an engine source in the tree loads, or builds a path for.

    Used to tell "this xclbin's slot is one the engine actually asks for" from
    "this artifact belongs to some other loader", instead of hardcoding a list of
    exceptions that would rot.

    Two things this got wrong once, both of which made the RISK branch unreachable
    for most of the op set (#2584):
      * the character class required three characters — "([A-Z][A-Z0-9_]{2,})" —
        while five of the six legacy slots are "O", "G", "U", "D" and "GU";
      * it read npu_engine_universal.cpp alone, so a slot named only in another
        engine (hybrid, cb, zaya_decode, the pool probes, the zero_copy tests)
        could not match either.
    Measured before the fix: 103 of the 146 committed final_i8_*.xclbin names fell
    to the "note" branch, and stripping the .txt of final_i8_D_qwen3_4b.xclbin
    printed "no engine slot token matches this name" with the summary still saying
    "0 of them belong to a slot the engine loads".
    """
    root = repo / "engine"
    if not root.is_dir():
        return set()
    tokens: set[str] = set()
    for pattern in _ENGINE_SOURCE_GLOBS:
        for src in root.rglob(pattern):
            text = _strip_c_comments(src.read_text(encoding="utf-8", errors="replace"))
            tokens.update(_SLOT_ARGUMENT.findall(text))
            for run in _PATH_LITERAL.findall(text):
                tokens.update(_prefixes(run))
    return tokens


def pairing_audit(xd: Path, repo: Path) -> int:
    """Report committed xclbins that ship without their instruction file.

    This is the other half of resolve(): init_i8() looks for the .txt first and
    falls back to the runtime generator when it is absent, and that generator
    emits SINGLE-CORE-ROW instructions which — against a multi-row (v27) xclbin —
    silently compute the WRONG result rather than merely a slower one. The warning
    is in the engine itself (npu_engine_universal.cpp:859), so a xclbin whose
    instruction file went missing is a wrong-answer path, not a slow one.
    """
    xcl = {f.name[len("final_i8_"):-len(".xclbin")]: f for f in sorted(xd.glob("final_i8_*.xclbin"))}
    insts = {f.name[len("insts_i8_"):-len(".txt")] for f in xd.glob("insts_i8_*.txt")}
    missing = sorted(set(xcl) - insts)
    tokens = engine_slot_tokens(repo)

    print(f"xclbin dir : {xd}")
    print(f"  {len(xcl)} final_i8_*.xclbin, {len(insts)} insts_i8_*.txt")
    print(f"  xclbins with no instruction file: {len(missing)}\n")

    risky = 0
    for name in missing:
        # Longest underscore prefix that the engine names as a literal slot token.
        parts = name.split("_")
        slot = next(("_".join(parts[:i]) for i in range(len(parts), 0, -1)
                     if "_".join(parts[:i]) in tokens), None)
        if slot:
            risky += 1
            print(f"  RISK   final_i8_{name}.xclbin — slot {slot!r} IS loaded by the engine,"
                  f" so this falls back to the runtime generator (wrong results)")
        else:
            print(f"  note   final_i8_{name}.xclbin — no engine slot token matches this name"
                  f" (another loader, or not this engine's artifact)")

    orphan = sorted(insts - set(xcl))
    if orphan:
        print(f"\n  {len(orphan)} instruction file(s) with no xclbin:")
        for o in orphan:
            print(f"    insts_i8_{o}.txt")
    print(f"\n  {risky} of them belong to a slot the engine loads.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / ".config/flm/models"),
                    help="directory of deployed model dirs")
    ap.add_argument("--xclbins", default=None,
                    help="xclbin directory (default: <repo>/engine/npu/xclbins)")
    ap.add_argument("--pairing", action="store_true",
                    help="audit the committed set: every xclbin needs its insts file")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    xd = Path(args.xclbins) if args.xclbins else repo / "engine" / "npu" / "xclbins"
    if args.pairing:
        if not xd.is_dir():
            sys.exit(f"npu_xclbin_census: no xclbin directory at {xd}")
        return pairing_audit(xd, repo)
    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"npu_xclbin_census: no model root at {root}")
    if not xd.is_dir():
        sys.exit(f"npu_xclbin_census: no xclbin directory at {xd}")

    models, missing_total, underivable = [], 0, []
    no_insts_total = 0
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
            ikind, ipath = resolve_insts(t, k, n, tag, xd)
            # The dangerous combination: the engine has an xclbin to run but no
            # instructions for it, so init_i8() takes the runtime generator.
            gen_fallback = kind != "MISSING" and ikind == "MISSING"
            if gen_fallback:
                no_insts_total += 1
            rows.append({"slot": t, "k": k, "n": n, "kind": kind, "path": path.name,
                         "insts_kind": ikind, "insts_path": ipath.name,
                         "generator_fallback": gen_fallback})
        models.append({"model": mdir.name, "tag": tag, "dims": d, "slots": rows})

    if args.json:
        print(json.dumps({"xclbins": str(xd), "models": models,
                          "generator_fallback_slots": no_insts_total,
                          "underivable": underivable}, indent=1))
        return 0

    print(f"xclbin dir : {xd}  ({len(list(xd.glob('*.xclbin')))} xclbins)")
    print(f"model root : {root}\n")
    for m in models:
        miss = [s for s in m["slots"] if s["kind"] == "MISSING"]
        gen = [s for s in m["slots"] if s["generator_fallback"]]
        d = m["dims"]
        flags = []
        if miss:
            flags.append(f"{len(miss)} NO XCLBIN")
        if gen:
            flags.append(f"{len(gen)} NO INSTS -> generator")
        flag = "OK " if not flags else " + ".join(flags)
        print(f"{m['model']:28} tag={m['tag']:22} H={d['H']:<5} NH={d['NH']:<3} "
              f"NKV={d['NKV']:<3} HD={d['HD']:<4} IM={d['IM']:<6} "
              f"gu_split={'Y' if d['IM'] * 2 > GU_SPLIT_LIMIT else 'n'}  {flag}")
        for s in m["slots"]:
            via = f"via {s['kind']}" if s["kind"] != "MISSING" else "MISSING"
            insts = (f"insts via {s['insts_kind']}" if s["insts_kind"] != "MISSING"
                     else "insts MISSING")
            note = "  <- RUNTIME GENERATOR (wrong results)" if s["generator_fallback"] else ""
            print(f"    {s['slot']:3} K={s['k']:<6} N={s['n']:<6} {via:9} "
                  f"{s['path']:44} {insts}{note}")
    if underivable:
        print("\nunderivable from config.json (needs the dims by hand):")
        for name, miss in underivable:
            print(f"    {name:28} missing {', '.join(miss)}")

    total_slots = sum(len(m["slots"]) for m in models)
    print(f"\n{len(models)} models, {total_slots} slots checked "
          f"(4 where GU is fused, 5 where it is split)")
    print(f"  {missing_total} slot(s) with no xclbin under any candidate name"
          f" — these STOP the engine (FAIL <slot>)")
    print(f"  {no_insts_total} slot(s) with an xclbin but NO insts file"
          f" — these silently use the runtime generator")
    return 0


if __name__ == "__main__":
    sys.exit(main())
