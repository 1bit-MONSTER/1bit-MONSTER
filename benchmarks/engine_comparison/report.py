#!/usr/bin/env python3
"""
Aggregate the per-cell JSON results into a markdown report + CSV.

Reads results/*.json (written by run_matrix.py) and produces:
  * docs/engine_comparison_report.md  - human-readable tables
  * results/results.csv                - flat machine-readable dump
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

# The engine under test - the numerator of every performance ratio. This fork
# defaults to 1bit-monster' zaya_server; override with BENCH_HERO_ENGINE.
HERO_ENGINE = os.environ.get("BENCH_HERO_ENGINE", "zaya")


def _peek_config_arg(argv) -> str | None:
    """`--config PATH` must be honored before `config` imports + loads its JSON."""
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--config="):
            return a.split("=", 1)[1]
    return None


_cfg_arg = _peek_config_arg(sys.argv[1:])
if _cfg_arg:
    os.environ["BENCH_CONFIG"] = _cfg_arg

import config

HERO_DISPLAY = config.ENGINES[HERO_ENGINE].display if HERO_ENGINE in config.ENGINES else HERO_ENGINE

RESULTS_DIR = config.RESULTS_DIR
REPORT_PATH = config.REPO_ROOT / "docs" / "engine_comparison_report.md"
CSV_PATH = RESULTS_DIR / "results.csv"

# Columns are (engine, backend) pairs discovered in the results, ordered by the
# config registries (engine outer, backend inner). Results may contain backend
# ids that are not in the current config (e.g. the legacy abstract gpu / cpu
# ids from an older run); those sort after the registry ids and are labeled by
# their raw id.
def _order_key(col) -> tuple:
    eng_order = list(config.ENGINES.keys())
    b_order = list(config.BACKENDS.keys())
    e, b = col
    return (eng_order.index(e) if e in eng_order else len(eng_order),
            b_order.index(b) if b in b_order else len(b_order), str(e), str(b))


def _col_label(col) -> str:
    e, b = col
    eng = config.ENGINES.get(e)
    spec = config.BACKENDS.get(b)
    return f"{eng.display if eng else e} · {spec.display if spec else b}"


# Performance-ratio comparisons: TensorSharp (numerator) vs a reference engine
# on the *same* backend, so the columns stay apples-to-apples. A ratio > 1.0×
# means TensorSharp is faster (for throughput metrics) / lower-latency (TTFT).
_RATIO_REF_ENGINES = ("llamacpp", "vllm")


def load_all() -> dict:
    """Returns (baseline, rows).

    baseline[model][scenario][(engine,backend)] = record   (only mtp-off,
    concurrency-1 cells, so the headline per-engine tables stay apples-to-apples).
    rows = every record (all mtp / concurrency axes), used by the MTP and
    concurrency sections.
    """
    out: dict = {}
    rows = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        eng, backend, model, scenario = d["engine"], d["backend"], d["model"], d["scenario"]
        rows.append(d)
        if d.get("mtp", False) or int(d.get("concurrency", 1) or 1) != 1:
            continue  # keep the baseline tables to the single-stream, no-MTP point
        out.setdefault(model, {}).setdefault(scenario, {})[(eng, backend)] = d
    return out, rows


def _cell(rec, metric) -> str:
    if rec is None:
        return "n/a"
    status = rec.get("status")
    if status == "skipped":
        return "—"
    if status != "ok":
        return "fail"
    v = rec.get(metric, 0.0) or 0.0
    return f"{v:.1f}" if v > 0 else "—"


def _present_columns(data: dict) -> list:
    seen = set()
    for scen_map in data.values():
        for col_map in scen_map.values():
            seen.update(col_map.keys())
    return sorted(seen, key=_order_key)


def _scenario_rows(scen_map: dict) -> list:
    """Scenario ids to render, in a stable order: declared scenarios first (in
    registry order), then any extra ids present in the data (e.g. on-the-fly
    `prefill_*` scenarios run against a config that did not declare them)."""
    declared = list(config.SCENARIOS.keys())   # real declared keys only
    ordered = [s for s in declared if s in scen_map]
    extra = sorted(s for s in scen_map if s not in declared)
    return ordered + extra


def metric_table(scen_map: dict, cols: list, metric: str) -> str:
    head = "| Scenario | " + " | ".join(_col_label(c) for c in cols) + " |"
    sep = "|---|" + "|".join(["---:"] * len(cols)) + "|"
    rows = [head, sep]
    for scenario_id in _scenario_rows(scen_map):
        col_map = scen_map[scenario_id]
        cells = [scenario_id]
        for c in cols:
            cells.append(_cell(col_map.get(c), metric))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _ok_value(rec, metric: str) -> float:
    """The metric value only when the cell actually ran; else 0.0."""
    if not rec or rec.get("status") != "ok":
        return 0.0
    return float(rec.get(metric, 0.0) or 0.0)


def _ratio_val(ts_rec, ref_rec, metric: str, higher_is_better: bool = True) -> float:
    """TensorSharp-relative speedup for one cell, or 0.0 if either side is
    missing/failed/zero. For throughput (higher is better) this is TS/ref; for
    latency like TTFT (lower is better) it is ref/TS — so a value > 1.0× always
    means TensorSharp won."""
    a = _ok_value(ts_rec, metric)
    b = _ok_value(ref_rec, metric)
    if a <= 0 or b <= 0:
        return 0.0
    return (a / b) if higher_is_better else (b / a)


def _fmt_ratio(r: float) -> str:
    return f"{r:.2f}×" if r > 0 else "—"


def _geomean(vals: list) -> float:
    vals = [v for v in vals if v > 0]
    if not vals:
        return 0.0
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


_RATIO_METRICS = (("decode_tps", True), ("prefill_tps", True), ("ttft_ms", False))


def _present_ratio_pairs(scen_map: dict) -> list:
    """Ratio pairs that yield at least one real number for this model — i.e. some
    scenario where both TensorSharp and the reference actually ran on the same
    backend. Built from whatever backends appear in the data, so new backend ids
    (ggml_vulkan, ...) get their own comparison columns automatically; drops e.g.
    an all-`—` vLLM column when that endpoint never produced a comparable cell."""
    cols = set()
    for col_map in scen_map.values():
        cols.update(col_map.keys())
    out = []
    for ts in sorted((c for c in cols if c[0] == HERO_ENGINE), key=_order_key):
        for ref_eng in _RATIO_REF_ENGINES:
            ref = (ref_eng, ts[1])
            if ref not in cols:
                continue
            if any(_ratio_val(col_map.get(ts), col_map.get(ref), m, hib) > 0
                   for col_map in scen_map.values()
                   for m, hib in _RATIO_METRICS):
                out.append((ts, ref, f"vs {_col_label(ref)}"))
    return out


def ratio_table(scen_map: dict, pairs: list, metric: str,
                higher_is_better: bool = True) -> str:
    head = "| Scenario | " + " | ".join(lbl for _, _, lbl in pairs) + " |"
    sep = "|---|" + "|".join(["---:"] * len(pairs)) + "|"
    rows = [head, sep]
    for scenario_id in _scenario_rows(scen_map):
        col_map = scen_map[scenario_id]
        cells = [scenario_id]
        for ts, ref, _ in pairs:
            cells.append(_fmt_ratio(_ratio_val(
                col_map.get(ts), col_map.get(ref), metric, higher_is_better)))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def summary_section(data: dict) -> str:
    """Headline geomean speedup of TensorSharp over each reference engine, per
    model, across all scenarios both engines ran. Geomean (not mean) so a single
    runaway scenario can't dominate the per-model number."""
    lines = ["| Model | Comparison | decode | prefill | TTFT |",
             "|---|---|---:|---:|---:|"]
    any_row = False
    for model_id in config.MODELS:
        if model_id not in data:
            continue
        scen_map = data[model_id]
        for ts, ref, lbl in _present_ratio_pairs(scen_map):
            d_ratios, p_ratios, t_ratios = [], [], []
            for col_map in scen_map.values():
                d = _ratio_val(col_map.get(ts), col_map.get(ref), "decode_tps")
                p = _ratio_val(col_map.get(ts), col_map.get(ref), "prefill_tps")
                t = _ratio_val(col_map.get(ts), col_map.get(ref), "ttft_ms",
                               higher_is_better=False)
                if d:
                    d_ratios.append(d)
                if p:
                    p_ratios.append(p)
                if t:
                    t_ratios.append(t)
            dg, pg, tg = _geomean(d_ratios), _geomean(p_ratios), _geomean(t_ratios)
            if dg <= 0 and pg <= 0 and tg <= 0:
                continue
            any_row = True
            lines.append(f"| {config.MODELS[model_id].display} | {lbl} | "
                         f"{_fmt_ratio(dg)} | {_fmt_ratio(pg)} | {_fmt_ratio(tg)} |")
    if not any_row:
        return f"_No overlapping {HERO_DISPLAY} / reference cells to compare._"
    return "\n".join(lines)


def versions_block() -> str:
    def _try(cmd):
        try:
            return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
        except Exception:
            return ""
    ts_rev = _try(["git", "-C", str(config.REPO_ROOT), "rev-parse", "--short", "HEAD"])
    dotnet_v = _try(["dotnet", "--version"])
    gpu = _try(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    ts_backends = " / ".join(b.ts_backend for b in config.BACKENDS.values() if b.ts_backend)
    return (
        "| Component | Version / detail |\n"
        "|---|---|\n"
        f"| {HERO_DISPLAY} | git `{ts_rev}` (backends: {ts_backends}) |\n"
        f"| llama.cpp | `{config.LLAMA_SERVER_EXE}` |\n"
        f"| vLLM | endpoint `{config.VLLM_BASE_URL}` (connect-only) |\n"
        f"| GPU | {gpu or 'unknown'} |\n"
    )


def _ok_decode(rec) -> float:
    if not rec or rec.get("status") != "ok":
        return 0.0
    return float(rec.get("decode_tps", 0.0) or 0.0)


def mtp_section(rows: list) -> str:
    """MTP/NextN on-vs-off decode comparison (single-stream, TensorSharp).

    Pairs the concurrency-1 records that share (engine, backend, model, scenario)
    but differ on `mtp`, and reports decode tok/s off → on plus the speedup."""
    # index: (engine, backend, model, scenario) -> {mtp: rec}
    idx: dict = {}
    for r in rows:
        if int(r.get("concurrency", 1) or 1) != 1:
            continue
        key = (r["engine"], r["backend"], r["model"], r["scenario"])
        idx.setdefault(key, {})[bool(r.get("mtp", False))] = r

    paired = [(k, v) for k, v in idx.items() if True in v and False in v]
    if not paired:
        return "_No MTP on/off pairs were run (use `--mtp off,on`)._"

    lines = ["| Engine · Backend · Model | Scenario | decode off | decode on | speedup |",
             "|---|---|---:|---:|---:|"]
    for (eng, backend, model, scenario), v in sorted(paired):
        off = _ok_decode(v[False])
        on = _ok_decode(v[True])
        if off > 0 and on > 0:
            spd = f"{on / off:.2f}×"
        else:
            spd = "—"
        off_s = f"{off:.1f}" if off > 0 else _cell(v[False], "decode_tps")
        on_s = f"{on:.1f}" if on > 0 else _cell(v[True], "decode_tps")
        lines.append(f"| {eng} · {backend} · {model} | {scenario} | {off_s} | {on_s} | {spd} |")
    return "\n".join(lines)


def concurrency_section(rows: list) -> str:
    """Parallel-request scaling: per (engine, backend, model, scenario, mtp),
    show per-request and system-aggregate decode throughput at each concurrency."""
    levels = sorted({int(r.get("concurrency", 1) or 1) for r in rows})
    if levels == [1]:
        return "_No parallel-request cells were run (use `--concurrency 1,4,8`)._"

    # index: (engine, backend, model, scenario, mtp) -> {concurrency: rec}
    idx: dict = {}
    for r in rows:
        key = (r["engine"], r["backend"], r["model"], r["scenario"], bool(r.get("mtp", False)))
        idx.setdefault(key, {})[int(r.get("concurrency", 1) or 1)] = r

    # Only keep series that actually exercise >1 concurrency.
    series = {k: v for k, v in idx.items() if any(c > 1 for c in v)}
    if not series:
        return "_No parallel-request cells were run (use `--concurrency 1,4,8`)._"

    head = ("| Engine · Backend · Model · Scenario | metric | "
            + " | ".join(f"c={c}" for c in levels) + " |")
    sep = "|---|---|" + "|".join(["---:"] * len(levels)) + "|"
    lines = [head, sep]
    for key, by_c in sorted(series.items()):
        eng, backend, model, scenario, mtp = key
        label = f"{eng} · {backend} · {model} · {scenario}" + (" · mtp" if mtp else "")

        def _row(metric_label, metric_key):
            cells = []
            for c in levels:
                rec = by_c.get(c)
                cells.append(_cell(rec, metric_key))
            return f"| {label} | {metric_label} | " + " | ".join(cells) + " |"

        lines.append(_row("decode/req t/s", "decode_tps"))
        lines.append(_row("aggregate t/s", "aggregate_decode_tps"))
    return "\n".join(lines)


def _is_image_edit_scenario(scenario_id: str) -> bool:
    try:
        return config.SCENARIOS[scenario_id].kind == "image_edit"
    except KeyError:
        return False


# (record key, table label). All values are engine-reported pipeline-phase
# timings in ms, rendered in seconds.
_EDIT_COLS = (
    ("edit_total_ms", "total (warm)"),
    ("edit_per_step_ms", "per step"),
    ("edit_sampling_ms", "sampling"),
    ("edit_text_encode_ms", "text encode"),
    ("edit_vae_encode_ms", "VAE encode"),
    ("edit_vae_decode_ms", "VAE decode"),
    ("edit_first_total_ms", "first request (cold)"),
)


def _edit_cell(rec, key) -> str:
    if rec is None:
        return "n/a"
    if rec.get("status") == "skipped":
        return "—"
    if rec.get("status") != "ok":
        return "fail"
    v = float(rec.get(key, 0.0) or 0.0)
    return f"{v / 1000.0:.2f} s" if v > 0 else "—"


def image_edit_section(rows: list) -> str:
    """Image-editing (stable-diffusion) results: one table per
    (model, scenario, backend) with engines as rows and pipeline phases as
    columns, plus a TensorSharp-vs-reference speedup line. All timings are the
    engines' own pipeline timers (weight-file load excluded on both sides)."""
    recs = [r for r in rows
            if _is_image_edit_scenario(r.get("scenario", ""))
            and not r.get("mtp", False)
            and int(r.get("concurrency", 1) or 1) == 1]
    if not recs:
        return "_No image-edit cells were run (see the `image_edit` scenario)._"

    groups: dict = {}
    for r in recs:
        groups.setdefault((r["model"], r["scenario"], r["backend"]), {})[r["engine"]] = r

    eng_order = list(config.ENGINES.keys())
    out = []
    for (model_id, scenario_id, backend), by_eng in sorted(groups.items()):
        model = config.MODELS.get(model_id)
        spec = config.BACKENDS.get(backend)
        ok = {e: r for e, r in by_eng.items() if r.get("status") == "ok"}
        if not ok and all(r.get("status") == "skipped" for r in by_eng.values()):
            continue                      # fully gated cell group (e.g. cpu backend)
        rep = next(iter(ok.values()), None)
        dims = (f", {rep['edit_width']}x{rep['edit_height']}, {rep.get('steps', 0)} steps"
                if rep and rep.get("edit_width") else "")
        title = model.display if model else model_id
        out.append(f"### {title} — `{scenario_id}` on "
                   f"{spec.display if spec else backend}{dims}\n")

        head = "| Engine | " + " | ".join(lbl for _, lbl in _EDIT_COLS) + " |"
        sep = "|---|" + "|".join(["---:"] * len(_EDIT_COLS)) + "|"
        lines = [head, sep]
        engines_here = sorted(by_eng, key=lambda e: (eng_order.index(e)
                                                     if e in eng_order else len(eng_order), e))
        for e in engines_here:
            r = by_eng[e]
            eng = config.ENGINES.get(e)
            cells = [_edit_cell(r, k) for k, _ in _EDIT_COLS]
            lines.append(f"| {eng.display if eng else e} | " + " | ".join(cells) + " |")
        out.append("\n".join(lines))
        out.append("")

        ts = ok.get(HERO_ENGINE)
        for ref_eng in ("sdcpp", "llamacpp"):
            ref = ok.get(ref_eng)
            if not (ts and ref):
                continue
            ratios = []
            for key, lbl in _EDIT_COLS:
                if key == "edit_first_total_ms":
                    continue              # no cold/warm distinction for a CLI engine
                a = float(ts.get(key, 0.0) or 0.0)
                b = float(ref.get(key, 0.0) or 0.0)
                if a > 0 and b > 0:
                    ratios.append(f"{lbl} **{b / a:.2f}×**")
            if ratios:
                ref_disp = config.ENGINES[ref_eng].display
                out.append(f"**{HERO_DISPLAY} vs {ref_disp}** (ratio = {ref_disp} time / "
                           f"TensorSharp time; > 1.0× = TensorSharp faster): "
                           + ", ".join(ratios) + "\n")
    return "\n".join(out) if out else "_No image-edit cells were run._"


# ---------------------------------------------------------------------------
# Output quality — cross-engine agreement on identical greedy requests
# ---------------------------------------------------------------------------
def _rec_output_text(rec) -> str:
    """Full captured output when available (new results), else the 300-char
    preview (older result files)."""
    if not rec:
        return ""
    return str(rec.get("output_text") or rec.get("output_preview") or "")


def _norm_ws(s: str) -> str:
    return " ".join((s or "").split())


def _json_valid(text: str):
    """Whether the output parses as JSON (directly, or the first-{ .. last-}
    slice, tolerating prose/fences around the object)."""
    t = (text or "").strip()
    if not t:
        return False
    candidates = [t]
    if "{" in t and "}" in t:
        candidates.append(t[t.find("{"): t.rfind("}") + 1])
    for cand in candidates:
        try:
            json.loads(cand)
            return True
        except Exception:
            pass
    return False


def _similarity_verdict(r: float) -> str:
    if r >= 0.90:
        return "high agreement"
    if r >= 0.75:
        return "moderate"
    if r >= 0.50:
        return "low"
    return "diverged"


def _mark(v) -> str:
    return "yes" if v else ("no" if v is False else "?")


def quality_section(data: dict) -> str:
    """Output-quality comparison of TensorSharp vs llama.cpp on the same
    backend. Both engines decode the same GGUF greedily (temperature=0), so
    their outputs should agree closely; low text similarity or a failed
    structural check (valid JSON in json_mode, tool call emitted in
    function_call) flags a quality problem on one side. Prefill scenarios are
    excluded (their 8-token outputs carry no quality signal)."""
    ref_engine = "llamacpp"
    rows = []       # (model_id, backend, scenario, ts_rec, ref_rec, sim)
    for model_id in config.MODELS:
        scen_map = data.get(model_id)
        if not scen_map:
            continue
        for scenario_id in _scenario_rows(scen_map):
            if scenario_id.startswith("prefill_") or _is_image_edit_scenario(scenario_id):
                continue
            col_map = scen_map[scenario_id]
            backends = sorted({b for (_, b) in col_map})
            for backend in backends:
                ts = col_map.get((HERO_ENGINE, backend))
                ref = col_map.get((ref_engine, backend))
                if not ts or not ref or ts.get("status") != "ok" or ref.get("status") != "ok":
                    continue
                a, b = _norm_ws(_rec_output_text(ts)), _norm_ws(_rec_output_text(ref))
                # A correct function_call answer often carries no text content
                # (tool_calls only) — keep the row for its structural check and
                # render the similarity as `—`.
                sim = difflib.SequenceMatcher(None, a, b).ratio() if a and b else None
                try:
                    kind = config.SCENARIOS[scenario_id].kind
                except KeyError:
                    kind = ""
                if sim is None and kind not in ("json_mode", "function_call"):
                    continue
                rows.append((model_id, backend, scenario_id, ts, ref, sim))
    if not rows:
        return "_No overlapping ok cells with captured output to compare._"

    out = ["| Model | Backend | Scenario | similarity | verdict | tokens (TS / ref) | "
           "finish (TS / ref) | checks |",
           "|---|---|---|---:|---|---|---|---|"]
    for model_id, backend, scenario_id, ts, ref, sim in rows:
        kind = ""
        try:
            kind = config.SCENARIOS[scenario_id].kind
        except KeyError:
            pass
        if kind == "json_mode":
            checks = (f"json valid: TS {_mark(_json_valid(_rec_output_text(ts)))} / "
                      f"ref {_mark(_json_valid(_rec_output_text(ref)))}")
        elif kind == "function_call":
            checks = (f"tool call: TS {_mark(ts.get('tool_call_ok'))} / "
                      f"ref {_mark(ref.get('tool_call_ok'))}")
        else:
            checks = "—"
        spec = config.BACKENDS.get(backend)
        sim_s = f"{sim:.2f}" if sim is not None else "—"
        verdict = _similarity_verdict(sim) if sim is not None else "—"
        out.append(f"| {model_id} | {spec.display if spec else backend} | {scenario_id} | "
                   f"{sim_s} | {verdict} | "
                   f"{ts.get('completion_tokens', 0)} / {ref.get('completion_tokens', 0)} | "
                   f"{ts.get('finish_reason', '') or '—'} / {ref.get('finish_reason', '') or '—'} | "
                   f"{checks} |")

    sims = [sim for *_, sim in rows if sim is not None]
    if sims:
        out.append("")
        out.append(f"Mean similarity across {len(sims)} compared cells: "
                   f"**{sum(sims) / len(sims):.2f}** "
                   f"(min {min(sims):.2f}, max {max(sims):.2f}).")

    # Side-by-side excerpts for eyeballing, lowest-agreement cells first.
    out.append("")
    with_text = [r for r in rows if r[5] is not None]
    for model_id, backend, scenario_id, ts, ref, sim in sorted(with_text, key=lambda r: r[5]):
        ts_txt = _rec_output_text(ts)[:600] or "(empty)"
        ref_txt = _rec_output_text(ref)[:600] or "(empty)"
        out.append(f"<details><summary>{model_id} · {backend} · {scenario_id} "
                   f"(similarity {sim:.2f})</summary>\n")
        out.append(f"**TensorSharp**\n\n```\n{ts_txt}\n```\n")
        out.append(f"**llama.cpp**\n\n```\n{ref_txt}\n```\n")
        out.append("</details>")
    return "\n".join(out)


def tool_summary(rows: list) -> str:
    fc = [r for r in rows if r["scenario"] == "function_call" and r["status"] == "ok"]
    if not fc:
        return "_No function-call cells were run._"
    lines = ["| Engine · Backend · Model | tool_call emitted |", "|---|:---:|"]
    for r in sorted(fc, key=lambda r: (r["engine"], r["backend"], r["model"])):
        ok = r.get("tool_call_ok")
        mark = "yes" if ok else ("no" if ok is False else "?")
        lines.append(f"| {r['engine']} · {r['backend']} · {r['model']} | {mark} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, metavar="PATH",
                    help="JSON benchmark config file (default: benchmark_config.json, "
                         "or set BENCH_CONFIG)")
    ap.add_argument("--results", default=None,
                    help="results directory to aggregate (default from config)")
    args = ap.parse_args()

    global RESULTS_DIR, CSV_PATH
    if args.results:
        RESULTS_DIR = Path(args.results)
        CSV_PATH = RESULTS_DIR / "results.csv"

    data, rows = load_all()
    if not rows:
        print(f"No results found in {RESULTS_DIR}. Run run_matrix.py first.")
        return
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    out = []
    out.append(f"# Engine comparison benchmark — {HERO_DISPLAY} vs reference engines\n")
    out.append("Same GGUF files, same host, one uniform OpenAI `/v1/chat/completions` "
               "surface, across text / image / audio / video / single-turn / multi-turn / "
               "function-call / structured-output scenarios on the selected compute "
               "backends (ggml_cuda / ggml_vulkan / ggml_metal / ggml_cpu / cpu / ...).\n")
    out.append("Numbers are tokens/second (higher is better). `—` = not applicable / skipped, "
               "`fail` = errored at runtime, `n/a` = combination never attempted.\n")

    out.append("## Software / hardware\n")
    out.append(versions_block())
    out.append("")

    out.append("## Methodology\n")
    out.append("- Each `(engine, backend, model)` group launches its server once; all of "
               "that group's scenarios run against it, so per-scenario timings exclude "
               "model-load cost.\n"
               "- Metrics come from the **streamed** response: `ttft` is time-to-first-token "
               "(prefill latency proxy), `prefill_tps = prompt_tokens / ttft`, and "
               "`decode_tps = (completion_tokens - 1) / (t_last - t_first)`.\n"
               "- DiffusionGemma denoises whole blocks (no token stream), so it is run "
               "non-streaming and its `decode_tps` is wall-clock tokens/second.\n"
               "- Greedy sampling (`temperature=0`); one warmup request per server is discarded.\n"
               "- The headline per-engine tables are the **single-stream, MTP-off** baseline. "
               "MTP on/off and parallel-request scaling are reported in their own sections "
               "below.\n")

    out.append(f"## Performance ratio — {HERO_DISPLAY} vs reference engines\n")
    out.append(f"Geomean of {HERO_DISPLAY}'s per-scenario speedup over each reference engine on "
               "the **same backend**, across every scenario both engines ran (single-stream, "
               "MTP-off). A value **> 1.0× means the hero engine is faster** (for decode / prefill "
               "throughput) or lower-latency (for TTFT); `—` = no overlapping cells. "
               "Per-scenario ratios are in each model's section below.\n")
    out.append(summary_section(data))
    out.append("")

    # Render declared models first, then any models present in the results but
    # not in the config registry (e.g. a run with a different config file).
    for model_id in list(config.MODELS) + [m for m in data if m not in config.MODELS]:
        if model_id not in data:
            continue
        model = config.MODELS.get(model_id)
        if model is not None and model.is_image_edit:
            continue    # no token metrics; rendered in the image-editing section
        scen_map = data[model_id]
        cols = _present_columns({model_id: scen_map})
        if not cols:
            continue
        display = model.display if model is not None else model_id
        out.append(f"## {display}  (`{model_id}`)\n")
        out.append("**Decode throughput (tok/s)**\n")
        out.append(metric_table(scen_map, cols, "decode_tps"))
        out.append("")
        out.append("**Prefill throughput (tok/s)**\n")
        out.append(metric_table(scen_map, cols, "prefill_tps"))
        out.append("")
        out.append("**Time to first token (ms, lower is better)**\n")
        out.append(metric_table(scen_map, cols, "ttft_ms"))
        out.append("")

        pairs = _present_ratio_pairs(scen_map)
        if pairs:
            out.append(f"**Performance ratio — {HERO_DISPLAY} vs reference "
                       "(> 1.0× = {HERO_DISPLAY} faster)**\n")
            out.append("_Decode throughput_\n")
            out.append(ratio_table(scen_map, pairs, "decode_tps"))
            out.append("")
            out.append("_Prefill throughput_\n")
            out.append(ratio_table(scen_map, pairs, "prefill_tps"))
            out.append("")
            out.append(f"_Time to first token (latency; > 1.0× = {HERO_DISPLAY} lower)_\n")
            out.append(ratio_table(scen_map, pairs, "ttft_ms", higher_is_better=False))
            out.append("")

    out.append(f"## Output quality — {HERO_DISPLAY} vs llama.cpp\n")
    out.append("Both engines decode the **same GGUF greedily** (temperature=0) on the same "
               "backend, so their outputs should agree closely. `similarity` is a "
               "whitespace-normalized SequenceMatcher ratio between the two outputs "
               "(1.00 = identical); low similarity, an invalid JSON object in `json_mode`, "
               "or a missing tool call in `function_call` flags an output-quality problem "
               "on one side. Prefill scenarios (8-token outputs) are excluded. "
               "Side-by-side excerpts follow the table, lowest agreement first.\n")
    out.append(quality_section(data))
    out.append("")

    out.append("## Image editing (stable-diffusion)\n")
    out.append("Same input image, prompt, resolution, step count, cfg and seed for every "
               "engine. Timings are each engine's **own pipeline timers**, so weight-file "
               "loading and HTTP/process overhead are excluded on both sides. Lower is better.\n")
    out.append(image_edit_section(rows))
    out.append("")

    out.append("## MTP / NextN speculative decoding (on vs off)\n")
    out.append("Single-stream decode tok/s with MTP/NextN speculative decoding off vs on "
               "(the hero engine only). Speedup `< 1.0×` means speculation cost more than it "
               "saved for that cell — expected when the fused full-model decode path is "
               "already the fast path.\n")
    out.append(mtp_section(rows))
    out.append("")

    out.append("## Parallel-request scaling (concurrency)\n")
    out.append("`decode/req` is the mean per-request decode tok/s; `aggregate` is the "
               "system-wide decode throughput (total generated tokens / the wall window "
               "during which any sequence was decoding) when N identical requests are fired "
               "at one server at once.\n")
    out.append(concurrency_section(rows))
    out.append("")

    out.append("## Function-calling correctness\n")
    out.append(tool_summary(rows))
    out.append("")

    REPORT_PATH.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")

    # Flat CSV
    fields = ["engine", "backend", "model", "scenario", "mtp", "concurrency",
              "status", "detail", "prompt_tokens", "completion_tokens", "ttft_ms",
              "prefill_tps", "decode_tps", "aggregate_decode_tps", "requests_ok",
              "total_wall_ms", "finish_reason", "tool_call_ok",
              "steps", "edit_total_ms", "edit_first_total_ms", "edit_text_encode_ms",
              "edit_vae_encode_ms", "edit_sampling_ms", "edit_per_step_ms",
              "edit_vae_decode_ms", "edit_width", "edit_height", "edit_image"]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
