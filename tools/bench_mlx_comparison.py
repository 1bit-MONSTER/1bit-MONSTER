#!/usr/bin/env python3
"""bench_mlx_comparison.py — integrated backend benchmark runner for lemonade issue #1642.

Runs the apples-to-apples comparison described in
`mlx-backend-benchmark-spec.md` (acceptance criteria for promoting the
`lemon-mlx` backend from experimental to supported):

    lemon-mlx/rocm  vs  vllm/rocm  vs  llamacpp/rocm  vs  llamacpp/vulkan

through a running lemonade server (lemond). Uses only the Python stdlib.
Streaming requests are measured client-side for TTFT and decode rate; the
concurrency sweep measures aggregate throughput per batch.

Usage
-----
    python3 bench_mlx_comparison.py \\
        --server http://127.0.0.1:13305 \\
        --models  "Qwen3.5-0.8B-MLX" "Qwen3-4B-Q4_K_M" \\
        --backends "lemon-mlx:rocm" "llamacpp:rocm" "llamacpp:vulkan" "vllm:rocm" \\
        --scenarios chat-short code-explain chat-long-output \\
        --concurrency 1 4 8 \\
        --runs 5 --warmup 2 \\
        --output-dir ./bench-results \\
        --install --auto-pull --cooldown 60

Notes
-----
- Model names are lemonade registry names (as listed by `lemonade models` /
  GET /api/v1/models). A model is only run against backends whose recipe
  matches the model's registry recipe (e.g. an `-MLX` model only runs on
  lemon-mlx; a GGUF model only on llamacpp).
- `--install` installs each listed backend first (idempotent).
- `--auto-pull` downloads models that are not yet downloaded (cache-first).
- Per-backend sequential execution with a cooldown pause is the default
  (thermal isolation per spec sec. 2). `--interleave` switches to full
  round-robin interleaving of backends (spec sec. 7.5) at the cost of
  load/unload churn.
- Raw JSON (all runs) + a markdown summary table are written to --output-dir.

Exit codes: 0 = ran to completion (some rows may have failed), 1 = fatal error.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Config defaults (mirror the benchmark spec)
# ---------------------------------------------------------------------------

DEFAULT_BACKENDS = ["lemon-mlx:rocm", "vllm:rocm", "llamacpp:rocm", "llamacpp:vulkan"]
DEFAULT_SCENARIOS = ["chat-short", "code-explain", "chat-long-output"]
DEFAULT_CONCURRENCY = [1, 4, 8]
DEFAULT_RUNS = 5
DEFAULT_WARMUP = 2
DEFAULT_COOLDOWN_S = 60
DEFAULT_REQUEST_TIMEOUT_S = 300
DEFAULT_LOAD_TIMEOUT_S = 1800  # first load may download a model

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "chat-short": {
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Hello! How are you today?"},
        ],
        "max_tokens": 20,
    },
    "code-explain": {
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {
                "role": "user",
                "content": (
                    "Explain what this code does:\n\n"
                    "```python\n"
                    "def merge_sort(arr):\n"
                    "    if len(arr) <= 1:\n"
                    "        return arr\n"
                    "    mid = len(arr) // 2\n"
                    "    left = merge_sort(arr[:mid])\n"
                    "    right = merge_sort(arr[mid:])\n"
                    "    return merge(left, right)\n"
                    "```"
                ),
            },
        ],
        "max_tokens": 128,
    },
    "chat-long-output": {
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {
                "role": "user",
                "content": (
                    "Explain how transformers work in deep learning, covering "
                    "attention mechanisms, encoder-decoder architecture, and "
                    "positional encoding in detail."
                ),
            },
        ],
        "max_tokens": 256,
    },
}


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only)
# ---------------------------------------------------------------------------

class ApiError(RuntimeError):
    def __init__(self, status: int, body: str, message: str):
        super().__init__(message)
        self.status = status
        self.body = body


class LemonadeClient:
    def __init__(self, base_url: str, api_key: str = "", timeout_s: float = 300.0):
        self.base = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout_s

    def _headers(self, content_type: Optional[str] = None) -> Dict[str, str]:
        h = {"Accept": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        if content_type:
            h["Content-Type"] = content_type
        return h

    def request(self, path: str, method: str = "GET", body: Optional[dict] = None,
                timeout_s: Optional[float] = None, stream: bool = False,
                extra_headers: Optional[Dict[str, str]] = None) -> Any:
        url = self.base + path
        data = None
        content_type = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            content_type = "application/json"
        headers = self._headers(content_type)
        if extra_headers:
            headers.update(extra_headers)
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        to = timeout_s if timeout_s is not None else self.timeout
        try:
            resp = urllib.request.urlopen(req, timeout=to)
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:2000]
            raise ApiError(e.code, msg, f"HTTP {e.code} on {method} {path}: {msg}") from None
        except urllib.error.URLError as e:
            raise RuntimeError(f"Connection error on {method} {path}: {e.reason}") from None
        if stream:
            return resp  # caller owns reading
        raw = resp.read().decode("utf-8", "replace")
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return {"_raw": raw}

    # -- health / info -----------------------------------------------------
    def health(self) -> bool:
        try:
            self.request("/api/v1/health", timeout_s=5)
            return True
        except Exception:
            return False

    def system_info(self) -> dict:
        return self.request("/api/v1/system-info", timeout_s=15)

    def list_models(self) -> Dict[str, dict]:
        out = {}
        data = self.request("/api/v1/models?show_all=true", timeout_s=15)
        models = data.get("data", data.get("models", [])) if isinstance(data, dict) else data
        for m in models or []:
            name = m.get("name") or m.get("id")
            if name:
                out[name] = m
        return out

    def system_stats(self) -> dict:
        try:
            return self.request("/api/v1/system-stats", timeout_s=10)
        except Exception:
            return {}

    # -- actions ------------------------------------------------------------
    def install_backend(self, recipe: str, backend: str) -> dict:
        return self.request("/api/v1/install", "POST",
                            {"recipe": recipe, "backend": backend},
                            timeout_s=3600)

    def pull_model(self, model_name: str) -> None:
        # Streaming progress events; read and discard, but surface failures.
        resp = self.request(
            "/api/v1/pull", "POST",
            {"model_name": model_name, "stream": True, "do_not_upgrade": True},
            timeout_s=3600, stream=True,
        )
        try:
            for _line in resp:
                pass
        finally:
            resp.close()

    def load_model(self, model_name: str, recipe_options: Optional[dict] = None,
                   pinned: bool = False) -> float:
        """POST /api/v1/load; returns wall-clock load duration in seconds."""
        body: dict = {"model_name": model_name, "save_options": False}
        if pinned:
            body["pinned"] = True
        if recipe_options:
            body.update(recipe_options)
        t0 = time.monotonic()
        self.request("/api/v1/load", "POST", body, timeout_s=DEFAULT_LOAD_TIMEOUT_S)
        return time.monotonic() - t0

    def unload_model(self, model_name: str) -> None:
        try:
            self.request("/api/v1/unload", "POST", {"model_name": model_name}, timeout_s=60)
        except Exception:
            pass  # unload is best-effort


# ---------------------------------------------------------------------------
# Streaming measurement
# ---------------------------------------------------------------------------

class RunResult:
    __slots__ = ("ttft_ms", "total_ms", "output_tokens", "prompt_tokens",
                 "decode_tok_s", "ok", "error", "reasoning_chunks")

    def __init__(self):
        self.ttft_ms: Optional[float] = None
        self.total_ms: Optional[float] = None
        self.output_tokens: Optional[int] = None
        self.prompt_tokens: Optional[int] = None
        self.decode_tok_s: Optional[float] = None
        self.ok = False
        self.error: str = ""
        self.reasoning_chunks = 0

    def to_json(self) -> dict:
        return {
            "ok": self.ok,
            "error": self.error,
            "ttft_ms": self.ttft_ms,
            "total_ms": self.total_ms,
            "output_tokens": self.output_tokens,
            "prompt_tokens": self.prompt_tokens,
            "decode_tok_s": self.decode_tok_s,
            "reasoning_chunks": self.reasoning_chunks,
        }


def _parse_sse_line(line: str) -> Optional[dict]:
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[len("data:"):].strip()
    if payload == "[DONE]":
        return {"done": True}
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def stream_one_request(client: LemonadeClient, path: str, body: dict,
                       deadline: float, timeout_s: float) -> RunResult:
    """Send one streaming request; measure TTFT / decode rate from the SSE stream."""
    res = RunResult()
    t_send = time.monotonic()
    remaining = max(1.0, deadline - t_send)
    try:
        resp = client.request(path, "POST", body, timeout_s=min(timeout_s, remaining),
                              stream=True,
                              extra_headers={"Accept": "text/event-stream"})
    except Exception as e:  # noqa: BLE001
        res.error = f"send failed: {e}"
        return res

    content_chunks = 0
    first_token_at: Optional[float] = None
    usage: Optional[dict] = None
    try:
        with resp:
            for raw in resp:
                if time.monotonic() > deadline:
                    res.error = "client deadline exceeded"
                    break
                ev = _parse_sse_line(raw.decode("utf-8", "replace"))
                if ev is None:
                    continue
                if ev.get("done"):
                    break
                choices = ev.get("choices") or []
                if not choices:
                    if isinstance(ev.get("usage"), dict):
                        usage = ev["usage"]
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("reasoning_content"):
                    res.reasoning_chunks += 1
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                if delta.get("content"):
                    content_chunks += 1
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                if isinstance(choices[0].get("usage"), dict):
                    usage = choices[0]["usage"]
                # Some backends put usage on the top-level of the final chunk
                if isinstance(ev.get("usage"), dict):
                    usage = ev["usage"]
    except Exception as e:  # noqa: BLE001
        res.error = f"stream read failed: {e}"
        return res

    t_done = time.monotonic()
    res.total_ms = (t_done - t_send) * 1000.0
    if first_token_at is not None:
        res.ttft_ms = (first_token_at - t_send) * 1000.0

    if usage:
        res.output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
        res.prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
    if res.output_tokens is None:
        # Fallback: one SSE content chunk ~= one token (backend-agnostic estimate)
        res.output_tokens = content_chunks + res.reasoning_chunks
    if res.output_tokens is None or res.output_tokens <= 0:
        res.error = res.error or "no output tokens observed"
        return res
    res.ok = True

    if res.ttft_ms is not None and res.total_ms is not None:
        decode_ms = res.total_ms - res.ttft_ms
        if decode_ms > 0:
            res.decode_tok_s = max(0.0, (res.output_tokens - 1)) / (decode_ms / 1000.0)
        else:
            res.decode_tok_s = res.output_tokens / max(res.total_ms, 1) * 1000.0
    return res


def run_batch(client: LemonadeClient, path: str, body: dict, concurrency: int,
              timeout_s: float) -> Tuple[List[RunResult], float]:
    """Fire `concurrency` identical requests at once; return results + batch wall time."""
    batch_start = time.monotonic()
    deadline = batch_start + timeout_s
    results: List[RunResult] = []

    barrier = threading.Barrier(concurrency)

    def worker() -> RunResult:
        barrier.wait(timeout=timeout_s)  # align request dispatch
        return stream_one_request(client, path, body, deadline, timeout_s)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(worker) for _ in range(concurrency)]
        for f in futs:
            try:
                results.append(f.result(timeout=timeout_s + 30))
            except Exception as e:  # noqa: BLE001
                r = RunResult()
                r.error = f"worker failed: {e}"
                results.append(r)
    batch_wall = time.monotonic() - batch_start
    return results, batch_wall


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def pct(vals: List[float], p: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return s[idx]


def summarize(vals: List[float]) -> dict:
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "p50": pct(vals, 50),
        "p95": pct(vals, 95),
        "min": min(vals),
        "max": max(vals),
    }


# ---------------------------------------------------------------------------
# Environment recording (best-effort, never fatal)
# ---------------------------------------------------------------------------

def _sh(cmd: List[str], max_lines: int = 8) -> List[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return []
        return [l for l in out.stdout.splitlines() if l.strip()][:max_lines]
    except Exception:
        return []


def record_environment(pins_file: Optional[str] = None) -> dict:
    env: dict = {"timestamp": datetime.now(timezone.utc).isoformat()}

    def put(key: str, val: Any) -> None:
        if val:
            env[key] = val

    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    put("os", line.split("=", 1)[1].strip().strip('"'))
    except OSError:
        pass
    put("kernel", (_sh(["uname", "-r"]) or [""])[0])
    put("cpu", _sh(["lscpu"], 2))
    put("gpu", _sh(["rocminfo"], 12))
    put("amdgpu_module", _sh(["modinfo", "amdgpu"], 2))
    put("rocm_version", _sh(["cat", "/opt/rocm/.info/version"], 1))
    if not env.get("rocm_version"):
        put("rocm_packages", _sh(["dpkg", "-l", "amdrocm*"], 4))

    if pins_file and os.path.isfile(pins_file):
        try:
            with open(pins_file) as f:
                pins = json.load(f)
            env["backend_versions"] = {
                k: v for k, v in pins.items() if k != "checksums"
            }
        except Exception:
            pass
    return env


# ---------------------------------------------------------------------------
# Main benchmark driver
# ---------------------------------------------------------------------------

def parse_backends(specs: List[str]) -> List[Tuple[str, str]]:
    out = []
    for s in specs:
        if ":" in s:
            recipe, backend = s.split(":", 1)
        else:
            recipe, backend = s, ""
        out.append((recipe, backend))
    return out


def model_recipe_hint(name: str) -> Optional[str]:
    low = name.lower()
    if "-mlx" in low or low.startswith("mlx-community"):
        return "lemon-mlx"
    if low.endswith(".gguf") or "q4_" in low or "q8_" in low or "k_m" in low:
        return "llamacpp"
    return None


def load_scenario(name: str) -> Optional[dict]:
    if name in SCENARIOS:
        return SCENARIOS[name]
    if os.path.isfile(name):
        try:
            with open(name) as f:
                return json.load(f)
        except Exception as e:
            print(f"  [warn] scenario file {name} unreadable: {e}", file=sys.stderr)
    return None


def build_plan(client: LemonadeClient, backends: List[Tuple[str, str]],
               model_names: List[str], all_models: bool, scenarios: List[str],
               install: bool, auto_pull: bool) -> dict:
    """Validate backends/models against system-info; return execution plan."""
    sysinfo = client.system_info()
    recipes = sysinfo.get("recipes", {}) if isinstance(sysinfo, dict) else {}
    known = client.list_models()

    installed: Dict[str, List[str]] = {}
    missing_backends: List[Tuple[str, str]] = []
    for recipe, backend in backends:
        rd = recipes.get(recipe, {})
        bd = (rd.get("backends") or {}).get(backend, {}) if backend else {}
        state = bd.get("state", "") if backend else rd.get("state", "")
        if backend and state in ("installed", "update_required", "update_available"):
            installed.setdefault(recipe, []).append(backend)
        elif not backend and state in ("installed", "update_required", "update_available"):
            installed.setdefault(recipe, []).append(recipe)
        elif install:
            print(f"Installing {recipe}:{backend} ...")
            client.install_backend(recipe, backend)
            installed.setdefault(recipe, []).append(backend)
        else:
            missing_backends.append((recipe, backend))

    if missing_backends:
        names = ", ".join(f"{r}:{b}" for r, b in missing_backends)
        raise RuntimeError(
            f"Backend(s) not installed: {names}. Re-run with --install "
            "(or `lemonade backends install <recipe>:<backend>`)."
        )

    if all_models:
        model_names = sorted(known.keys())
    if not model_names:
        raise RuntimeError(
            "No models to benchmark. Pass --models NAME... (lemonade registry "
            "names, see `lemonade models`) or --all-models."
        )

    plan: Dict[str, Any] = {"backends": {}, "models": {}}
    for name in model_names:
        mi = known.get(name)
        if mi is None:
            print(f"  [warn] model '{name}' not found in registry; it will be "
                  f"skipped unless --auto-pull can fetch it.", file=sys.stderr)
            if not auto_pull:
                continue
        recipe = (mi or {}).get("recipe") or model_recipe_hint(name) or ""
        hint = model_recipe_hint(name)
        cand = [r for r in installed if recipe and r == recipe] or \
               ([r for r in installed if hint and r == hint] if hint else []) or \
               list(installed.keys())
        plan["models"][name] = {"recipe": recipe, "download_state": (mi or {}).get("download_state", ""),
                                "candidate_recipes": cand}
        for r in cand:
            plan["backends"].setdefault(r, [])
            for b in installed[r]:
                if b not in plan["backends"][r]:
                    plan["backends"][r].append(b)

    scenarios_out = []
    for s in scenarios:
        sc = load_scenario(s)
        if sc is None:
            print(f"  [warn] unknown scenario '{s}'; skipping", file=sys.stderr)
            continue
        scenarios_out.append({"name": s, **sc})
    plan["scenarios"] = scenarios_out
    return plan


def run_scenario_sweep(client: LemonadeClient, model: str, scenario: dict,
                       concurrency: List[int], runs: int, warmup: int,
                       timeout_s: float, response_log: Optional[list]) -> dict:
    """Run one scenario at each concurrency level against a loaded model."""
    path = "/api/v1/chat/completions"
    out: dict = {"scenario": scenario["name"], "concurrency": {}}
    body_template = {
        "model": model,
        "messages": scenario["messages"],
        "max_tokens": scenario.get("max_tokens", 128),
        "stream": True,
        "temperature": 0.0,  # deterministic-ish for repeatability
    }

    # Warmup (concurrency 1, discarded)
    for _ in range(warmup):
        run_batch(client, path, dict(body_template), 1, timeout_s)

    for c in concurrency:
        rows: List[RunResult] = []
        batch_walls: List[float] = []
        for _ in range(runs):
            results, batch_wall = run_batch(client, path, dict(body_template), c, timeout_s)
            rows.extend(results)
            batch_walls.append(batch_wall)
            if response_log is not None:
                for r in results:
                    response_log.append({
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "model": model, "scenario": scenario["name"],
                        "concurrency": c, **r.to_json(),
                    })

        ok = [r for r in rows if r.ok]
        failed = [r.error for r in rows if not r.ok]
        ttfts = [r.ttft_ms for r in ok if r.ttft_ms is not None]
        decodes = [r.decode_tok_s for r in ok if r.decode_tok_s is not None]
        # aggregate tok/s per batch: sum output tokens / batch wall time
        agg_rates = []
        idx = 0
        for wall in batch_walls:
            batch_rows = rows[idx:idx + c]
            idx += c
            toks = sum(r.output_tokens or 0 for r in batch_rows if r.ok)
            agg_rates.append(toks / wall if wall > 0 else 0.0)

        out["concurrency"][str(c)] = {
            "runs": [r.to_json() for r in rows],
            "ok": len(ok),
            "failed": len(failed),
            "errors": failed[:5],
            "ttft_ms": summarize(ttfts),
            "decode_tok_s": summarize(decodes),
            "aggregate_tok_s": summarize(agg_rates),
        }
        print(f"    concurrency {c}: ok={len(ok)}/{len(rows)} "
              f"ttft_mean={summarize(ttfts).get('mean'):.0f}ms "
              f"decode={summarize(decodes).get('mean'):.1f} t/s "
              f"agg={summarize(agg_rates).get('mean'):.1f} t/s", flush=True)
    return out


def run_model_on_backend(client: LemonadeClient, model: str, recipe: str, backend: str,
                         scenarios: List[dict], concurrency: List[int], runs: int,
                         warmup: int, timeout_s: float, response_log: Optional[list],
                         reload_between_scenarios: bool) -> dict:
    """Load model on backend, run all scenarios, unload. Records load duration."""
    recipe_options = {f"{recipe}_backend": backend}

    def load() -> float:
        return client.load_model(model, recipe_options, pinned=True)

    print(f"  load {model} on {recipe}:{backend} ...", flush=True)
    load_s = load()
    mem_before = client.system_stats()

    out: dict = {"model": model, "recipe": recipe, "backend": backend,
                 "load_duration_s": round(load_s, 2), "scenarios": {}}

    for sc in scenarios:
        print(f"    scenario {sc['name']} ...", flush=True)
        if reload_between_scenarios and out["scenarios"]:
            # Avoid cross-scenario prefix-cache reuse (e.g. vLLM prefix cache)
            client.unload_model(model)
            load_s2 = load()
            out["load_duration_s"] = round(load_s2, 2)
        out["scenarios"][sc["name"]] = run_scenario_sweep(
            client, model, sc, concurrency, runs, warmup, timeout_s, response_log)

    mem_after = client.system_stats()
    out["memory_delta"] = _memory_delta(mem_before, mem_after)
    client.unload_model(model)
    return out


def _memory_delta(before: dict, after: dict) -> Optional[dict]:
    """Best-effort memory delta from system-stats; returns {} if unavailable."""
    try:
        def grab(d: dict, keys: List[str]) -> Optional[float]:
            for k in keys:
                if isinstance(d, dict) and k in d:
                    v = d[k]
                    if isinstance(v, (int, float)):
                        return float(v)
            return None

        b = grab(before, ["memory_used_mb", "used_mb", "mem_used", "gpu_memory_used_mb"])
        a = grab(after, ["memory_used_mb", "used_mb", "mem_used", "gpu_memory_used_mb"])
        if b is None or a is None:
            return None
        return {"delta_mb": round(a - b, 1), "before_mb": b, "after_mb": a}
    except Exception:
        return None


def render_markdown(results: List[dict], env: dict) -> str:
    L = ["# lemon-mlx vs vLLM vs llamacpp ROCm/Vulkan — benchmark summary", ""]
    L.append(f"Environment: `{env.get('os', '?')}`, kernel `{env.get('kernel', '?')}`, "
             f"run at `{env.get('timestamp', '?')}`")
    L.append("")
    L.append("| Backend | Model | Scenario | Conc | TTFT ms (mean±std) | Decode t/s | Aggregate t/s | OK |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        for sname, sc in r["scenarios"].items():
            for c, s in sorted(sc["concurrency"].items(), key=lambda kv: int(kv[0])):
                t = s["ttft_ms"]
                d = s["decode_tok_s"]
                a = s["aggregate_tok_s"]
                tt = f"{t.get('mean', float('nan')):.0f}±{t.get('std', 0):.0f}" if t.get("n") else "-"
                de = f"{d.get('mean', float('nan')):.1f}" if d.get("n") else "-"
                ag = f"{a.get('mean', float('nan')):.1f}" if a.get("n") else "-"
                L.append(f"| {r['recipe']}:{r['backend']} | {r['model']} | {sname} | "
                         f"{c} | {tt} | {de} | {ag} | {s['ok']}/{s['ok'] + s['failed']} |")
    L.append("")
    L.append("> Raw per-run JSON in the sibling `results-*.json` file. See the benchmark spec "
             "for methodology (environment pinning, format disclosure, workload-regime reading).")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Apples-to-apples backend benchmark for lemonade (issue #1642).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--server", default="http://127.0.0.1:13305",
                    help="lemond base URL")
    ap.add_argument("--api-key", default="", help="API key if lemond requires one")
    ap.add_argument("--models", nargs="*", default=[],
                    help="lemonade registry model names to benchmark "
                         "(e.g. 'Qwen3.5-0.8B-MLX' 'Qwen3-4B-Q4_K_M')")
    ap.add_argument("--all-models", action="store_true",
                    help="benchmark every model currently in the registry")
    ap.add_argument("--backends", nargs="*", default=DEFAULT_BACKENDS,
                    help="recipe:backend pairs, e.g. lemon-mlx:rocm")
    ap.add_argument("--scenarios", nargs="*", default=DEFAULT_SCENARIOS,
                    help="scenario names (chat-short, code-explain, chat-long-output) or JSON file paths")
    ap.add_argument("--concurrency", nargs="*", type=int, default=DEFAULT_CONCURRENCY,
                    help="concurrency levels for the batch sweep")
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                    help="measurement runs per (backend x model x scenario x concurrency)")
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                    help="discarded warmup completions (concurrency 1)")
    ap.add_argument("--timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT_S,
                    help="per-request timeout in seconds")
    ap.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN_S,
                    help="pause between backends (thermal isolation)")
    ap.add_argument("--install", action="store_true",
                    help="install listed backends first (idempotent)")
    ap.add_argument("--auto-pull", action="store_true",
                    help="pull models that are not downloaded yet (cache-first)")
    ap.add_argument("--interleave", action="store_true",
                    help="round-robin interleave backends instead of sequential+cooldown")
    ap.add_argument("--reload", action="store_true",
                    help="reload the model between scenarios (avoids cross-scenario "
                         "prefix-cache reuse, e.g. vLLM; costs load time)")
    ap.add_argument("--output-dir", default="./bench-results",
                    help="directory for raw JSON + markdown summary")
    ap.add_argument("--pins-file", default="src/cpp/resources/backend_versions.json",
                    help="path to lemonade backend_versions.json for pin recording")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate and print the plan without running anything")
    args = ap.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_path = os.path.join(args.output_dir, f"results-{ts}.json")
    summary_path = os.path.join(args.output_dir, f"summary-{ts}.md")

    env = record_environment(args.pins_file)
    client = LemonadeClient(args.server, args.api_key, timeout_s=args.timeout)

    if not client.health():
        print(f"lemond not reachable at {args.server}. Start it (e.g. `lemond`) "
              f"or point --server at a running instance.", file=sys.stderr)
        return 1
    print(f"Connected to {args.server}")

    try:
        plan = build_plan(client, parse_backends(args.backends), args.models,
                          args.all_models, args.scenarios, args.install, args.auto_pull)
    except RuntimeError as e:
        print(f"Plan error: {e}", file=sys.stderr)
        return 1

    if not plan["scenarios"]:
        print("No valid scenarios; nothing to run.", file=sys.stderr)
        return 1

    print("\nPlan:")
    for name, mi in plan["models"].items():
        print(f"  model {name}: recipe={mi['recipe'] or 'auto'} -> recipes {mi['candidate_recipes']}")
    for r, bs in plan["backends"].items():
        print(f"  backend {r}: {', '.join(bs)}")
    print(f"  scenarios: {[s['name'] for s in plan['scenarios']]}")
    print(f"  concurrency: {args.concurrency}, runs: {args.runs}, warmup: {args.warmup}")
    if args.dry_run:
        print("\n[dry-run] not executing.")
        return 0

    if args.auto_pull:
        for name in plan["models"]:
            if plan["models"][name]["download_state"] not in ("downloaded", "ready", ""):
                print(f"Pulling {name} ...")
                try:
                    client.pull_model(name)
                except Exception as e:
                    print(f"  [warn] pull failed for {name}: {e}", file=sys.stderr)

    results: List[dict] = []
    response_log: List[dict] = []

    def run_one(name: str, r: str, b: str, scs: List[dict]) -> None:
        try:
            results.append(run_model_on_backend(
                client, name, r, b, scs, args.concurrency, args.runs,
                args.warmup, args.timeout, response_log, args.reload))
        except Exception as e:  # noqa: BLE001
            print(f"  [fail] {name} on {r}:{b}: {e}", file=sys.stderr)

    if args.interleave:
        # Round-robin: one pass per (model x scenario) per backend (spec sec. 7.5).
        # Load/unload churn is the cost; thermal drift averaging is the benefit.
        combos = [
            (name, r, b, [sc])
            for name, mi in plan["models"].items()
            for r in mi["candidate_recipes"]
            for b in plan["backends"].get(r, [])
            for sc in plan["scenarios"]
        ]
        for i, combo in enumerate(combos):
            if i > 0:
                time.sleep(args.cooldown)
            print(f"\n[{i+1}/{len(combos)}] {combo[0]} on {combo[1]}:{combo[2]} / "
                  f"{combo[3][0]['name']}", flush=True)
            run_one(*combo)
    else:
        # Sequential per backend with a cooldown pause between backends
        # (thermal isolation, spec sec. 2). Backend order follows the plan.
        backend_order: List[Tuple[str, str]] = []
        for name, mi in plan["models"].items():
            for r in mi["candidate_recipes"]:
                for b in plan["backends"].get(r, []):
                    if (r, b) not in backend_order:
                        backend_order.append((r, b))
        for pos, (r, b) in enumerate(backend_order):
            print(f"\n=== backend {r}:{b} ===", flush=True)
            for name, mi in plan["models"].items():
                if r not in mi["candidate_recipes"]:
                    continue
                print(f"\n--- {name} ---", flush=True)
                run_one(name, r, b, plan["scenarios"])
            if pos < len(backend_order) - 1:
                print(f"\n[cooldown {args.cooldown}s between backends]", flush=True)
                time.sleep(args.cooldown)

    payload = {
        "issue": "https://github.com/lemonade-sdk/lemonade/issues/1642",
        "spec": "mlx-backend-benchmark-spec.md",
        "environment": env,
        "config": {
            "server": args.server,
            "backends": args.backends,
            "models": args.models or "(all)",
            "scenarios": [s["name"] for s in plan["scenarios"]],
            "concurrency": args.concurrency,
            "runs": args.runs,
            "warmup": args.warmup,
            "timeout_s": args.timeout,
        },
        "results": results,
        "response_log": response_log,
    }
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    md = render_markdown(results, env)
    with open(summary_path, "w") as f:
        f.write(md + "\n")

    print("\n=== SUMMARY ===")
    print(md)
    print(f"\nRaw JSON: {results_path}")
    print(f"Summary : {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
