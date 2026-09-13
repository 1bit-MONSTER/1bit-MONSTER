# RESULTS — FLM-parity harness smoke test (task-1, 2026-09-09)

Harness: `benchmarks/flm_parity.sh`. Goal `mttxt22c-a6rv75` task-1.

## What was validated

- **FLM on-box measurement** works via the hidden `flm bench` command
  (v1.0.4 still has it; not in `--help`). `-i <config.json>` + `--bench-iterations`
  control the sweep; it writes `bench_<tag>_<yyyymmdd>.csv` (ttft/prefill/decode
  per context length).
- **Native measurement** works: `npu_engine_<model> model.q4nx <decode_tokens>
  <ids_file>` (whitespace-separated token IDs from
  `engine/npu/tokenizer/tokenize`, comma output → `tr ',' ' '`). Parsed markers:
  `=== Prefill N ===`, `Prefill: Xms (Y ms/tok)`, final `(Z tok/s)`.
- **Required env**: `NPU_XCLBIN_DIR=$PWD/engine/npu/xclbins` — the pre-built
  `npu_engine_*` binaries carry a stale compiled-in path
  (`/home/bcloud/1bit-MONSTER-pi/...`); the harness exports this automatically.
- Fixed a `set -euo pipefail` + `| head -1` SIGPIPE race that intermittently
  zeroed the native parse (replaced with `grep -m1` + `|| true`).

## On-box reference + native numbers (Qwen3-0.6B, ctx=1k, 3 runs)

| metric | native | FLM (Strix Halo, v1.0.4) | published (Kraken Pt, v0.9.31) |
|---|---:|---:|---:|
| decode tok/s | **2** | 70.8–74.2 | 66.5 |
| prefill tok/s | **125** (128 tok, capped) | 1163–1410 | 1494 |
| TTFT (s) | **1.07–1.08** (128 tok) | 0.70–0.84 (~1928 tok) | — |

## Findings (feed tasks 2–6)

1. **Decode gap is ~35–37×** on Qwen3-0.6B: native 2 tok/s vs FLM ~71 tok/s.
   This is the core "match FLM performance" gap.
2. **Native prefill is hard-capped at 128 tokens** (`npt > XM → npt = XM`,
   `engine/npu/src/npu_engine_universal.cpp:3520`). Until the M=128-bake is
   lifted, native prefill/TTFT cannot be compared fairly against FLM at 1k–32k —
   this is a prerequisite for tasks 3–5, not just an optimization.
3. **Cross-hardware drift is real**: FLM v1.0.4 on Strix Halo decodes ~7% faster
   than the published Kraken Point table. The harness's primary bar is therefore
   the **on-box FLM measurement**, with the published tables kept for reporting.

## Known harness limitations (next iterations)

- Only the Qwen3 family has embedded published tables; other families need their
  rows added or read live from FLM.
- Decode uses a short 8-token window in the smoke test; real runs should use 32+
  tokens for stable decode tok/s.
- The native prefill-cap is reported but not yet worked around.
