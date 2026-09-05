# FastFlowLM Upstream Report Package

Packaged upstream-report candidates for [ROCm/FastFlowLM](https://github.com/ROCm/FastFlowLM)
(upstream of the vendored `third_party/FastFlowLM` submodule @ `17a35cb7`).

Two runtime anomalies were closed on our side as intrinsic/one-off but are
genuine upstream behavioral questions with minimal repros. Both are fully
documented with captures in `npu-infer/docs/txn-decode-findings.md` and were
tracked here as issues #2052 / #2065 (closed) and #2083 (packaging).

| Report | Anomaly | Repo issue | Status |
|--------|---------|------------|--------|
| [`01-batched-prefill-rope-divergence.md`](01-batched-prefill-rope-divergence.md) | Batched `prefill(ids)` produces different KV/rope state than N×`forward(tok)` on the same prompt — an internal inconsistency of the runtime's own decode path | #2052 (closed), R39/R67 | Ready to file |
| [`02-reboot-drift.md`](02-reboot-drift.md) | Byte-identical runtime outputs move across one reboot (deterministic per boot, reverts on next fresh boot) | #2065 (closed), R40/R41/R66 | Ready to file |

## Packaging decision (issue #2083)

File **two separate upstream issues** — the mechanisms are unrelated
(runtime-internal prefill/rope inconsistency vs. cross-reboot kernel/driver
state sensitivity). Each report is self-contained for an external maintainer.

Filing checklist per report:

- [ ] Repo + issue title from the report header
- [ ] Environment list included (xclbins, XRT runlist stack, kernel/driver IDs)
- [ ] Repro harness: `npu-infer/tools/capture/run_qwen3_npu`-style, tokens `1000`+`1001`
- [ ] Capture artifacts from `/home/bcloud/.cache/rtcap/` on the dev box attached
      or referenced (rtcap captures + byte-parity comparison scripts from R39/R41)
- [ ] Reference the detailed tracking doc `npu-infer/docs/txn-decode-findings.md`
