#!/usr/bin/env python3
"""runner_label_selfcheck.py — a job that needs the NPU must not be scheduled by a bare label.

Why this exists, measured 2026-09-18:

  * This repo has TWO self-hosted runners and only one has the NPU:

        ryzen-pr-agent      self-hosted,Linux,X64,pr-agent,ryzen
        strixhalo-pr-agent  self-hosted,Linux,X64,pr-agent,strix-halo

  * `bench.yml` said `runs-on: self-hosted`, so it matched both. On ryzen-pr-agent
    `ldconfig` finds no XRT, `Check XRT` set `XRT_OK=false`, and every step below it
    carried `if: env.XRT_OK == 'true'` — so the engine build and the benchmark were
    SKIPPED and the job reported SUCCESS. 42 of the last 60 runs of that workflow were
    `success` with a median duration of 0.6 min, which is not enough time to build or
    measure anything (#2508). The gate that reported green had never run.

  * It was the only workflow still on the bare label. end-to-end-smoke.yml,
    kernel-object-lint.yml, npu-reset.yml and bundle-decoder-check.yml all pin
    `[self-hosted, strix-halo]`.

So the rule is narrow and mechanical: a workflow that names NPU/XRT specifics must not
select its runner by the bare label, or it can be scheduled onto a box that cannot do the
work and report success for having skipped it.

That rule cannot be checked by running the real tree alone, because after the fix no real
file violates it — so this script builds its OWN fixtures (one violating, one correct) and
requires its detector to fail the first and pass the second. A detector that cannot fail
would otherwise pass forever on a clean tree, which is the mistake this whole file is
about.

Run: python3 Testing/runner_label_selfcheck.py [DIR]
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# A workflow that mentions any of these is doing NPU/XRT work, i.e. it can only run on the
# box that has the device.
HARDWARE_MARKERS = (
    "xrt_coreutil",
    "/dev/accel",
    "npu_engine_universal",
    "xclbin",
    "strix-halo",
)
# The correct form, and the bare form that matches every self-hosted runner.
PINNED = re.compile(r"runs-on:\s*\[\s*self-hosted\s*,\s*[\w-]+\s*\]")
BARE = re.compile(r"runs-on:\s*self-hosted\s*$")

fixtures = {
    "violating": """name: Synthetic NPU job
on: push
jobs:
  bench:
    runs-on: self-hosted
    steps:
      - run: if ! ldconfig -p | grep -q xrt_coreutil; then exit 1; fi
""",
    "correct": """name: Synthetic NPU job
on: push
jobs:
  bench:
    runs-on: [self-hosted, strix-halo]
    steps:
      - run: if ! ldconfig -p | grep -q xrt_coreutil; then exit 1; fi
""",
    "correct_unrelated": """name: Synthetic CPU job
on: push
jobs:
  lint:
    runs-on: self-hosted
    steps:
      - run: echo "no device needed here"
""",
}


def violations(text: str) -> list[str]:
    """Lines that select a self-hosted runner by the bare label in a workflow that does
    NPU/XRT work. `runs-on: self-hosted` in a workflow with no hardware marker is fine —
    the defect is the pair, not the label on its own."""
    if not any(m in text for m in HARDWARE_MARKERS):
        return []
    return [l.strip() for l in text.splitlines() if BARE.search(l)]


def scan(directory: Path) -> list[tuple[Path, str]]:
    found = []
    for wf in sorted(directory.glob("*.yml")) + sorted(directory.glob("*.yaml")):
        for line in violations(wf.read_text(encoding="utf-8", errors="replace")):
            found.append((wf, line))
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", nargs="?", default=str(REPO / ".github" / "workflows"))
    args = ap.parse_args(argv)

    fails = 0
    # ---- controls first: prove the detector can fail, on fixtures we control ----
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        for name, body in fixtures.items():
            (tdp / f"{name}.yml").write_text(body, encoding="utf-8")
        caught = {p.stem for p, _ in scan(tdp)}
        def check(ok: bool, label: str) -> None:
            nonlocal fails
            if not ok:
                print(f"  FAIL {label}")
                fails += 1
        check("violating" in caught, "the bare label + a hardware marker is reported")
        check("correct" not in caught, "the pinned label is not reported")
        check("correct_unrelated" not in caught,
              "the bare label alone (no hardware marker) is not reported")

    # ---- then the real tree, which must be clean ----
    real = scan(Path(args.dir))
    for wf, line in real:
        print(f"  FAIL {wf.relative_to(REPO) if REPO in wf.parents else wf}: "
              f"'{line}' — this workflow does NPU/XRT work, so it must pin the runner "
              f"that has the device: runs-on: [self-hosted, strix-halo]")
        fails += 1

    if fails:
        print(f"\n{fails} FAILURE(S) — a job that needs the NPU can be scheduled onto a "
              f"runner that does not have it, and skip its own work while reporting success")
        return 1
    print(f"runner labels ok ({len(list(Path(args.dir).glob('*.y*ml')))} workflow(s) checked, "
          f"3 controls)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
