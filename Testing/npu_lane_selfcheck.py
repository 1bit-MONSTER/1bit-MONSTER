#!/usr/bin/env python3
"""npu_lane_selfcheck.py — the NPU lane is the engine's own worker, and it must be
described the same way by the installer, the run time, and the docs.

The NPU lane runs on the engine's own FLM-free C++ engine: `src/backend_npu.cpp`
fork/execs `npu_engine_universal` (resolved from `NPU_ENGINE_BIN`, else
`./npu_engine_universal`) and drives the pre-compiled xclbins. FastFlowLM is a
*separate, optional* lane (`npu_flm`) — but the tree had drifted:

  * `tests/backends/backend_npu.cpp`, the legacy FLM **test harness** compiled into
    the binary, printed `NPU: FLM not installed` at probe time, so a missing
    optional runtime read as "the NPU is broken" (issue #2358);
  * `install.sh` never built or mentioned `npu_engine_universal` at all, so a
    source install ended with no NPU lane and no explanation.

A compiler cannot see either problem, so this check pins the contract instead:
the installer probes the same worker the run time does, the legacy FLM probe stays
quiet unless asked, and no user-facing doc claims the NPU needs FastFlowLM.

Run: python3 Testing/npu_lane_selfcheck.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NATIVE = ROOT / "src" / "backend_npu.cpp"                  # npu_xrt — the engine's own lane
LEGACY_HARNESS = ROOT / "tests" / "backends" / "backend_npu.cpp"   # legacy FLM test backend
FLM_LANE = ROOT / "src" / "backend_npu_flm.cpp"            # npu_flm — optional FLM lane
INSTALLER = ROOT / "install.sh"
SITE_INSTALLER = ROOT / "site" / "install.sh"
DOCS = [
    ROOT / "docs" / "wiki" / "Installation.md",
    ROOT / "docs" / "guides" / "getting-started.md",
    ROOT / "docs" / "wiki" / "npu-architecture.md",
]

failures: list[str] = []
checks = 0


def check(ok: bool, label: str, detail: str = "") -> None:
    global checks
    checks += 1
    if not ok:
        failures.append(f"{label}{(': ' + detail) if detail else ''}")


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


native = text(NATIVE)
harness = text(LEGACY_HARNESS)
flm = text(FLM_LANE)
installer = text(INSTALLER)
site_installer = text(SITE_INSTALLER)

for label, src in (("src/backend_npu.cpp", native),
                   ("tests/backends/backend_npu.cpp", harness),
                   ("src/backend_npu_flm.cpp", flm),
                   ("install.sh", installer)):
    check(bool(src), f"{label} is readable")

# ── 1. the native lane is the engine's own worker ─────────────────────────────
check("NPU_ENGINE_BIN" in native and "npu_engine_universal" in native,
      "the native NPU lane drives npu_engine_universal via NPU_ENGINE_BIN")

# The installer must look where the worker actually lands, and say how to enable it.
check("npu_engine_universal" in installer,
      "install.sh knows about the native NPU worker",
      "install.sh never mentions npu_engine_universal")
check("NPU_ENGINE_BIN" in installer,
      "install.sh tells the user how to point at the worker (NPU_ENGINE_BIN)")
check("build/npu_engine_universal" in installer,
      "install.sh looks where a source build puts the worker (build/npu_engine_universal)")
check(installer == site_installer, "site/install.sh is in sync with install.sh",
      "run `make package-site`, or edit install.sh and re-copy")

# ── 2. the legacy FLM harness cannot masquerade as the NPU gate ───────────────
check("NPU: FLM not installed" not in harness,
      "the bare `NPU: FLM not installed` line is gone from the legacy harness")
# Every stderr write in the harness's availability probe must be env-gated.
idx = harness.find("legacy FLM test backend disabled")
check(idx > 0, "the legacy harness still explains itself (behind a flag)")
check(idx > 0 and "NPU_FLM_TEST_VERBOSE" in harness[max(0, idx - 400):idx],
      "that explanation is behind the NPU_FLM_TEST_VERBOSE gate",
      "fprintf reaches stderr without passing the getenv gate")

# ── 3. the optional FLM lane says it is the FLM lane ─────────────────────────
for m in re.finditer(r'fprintf\(\s*stderr,?\s*((?:"(?:[^"\\]|\\.)*"\s*)+)', flm):
    msg = " ".join(re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1)))
    low = msg.lower()
    if "not found" not in low:
        continue
    check("flm" in low, "the FLM-lane message names FLM", msg[:80])
    check("does not need" in low or "native" in low,
          "the FLM-lane message says the native NPU lane is unaffected", msg[:80])

# ── 4. no doc claims the NPU requires FastFlowLM ─────────────────────────────
docs_blob = "\n".join(text(d) for d in DOCS)
for bad in ("required for the NPU lane", "NPU lane needs FastFlowLM",
            "NPU lane is executed by FastFlowLM"):
    check(bad not in docs_blob, f"docs do not claim the NPU needs FLM ({bad!r})")

check("NPU_ENGINE_BIN" in docs_blob,
      "the native worker is documented (NPU_ENGINE_BIN)")
check("npu_engine_universal" in docs_blob,
      "docs name npu_engine_universal as the NPU runtime")
check(("optional" in docs_blob.lower() and "FastFlowLM" in docs_blob),
      "docs describe FastFlowLM as optional, not required")

# ── report ────────────────────────────────────────────────────────────────────
if failures:
    print("NPU lane contract FAILED:")
    for f in failures:
        print(f"  ✗ {f}")
    sys.exit(1)

print(f"NPU lane contract OK ({checks} checks: native worker via NPU_ENGINE_BIN, "
      f"legacy FLM probe quiet, FLM documented as optional)")
