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


# ── 5. the native lane is registered, and it is routed/priced ahead of FLM ────
# `npu_xrt` was declared in discover() and never pushed into `backends_` — its
# block was left unclosed, so it printed a "✅ detected" banner line and then
# existed in no list a consumer walks. That is why the Q4NX route named no native
# entry (#2358). These checks make the omission impossible to reintroduce quietly.
import re as _re

manager = text(ROOT / "src" / "backend_manager.cpp")
router = text(ROOT / "src" / "model_router.cpp")

lines = manager.splitlines()
try:
    start = next(i for i, l in enumerate(lines) if l.strip().startswith("void BackendManager::discover"))
    end = next(i for i, l in enumerate(lines[start + 1:], start + 1) if _re.match(r"^}", l))
except StopIteration:
    start = end = -1
check(start > 0 and end > start, "discover() was located in backend_manager.cpp")

if start > 0:
    ids = [(m.group(1), i) for i, l in enumerate(lines)
           if (m := _re.search(r'info\.id = "([^"]+)"', l)) and start < i < end]
    ids.append(("<end>", end))
    unregistered = []
    for (name, i), (_, j) in zip(ids, ids[1:]):
        if "backends_.push_back(info)" not in "\n".join(lines[i:j]):
            unregistered.append(name)
    check(len(ids) - 1 >= 15, "discover() declares the expected set of lanes",
          f"only {len(ids) - 1} lanes found")
    check(not unregistered, "every lane declared in discover() is registered in backends_",
          f"declared but never pushed: {', '.join(unregistered)}")
    check("npu_xrt" in [n for n, _ in ids], "the native npu_xrt lane is declared")

# Priority: the native lane must rank above the optional FLM lane.
def _prio(ident: str) -> int | None:
    i = manager.find(f'info.id = "{ident}"')
    if i < 0:
        return None
    m = _re.search(r"info\.priority = tier_priority\(info\.tier\) \+ (\d+)", manager[i:i + 1200])
    return int(m.group(1)) if m else None

prio_xrt, prio_flm = _prio("npu_xrt"), _prio("npu_flm")
check(prio_xrt is not None and prio_flm is not None,
      "both NPU lanes declare a tier_priority offset", f"xrt={prio_xrt} flm={prio_flm}")
if prio_xrt is not None and prio_flm is not None:
    check(prio_flm < prio_xrt, "the optional FLM lane ranks below the native lane",
          f"npu_flm +{prio_flm} vs npu_xrt +{prio_xrt}")

# Route order for Q4NX: native first, FLM as fallback, CPU last.
m = _re.search(r'cfg\.format == ModelFormat::Q4NX\)\s*\n\s*return \{\{([^}]*)\}', router)
check(m is not None, "the qwen3 Q4NX route is present in model_router.cpp")
if m:
    order = [x.strip().strip('"') for x in m.group(1).split(",")]
    check(order[:2] == ["npu_xrt", "npu_flm"],
          "the Q4NX route tries the native worker before the FLM lane", str(order))
    check("cpu_generic" in order, "the Q4NX route keeps a CPU fallback", str(order))

# ── report ────────────────────────────────────────────────────────────────────
if failures:
    print("NPU lane contract FAILED:")
    for f in failures:
        print(f"  ✗ {f}")
    sys.exit(1)

print(f"NPU lane contract OK ({checks} checks: native worker via NPU_ENGINE_BIN, "
      f"legacy FLM probe quiet, FLM documented as optional)")
