#!/usr/bin/env python3
"""npu_worker_bundle_selfcheck.py — the NPU worker bundle stays shippable.

The native NPU worker (`npu_engine_universal`) cannot be compiled in CI: the
release runner's only XRT is Ubuntu noble's `libxrt-dev` (XRT 2.13, header layout
`xrt/experimental/*`), while this tree includes `xrt/xrt_device.h` (XRT >= 2.14,
AMD's `/opt/xilinx/xrt` layout), and AMD ships XRT as source/RPM only. So the
worker ships as a **vendored prebuilt** that packaging prefers to override with a
freshly built one — and a prebuilt binary is exactly the kind of thing that rots
silently. This gate pins the contract:

  * the manifest's sha256 matches the committed binaries (a bumped file with a
    stale manifest fails here, not in a user's install);
  * the manifest still records where it came from and how to regenerate it;
  * the worker's RUNPATH can find the bundled OpenMP runtime in the layouts we
    actually ship (`$ORIGIN` and `$ORIGIN/../lib/1bit`) — the bug this all exists
    for was a worker that only started on the machine that built it;
  * both staging paths (packaging/Makefile, .github/workflows/release.yml) still
    stage the pair together, so a future refactor cannot quietly drop libomp.

Run: python3 Testing/npu_worker_bundle_selfcheck.py
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREBUILT = ROOT / "packaging" / "prebuilt"
MANIFEST = PREBUILT / "manifest.json"
WORKER = PREBUILT / "npu_engine_universal"
LIBOMP = PREBUILT / "libomp.so"
MAKEFILE = ROOT / "packaging" / "Makefile"
RELEASE = ROOT / ".github" / "workflows" / "release.yml"

failures: list[str] = []
checks = 0


def check(ok: bool, label: str, detail: str = "") -> None:
    global checks
    checks += 1
    if not ok:
        failures.append(f"{label}{(': ' + detail) if detail else ''}")


def sha256(p: Path) -> str:
    return subprocess.run(["sha256sum", str(p)], capture_output=True, text=True).stdout.split()[0]


# ── 1. the committed bundle matches its manifest ──────────────────────────────
check(WORKER.is_file(), "the vendored worker is present", str(WORKER))
check(LIBOMP.is_file(), "the bundled libomp.so is present", str(LIBOMP))
check(MANIFEST.is_file(), "the prebuilt manifest is present", str(MANIFEST))

if WORKER.is_file() and LIBOMP.is_file() and MANIFEST.is_file():
    man = json.loads(MANIFEST.read_text())
    files = man.get("files", {})
    for name, path in (("npu_engine_universal", WORKER), ("libomp.so", LIBOMP)):
        want = (files.get(name) or {}).get("sha256")
        got = sha256(path)
        check(want == got, f"manifest sha256 matches {name}",
              f"manifest {str(want)[:16]}… != file {got[:16]}… (regenerate and bump the manifest)")
        declared = (files.get(name) or {}).get("size")
        check(declared == path.stat().st_size, f"manifest size matches {name}",
              f"{declared} != {path.stat().st_size}")

    # provenance a future maintainer needs to rebuild it
    for field in ("built_from_commit", "built_on", "build_command", "toolchain", "why", "regenerate"):
        check(bool(man.get(field)), f"manifest records {field}")
    check("xrt" in json.dumps(man).lower(), "manifest names the XRT runtime dependency")

# ── 2. the worker can find its bundled runtime in the layouts we ship ────────
if WORKER.is_file():
    runpath = ""
    if shutil.which("readelf"):
        out = subprocess.run(["readelf", "-d", str(WORKER)], capture_output=True, text=True).stdout
        m = re.search(r"\((?:RUNPATH|RPATH)\)\s+Library \w+ path: \[([^\]]*)\]", out)
        runpath = m.group(1) if m else ""
    if runpath:
        check("$ORIGIN" in runpath.split(":")[0],
              "the worker searches its own directory first ($ORIGIN)", runpath)
        check("$ORIGIN/../lib/1bit" in runpath,
              "the worker also searches ../lib/1bit (tarball + .deb layouts)", runpath)
        check("/opt/xilinx/xrt/lib" in runpath,
              "the worker still searches the system XRT directory", runpath)
    else:
        # No binutils: fall back to the string in .dynstr, which is what the
        # loader reads anyway.
        blob = WORKER.read_bytes()
        check(b"$ORIGIN/../lib/1bit" in blob,
              "the worker carries the $ORIGIN/../lib/1bit runpath (no readelf available)")

# ── 3. both staging paths still ship the pair together ────────────────────────
mk = MAKEFILE.read_text() if MAKEFILE.is_file() else ""
rel = RELEASE.read_text() if RELEASE.is_file() else ""
check("prebuilt/npu_engine_universal" in mk, "packaging/Makefile falls back to the vendored worker")
check("prebuilt/libomp.so" in mk, "packaging/Makefile stages the bundled libomp.so")
check(re.search(r"ln -sf\s+\.\./lib/1bit/npu_engine_universal", mk) is not None,
      "packaging/Makefile symlinks usr/bin/1bit-npu at the worker (so $ORIGIN resolves)")
check("prebuilt/npu_engine_universal" in rel, "release.yml falls back to the vendored worker")
check("packaging/prebuilt/libomp.so" in rel, "release.yml stages the bundled libomp.so")
check("dist/lib/1bit/libomp.so" in rel, "release.yml puts libomp where the RUNPATH looks")
check("/usr/lib/1bit" in rel, "the .deb also installs /usr/lib/1bit (the $ORIGIN/../lib/1bit target)")

# ── report ────────────────────────────────────────────────────────────────────
if failures:
    print("NPU worker bundle FAILED:")
    for f in failures:
        print(f"  ✗ {f}")
    sys.exit(1)

print(f"NPU worker bundle OK ({checks} checks: manifest ↔ binaries, RUNPATH, both staging paths)")
