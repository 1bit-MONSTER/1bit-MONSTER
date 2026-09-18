#!/usr/bin/env python3
"""zaya launch resolution — does the harness find the engine on a bare build?

Why this exists: the benchmark harness launches its primary engine as
`build/zaya_server`, which is not a build product. It is an argv[0] symlink that
install.sh and packaging/Makefile create so the single ELF can answer to the
legacy name. `cmake --build build --target onebin` emits only `build/1bit`, so on
a fresh clone the harness died on a missing file — after the README's own build
step, which until #2477 named a target that no longer existed at all (#2478).

Reading the source cannot catch this: the path is a string in a config default
and the failure is a spawn of something absent. So this check builds each layout
in a temp dir and asserts what config.zaya_launch_prefix returns for it — the
installer layout, the bare cmake layout, and the nothing-built layout.

Stdlib only, no build, no device — safe to run in the host-only suite.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks" / "engine_comparison"))

import config  # noqa: E402

FAILED: list[str] = []


def expect(what: str, got, want) -> None:
    if got == want:
        print(f"  ok   {what}")
    else:
        print(f"  FAIL {what}\n         got:  {got!r}\n         want: {want!r}")
        FAILED.append(what)


def expect_raises(what: str, fn, *needles: str) -> None:
    try:
        got = fn()
    except FileNotFoundError as exc:
        text = str(exc)
        missing = [n for n in needles if n not in text]
        if missing:
            print(f"  FAIL {what}\n         message lacks {missing}: {text!r}")
            FAILED.append(what)
        else:
            print(f"  ok   {what}")
    except Exception as exc:  # noqa: BLE001 - report anything unexpected
        print(f"  FAIL {what}\n         wrong exception: {type(exc).__name__}: {exc}")
        FAILED.append(what)
    else:
        print(f"  FAIL {what}\n         returned instead of raising: {got!r}")
        FAILED.append(what)


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        build = Path(td) / "build"
        build.mkdir()
        legacy = build / "zaya_server"
        onebin = build / "1bit"
        # The default config path is repo-relative; point it at the fixture.
        config.ZAYA_SERVER_EXE = legacy
        # An unknown backend resolves to the global default (spec is None).
        default = "_selfcheck_default"

        legacy.write_bytes(b"")
        expect("argv[0] symlink present -> the configured path",
               config.zaya_launch_prefix(default), [str(legacy)])

        legacy.unlink()
        onebin.write_bytes(b"")
        expect("bare cmake build -> '<1bit> zaya'",
               config.zaya_launch_prefix(default), [str(onebin), "zaya"])

        onebin.unlink()
        expect_raises("nothing built -> actionable error, not a spawn",
                      lambda: config.zaya_launch_prefix(default), "onebin", "install.sh")

        # A missing *custom* per-backend exe is a configuration error, and must
        # not be papered over with the sibling 1bit: that would run a different
        # backend than the one under test and silently corrupt a comparison.
        onebin.write_bytes(b"")
        custom = build / "zaya_vulkan"
        config.BACKENDS["_selfcheck_custom"] = SimpleNamespace(zaya_exe=custom)
        expect_raises("missing custom per-backend exe -> NOT substituted",
                      lambda: config.zaya_launch_prefix("_selfcheck_custom"), str(custom))

    if FAILED:
        print(f"zaya_launch_selfcheck: {len(FAILED)} failed")
        return 1
    print("zaya_launch_selfcheck: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
