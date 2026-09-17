#!/usr/bin/env python3
"""zaya_launch_selfcheck.py — the benchmark harness must be able to launch the engine it can build.

WHY
---
benchmarks/engine_comparison launched `build/zaya_server`, which **no cmake target produces**: it
is an argv[0] symlink that only `site/install.sh` (and the packager) create. A plain
`cmake --build build --target onebin` produces `build/1bit`, and the dispatcher in
`tools/onebin.cpp` reads `argv[1]` as a subcommand — so `build/1bit --model … --port …` prints
its usage text and exits, and pointing `BENCH_ZAYA_SERVER` at it did not help for the same
reason. The documented build command in the harness README named a `zaya_server` *target* that
does not exist either (#2478).

The cases below assert the argv prefix `config.resolve_zaya_launch()` returns for each shape that
exists on a real machine, and then that `ZayaEngine.start()` actually uses it — the defect was in
the command the engine builds, not only in the path it resolves. No server is started and no
model is needed: `_spawn` is captured, so this runs anywhere.

EXIT CODES
----------
0  ok          1  violation          2  usage or environment error (harness missing/unimportable)
"""

from __future__ import annotations

import dataclasses
import os
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "benchmarks" / "engine_comparison"

# Floor: the integration case needs at least one backend that can run zaya, or it proves nothing.
MIN_ZAYA_BACKENDS = 1


class EnvError(Exception):
    """Cannot judge (exit 2)."""


def load_harness():
    """Import the harness's config + engines with `requests` stubbed.

    engines.py imports requests at module scope but only calls it from HTTP helpers, so a stub
    keeps this check runnable on a runner that has no third-party Python packages.
    """
    if not (HARNESS / "config.py").exists() or not (HARNESS / "engines.py").exists():
        raise EnvError(f"no harness at {HARNESS}")
    sys.modules.setdefault("requests", types.ModuleType("requests"))
    sys.path.insert(0, str(HARNESS))
    import config  # noqa: E402
    import engines  # noqa: E402
    return config, engines


def touch(path: Path, mode: int = 0o755) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(mode)
    return path


def main() -> int:
    try:
        config, engines = load_harness()
    except EnvError as exc:
        print(f"ENVIRONMENT: {exc}", file=sys.stderr)
        return 2
    except ImportError as exc:
        print(f"ENVIRONMENT: cannot import the harness: {exc}", file=sys.stderr)
        return 2

    if not hasattr(config, "resolve_zaya_launch"):
        print("VIOLATION: config.resolve_zaya_launch() is gone - the harness builds its argv "
              "somewhere this check cannot see", file=sys.stderr)
        return 1

    fail = 0

    def expect(name: str, got, want) -> None:
        nonlocal fail
        if got == want:
            print(f"  ok   {name:56s} {got!r}")
        else:
            print(f"  FAIL {name:56s} {got!r} != {want!r}")
            fail = 1

    zaya_backends = [b for b in config.BACKENDS if getattr(config.BACKENDS[b], "zaya_enabled", False)]
    if len(zaya_backends) >= MIN_ZAYA_BACKENDS:
        print(f"  ok   {'harness floor':56s} {len(zaya_backends)} zaya-capable backend(s), "
              f"{len(config.BACKENDS)} total")
    else:
        print(f"  FAIL {'harness floor':56s} no zaya-capable backend: nothing below can run")
        fail = 1

    with tempfile.TemporaryDirectory() as td:
        build = Path(td) / "build"
        # 1. neither exists: the fresh clone that was only built, with the symlink name still in
        #    the config - it must name both paths rather than raise FileNotFoundError. Checked
        #    before anything is created, in a tree of its own.
        empty = Path(td) / "empty" / "build" / "zaya_server"
        try:
            config.resolve_zaya_launch(empty)
            print(f"  FAIL {'no binary at all is an error':56s} returned a command")
            fail = 1
        except RuntimeError as exc:
            msg = str(exc)
            want = (str(empty), str(empty.with_name("1bit")),
                    "cmake --build build --target onebin", "BENCH_ZAYA_SERVER")
            expect("no binary at all is an error naming both",
                   [p for p in want if p not in msg], [])
        one_bit = touch(build / "1bit")
        one_bit_server = touch(build / "1bit-server")
        legacy = build / "zaya_server"
        symlink_target = legacy  # resolved below

        # 2. the multi-tool exists where the symlink was expected: add the subcommand
        expect("default exe missing, build/1bit present",
               config.resolve_zaya_launch(legacy), [str(one_bit), config.ZAYA_SUBCOMMAND])

        # 3. BENCH_ZAYA_SERVER pointed straight at the multi-tool
        expect("configured exe is build/1bit",
               config.resolve_zaya_launch(one_bit), [str(one_bit), config.ZAYA_SUBCOMMAND])

        # 4. the other multi-tool spelling the dispatcher accepts (prog == "1bit-server")
        expect("configured exe is 1bit-server",
               config.resolve_zaya_launch(one_bit_server),
               [str(one_bit_server), config.ZAYA_SUBCOMMAND])

        # 5. install.sh ran: an argv[0] symlink resolves the mode itself, no subcommand
        symlink_target.symlink_to(one_bit)
        expect("argv[0] symlink present (install.sh ran)",
               config.resolve_zaya_launch(legacy), [str(legacy)])

        # 6. the command the engine actually builds. The bug lived here: argv[1] must be the
        #    subcommand, because the dispatcher reads it.
        backend = zaya_backends[0]
        spec = dataclasses.replace(config.BACKENDS[backend],
                                   zaya_exe=None, zaya_extra_args=("--extra", "1"), zaya_env={})
        config.BACKENDS[backend] = spec
        symlink_target.unlink()                      # only build/1bit remains
        config.ZAYA_SERVER_EXE = legacy              # what config.py computes by default
        engines._port_open = lambda *a, **k: False   # no real bind check in a unit test

        class Capture(engines.ZayaServer):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.captured = None

            def _spawn(self, cmd, cwd=None, env=None):
                self.captured = [str(c) for c in cmd]

        model = types.SimpleNamespace(gguf=Path("/nonexistent/model.gguf"))
        engine = Capture(model, backend, Path(td) / "zaya.log")
        engine.start()
        cmd = engine.captured or []
        expect("start() passes the subcommand as argv[1]",
               cmd[:2], [str(one_bit), config.ZAYA_SUBCOMMAND])
        expect("start() keeps --model/--port in order",
               cmd[2:6], ["--model", str(model.gguf), "--port", str(config.ZAYA_PORT)])
        expect("start() still appends zaya_extra_args last", cmd[6:], ["--extra", "1"])

    if fail:
        print("\nVIOLATION: the harness cannot launch the binary a plain cmake build produces "
              "(see benchmarks/engine_comparison/config.py: resolve_zaya_launch).", file=sys.stderr)
        return 1
    print("OK: the launcher covers the argv[0] symlink, build/1bit and 1bit-server")
    return 0


if __name__ == "__main__":
    sys.exit(main())
