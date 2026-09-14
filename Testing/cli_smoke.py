#!/usr/bin/env python3
"""Runtime counterpart to dispatch_selfcheck.py: does the BUILT binary actually
reach a handler for every entry point we advertise?

dispatch_selfcheck.py reads the source lists (packaged symlinks vs `prog ==`
branches, onebit's command tokens vs `cmd ==` branches) and cannot see whether
the linked binary really routes them — that is how `./run.sh chat` shipped
printing the usage text while every static check passed. This runs the artifact:

  * every subcommand tools/onebin.cpp dispatches, and
  * every symlink name packaging/Makefile installs,

each probed with `--help` under a timeout. The failure signature is exact: the
top-level fallback always begins `usage: 1bit <subcommand> [args...]`, and no
handler prints that. A command that ignores `--help` and starts work is NOT a
failure here (some tools have no help flag) — it is reported as a note, and the
process is killed by the timeout.

Exit 0 = every advertised entry point reached a handler. If no binary is present
the check SKIPS (exit 0 with a message) unless --require is passed, so it can sit
in the host-only suite, which does not build one.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FALLBACK = "usage: 1bit <subcommand>"


def read(rel: str) -> str:
    p = ROOT / rel
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def advertised_entries() -> tuple[list[str], list[str]]:
    onebin = read("tools/onebin.cpp")
    # flags are not subcommands: `1bit --version --help` is not a probe, and
    # the flag forms are covered by dispatch_selfcheck.py's source lists.
    subs = sorted(t for t in set(re.findall(r'cmd\s*==\s*"([\w.-]+)"', onebin))
                  if not t.startswith("-") and t != "help")
    makefile = read("packaging/Makefile")
    links: set[str] = set()
    for m in re.finditer(r"for\s+name\s+in\s+([^;]+);", makefile):
        links |= {t for t in m.group(1).split() if re.fullmatch(r"[\w.-]+", t)}
    return subs, sorted(links)


def probe(argv: list[str], timeout: int) -> tuple[str, str]:
    """Run a probe; return (verdict, note). verdict ∈ ok|fallthrough|timeout."""
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        out = (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired as e:
        out = ((e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")) \
            + ((e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or ""))
        if FALLBACK in out:
            return "fallthrough", "printed the top-level usage, then did not exit"
        return "timeout", "did not exit on --help (handler reached; killed)"
    except OSError as e:
        return "fallthrough", f"could not execute: {e}"
    first = next((l for l in out.splitlines() if l.strip()), "")
    if first.strip().startswith(FALLBACK):
        return "fallthrough", "printed the top-level usage text"
    return "ok", ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=str(ROOT / "build" / "1bit"))
    ap.add_argument("--require", action="store_true", help="fail if the binary is missing")
    ap.add_argument("--timeout", type=int, default=20)
    a = ap.parse_args()

    binary = Path(a.binary)
    if not binary.exists():
        msg = f"cli_smoke: no binary at {binary} — skipped (the host-only suite does not build one)"
        print(msg)
        return 1 if a.require else 0

    subs, links = advertised_entries()
    failures: list[str] = []
    notes: list[str] = []

    for sub in subs:
        verdict, note = probe([str(binary), sub, "--help"], a.timeout)
        if verdict == "fallthrough":
            failures.append(f"1bit {sub} --help: {note}")
        elif verdict == "timeout":
            notes.append(f"1bit {sub}: {note}")

    tmp = tempfile.mkdtemp(prefix="cli_smoke_")
    try:
        for name in links:
            link = Path(tmp) / name
            try:
                link.symlink_to(binary)
            except OSError:
                shutil.copy2(binary, link)
            verdict, note = probe([str(link), "--help"], a.timeout)
            if verdict == "fallthrough":
                failures.append(f"{name} (symlink, argv[0] dispatch): {note}")
            elif verdict == "timeout":
                notes.append(f"{name}: {note}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    for n in notes:
        print(f"  note  {n}")
    if failures:
        print(f"cli_smoke FAILED — {len(failures)} advertised entry point(s) do not reach a handler:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"cli_smoke ok: {len(subs)} subcommand(s) and {len(links)} symlink(s) reach a handler")
    return 0


if __name__ == "__main__":
    sys.exit(main())
