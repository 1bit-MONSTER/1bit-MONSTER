#!/usr/bin/env python3
"""CLI dispatch coverage — does everything that ships reach a handler?

Why this exists: tools/onebit.cpp implements `chat`, `pull`, `list`, `status`
and the rest of the agent CLI, and `onebit_main` is compiled into the single ELF
under ONE_BIN_DISPATCH. tools/onebin.cpp *declared* it and never called it, so
`./run.sh chat` — the quick start printed by packaging/tarball-run.sh and quoted
in docs/guides/getting-started.md — printed the top-level usage text instead.
The `onebit` symlink that packaging/Makefile and the flatpak manifest ship had no
argv[0] handler either.

Neither failure is visible to a compiler: a declared-but-uncalled function and a
symlink with no branch are both perfectly valid C++. So this check reads the
lists from their sources and fails when they stop agreeing.

Stdlib only, no build, no device — safe to run in the host-only suite.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Symlinks that intentionally reach something other than a dispatcher branch.
# Each needs a reason: an unexplained entry here is how the next one hides.
# EMPTY, and that is the point: `1bit-server` used to sit here as "two programs,
# one name". It is now dispatched to the zaya server, because the standalone
# packaging/binary/server.cpp is Windows-only and zaya_server already parses the
# bare positional port that packaging/ollama/README.md documents.
ALLOWED_UNHANDLED: dict[str, str] = {}


def read(rel: str) -> str:
    p = ROOT / rel
    if not p.exists():
        sys.exit(f"dispatch_selfcheck: missing {rel} (run from the repo root)")
    return p.read_text(encoding="utf-8", errors="replace")


def quoted_tokens(text: str) -> set[str]:
    return set(re.findall(r'"([A-Za-z0-9_.-]+)"', text))


def main() -> int:
    onebin = read("tools/onebin.cpp")
    onebit = read("tools/onebit.cpp")
    makefile = read("packaging/Makefile")
    flatpak = read("packaging/flatpak/monster.onebit.Engine.yml")

    problems: list[str] = []

    # ── 1. onebit_main must be CALLED, not merely declared ──
    if "onebit_main(" not in onebin:
        problems.append(
            "tools/onebin.cpp never mentions onebit_main — the whole agent CLI "
            "(chat/pull/list/status/…) is unreachable"
        )
    elif not re.search(r"return\s+onebit_main\s*\(", onebin):
        n = len(re.findall(r"onebit_main\s*\(", onebin))
        problems.append(
            f"tools/onebin.cpp names onebit_main {n}x but never calls it — "
            "a declaration alone leaves every one of its commands unreachable"
        )

    # ── 2. every packaged symlink reaches a handler ──
    handled = set(re.findall(r'prog\s*==\s*"([A-Za-z0-9_.-]+)"', onebin))
    symlinks: set[str] = set()
    for m in re.finditer(r"for\s+name\s+in\s+([^;]+);", makefile):
        symlinks |= quoted_tokens(m.group(0)) | set(m.group(1).split())
    for m in re.finditer(r"for\s+s\s+in\s+([^;]+);", flatpak):
        symlinks |= set(m.group(1).split())
    symlinks = {s for s in symlinks if re.fullmatch(r"[A-Za-z0-9_.-]+", s)}
    for name in sorted(symlinks - handled - set(ALLOWED_UNHANDLED)):
        problems.append(
            f"packaging ships a `{name}` symlink but tools/onebin.cpp has no "
            f'prog == "{name}" handler — running it prints the usage text'
        )

    # ── 3. every command onebit.cpp accepts is dispatched ──
    parser = re.search(
        r"if\s*\(arg\s*==\s*\"chat\".*?\)\s*\{", onebit, re.S
    )
    if not parser:
        problems.append(
            "tools/onebit.cpp no longer has the recognisable command list this "
            "check parses — update the check with the new shape"
        )
    else:
        accepted = quoted_tokens(parser.group(0))
        dispatched = set(re.findall(r'cmd\s*==\s*"([A-Za-z0-9_.-]+)"', onebin))
        for name in sorted(accepted - dispatched):
            problems.append(
                f"tools/onebit.cpp accepts `{name}` but tools/onebin.cpp does not "
                "dispatch it — the command falls through to the usage text"
            )

    if problems:
        print("dispatch coverage FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(
        f"dispatch coverage ok: {len(symlinks)} symlink name(s), "
        f"{len(accepted)} command token(s), onebit_main called"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
