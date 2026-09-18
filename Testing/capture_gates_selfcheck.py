#!/usr/bin/env python3
"""capture_gates_selfcheck.py — every capture gate the docs tell you to set must be implemented, or refused.

Why this exists (#2528): `docs/AGENT-COORDINATION.md` tells the next capture to set
`CAP_NO_SYNC=1`, because "without it my first verification run wrote 181 GB in about
five minutes", and `site/1bit-post-byte-identical.html` publishes the same advice. main's
interposer read `CAP_DIR` and `CAP_POSTRUN_ACT` and ignored every dump gate, so an
operator who followed that instruction got the 181 GB run while their manifest looked
like a capture that had honoured the gate. A documented gate that does nothing is worse
than no gate, because it is trusted.

The invariant: every `CAP_*` name the documentation tells an operator to set is either

  * read by `getenv()` in `npu-infer/tools/capture/cap_interposer.cpp`, or
  * in that file's `refused[]` list, which exits 2 at load time before a byte is dumped.

The interposer carries two machine-readable markers for its own lists, and this check
verifies they agree with the source, so a marker cannot drift away from the code.
These are the current lines, not an illustration — when the marker was two names and
the file read nine, this check was red on main and the example here was part of the
reason the drift went unnoticed:

    // capture-gates-read: CAP_BIG_MAX CAP_DIR CAP_DUMP_BIG CAP_MM_W CAP_NO_SYNC CAP_POSTRUN_ACT CAP_POSTRUN_KV CAP_RUNLIST_KV CAP_SKIP_BIG
    // capture-gates-refused: CAP_NO_SYNC CAP_SKIP_BIG CAP_DUMP_BIG

Sources scanned for documented gates: the coordination protocol and every published
`site/1bit-post-*.html`. The floor assertions below exist so an empty or broken parse
cannot pass this check vacuously.

Run: python3 Testing/capture_gates_selfcheck.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INTERPOSER = REPO / "npu-infer" / "tools" / "capture" / "cap_interposer.cpp"
DOCS = [REPO / "docs" / "AGENT-COORDINATION.md"] + sorted(
    (REPO / "site").glob("1bit-post-*.html")
)
GATE = re.compile(r"\bCAP_[A-Z_]+\b")

failures: list[str] = []
checks = 0


def check(ok: bool, label: str, detail: str = "") -> None:
    global checks
    checks += 1
    if not ok:
        failures.append(f"{label}{(': ' + detail) if detail else ''}")


def gates_in(text: str) -> set[str]:
    return set(GATE.findall(text))


def source_lists(src: str) -> tuple[set[str], set[str], set[str], set[str]]:
    """(read via getenv, refused[], declared-read marker, declared-refused marker)"""
    read = set(re.findall(r'getenv\(\s*"(CAP_[A-Z_]+)"\s*\)', src))
    m = re.search(r"refused\[\]\s*=\s*\{([^}]*)\}", src, re.S)
    refused = set(re.findall(r'"(CAP_[A-Z_]+)"', m.group(1))) if m else set()
    def marker(kind: str) -> set[str]:
        mm = re.search(rf"^//\s*capture-gates-{kind}:(.*)$", src, re.M)
        return set(GATE.findall(mm.group(1))) if mm else set()
    return read, refused, marker("read"), marker("refused")


def coverage_problems(documented: set[str], handled: set[str]) -> list[str]:
    return sorted(documented - handled)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=None, help=f"interposer to check (default: {INTERPOSER})")
    args = ap.parse_args(argv)
    source = Path(args.source) if args.source else INTERPOSER
    if not source.is_file():
        print(f"FAIL: no {source}", file=sys.stderr)
        return 1
    src = source.read_text(encoding="utf-8", errors="replace")

    documented: dict[str, str] = {}
    for path in DOCS:
        if not path.is_file():
            continue
        for name in gates_in(path.read_text(encoding="utf-8", errors="replace")):
            documented.setdefault(name, path.relative_to(REPO).as_posix())

    read, refused, decl_read, decl_refused = source_lists(src)
    handled = read | refused

    # ---- floor: a broken parse must not pass ---------------------------------
    check(len(documented) >= 2, "floor: >=2 documented gates found",
          f"found {len(documented)}: {sorted(documented)}")
    check(len(read) >= 2, "floor: >=2 gates read via getenv", f"found {sorted(read)}")
    check(len(refused) >= 1, "floor: >=1 gate refused", f"found {sorted(refused)}")

    # ---- the markers must agree with the code they describe -------------------
    check(decl_read == read, "marker capture-gates-read == getenv() set",
          f"marker {sorted(decl_read)} vs source {sorted(read)}")
    check(decl_refused == refused, "marker capture-gates-refused == refused[] list",
          f"marker {sorted(decl_refused)} vs source {sorted(refused)}")

    # ---- every documented gate is handled -------------------------------------
    missing = coverage_problems(set(documented), handled)
    check(not missing, "every documented CAP_* gate is read or refused",
          "; ".join(f"{n} ({documented.get(n, '?')})" for n in missing))

    # ---- controls: each matcher must be able to fail ---------------------------
    # An emptied refused[] list cannot be caught *through coverage*: every name in
    # refused[] is also read via getenv() (refused <= read), so dropping the list
    # leaves the documented set covered. The previous form of this control asserted
    # exactly that catch and therefore could never fire — it was reporting the truth
    # about its own premise, which is why the run below failed on it. What does catch
    # the edit is the marker comparison, so this control constructs the emptied list
    # from the source and asserts THAT rule fires.
    #
    # It is written against the CONSTRUCTED list rather than against a difference from
    # the shipped text, so a source that is already empty still reports the substantive
    # failure (marker != refused[]) once, instead of two misattributed control failures.
    empty_src = re.sub(r"refused\[\]\s*=\s*\{[^}]*\}", "refused[] = {}", src,
                       count=1, flags=re.S)
    _, refused_empty, _, _ = source_lists(empty_src)
    controls = [
        ("a documented gate nothing handles",
         coverage_problems(set(documented) | {"CAP_FAKE_GATE"}, handled)),
        ("an emptied refused[] list",
         [1] if (refused_empty == set() and decl_refused != refused_empty) else []),
        ("a marker that disagrees with the source",
         [1] if decl_read != (read - {"CAP_DIR"}) else []),
    ]
    for label, result in controls:
        check(bool(result), f"control: '{label}' is caught")

    if failures:
        for f in failures:
            print(f"  FAIL {f}")
        print(f"\n{len(failures)} FAILURE(S) — documented capture gates and the interposer disagree")
        return 1
    print(f"capture gates ok ({checks} checks, {len(controls)} negative controls)")
    print(f"  documented: {sorted(documented)}")
    print(f"  read:       {sorted(read)}")
    print(f"  refused:    {sorted(refused)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
