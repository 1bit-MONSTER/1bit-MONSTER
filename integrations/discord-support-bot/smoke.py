#!/usr/bin/env python3
"""smoke.py — validate the /docs answer pipeline without needing secrets.

Checks:
  1. CONTEXT7_LIBRARY_ID is configured (has a sane default).
  2. Context7 retrieval returns snippets for a sample question (uses
     CONTEXT7_API_KEY if set, otherwise anonymous).
  3. If DEEPSEEK_API_KEY is set, generates a real grounded answer and reports
     its length; otherwise it's skipped (marked SKIP, not a failure).

Exit code 0 if all required checks pass, 1 otherwise.
"""
from __future__ import annotations

import os
import sys

import context7
import llm

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = PASS if ok else FAIL
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        return False
    return True


def main() -> int:
    load_dotenv()
    ok = True

    library_id = os.getenv("CONTEXT7_LIBRARY_ID", "/1bit-monster/1bit-monster")
    ok &= check("CONTEXT7_LIBRARY_ID configured", bool(library_id), library_id)

    # --- Context7 retrieval (no key needed) ---
    try:
        data = context7.get_context(library_id, "How do I build the engine?")
        block = context7.format_context(data)
        info = len(data.get("infoSnippets", []))
        code = len(data.get("codeSnippets", []))
        ok &= check(
            "Context7 retrieval returns grounded docs",
            len(block.strip()) > 0 and (info + code) > 0,
            f"{info} info + {code} code snippets, {len(block)} chars",
        )
    except Exception as exc:  # noqa: BLE001
        ok &= check("Context7 retrieval returns grounded docs", False, f"{type(exc).__name__}: {exc}")

    # --- Optional DeepSeek answer (only if key present) ---
    if os.getenv("DEEPSEEK_API_KEY"):
        try:
            ans = llm.generate("How do I build the engine?", block, os.getenv("DEEPSEEK_API_KEY"))
            ok &= check("DeepSeek grounded answer", len(ans.strip()) > 0, f"{len(ans.strip())} chars")
        except Exception as exc:  # noqa: BLE001
            ok &= check("DeepSeek grounded answer", False, f"{type(exc).__name__}: {exc}")
    else:
        print(f"[{SKIP}] DeepSeek grounded answer — DEEPSEEK_API_KEY not set (skipped; retrieval already verified)")

    print("\n" + ("SMOKE PASS" if ok else "SMOKE FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
