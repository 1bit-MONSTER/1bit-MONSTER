#!/usr/bin/env python3
"""npu_pack_stride_selfcheck.py — a fused-weight transpose_pack offset must scale by the row stride.

WHY
---
transpose_pack(src, out_f, in_f, dst, dst_stride, dst_offset) reads

    dst[i * dst_stride + dst_offset + o] = src[o * in_f + i]      (o < out_f, i < in_f)

so `src` is the head of a row-major [out_f][in_f] block and `in_f` is the row stride. The
callers pass the same quantity as `out_f` and as `dst_offset` - both are counts of ROWS of the
fused weight matrix - while `in_f` is H. A call that packages the block starting at row R must
therefore pass `qkv_w + R * in_f`: `qkv_w + R` advances R floats, i.e. R/H rows, and reads a
sliding window inside the first rows of the buffer. Row 0 is right by accident, which is what
makes this defect present as "head 0 is fine, every other head is noise" (#2451).

Two such offsets sat in engine/npu/src/npu_engine_universal.cpp (the GDN K and V blocks) and two
more in the STD per-head loop, which are exempt here because their fix is not an offset scaling:
the converter flattens full-attention q/gate to [q_all | gate_all]
(third_party/FLM_Q4NX_Converter/q4nx/models/qwen35.py, "(g p h) -> (p g h)" with p=2), so that
branch needs a restructure. The exemption is self-expiring: if those sites are rewritten, this
check fails and asks for the exemption to be deleted rather than leaving a stale allowance.

This is a source-level guard because the packing lives inside a 4000-line function - there is no
unit to call. It fails when:
  * an offset call `buf + <expr>` does not scale <expr> by H and is not one of the exemptions;
  * an exemption no longer matches a call site (fixed/renamed), or no longer exists in the tree;
  * the number of call sites or offset calls drops below the floors, which is how a scan that
    silently stopped matching would otherwise pass.

EXIT CODES
----------
0  ok          1  violation          2  usage or environment error
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Every file with a transpose_pack, i.e. every place a fused pack can be written. Both
# definitions are `static`, so the call sites are in the same translation unit.
FILES = (
    "engine/npu/src/npu_engine_universal.cpp",
    "engine/npu/src/npu_engine_cb.cpp",
)

# Floors: a regex that stopped matching would otherwise pass this check by finding nothing.
# Today: 19 call sites (14 in the universal engine, 5 in the cb worker) and 4 offset calls.
MIN_CALLS = 15
MIN_OFFSET_CALLS = 2

# Call sites whose offset is deliberately NOT scaled by H, with the reason. Keyed by the exact
# first argument (whitespace normalised). See the module docstring: the STD q/gate loop is the
# per-head interleave the converter does not produce, so its fix is a restructure, not `* H`.
KNOWN_UNSCALED = {
    "qkv_w + h * 2 * std_hd[l]": (
        "STD q block, per-head interleave; the correct fix is the [q_all | gate_all] "
        "restructure from the converter, not an offset scaling (#2451)"
    ),
    "qkv_w + h * 2 * std_hd[l] + std_hd[l]": (
        "STD gate block, same restructure as the q block (#2451)"
    ),
}

CALL_RE = re.compile(r"\btranspose_pack\s*\(")
OFFSET_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\+\s*(.+)$", re.S)
STRIDE_RE = re.compile(r"\*\s*H\b")


class EnvError(Exception):
    """The scan could not be trusted (exit 2)."""


def calls(text: str) -> list[str]:
    """First argument of every transpose_pack call in `text`, whitespace-normalised."""
    out = []
    for m in CALL_RE.finditer(text):
        i = m.end()
        depth = 1
        start = i
        while i < len(text) and depth:
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if depth:
            raise EnvError("unbalanced parentheses after a transpose_pack call")
        args = text[start:i]
        first = args.split(",", 1)[0]
        out.append(" ".join(first.split()))
    return out


def main() -> int:
    total = 0
    offset_calls: list[str] = []
    problems: list[str] = []
    seen_exempt: set[str] = set()

    for rel in FILES:
        path = ROOT / rel
        try:
            text = path.read_text()
        except OSError as exc:
            print(f"ENVIRONMENT: cannot read {rel}: {exc}", file=sys.stderr)
            return 2
        # the definitions are not calls
        body = re.sub(r"static\s+void\s+transpose_pack\s*\([^)]*\)\s*\{", "", text)
        found = calls(body)
        total += len(found)
        for first in found:
            m = OFFSET_RE.match(first)
            if not m:
                continue
            offset_calls.append(first)
            if STRIDE_RE.search(m.group(2)):
                continue
            if first in KNOWN_UNSCALED:
                seen_exempt.add(first)
                continue
            problems.append(
                f"{rel}: transpose_pack({first}, ...) advances the source pointer by a row "
                f"count without scaling it by the row stride H - the block it packs is read "
                f"from the wrong rows ({first.split('+', 1)[1].strip()} floats instead of "
                f"that many rows)"
            )

    for exempt, why in sorted(KNOWN_UNSCALED.items()):
        if exempt not in seen_exempt:
            problems.append(
                f"the KNOWN_UNSCALED exemption {exempt!r} no longer matches a call site: the "
                f"site was fixed or rewritten, so delete the exemption ({why})"
            )

    print(f"npu pack stride - {len(FILES)} file(s), {total} transpose_pack call(s), "
          f"{len(offset_calls)} with a source offset")
    for exempt in sorted(KNOWN_UNSCALED):
        state = "exempt (still unscaled)" if exempt in seen_exempt else "MISSING"
        print(f"  {state}: {exempt}")

    if total < MIN_CALLS:
        problems.append(
            f"only {total} transpose_pack call(s) found, floor is {MIN_CALLS}: the scan "
            f"stopped matching, so a clean result here would mean nothing"
        )
    if len(offset_calls) < MIN_OFFSET_CALLS:
        problems.append(
            f"only {len(offset_calls)} call(s) pass a source offset, floor is "
            f"{MIN_OFFSET_CALLS}: nothing here would catch an unscaled offset"
        )

    if problems:
        print("VIOLATION:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print(f"OK: every source offset is scaled by H ({len(offset_calls) - len(KNOWN_UNSCALED)} "
          f"scaled, {len(KNOWN_UNSCALED)} exempt)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except EnvError as exc:
        print(f"ENVIRONMENT: {exc}", file=sys.stderr)
        sys.exit(2)
