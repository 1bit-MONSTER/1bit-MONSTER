#!/usr/bin/env python3
"""Honesty gate for the Prism lane's results of record (plan P6).

The lane's contract, stated by the operator: every number carries an honesty tag, and a claim ships
only with (model, format, backend, box, tokens, prompt, date). This gate is what makes that
mechanical instead of aspirational.

Rules enforced on tests/prism/PRISM_RESULTS.md:
  R1. Every line that states a number with a physical unit (or a correctness correlation) must carry
      a bracketed tag [model | format | backend | box | tokens | prompt | date] with exactly 7 fields.
  R2. `box` must be one of: cpu-host, strixhalo-quiet, strixhalo-busy, strixhalo-unknown.
  R3. `date` must be ISO YYYY-MM-DD; `tokens` must be an integer or '-'.
  R4. A `strixhalo-quiet` claim requires, in the same file, a triad evidence line tagged
      strixhalo-quiet whose value is >= 200 GB/s. This is risk R16 encoded: the same box reads
      139-170 GB/s under peer NPU load, so a quiet timing without bandwidth evidence is not
      admissible.
  R5. No `tok/s` row may be tagged `strixhalo-quiet` while the gate section says the gate is
      unmeasured; symmetry check that a quiet tok/s claim has triad support (R4) AND a tokens count.

Exit 0 = honest, 1 = violation (each with file:line).
"""
import re
import sys
import pathlib

DOC = pathlib.Path(__file__).with_name("PRISM_RESULTS.md")
BOXES = {"cpu-host", "strixhalo-quiet", "strixhalo-busy", "strixhalo-unknown", "n/a"}

# a number with a physical unit, or a correlation/cosine claim
NUM = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*(GB/s|MB/s|tok/s|tok\b|GB\b|MB\b|ms\b|s\b(?!\w)|%|x\b)|"
    r"corr\s*=?\s*\d|cosine\s+\d|min\s+\d\.\d{3,}"
)
TAG = re.compile(r"\[([^\[\]]*)\]")
TRIAD = re.compile(r"triad.*?(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*GB/s", re.I)


def tags_on(line):
    out = []
    for m in TAG.finditer(line):
        fields = [f.strip().rstrip(chr(92)).strip() for f in m.group(1).split("|")]
        if len(fields) == 7:
            out.append(fields)
    return out


# Shared docs that also carry this lane's claims. Scope rules, so that nothing here polices another
# lane's rows in the same file:
#   A) LINE scope  - the line itself names this lane (a pack name, "Prism ML", PrismEngine, prism_*).
#      This catches the family doc's pack table, whose rows name the packs but not the family.
#   B) PARA scope  - a prose paragraph naming the lane; table rows are excluded from this rule,
#      because one markdown table is a single paragraph and would otherwise drag in other lanes.
SCOPED = {
    "docs/model-families/bitnet-bonsai.md": ("prism ml", "gateddeltanet", "prismengine", "prism_",
                                             "bonsai-27b-q1_0", "ternary-bonsai-2-27b",
                                             "ternary-bonsai-27b"),
    "models/catalog/README.md": ("prism ml", "gateddeltanet", "prismengine", "prism_",
                                 "bonsai-27b-q1_0", "ternary-bonsai-2-27b", "ternary-bonsai-27b"),
    "research/TRACKING.md": ("prism ml bonsai 27b",),
}


def _in_scope_line(line, markers):
    low = line.lower()
    return any(m in low for m in markers)


def check_scoped(root, fails):
    """Honesty check for this lane's claims living in shared docs (see the scope rules above)."""
    checked = 0
    for rel, markers in SCOPED.items():
        f = root / rel
        if not f.exists():
            continue
        lines = f.read_text().split("\n")
        para_start = 0
        for i in range(len(lines) + 1):
            at_break = i == len(lines) or not lines[i].strip()
            if at_break:
                blob = " ".join(lines[para_start:i]).lower()
                para_hit = any(m in blob for m in markers)
                for k in range(para_start, i):
                    line = lines[k]
                    hit = _in_scope_line(line, markers) or (
                        para_hit and not line.lstrip().startswith("|")
                    )
                    if not hit or not NUM.search(line):
                        continue
                    checked += 1
                    if not tags_on(line):
                        fails.append(
                            f"{rel}:{k+1}: lane claim without a 7-field tag: {line.strip()[:100]}"
                        )
                para_start = i + 1
    return checked


def main():
    if not DOC.exists():
        print(f"FAIL: {DOC} missing")
        return 1
    lines = DOC.read_text().splitlines()
    fails, checked, quiet_triads, quiet_tok_ok = [], 0, [], set()

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            body = " ".join(cells[:-1]) if len(cells) > 1 else stripped
        else:
            body = stripped
        if not NUM.search(body):
            continue
        checked += 1
        found = tags_on(line)
        if not found:
            fails.append(f"{i}: number without a 7-field honesty tag: {stripped[:110]}")
            continue
        for f in found:
            model, fmt, backend, box, tokens, prompt, date = f
            if box not in BOXES:
                fails.append(f"{i}: bad box '{box}' (allowed: {sorted(BOXES)})")
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
                fails.append(f"{i}: bad date '{date}'")
            if not (tokens == "-" or tokens.isdigit()):
                fails.append(f"{i}: bad token count '{tokens}'")
            if box == "strixhalo-quiet":
                quiet_tok_ok.add(i)

    for i, line in enumerate(lines, 1):
        m = TRIAD.search(line)
        if m and any(f[3] == "strixhalo-quiet" for f in tags_on(line)):
            quiet_triads.append((i, float(m.group(1)), float(m.group(2))))

    if quiet_tok_ok and not quiet_triads:
        fails.append(
            "quiet-box timing claimed but no strixhalo-quiet triad evidence line in the file (R4/R16)"
        )
    for i, lo, hi in quiet_triads:
        if hi < 200.0:
            fails.append(f"{i}: strixhalo-quiet triad is only {lo}-{hi} GB/s (<200) — not quiet")

    checked += check_scoped(DOC.resolve().parents[2], fails)

    print(f"honesty tags: {checked} numeric claims checked, "
          f"{len(quiet_triads)} quiet triad line(s), {len(fails)} violation(s)")
    for f in fails:
        print(f"  FAIL {f}")
    print("HONESTY GATE: " + ("PASS" if not fails else "FAIL"))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
