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
    """Honesty check for this lane's claims living in shared docs (see the scope rules above).

    Returns (total, per_file_counts) so the self-test can assert coverage per file.
    """
    checked = 0
    counts = {}
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
                    counts[rel] = counts.get(rel, 0) + 1
                    if not tags_on(line):
                        fails.append(
                            f"{rel}:{k+1}: lane claim without a 7-field tag: {line.strip()[:100]}"
                        )
                para_start = i + 1
    return checked, counts


def scan_strict(lines, fails=None):
    """Whole-file rule: any line stating a number with a unit needs a 7-field tag.

    Extracted so the self-test below exercises the *same* code that polices the real file: a
    self-test against a copy of the rule would prove only that the copy works.
    """
    fails = fails if fails is not None else []
    checked, quiet_ok = 0, set()
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
                quiet_ok.add(i)
    return checked, fails, quiet_ok


# Per-file coverage floors for the scoped scan. A too-narrow scope pattern under-policies *silently*
# — it happened: the first version of the paragraph rule skipped the family doc's pack table entirely
# and still printed PASS. A scope regression must therefore fail loudly instead of looking clean.
# The floors sit well below the current counts on purpose: they catch a rule that stopped matching,
# they do not pin today's numbers.
SCOPED_FLOOR = {
    "docs/model-families/bitnet-bonsai.md": 8,
    "models/catalog/README.md": 3,
}
DOC_FLOOR = 15


def self_test(fails, counts, doc_checked):
    """Prove the gate can fail, can pass, and is still looking where it claims to look.

    Recorded because this bit twice in one day, in this lane and in a peer's: an absent marker is not
    evidence of an absent claim. A too-narrow grep/scope pattern produces a clean-looking PASS while
    nothing is actually checked, and that is strictly worse than no gate — it manufactures trust.
    """
    good = ["a claim 1.5 GB/s [m | f | b | cpu-host | - | - | 2026-09-18]"]
    bad = ["a claim 1.5 GB/s and no tag", "corr=0.99 and no tag"]
    gc, gf, _ = scan_strict(good)
    bc, bf, _ = scan_strict(bad)
    if gc != 1 or gf:
        fails.append(f"self-test: a correctly tagged fixture must pass (checked={gc}, failures={gf})")
    if bc != 2 or len(bf) != 2:
        fails.append(f"self-test: untagged fixtures must be flagged (checked={bc}, flagged={len(bf)})")
    fx = ["<!-- gate-facts\na=10\nb=4\nc=(a - b)\n-->", "| row | 5 | <!-- derive: c -->"]
    bad_facts = load_facts("\n".join(fx))
    bfails = []
    check_derivations([fx[1]], bad_facts, bfails)
    if not bfails:
        fails.append("self-test: a wrong derived value must be flagged (derivation check not firing)")
    gx = ["<!-- gate-facts\na=10\nb=4\nc=(a - b)\n-->", "| row | 6 | <!-- derive: c -->"]
    gfails = []
    check_derivations([gx[1]], load_facts("\n".join(gx)), gfails)
    if gfails:
        fails.append("self-test: a correct derived value must pass the derivation check")
    if doc_checked < DOC_FLOOR:
        fails.append(
            f"self-test: results of record scanned only {doc_checked} claims (<{DOC_FLOOR}) — the "
            f"detector or the file shrank, and an absent claim is not an absent problem"
        )
    for rel, floor in SCOPED_FLOOR.items():
        got = counts.get(rel, 0)
        if got < floor:
            fails.append(
                f"self-test: {rel} scanned only {got} lane claims (<{floor}) — a scope rule stopped "
                f"matching, and a silent under-check reads as PASS"
            )
    return 0 if not fails else 1


FACTS_RE = re.compile(r"<!--\s*gate-facts(.*?)-->", re.S)
DERIVE_RE = re.compile(r"<!--\s*derive:\s*([a-z_0-9 ]+?)\s*-->")


def load_facts(text):
    """Machine-readable operands for derived numbers, so a derived figure can be reproduced.

    Blunt but sufficient: values and expressions confined to names defined in the same block plus
    arithmetic. A name may reference an earlier name.
    """
    m = FACTS_RE.search(text)
    if not m:
        return {}
    env = {}
    for raw in m.group(1).strip().split("\n"):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", k):
            continue
        if not re.fullmatch(r"[0-9A-Za-z_\s+*/().-]+", v):
            raise SystemExit(f"gate-facts: refused to evaluate {k} = {v}")
        try:
            env[k] = eval(v, {"__builtins__": {}}, env)  # noqa: S307 - restricted namespace
        except Exception as exc:
            raise SystemExit(f"gate-facts: cannot evaluate {k} = {v}: {exc}")
    return env


def check_derivations(lines, facts, fails):
    """A row marked `<!-- derive: name -->` must contain that fact's value among its numbers.

    Motivating failure, which this exact rule would have caught: I recorded a corrected figure after
    checking that it divided correctly into its denominator (1452 of 1541) but never that the
    numerator was itself a difference. Internal consistency passed, the input was wrong, and a peer
    had to withdraw the premise before I noticed. Checking the ratio is not checking the provenance.
    """
    checked = 0
    for i, line in enumerate(lines, 1):
        names = DERIVE_RE.findall(line)
        if not names:
            continue
        nums = [float(x) for x in re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)", DERIVE_RE.sub("", line))]
        for name in " ".join(names).split():
            if name not in facts:
                fails.append(f"{i}: derive marker '{name}' has no matching gate-fact")
                continue
            checked += 1
            val = float(facts[name])
            # a row may quote a fractional fact rounded to an integer (27.125 -> "27%"), so the
            # tolerance is half a unit; a genuinely wrong figure (418 -> 1452) is length-scale away.
            if not any(abs(n - val) <= 0.5 for n in nums):
                fails.append(
                    f"{i}: derived {name} = {round(val, 3)} does not appear in its row "
                    f"(numbers there: {nums[:6]}) - provenance not reproducible from the operands"
                )
    return checked


def main():
    if not DOC.exists():
        print(f"FAIL: {DOC} missing")
        return 1
    lines = DOC.read_text().splitlines()
    fails, quiet_triads = [], []
    checked, fails, quiet_tok_ok = scan_strict(lines, fails)

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

    facts = load_facts(DOC.read_text())
    derived = check_derivations(lines, facts, fails)
    scoped_checked, counts = check_scoped(DOC.resolve().parents[2], fails)
    checked += scoped_checked
    st = self_test(fails, counts, len(lines) and checked - scoped_checked)

    print(f"honesty tags: {checked} numeric claims checked, "
          f"{derived} derivation(s) reproduced, "
          f"{len(quiet_triads)} quiet triad line(s), {len(fails)} violation(s); "
          f"self-test {'ok' if st == 0 else 'FAILED'} (detector fires, tagged fixture passes, coverage floors)")
    for f in fails:
        print(f"  FAIL {f}")
    print("HONESTY GATE: " + ("PASS" if not fails else "FAIL"))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
