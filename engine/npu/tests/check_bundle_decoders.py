#!/usr/bin/env python3
"""check_bundle_decoders.py — every *-NPU2 bundle must decode with the
convention its family is known to use.

Why this exists: a 4-bit bundle can be decoded four ways (scale index layout x
nibble signedness), and a WRONG pairing does not look wrong. It still produces a
weight distribution of the right shape and scale, so the model loads, runs, and
answers confidently with the wrong token. LFM2-1.2B and 2.6B were both in that
state: neither of the two decoders in dequant_q4nx.cpp decoded them (they need
group-major scales AND signed nibbles), and nothing anywhere failed.

The oracle is the model's own tying: with tie_word_embeddings=true, lm_head must
equal embed_tokens, so decoding lm_head under each convention and correlating
against the BF16 embedding rows names the convention. Measured, not read — and
the same trick classifies any future family for free.

Exit: 0 = every classifiable bundle matches its family's expected convention
      1 = at least one mismatch (converter drift, or a new family that needs a
          convention nobody implemented yet — decide, then update EXPECTED)
      2 = environment: no model store to check (not a verdict about the bundles)

Usage: engine/npu/tests/check_bundle_decoders.py [models-dir]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from q4nx_decoder_map import probe  # noqa: E402  (one implementation, two callers)

# family prefix -> convention its bundles must decode with. Measured 2026-09-13:
# LFM2 is the only family here that needs group-major scales + signed nibbles;
# every other classifiable bundle uses group-major scales + unsigned nibbles,
# which is what npu_engine_universal.cpp calls (lines 913/948/1523).
EXPECTED = [
    ("LFM2-", "group+signed"),
]
DEFAULT_EXPECTED = "group+unsigned"
MIN_CORR = 0.9        # a convention must reach this to be named the winner
UNTIED_CEILING = 0.5  # at or below this, no convention correlates at all: untied model.
                      # Between the two, something correlates but nothing fits — see main().


def expected_for(bundle):
    for prefix, conv in EXPECTED:
        if bundle.startswith(prefix):
            return conv
    return DEFAULT_EXPECTED


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.config/flm/models")
    if not os.path.isdir(d):
        print(f"ERROR: no model store at {d} — cannot judge the bundles.", file=sys.stderr)
        return 2

    rows, failed, skipped, suspicious = [], 0, 0, 0
    classified_conventions = set()
    for name in sorted(os.listdir(d)):
        md = os.path.join(d, name)
        if not os.path.isdir(md) or not os.path.exists(os.path.join(md, "model.q4nx")):
            continue
        r = probe(md)
        if r is None:
            continue
        if isinstance(r, str):                    # no tied pair / geometry we skip
            rows.append((name, "skipped", r))
            skipped += 1
            continue
        winner, corr, scores = r
        if winner == "UNTIED-OR-UNKNOWN" and corr < UNTIED_CEILING:
            # The oracle is blind here, and that is not a verdict: with tying off,
            # lm_head and embed_tokens are independent matrices, so all four
            # conventions score ~0. Reported, never failed.
            rows.append((name, "unclassifiable", f"untied (best corr {corr:+.4f} < {UNTIED_CEILING})"))
            skipped += 1
            continue
        if winner == "UNTIED-OR-UNKNOWN":
            # Not ~0 and not >= MIN_CORR: something correlates with the embedding
            # but no single convention explains it. That is a partially-corrupted
            # bundle or a geometry change, not an untied model — and calling it
            # "unclassifiable" would let it pass silently.
            rows.append((name, "SUSPICIOUS",
                         f"no convention fits, but best corr {corr:+.4f} is not ~0 "
                         f"(>= {UNTIED_CEILING}); partial corruption or geometry change?"))
            suspicious += 1
            continue
        want = expected_for(name)
        ok = (winner == want)
        if not ok:
            failed += 1
        else:
            classified_conventions.add(want)
        rows.append((name, "ok" if ok else "MISMATCH",
                     f"{winner} (corr {corr:+.4f}), expected {want}"))

    print("bundle                           verdict   detail")
    for name, verdict, detail in rows:
        print(f"{name:32s} {verdict:9s} {detail}")

    classifiable = len(rows) - skipped
    # Coverage floors. Without them this check passes while verifying nothing: an
    # empty or relocated store, or a rewrite that stops finding the tied pairs,
    # would print "classified=0" and exit 0 — a zero read as clean, which is the
    # exact mistake this gate exists to prevent one level down.
    missing_conv = [c for _, c in EXPECTED if c not in classified_conventions]
    if DEFAULT_EXPECTED not in classified_conventions:
        missing_conv.append(DEFAULT_EXPECTED)
    coverage_lost = classifiable == 0 or missing_conv

    print(f"\nSUMMARY: classified={classifiable} skipped={skipped} "
          f"mismatched={failed} suspicious={suspicious}")
    if failed or suspicious or coverage_lost:
        print("RESULT: FAIL")
        if coverage_lost:
            print(f"COVERAGE LOST: classified={classifiable}"
                  + (f", no bundle classified as {', '.join(missing_conv)}" if missing_conv else "")
                  + " — this check verified nothing. Fix the store path or the oracle before"
                    " reading a PASS out of it.")
        if suspicious:
            print("At least one bundle correlates with its embedding but matches no convention:")
            print("that is partial corruption or a geometry change, not an untied model.")
        print("A bundle no longer decodes with its family's convention. If this is a new")
        print("family, measure it with tools/q4nx_decoder_map.py, implement the decoder it")
        print("needs, and add its prefix to EXPECTED in this file — do not widen MIN_CORR.")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
