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
MIN_CORR = 0.9   # below this the oracle cannot name a convention at all


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

    rows, failed, skipped = [], 0, 0
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
        if winner == "UNTIED-OR-UNKNOWN":
            # The oracle is blind here, and that is not a verdict: with tying
            # off, lm_head and embed_tokens are independent matrices, so all
            # four conventions score ~0. Reported, never failed.
            rows.append((name, "unclassifiable", f"untied (best corr {corr:+.4f} < {MIN_CORR})"))
            skipped += 1
            continue
        want = expected_for(name)
        ok = (winner == want)
        if not ok:
            failed += 1
        rows.append((name, "ok" if ok else "MISMATCH",
                     f"{winner} (corr {corr:+.4f}), expected {want}"))

    print("bundle                           verdict   detail")
    for name, verdict, detail in rows:
        print(f"{name:32s} {verdict:9s} {detail}")

    classifiable = len(rows) - skipped
    print(f"\nSUMMARY: classified={classifiable} skipped={skipped} mismatched={failed}")
    if failed:
        print("RESULT: FAIL")
        print("A bundle no longer decodes with its family's convention. If this is a new")
        print("family, measure it with tools/q4nx_decoder_map.py, implement the decoder it")
        print("needs, and add its prefix to EXPECTED in this file — do not widen MIN_CORR.")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
