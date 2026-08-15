#!/usr/bin/env python3
"""Regenerate Testing/census_full_summary.json from the raw census counts +
the ACTUAL engine arch registry (rcpp_arch_from_string). This is the Phase-4
gate: "sweep script output == documented count". Run:

    python3 Testing/census_coverage.py

The script compiles a tiny probe against include/rocm_cpp/bitnet_model.h so
coverage is measured by the real mapping function, not a copy. The strip logic
mirrors src/safetensors_reader.cpp (lowercase + suffix strip).
"""
import json, os, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COUNTS = os.path.join(ROOT, "Testing", "census_arch_counts.json")
OUT = os.path.join(ROOT, "Testing", "census_full_summary.json")

STRIP_SUFFIXES = ("forcausallm", "lmheadmodel", "model",
                  "forconditionalgeneration", "forvisiontext2text")

def strip_arch(arch: str) -> str:
    low = arch.lower()
    for suf in STRIP_SUFFIXES:
        if len(low) > len(suf) and low.endswith(suf):
            return low[: -len(suf)]
    return low

def build_mapper():
    """Compile a stdin->token probe linked against the real header."""
    src = os.path.join(ROOT, "Testing", "arch_mapper_probe.cpp")
    with open(src, "w") as f:
        f.write(r'''#include <cstdio>
#include <cstring>
#include "rocm_cpp/bitnet_model.h"
int main() {
    char buf[512];
    while (fgets(buf, sizeof buf, stdin)) {
        buf[strcspn(buf, "\n")] = 0;
        std::printf("%d\n", (int)rcpp_arch_from_string(buf));
    }
    return 0;
}
''')
    exe = os.path.join(tempfile.gettempdir(), "arch_mapper_probe")
    subprocess.run(["g++", "-std=c++17", "-I", os.path.join(ROOT, "include"),
                    src, "-o", exe], check=True)
    return exe

def main():
    raw = json.load(open(COUNTS))
    counts = raw["counts"]
    stripped = {}
    for arch, cnt in counts.items():
        if arch == "<none>":
            continue
        s = strip_arch(arch)
        stripped[s] = stripped.get(s, 0) + cnt

    mapper = build_mapper()
    probe = subprocess.run([mapper], input="\n".join(stripped), text=True,
                           capture_output=True, check=True)
    toks = [int(t) for t in probe.stdout.split()]

    # token id -> family name (mirror bitnet_model.h enum)
    hdr = open(os.path.join(ROOT, "include", "rocm_cpp", "bitnet_model.h")).read()
    names = {}
    import re
    for m in re.finditer(r"RCPP_ARCH_(\w+)\s*=\s*(\d+)", hdr):
        names[int(m.group(2))] = m.group(1)
    names[255] = "UNKNOWN"

    family_counts = {}
    covered = 0
    for s, tok in zip(stripped, toks):
        if tok == 255:
            continue
        fam = names.get(tok, f"TOKEN{tok}")
        family_counts[fam] = family_counts.get(fam, 0) + stripped[s]
        covered += stripped[s]

    with_arch = sum(stripped.values())
    summary = {
        "total": raw.get("total", 0),
        "with_arch": with_arch,
        "no_arch": raw.get("no_arch", counts.get("<none>", 0)),
        "n_archs": len(stripped),
        "registry_covered": covered,
        "family_counts": {k: family_counts[k] for k in sorted(family_counts)},
    }
    json.dump(summary, open(OUT, "w"), indent=1, sort_keys=True)
    print(f"total={summary['total']} with_arch={with_arch} "
          f"registry_covered={covered} ({100*covered/with_arch:.2f}%) -> {OUT}")

if __name__ == "__main__":
    main()
