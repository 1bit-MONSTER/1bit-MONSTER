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

# model_type values map directly through rcpp_arch_from_string (snake_case
# aliases live in bitnet_model.h); the reader falls back to model_type when
# the class name maps UNKNOWN. This mirrors src/safetensors_reader.cpp.
MT_MAP = {
    "gpt_neox": "gptneox", "gpt_neo": "gptneo", "gpt_j": "gptj",
    "gpt_bigcode": "gptbigcode", "qwen2_vl": "qwen2vl", "qwen3_vl": "qwen3vl",
    "qwen3_moe": "qwen3moe", "qwen2_moe": "qwen2moe", "mistral_moe": "mixtral",
    "granite_moe": "granitemoe", "gemma3_text": "gemma3", "gemma4_text": "gemma4",
    "llava": "llava", "llava_llama": "llavallama", "llava_qwen2": "llavaqwen2",
    "deepseek_v2": "deepseekv2", "deepseek_v3": "deepseekv3",
    "deepseek_v4": "deepseekv4", "stablelm_epoch": "stablelmepoch",
    "openelm": "openelm", "cohere": "cohere", "cambrian_qwen": "cambrianqwen",
    "hunyuan_v1_dense": "hunyuandensev1", "exaone4": "exaone4",
    "nemotron": "nemotron", "fp8_qwen3": "fp8_qwen3", "fp8_qwen2": "fp8_qwen2",
    "fp8_llama": "fp8_llama", "bit_llama": "bitllama",
}

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
    # class-name mapping first
    class_mapped = {s: t != 255 for s, t in zip(stripped, toks)}
    # model_type fallback (mirror the reader): unknown class -> try model_type
    # NOTE: census_arch_counts.json only stores class names, so model_type
    # fallback is modeled per-class from the known MT_MAP (the full per-model
    # extraction lives in /tmp/census_full_data.jsonl + docs).
    fallback = {}
    for mt, alias in MT_MAP.items():
        if mt in stripped and not class_mapped.get(mt):
            fallback[mt] = stripped[mt]  # count via the model_type-as-class proxy
    # Re-derive per-class counts with fallback applied.
    merged = {}
    for s, c in stripped.items():
        key = s
        if not class_mapped.get(s) and MT_MAP.get(s) in stripped:
            key = MT_MAP[s]  # route through the alias token
        merged[key] = merged.get(key, 0) + c

    merged_lines = "\n".join(merged)
    probe2 = subprocess.run([mapper], input=merged_lines, text=True,
                            capture_output=True, check=True)
    merged_toks = [int(t) for t in probe2.stdout.split()]

    # token id -> family name (mirror bitnet_model.h enum)
    hdr = open(os.path.join(ROOT, "include", "rocm_cpp", "bitnet_model.h")).read()
    names = {}
    import re
    for m in re.finditer(r"RCPP_ARCH_(\w+)\s*=\s*(\d+)", hdr):
        names[int(m.group(2))] = m.group(1)
    names[255] = "UNKNOWN"

    family_counts = {}
    covered = 0
    for s, tok in zip(merged, merged_toks):
        if tok == 255:
            continue
        fam = names.get(tok, f"TOKEN{tok}")
        family_counts[fam] = family_counts.get(fam, 0) + merged[s]
        covered += merged[s]

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
