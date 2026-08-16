#!/usr/bin/env python3
"""Regenerate Testing/census_full_summary.json from the raw census counts +
the ACTUAL engine arch registry (rcpp_arch_from_string). This is the Phase-4
gate: "sweep script output == documented count". Run:

    python3 Testing/census_coverage.py

The script compiles a tiny probe against include/rocm_cpp/bitnet_model.h so
coverage is measured by the real mapping function, not a copy. The strip logic
mirrors src/safetensors_reader.cpp (lowercase + suffix strip), and the
model_type fallback mirrors the reader's 2026-08-15 behavior: an unknown
class name resolves through its config's model_type (as-is, then
underscore/dash-stripped) — the dominant model_type per class comes from
census_modeltype_aggregate.json.
"""
import json, os, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COUNTS = os.path.join(ROOT, "Testing", "census_arch_counts.json")
AGG = os.path.join(ROOT, "Testing", "census_modeltype_aggregate.json")
OUT = os.path.join(ROOT, "Testing", "census_full_summary.json")

STRIP_SUFFIXES = ("forcausallm", "lmheadmodel", "model",
                  "forconditionalgeneration", "forvisiontext2text")

# #1676 (2026-08-15): NOT causal text decoders — TTS, encoder-decoder, and
# masked-LM classes sit in the census only because they carry an architectures
# string. The plan declared encoder-decoders OUT of scope; the engine has no
# cross-attention or TTS path. Excluded from the with_arch denominator.
NON_TEXT_GEN = {
    "parlertts",      # TTS (Parler-TTS)
    "t5", "mt5", "umt5",              # encoder-decoder
    "bart", "mbart", "marian", "longformerbart",  # encoder-decoder
    "t5gemma",                            # T5+Gemma hybrid (encoder-decoder)
    "bert", "roberta", "xlmroberta", "xlnet",  # masked-LM / encoder heads
    "morpht5auto", "morpht5concat", "morpht5sum",  # MorphT5 (encoder-decoder)
    "m2m100",                             # M2M-100 (encoder-decoder seq2seq)
    "slidingwindow", "blenderbot",         # encoder-decoder (verify is_encoder_decoder)
    "alfredunimodel",  # encoder-decoder / masked-LM variant
    "automodelforseq2seqlm",  # encoder-decoder / masked-LM variant
    "bartencodec",  # encoder-decoder / masked-LM variant
    "bartforsequenceclassification",  # encoder-decoder / masked-LM variant
    "bartprefixprop",  # encoder-decoder / masked-LM variant
    "bertformtsparseffdedit",  # encoder-decoder / masked-LM variant
    "bertforsequenceclassification",  # encoder-decoder / masked-LM variant
    "bertfortokenclassification",  # encoder-decoder / masked-LM variant
    "bertgenerationdecoder",  # encoder-decoder / masked-LM variant
    "bottleneckt5lmwithperturb",  # encoder-decoder / masked-LM variant
    "clipt5",  # encoder-decoder / masked-LM variant
    "cpt",  # encoder-decoder / masked-LM variant
    "custombart",  # encoder-decoder / masked-LM variant
    "decoderonlyt5",  # encoder-decoder / masked-LM variant
    "dnikud",  # encoder-decoder / masked-LM variant
    "e5_base_ctseg",  # encoder-decoder / masked-LM variant
    "efflongt5",  # encoder-decoder / masked-LM variant
    "efft5",  # encoder-decoder / masked-LM variant
    "elbart",  # encoder-decoder / masked-LM variant
    "fiphi-neuralmark-v3",  # encoder-decoder / masked-LM variant
    "fsdpt5",  # encoder-decoder / masked-LM variant
    "fusionindecoder",  # encoder-decoder / masked-LM variant
    "gistt5",  # encoder-decoder / masked-LM variant
    "grapht5transformer",  # encoder-decoder / masked-LM variant
    "latr",  # encoder-decoder / masked-LM variant
    "llavat5",  # encoder-decoder / masked-LM variant
    "longformerbartwithdoctype",  # encoder-decoder / masked-LM variant
    "longformerencoderbartdecoder",  # encoder-decoder / masked-LM variant
    "longformerencoderdecoder",  # encoder-decoder / masked-LM variant
    "longt5",  # encoder-decoder / masked-LM variant
    "lsgbart",  # encoder-decoder / masked-LM variant
    "marianmt",  # encoder-decoder / masked-LM variant
    "mlongformerencoderdecoder",  # encoder-decoder / masked-LM variant
    "modernmarianmt",  # encoder-decoder / masked-LM variant
    "mt5lsaalibi",  # encoder-decoder / masked-LM variant
    "myt5",  # encoder-decoder / masked-LM variant
    "narbart",  # encoder-decoder / masked-LM variant
    "omflaxt5",  # encoder-decoder / masked-LM variant
    "pipelinedbart",  # encoder-decoder / masked-LM variant
    "pipelinedt5",  # encoder-decoder / masked-LM variant
    "poptorchpipelinedbart",  # encoder-decoder / masked-LM variant
    "poptorchpipelinedt5",  # encoder-decoder / masked-LM variant
    "prophetnet",  # encoder-decoder / masked-LM variant
    "quantizedt5",  # encoder-decoder / masked-LM variant
    "robertaforcl",  # encoder-decoder / masked-LM variant
    "robertaformaskedlm",  # encoder-decoder / masked-LM variant
    "robertaprelayernorm",  # encoder-decoder / masked-LM variant
    "ropemhdat5",  # encoder-decoder / masked-LM variant
    "rr",  # encoder-decoder / masked-LM variant
    "sonar",  # encoder-decoder / masked-LM variant
    "spectus",  # encoder-decoder / masked-LM variant
    "svdcompressedbartforconditiongeneration",  # encoder-decoder / masked-LM variant
    "t5encoder",  # encoder-decoder / masked-LM variant
    "t5forsequenceclassification",  # encoder-decoder / masked-LM variant
    "t5gemma2",  # encoder-decoder / masked-LM variant
    "t5graph",  # encoder-decoder / masked-LM variant
    "t5la",  # encoder-decoder / masked-LM variant
    "texttotext",  # encoder-decoder / masked-LM variant
    "trainablem2m",  # encoder-decoder / masked-LM variant
    "ttcompressedbartforconditiongeneration",  # encoder-decoder / masked-LM variant
    "udopunimodel",  # encoder-decoder / masked-LM variant
    "unilm",  # encoder-decoder / masked-LM variant
    "vilt5",  # encoder-decoder / masked-LM variant
    "xlmrobertaforsequenceclassification",  # encoder-decoder / masked-LM variant
    "xlmrobertafortokenclassification",  # encoder-decoder / masked-LM variant
    "xlmrobertaxl",  # encoder-decoder / masked-LM variant

    "bertformaskedlm", "distilbertforsequenceclassification", "electra",
    "camembert",  # encoder-only classes (not causal decoders)

    # ── 2026-08-15 pass 2: verify/dump-config-verified encoder-decoder,
    # diffusion, and audio/TTS classes (not causal text decoders, #1676) ──
    "audioonlythinker", "av2text", "blenderbot", "cerpt", "cerptmultimodal",
    "clipvisionmarian", "clipvisionmbart", "codet5pbimodal", "deltalm",
    "detime", "dicow", "diffusiongemmaforblockdiffusion", "diffusionllm",
    "discretediffusion", "elastict5", "encoderdecoder",
    "expivmefordiffusionlmhub", "fairseqt5", "florence2",
    "giddfordiffusionlm", "granitespeech", "hed", "hftransformer",
    "locost", "membart", "microloopfordiffusionlm", "mvp", "needle",
    "nemotronlabsdiffusion", "nort5", "onebittts", "openba",
    "pathummaaudio", "pix2seq", "qwen2audio", "qwen2audiotime",
    "rotobart", "sealionaudio", "seamlessm4tv2fortexttotext", "sled",
    "slidingwindow", "songgendualtrack", "songgenmixed", "speech2text2",
    "speech2texttransformer", "speechlmm", "speechunit", "stepaudio2",
    "t5with", "tabletransformerforobjectdetection", "typhoonaudio",
    "vaswanirope", "visionencoderdecoder", "vitgpt2lm", "whisperaccent",

    # ── 2026-08-15 pass-3: encoder-decoder / audio / non-transformer / diffusion ──
    "bigbird", "bigbirdpegasus", "longformerforsequenceclassification",
    "blenderbotsmall", "vibevoice", "qwen3asr", "lstm", "prot2text",
    "bd3lm",  # BD3-LM — block discrete DENOISING DIFFUSION LM (not causal decoder, verified 2026-08-15)
    "plusmodel",  # LiltForTokenClassification (token classification, not causal LM)
    "kosine",  # SpeechT5 TTS
    "helloworld",  # junk test repo (model_type custom)
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


def probe(mapper, keys):
    if not keys:
        return []
    out = subprocess.run([mapper], input="\n".join(keys), text=True,
                         capture_output=True, check=True)
    return [int(t) for t in out.stdout.split()]


def main():
    raw = json.load(open(COUNTS))
    counts = raw["counts"]
    stripped = {}
    for arch, cnt in counts.items():
        if arch == "<none>":
            continue
        s = strip_arch(arch)
        stripped[s] = stripped.get(s, 0) + cnt

    # #1676: drop non-causal-decoder classes from the denominator BEFORE the
    # probe so they can never inflate coverage (all are UNKNOWN today).
    excluded = {}
    for s in list(stripped):
        if s in NON_TEXT_GEN:
            excluded[s] = stripped.pop(s)

    agg = json.load(open(AGG))
    # stripped class -> dominant (model_type, count). Sources in preference
    # order: (1) the verify pass's real fetched configs (census_tail_verify.json
    # — per-class model_type from actual config.json files), (2) the full
    # per-model dump (/tmp/census_full_data.jsonl — regenerated by the census
    # sweep; model_type filled for ~81%), (3) the aggregate's per-bucket top
    # archs (truncated at 5).
    mt_of = {}
    VERIFY = os.path.join(ROOT, "Testing", "census_tail_verify.json")
    if os.path.exists(VERIFY):
        v = json.load(open(VERIFY))
        for s, r in v.items():
            if r.get("decision") == "NO_CONFIG":
                continue  # no real config fetched — its model_type is the
                # aggregate's echo; let the restore/aggregate sources decide
            mt = r.get("model_type")
            if mt:
                mt_of[s] = (mt, 10)  # high priority: real config evidence
    DUMP = "/tmp/census_full_data.jsonl"
    if os.path.exists(DUMP):
        from collections import Counter as _C
        mt_counts = {}
        with open(DUMP) as f:
            for ln in f:
                try:
                    d = json.loads(ln)
                except Exception:
                    continue
                mt = d.get("model_type")
                if not mt:
                    continue
                for a in d.get("architectures") or []:
                    sk = strip_arch(str(a))
                    if sk not in stripped:
                        continue
                    mt_counts.setdefault(sk, _C())[mt] += 1
        for sk, cc in mt_counts.items():
            mt, n = cc.most_common(1)[0]
            if sk not in mt_of or n > mt_of[sk][1]:
                mt_of[sk] = (mt, n)
    # (3b) dump-evidence restore (census_dump_evidence_restore.json — the
    # per-class dominant model_types captured from /tmp/census_full_data.jsonl
    # before it was deleted; preserves the dump's fallback contribution).
    RESTORE = os.path.join(ROOT, "Testing", "census_dump_evidence_restore.json")
    if os.path.exists(RESTORE):
        for s, mt in json.load(open(RESTORE)).items():
            if s not in mt_of:
                mt_of[s] = (mt, 5)
    # (4) crawl index (census_model_index.json — model_id -> _text tags):
    # substring match for classes with no other evidence (the model's tag is
    # its config model_type). Regenerated auth-free via the Link-header cursor.
    INDEX = os.path.join(ROOT, "Testing", "census_model_index.json")
    if os.path.exists(INDEX):
        missing = [s for s in stripped if s not in mt_of]
        if missing:
            import re as _re
            index = json.load(open(INDEX))
            pat = _re.compile("|".join(sorted(missing, key=len, reverse=True)))
            for mid, mts in index.items():
                m = pat.search(mid)
                if m and m.group(0) not in mt_of:
                    mts2 = [t for t in mts if t != "none"]
                    if len(mts2) == 1:
                        mt_of[m.group(0)] = (mts2[0], 1)
    for mt, info in agg.items():
        for a in info.get("top_archs", []):
            sk = strip_arch(a["arch"])
            if sk not in mt_of or a["n"] > mt_of[sk][1]:
                mt_of[sk] = (mt, a["n"])

    # secondary evidence: the crawl index (census_model_index.json, id ->
    # _text tags) supplies model_type for classes the aggregate truncated.
    INDEX = os.path.join(ROOT, "Testing", "census_model_index.json")
    if os.path.exists(INDEX):
        import re as _re
        missing = [s for s in stripped if s not in mt_of]
        if missing:
            index = json.load(open(INDEX))
            pat = _re.compile("|".join(sorted(missing, key=len, reverse=True)))
            for mid, mts in index.items():
                m = pat.search(mid)
                if m and m.group(0) not in mt_of:
                    mts2 = [t for t in mts if t != "none"]
                    if len(mts2) == 1:
                        mt_of[m.group(0)] = (mts2[0], 1)

    mapper = build_mapper()
    toks = probe(mapper, list(stripped))
    class_mapped = {s: t != 255 for s, t in zip(stripped, toks)}

    # model_type fallback (mirror the reader): unknown class -> model_type
    # as-is -> underscore/dash-stripped. Batch-probe all distinct model_types.
    mts = set()
    for s in stripped:
        if class_mapped.get(s):
            continue
        e = mt_of.get(s)
        if e and e[0] != "<none>":
            mts.add(e[0])
            mts.add(e[0].replace("_", "").replace("-", ""))
    mt_toks = dict(zip(mts, probe(mapper, list(mts))))

    merged = {}
    for s, c in stripped.items():
        if class_mapped.get(s):
            merged[s] = merged.get(s, 0) + c
            continue
        e = mt_of.get(s)
        if not e or e[0] == "<none>":
            continue  # no model_type evidence — stays uncovered
        mt = e[0]
        if mt_toks.get(mt, 255) != 255:
            merged[mt] = merged.get(mt, 0) + c
        else:
            n = mt.replace("_", "").replace("-", "")
            if mt_toks.get(n, 255) != 255:
                merged[n] = merged.get(n, 0) + c
            # else: model_type unknown to registry — stays uncovered

    merged_toks = probe(mapper, list(merged))

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
        "excluded_non_text_gen": excluded,   # #1676: not causal decoders
        "family_counts": {k: family_counts[k] for k in sorted(family_counts)},
    }
    json.dump(summary, open(OUT, "w"), indent=1, sort_keys=True)
    print(f"total={summary['total']} with_arch={with_arch} "
          f"registry_covered={covered} ({100*covered/with_arch:.2f}%) -> {OUT}")


if __name__ == "__main__":
    main()
