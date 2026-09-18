#!/usr/bin/env bash
# run_all.sh — One Bit Systems: run the full HF-native bring-up test suite.
# Compiles + runs every self-check and the real-checkpoint e2e families.
# (mirrors the bring-up arc documented in ~/onebit-modular-research.md §1-23)
set -u
cd "$(dirname "$0")/.." || exit 1   # repo root
CXX="${CXX:-g++}"; FLAGS="-std=c++17 -Iinclude -Isrc -O2"
PYTHON="${PYTHON:-python3}"
BIN=/tmp/onebit_tests; mkdir -p "$BIN"
fail=0; total=0; skip=0

# ── tripwire: every selfcheck in Testing/ must be invoked by name ──
# Two selfchecks sat here invoked by nothing (hrx_backend_selfcheck.cpp,
# lse_backend_selfcheck.cpp), and the HRX one had rotted: its own documented
# compile line omitted src/hrx_inprocess.cpp, so it could not have linked. Nothing
# noticed, because nothing ran it. This fails the moment another one appears —
# either wire it into this script (or a workflow), or say above why it is manual.
total=$((total+1))
_orphans=""
_corpus="$(cat Testing/run_all.sh .github/workflows/*.yml 2>/dev/null)"
for _f in Testing/*_selfcheck.*; do
    _b="$(basename "$_f")"
    printf '%s' "$_corpus" | grep -q -- "$_b" || _orphans="$_orphans $_b"
done
if [ -n "$_orphans" ]; then
    echo "✗ selfcheck wiring: nothing invokes:$_orphans"
    fail=$((fail+1))
else
    echo "✓ selfcheck wiring (every Testing/*_selfcheck.* is invoked)"
fi

run() {  # run <name> <compile-args...> -- <run-args...>
    local name="$1"; shift
    local src=(); local runargs=()
    while [ "${1:-}" != "--" ] && [ $# -gt 0 ]; do src+=("$1"); shift; done
    [ $# -gt 0 ] && shift
    while [ $# -gt 0 ]; do runargs+=("$1"); shift; done
    total=$((total+1))
    # Keep the compiler's own words on failure: a bare "COMPILE FAILED" is a red
    # gate that names nothing, and in CI nobody can walk over and re-run it by hand.
    local log
    if ! log=$("$CXX" $FLAGS "${src[@]}" -o "$BIN/$name" 2>&1); then
        echo "✗ $name: COMPILE FAILED"
        printf '%s\n' "$log" | tail -5 | sed 's/^/    /'
        fail=$((fail+1)); return
    fi
    if "$BIN/$name" "${runargs[@]}" >/dev/null 2>&1; then
        echo "✓ $name"; else echo "✗ $name: CHECK FAILED"; fail=$((fail+1)); fi
}

echo "== fixture self-checks =="
run arch      Testing/arch_mapping_selfcheck.cpp --
run discovery Testing/discovery_selfcheck.cpp src/model_discovery.cpp src/gguf_reader.cpp src/q4nx_reader.cpp src/safetensors_reader.cpp
run router    Testing/router_selfcheck.cpp src/model_router.cpp
run dtypes    Testing/safetensors_weights_selfcheck.cpp src/safetensors_reader.cpp src/q4nx_reader.cpp
run sharded   Testing/sharded_reader_selfcheck.cpp src/safetensors_reader.cpp src/q4nx_reader.cpp
run rotation  Testing/rotation_table_selfcheck.cpp

# Where the NPU worker is looked up: the lane fork/execs `npu_engine_universal`, and
# resolving it relative to the cwd meant a service started elsewhere silently had no
# NPU lane at all (no package ships the worker either — issue #2360). Pins the order,
# so the legacy ./ and build/ paths can never shadow an installed worker.
run npu_worker Testing/npu_worker_path_selfcheck.cpp --

# The worker itself: CI cannot compile it (no usable XRT), so it ships as a
# vendored prebuilt that packaging prefers to override with a fresh build. A
# prebuilt binary rots silently — this pins manifest↔binary shas, the RUNPATH that
# lets it find its bundled libomp in every layout we ship, and both staging paths.
total=$((total+1))
if bundle_out=$("$PYTHON" Testing/npu_worker_bundle_selfcheck.py 2>&1); then
    echo "✓ npu_bundle"
else
    echo "✗ npu_bundle"
    printf '%s\n' "$bundle_out" | tail -6 | sed 's/^/    /'
    fail=$((fail+1))
fi
run iq1       Testing/iq1_selfcheck.cpp --

# Padded-vocab embedding gate: some 1BP artifacts declare the checkpoint's padded
# vocab (262272) while shipping the unpadded table (262147 rows) — that is the
# published v1 ZAYA1-8B upload, and the engine used to refuse it outright. The gate
# adopts the table's rows as the vocab and still refuses real truncation, so this
# pins the boundary in one place (#1521 producer side, #1606 truncation).
run embed_pad Testing/embed_pad_gate_selfcheck.cpp --
run tq2nz     Testing/tq2nz_e4m3_selfcheck.cpp --
# NPU artifact key contract (issue #2193): the header-window regression and the
# per-family GEMM names, both verifiable without a device.
run npu_keys  Testing/npu_key_contract_selfcheck.cpp src/q4nx_reader.cpp --
# NPU path resolution: an override naming a path this machine does not have must
# not be used (a stale NPU_XCLBIN_DIR in the shell silently broke every NPU run).
run npu_paths Testing/npu_paths_selfcheck.cpp --

# NPU fused-weight packing: transpose_pack's `in_f` is the ROW STRIDE of its source, so an
# offset that names a row block has to be scaled by it. The GDN K and V blocks passed a row
# count unscaled, so every row past 0 was read from a sliding window inside q's first rows —
# invisible to every other check, because the packing runs inside a 4000-line function and a
# wrong block still has the right shape (#2451). Source-level, no device needed.
echo "== npu pack stride =="
total=$((total+1))
if pack_stride_out=$("$PYTHON" Testing/npu_pack_stride_selfcheck.py 2>&1); then
    printf '%s\n' "$pack_stride_out" | sed 's/^/  /'
    echo "✓ npu_pack_stride"
else
    echo "✗ npu_pack_stride"
    printf '%s\n' "$pack_stride_out" | tail -6 | sed 's/^/    /'
    fail=$((fail+1))
fi

# CLI dispatch coverage: tools/onebit.cpp's whole command set (chat, pull, list,
# status, …) is compiled into the single ELF, but tools/onebin.cpp declared
# onebit_main and never called it — so the documented `./run.sh chat` printed the
# top-level usage, and so did every other command. A compiler cannot see a
# declared-but-uncalled function or a symlink with no branch, so the three lists
# (packaged symlinks, accepted commands, dispatch branches) are compared here.
echo "== CLI dispatch coverage =="
total=$((total+1))
if dispatch_out=$("$PYTHON" Testing/dispatch_selfcheck.py 2>&1); then
    echo "✓ cli_dispatch"
else
    echo "✗ cli_dispatch"
    printf '%s\n' "$dispatch_out" | tail -6 | sed 's/^/    /'
    fail=$((fail+1))
fi

# The same question asked of the ARTIFACT rather than the source lists: the static
# check cannot tell whether the linked binary really routes those names (that is
# how `./run.sh chat` shipped printing usage while every static check passed). It
# runs only when a built binary is present — this suite is host-only and does not
# build one — so the CI job that DOES build the binary runs it with --require.
echo "== CLI entry points (built binary) =="
total=$((total+1))
if [ -x build/1bit ]; then
    if smoke_out=$("$PYTHON" Testing/cli_smoke.py 2>&1); then
        echo "✓ cli_smoke"
        printf '%s\n' "$smoke_out" | grep -E "^  note" | sed 's/^/  /'
    else
        echo "✗ cli_smoke"
        printf '%s\n' "$smoke_out" | tail -6 | sed 's/^/    /'
        fail=$((fail+1))
    fi
else
    echo "  - cli_smoke: no build/1bit — skipped (run it where the binary is built)"; skip=$((skip+1))
fi

# Docs-and-repo consistency: links that point at nothing, paths that resolve
# nowhere, documented commands whose target does not exist, CI that invokes a
# missing script. Every one of these shipped in a form a compiler cannot see —
# the README told you to run a model you could not download, and `./run.sh chat`
# printed the usage text — so they are checked here, against the gated surface
# only (front page, guides, wiki, packaging/site READMEs).
echo "== repo consistency =="
total=$((total+1))
if docs_out=$("$PYTHON" Testing/repo_docs_selfcheck.py 2>&1); then
    echo "✓ repo_consistency"
    printf '%s\n' "$docs_out" | grep -E "^  \(" | sed 's/^/  /'
else
    echo "✗ repo_consistency"
    printf '%s\n' "$docs_out" | tail -8 | sed 's/^/    /'
    fail=$((fail+1))
fi
# Version manifest sync: Version consistency is a REQUIRED check (ruleset 19117606), and its
# script began with `[ -f "$file" ] || return 0` — so a manifest that moved out of the tree
# took its version check with it, silently. Two of its entries named a Homebrew formula that
# has never existed here, and with every manifest deleted it still exited 0 (issue #2488).
# The fix gives absence a failure path; these cases are what keeps it one.
echo "== version manifest sync =="
total=$((total+1))
if vsync_out=$(bash Testing/version_sync_selfcheck.sh 2>&1); then
    printf '%s\n' "$vsync_out" | sed 's/^/  /'
    echo "✓ version_sync"
else
    echo "✗ version_sync"
    printf '%s\n' "$vsync_out" | tail -8 | sed 's/^/    /'
    fail=$((fail+1))
fi
# Census diagnostics: one repo root, one policy set. Three scripts pinned ROOT
# to the shared checkout and two carried a stale NON_TEXT_GEN copy (#2387), so a
# worktree run read the wrong inputs and wrote the wrong tree — invisible,
# because the run succeeds against them.
total=$((total+1))
if census_out=$("$PYTHON" Testing/census_scripts_selfcheck.py 2>&1); then
    echo "✓ census_scripts"
else
    echo "✗ census_scripts"
    printf '%s\n' "$census_out" | tail -8 | sed 's/^/    /'
    fail=$((fail+1))
fi
# The family manifest's declared mappings. `Testing/bringup_runner.sh` step 1 does
# exactly this comparison and nothing invokes bringup_runner.sh (its step 3 needs
# fixtures and torch), which is how two families came to declare MIMO and GLM —
# tokens that do not exist in the enum (#2511). This runs the half that needs
# neither fixtures nor a device.
total=$((total+1))
if manifest_out=$("$PYTHON" Testing/manifest_mapping_selfcheck.py 2>&1); then
    echo "✓ manifest_mappings"
    printf '%s\n' "$manifest_out" | grep -E "^  " | sed 's/^/  /'
else
    echo "✗ manifest_mappings"
    printf '%s\n' "$manifest_out" | tail -8 | sed 's/^/    /'
    fail=$((fail+1))
fi
# The alias autopr must never treat a NAME as a family. Its fuzzy rules filed
# two draft PRs (`language`, `picolm`) that were closed unjustified (#2443,
# #2444), because the mapping table is an exact-match dispatch table holding
# 1-4 char aliases (`h` -> LLAMA) that a prefix rule turned into wildcards. The
# self-test pins both over-fires, the exact-match path that should still file,
# and the table-vs-engine agreement — and fails if it can read no table at all,
# so "could not determine" cannot print as a pass.
total=$((total+1))
if autopr_out=$("$PYTHON" Testing/census_autopr.py --self-test 2>&1); then
    echo "✓ census_autopr"
else
    echo "✗ census_autopr"
    printf '%s\n' "$autopr_out" | tail -8 | sed 's/^/    /'
    fail=$((fail+1))
fi
# Published coverage claims must equal the census. seo_sync rewrites them in the
# daily apply workflows, but that is not a gate — four false-claim shapes
# survived for months in wordings its patterns did not know (#2389 -> #2397).
# This checks the content of site/*.html and README.md, not the patterns.
total=$((total+1))
if claims_out=$("$PYTHON" Testing/seo_claim_selfcheck.py 2>&1); then
    echo "✓ seo_claims"
else
    echo "✗ seo_claims"
    printf '%s\n' "$claims_out" | tail -10 | sed 's/^/    /'
    fail=$((fail+1))
fi
# The claim gate itself must be able to fail. validate_claims.py --check-readme
# scanned README.md for five engine names; the figures moved to the wiki and the
# README became a landing page, so the scan read zero rows and returned [] for
# months while the page that inherited the numbers stamped a quarantined tok/s
# figure "validated" (#2476). A zero-row scan is not an error, so nothing failed.
# This injects a bad row into each claim page and requires the gate to catch it.
total=$((total+1))
if gate_out=$("$PYTHON" Testing/claims_gate_selfcheck.py 2>&1); then
    echo "✓ claims_gate"
    printf '%s\n' "$gate_out" | grep -E "^  note" | sed 's/^/  /'
else
    echo "✗ claims_gate"
    printf '%s\n' "$gate_out" | tail -8 | sed 's/^/    /'
    fail=$((fail+1))
fi
# v4 dedup e2e: synthetic GGUF with duplicated tensors -> converter -> loaders
DEDUP_DIR=/tmp/onebit_dedup; mkdir -p "$DEDUP_DIR"
total=$((total+1))
# Both halves report separately: "build/generate failed" covered a missing
# python package and a compile error alike, and said which of them it was to
# nobody. The fixture is stdlib-only now, but the next cause must be readable.
gen_log=$("$PYTHON" Testing/make_mini_gguf.py "$DEDUP_DIR/mini.gguf" 2>&1); gen_rc=$?
cc_log=$("$CXX" $FLAGS src/gguf_to_onebp.cpp src/gguf_reader.cpp src/q4nx_reader.cpp src/safetensors_reader.cpp \
    -o "$BIN/g2o" 2>&1); cc_rc=$?
if [ $gen_rc -eq 0 ] && [ $cc_rc -eq 0 ]; then
    conv_out=$("$BIN/g2o" "$DEDUP_DIR/mini.gguf" "$DEDUP_DIR/mini.1bp" 2>&1)
    if [ $? -eq 0 ] && printf '%s' "$conv_out" | grep -q 'dedup: blk.1.attn_q.weight'; then
        echo "✓ dedup converter (alias emitted)"
        run dedup_e2e Testing/dedup_loader_check.cpp src/onebp_model.cpp -- "$DEDUP_DIR/mini.1bp"
    else
        echo "✗ dedup converter: no alias emitted"; fail=$((fail+1))
    fi
else
    echo "✗ dedup converter: build/generate failed"
    [ $gen_rc -ne 0 ] && printf '%s\n' "$gen_log" | tail -5 | sed 's/^/    fixture:  /'
    [ $cc_rc -ne 0 ] && printf '%s\n' "$cc_log" | tail -5 | sed 's/^/    compiler: /'
    fail=$((fail+1))
fi

# NPU engine build script: nothing in CI invokes engine/npu/build_npu.sh, so a
# fresh clone could not build the NPU engine at all — it used its build dir 30
# lines before creating it — and no job noticed until someone cloned (#2440).
# A real build needs XRT, which a hosted runner has not got; this needs none,
# because it stubs the compilers and tests the script's own file handling.
echo "== NPU engine build script (stubbed compilers) =="
total=$((total+1))
if npu_build_out=$(bash Testing/npu_build_script_selfcheck.sh 2>&1); then
    printf '%s\n' "$npu_build_out" | sed 's/^/  /'
    echo "✓ npu_build_script"
else
    echo "✗ npu_build_script"
    printf '%s\n' "$npu_build_out" | tail -8 | sed 's/^/    /'
    fail=$((fail+1))
fi

# NPU artifact lookup: the engine resolves an xclbin (xp) and an instruction
# file (ip) per tensor slot. When ip() had no dimension-keyed fallback while
# xp() did, slots whose instructions are committed under their shape rather than
# a model tag silently fell back to the runtime generator, whose output is
# single-core-row and wrong against a multi-row xclbin. Qwen3-4B's QKV and O hit
# this. Needs no device and no compiler; has a built-in pre-fix control.
echo "== NPU insts/xclbin lookup parity =="
total=$((total+1))
if insts_lookup_out=$(PYTHON="$PYTHON" bash Testing/npu_insts_lookup_selfcheck.sh 2>&1); then
    printf '%s\n' "$insts_lookup_out" | sed 's/^/  /'
    echo "✓ npu_insts_lookup"
else
    echo "✗ npu_insts_lookup"
    printf '%s\n' "$insts_lookup_out" | tail -12 | sed 's/^/    /'
    fail=$((fail+1))
fi

# Provenance writer: build.toolchain can only come from the build that produced the
# artifacts, and the committed manifest is null because the last rebuild ran the
# documented command without --toolchain (#2262). The write path now refuses that,
# so the omission cannot silently repeat; this pins the refusal and its escape hatches.
echo "== xclbin provenance toolchain guard =="
total=$((total+1))
if toolchain_out=$(bash Testing/xclbin_toolchain_gate_selfcheck.sh 2>&1); then
    printf '%s\n' "$toolchain_out" | sed 's/^/  /'
    echo "✓ xclbin_toolchain_guard"
else
    echo "✗ xclbin_toolchain_guard"
    printf '%s\n' "$toolchain_out" | tail -8 | sed 's/^/    /'
    fail=$((fail+1))
fi

# NPU lane contract: the NPU runs on the engine's own worker (src/backend_npu.cpp
# → npu_engine_universal, FLM-free). install.sh never built or mentioned it, and
# the legacy FLM test harness printed "FLM not installed" as if the NPU were
# broken. A compiler cannot see a missing install step or a mislabelled lane (#2358).
echo "== NPU lane contract =="
total=$((total+1))
if npu_lane_out=$("$PYTHON" Testing/npu_lane_selfcheck.py 2>&1); then
    echo "✓ npu_lane"
else
    echo "✗ npu_lane"
    printf '%s\n' "$npu_lane_out" | tail -6 | sed 's/^/    /'
    fail=$((fail+1))
fi

echo "== backend compile =="
total=$((total+1))
if "$CXX" $FLAGS -c src/backend_generic.cpp -o "$BIN/bg.o" 2>/dev/null; then
    echo "✓ backend_generic.cpp"; else echo "✗ backend_generic.cpp"; fail=$((fail+1)); fi

echo "== e2e (needs model fixtures in /tmp/onebit-e2e — skipped if absent) =="

# The manifest's own runner: 29 of the 32 families in Testing/models_manifest.json carry
# `validated`, and Testing/bringup_runner.sh is what those statuses name — but nothing invoked
# it, and on a box without the host fixtures it reported "0/4 generation gates passed" with the
# other 25 families missing from the denominator entirely (with no gate commands and no
# fixtures it printed 0/0 and exited 0). Its verdict helpers are what this pins; see #2520.
echo "== manifest gate runner =="
total=$((total+1))
if bringup_out=$(bash Testing/bringup_runner_selfcheck.sh 2>&1); then
    printf '%s\n' "$bringup_out" | sed 's/^/  /'
    echo "✓ bringup_runner"
else
    echo "✗ bringup_runner"
    printf '%s\n' "$bringup_out" | tail -8 | sed 's/^/    /'
    fail=$((fail+1))
fi

e2e() {  # e2e <name> <model_dir> <oracle.gguf> [expect-torch-string]
    local name="$1" dir="$2" gguf="$3"
    if [ ! -f "$gguf" ]; then echo "  - $name: fixtures absent, skipped"; total=$((total+1)); skip=$((skip+1)); return; fi
    total=$((total+1))
    if ! "$CXX" $FLAGS src/backend_generic.cpp src/model_discovery.cpp src/gguf_reader.cpp \
        src/q4nx_reader.cpp src/safetensors_reader.cpp Testing/e2e_safetensors_selfcheck.cpp \
        -o "$BIN/e2e" 2>/dev/null; then echo "✗ $name e2e: COMPILE FAILED"; fail=$((fail+1)); return; fi
    if "$BIN/e2e" "$dir" "$gguf" >/dev/null 2>&1; then
        echo "✓ $name e2e"; else echo "✗ $name e2e: loader mismatch"; fail=$((fail+1)); fi
}
e2e llama  /tmp/onebit-e2e/smollm  /tmp/onebit-e2e/smollm/oracle-q8.gguf
e2e qwen2  /tmp/onebit-e2e/qwen2    /tmp/onebit-e2e/qwen2/oracle-q8.gguf
e2e gemma  /tmp/onebit-e2e/gemma    /tmp/onebit-e2e/gemma/oracle-q8.gguf
e2e qwen3  /tmp/onebit-e2e/qwen3    /tmp/onebit-e2e/qwen3/oracle-q8.gguf

# Instella-MoE (DeepSeek-V3 clone): gated MLA + FarSkip dual-residual + sigmoid
# router engine gate — mini fixture (real tokenizer, mini dims) vs HF logits.
# Fixtures live in 1bit-monster/models/kl-test/ (mini-full-f16.gguf + mini-full-hf.pt).
instella_mini=/tmp/onebit-instella/mini-full-f16.gguf
instella_ref=/tmp/onebit-instella/hf.npy
if [ -f "$instella_mini" ] && [ -f "$instella_ref" ]; then
    total=$((total+1))
    if ! "$CXX" $FLAGS Testing/cmp_instella.cpp src/deepseek.cpp src/gguf_reader.cpp \
        -o "$BIN/cmp_instella" 2>/dev/null; then echo "✗ instella: COMPILE FAILED"; fail=$((fail+1));
    elif "$BIN/cmp_instella" "$instella_mini" /tmp/onebit-instella/ids.txt "$instella_ref" 20 18 >/dev/null 2>&1; then
        echo "✓ instella engine (gated MLA + FarSkip)";
    else echo "✗ instella engine: top-20 mismatch vs HF"; fail=$((fail+1)); fi
else
    echo "  - instella: fixtures absent, skipped (cp -r 1bit-monster/models/kl-test/mini-full* /tmp/onebit-instella/)"
    total=$((total+1)); skip=$((skip+1))
fi


# Instella 1bp loader gate (Gate 5): convert mini GGUF -> Q4NX 1bp -> load via
# DeepSeekModel::load_from_1bp, assert MLA dims + gated/farskip flags round-trip
# and the forward runs. Golden top1 = the fp16 GGUF engine's own output (mini
# weights are random ~0.01 so quantization drifts top-1 — the STRUCTURE is the
# gate here; the real-16B Q4NX top1=128804 == fp16 is the manual validation).
instella_mini=/tmp/onebit-instella/mini-full-f16.gguf
instella_1bp=/tmp/onebit-instella/mini-full-q4nx.1bp
if [ -f "$instella_mini" ]; then
    total=$((total+1))
    if ! "$CXX" $FLAGS src/gguf_to_onebp.cpp src/gguf_reader.cpp -o "$BIN/gguf_to_onebp" 2>/dev/null; then
        echo "✗ instella-1bp: converter COMPILE FAILED"; fail=$((fail+1))
    else
        "$BIN/gguf_to_onebp" "$instella_mini" "$instella_1bp" >/dev/null 2>&1
        if ! "$CXX" $FLAGS Testing/cmp_instella_1bp.cpp src/deepseek.cpp src/gguf_reader.cpp \
            -o "$BIN/cmp_instella_1bp" 2>/dev/null; then echo "✗ instella-1bp: COMPILE FAILED"; fail=$((fail+1));
        elif "$BIN/cmp_instella_1bp" "$instella_1bp" /tmp/onebit-instella/ids.txt 178 >/dev/null 2>&1; then
            echo "✓ instella 1bp loader (MLA dims + gated + farskip round-trip)";
        else echo "✗ instella 1bp loader: structure/top1 mismatch"; fail=$((fail+1)); fi
    fi
else
    echo "  - instella-1bp: fixture absent, skipped"
    total=$((total+1)); skip=$((skip+1))
fi

# ── DeepSeek V4 gate (mini fixture, HF safetensors oracle) ──
total=$((total+1))
dsv4_dir=/tmp/onebit-dsv4
if [ -f "$dsv4_dir/logits_last.npy" ] && [ -f "$dsv4_dir/model.safetensors" ]; then
    echo "5 7 9 11 3" > /tmp/onebit-dsv4-ids.txt
    if ! "$CXX" $FLAGS Testing/cmp_deepseek_v4.cpp src/deepseek_v4.cpp src/safetensors_reader.cpp src/q4nx_reader.cpp \
        -o "$BIN/cmp_dsv4" 2>/dev/null; then echo "✗ deepseek_v4: COMPILE FAILED"; fail=$((fail+1));
    elif "$BIN/cmp_dsv4" "$dsv4_dir" /tmp/onebit-dsv4-ids.txt "$dsv4_dir/logits_last.npy" 20 18 >/dev/null 2>&1; then
        echo "✓ deepseek_v4 engine (Shared-KV MQA + mHC + hash-MoE)";
    else echo "✗ deepseek_v4 engine: top-20 mismatch vs HF"; fail=$((fail+1)); fi
else
    echo "  - deepseek_v4: fixture absent, skipped (python3 Testing/make_mini_deepseek_v4.py /tmp/onebit-dsv4)"; skip=$((skip+1))
fi

# ── ws13: DeepSeek V4/V4.1 architecture gates (fixtures: Testing/make_ws13_fixtures.sh) ──
# Every gate here compares against the REFERENCE'S OWN values captured by forward hooks.
# Skipped when the fixtures are absent, like the V4 gate above. The compressed gates all
# use a NON-SELECTIVE indexer (--index-topk 64): with the configured index_topk, which of
# several exactly-equal scores wins is implementation-defined on the reference side, so
# an index-equality gate would fail a correct implementation (see ws13 FINDINGS.md).
ws13="${WS13_FIXTURE_DIR:-/tmp/onebit-ws13}"
if [ -f "$ws13/csa_nt/comp_ref_L1.npy" ]; then
    "$CXX" $FLAGS Testing/cmp_deepseek_v4_compressor.cpp src/deepseek_v4.cpp src/safetensors_reader.cpp src/q4nx_reader.cpp -o "$BIN/cmp_comp" 2>/dev/null || { echo "✗ ws13 compressor: COMPILE FAILED"; fail=$((fail+1)); }
    "$CXX" $FLAGS Testing/cmp_deepseek_v4_compressor_incremental.cpp src/deepseek_v4.cpp src/safetensors_reader.cpp src/q4nx_reader.cpp -o "$BIN/cmp_comp_inc" 2>/dev/null || { echo "✗ ws13 compressor-incremental: COMPILE FAILED"; fail=$((fail+1)); }
    "$CXX" $FLAGS Testing/cmp_deepseek_v4_indexer.cpp src/deepseek_v4.cpp src/safetensors_reader.cpp src/q4nx_reader.cpp -o "$BIN/cmp_indexer" 2>/dev/null || { echo "✗ ws13 indexer: COMPILE FAILED"; fail=$((fail+1)); }
    # build our own comparator: the V4 gate above only builds $BIN/cmp_dsv4 when ITS
    # fixture is present, so depending on it would make these gates un-runnable alone.
    "$CXX" $FLAGS Testing/cmp_deepseek_v4.cpp src/deepseek_v4.cpp src/safetensors_reader.cpp src/q4nx_reader.cpp -o "$BIN/cmp_dsv4_ws13" 2>/dev/null || { echo "✗ ws13 e2e: COMPILE FAILED"; fail=$((fail+1)); }

    # compressor maths, batched and incremental (CSA two-series, HCA single-series)
    total=$((total+1))
    if "$BIN/cmp_comp" "$ws13/csa_nt" 1 "$ws13/csa_nt/attn_input_L1.npy" "$ws13/csa_nt/comp_ref_L1.npy" 1e-6 >/dev/null 2>&1 \
       && "$BIN/cmp_comp" "$ws13/hca160" 2 "$ws13/hca160/attn_input_L2.npy" "$ws13/hca160/comp_ref_L2.npy" 1e-6 >/dev/null 2>&1; then
        echo "✓ ws13 compressor (CSA batched + HCA batched)"
    else echo "✗ ws13 compressor: mismatch vs the reference"; fail=$((fail+1)); fi

    total=$((total+1))
    if "$BIN/cmp_comp_inc" "$ws13/csa_nt" 1 "$ws13/csa_nt/attn_input_L1.npy" "$ws13/csa_nt/comp_ref_L1.npy" 1e-6 >/dev/null 2>&1 \
       && "$BIN/cmp_comp_inc" "$ws13/hca160" 2 "$ws13/hca160/attn_input_L2.npy" "$ws13/hca160/comp_ref_L2.npy" 1e-6 >/dev/null 2>&1; then
        echo "✓ ws13 compressor incremental (token-by-token == batched reference)"
    else echo "✗ ws13 compressor incremental: mismatch"; fail=$((fail+1)); fi

    # compressed attention integrated: non-selective indexer -> exact per layer
    total=$((total+1))
    if "$BIN/cmp_dsv4_ws13" "$ws13/csa_nt" "$ws13/csa_nt/ids.txt" "$ws13/csa_nt/logits_last.npy" 20 18 "$BIN/ws13_nt.bin" >/dev/null 2>&1 \
       && "$BIN/cmp_dsv4_ws13" "$ws13/odd_nt" "$ws13/odd_nt/ids.txt" "$ws13/odd_nt/logits_last.npy" 20 18 "$BIN/ws13_odd.bin" >/dev/null 2>&1; then
        echo "✓ ws13 compressed attention (integration + shape-agnostic, non-selective)"
    else echo "✗ ws13 compressed attention: mismatch"; fail=$((fail+1)); fi

    # python-level gates (per-layer bound, indexer scores) — need numpy
    if "$PYTHON" -c 'import numpy' >/dev/null 2>&1; then
        total=$((total+1))
        if "$PYTHON" Testing/cmp_deepseek_v4_layers.py "$ws13/csa_nt" "$BIN/ws13_nt.bin" --tol 1e-6 >/dev/null 2>&1; then
            echo "✓ ws13 per-layer bound (<=1e-6, non-selective indexer)"
        else echo "✗ ws13 per-layer bound: exceeded"; fail=$((fail+1)); fi
        total=$((total+1))
        if "$BIN/cmp_indexer" "$ws13/csa" 1 "$ws13/csa/attn_input_L1.npy" "$ws13/csa/indexer_ref_L1.npy" "$BIN/ws13_ix.bin" >/dev/null 2>&1 \
           && "$PYTHON" Testing/cmp_deepseek_v4_indexer_scores.py "$ws13/csa" "$BIN/ws13_ix.bin" --layer 1 --tol 1e-6 >/dev/null 2>&1; then
            echo "✓ ws13 indexer scores (<=1e-6 + order-independent selection validity)"
        else echo "✗ ws13 indexer scores: mismatch"; fail=$((fail+1)); fi
    else
        echo "  - ws13 python gates skipped (no numpy in $PYTHON)"; total=$((total+2)); skip=$((skip+2))
    fi
else
    echo "  - ws13: fixtures absent, skipped (PYTHON=<torch env> Testing/make_ws13_fixtures.sh)"; total=$((total+1)); skip=$((skip+1))
fi

# ── GLM-MoE-DSA gate (mini fixture, HF safetensors oracle) ──
total=$((total+1))
# Committed fixture (Testing/fixtures/glmdsa) so the gate runs in CI without torch;
# GLMDSA_FIXTURE_DIR overrides for a locally regenerated /tmp fixture.
glmdsa_dir=${GLMDSA_FIXTURE_DIR:-Testing/fixtures/glmdsa}
[ -f "$glmdsa_dir/logits_last.npy" ] && [ -f "$glmdsa_dir/model.safetensors" ] || glmdsa_dir=/tmp/onebit-glmdsa
ids_file=/tmp/onebit-glmdsa-ids.txt
if [ -f "$glmdsa_dir/logits_last.npy" ] && [ -f "$glmdsa_dir/model.safetensors" ]; then
    echo "5 7 9 11 3" > "$ids_file"
    if ! "$CXX" $FLAGS Testing/cmp_glm_moe_dsa.cpp src/glm_moe_dsa.cpp src/safetensors_reader.cpp src/q4nx_reader.cpp \
        -o "$BIN/cmp_glmdsa" 2>/dev/null; then echo "✗ glm_moe_dsa: COMPILE FAILED"; fail=$((fail+1));
    elif "$BIN/cmp_glmdsa" "$glmdsa_dir" "$ids_file" "$glmdsa_dir/logits_last.npy" 20 18 >/dev/null 2>&1; then
        echo "✓ glm_moe_dsa engine (V3-MLA + DSA indexer + group-topk MoE)";
    else echo "✗ glm_moe_dsa engine: top-20 mismatch vs HF"; fail=$((fail+1)); fi
else
    echo "  - glm_moe_dsa: fixture absent, skipped (python3 Testing/make_mini_glm_moe_dsa.py /tmp/onebit-glmdsa)"; skip=$((skip+1))
fi

# ── GLM-MoE-DSA "shared" indexer configuration (issue #2423) ──
# Uses the same fixture and skips itself when that fixture is absent, so its
# result is folded into this suite's counters rather than reported separately.
total=$((total+1))
shared_out=$(bash Testing/check_glmdsa_shared_config.sh 2>&1) && shared_rc=0 || shared_rc=$?
printf '%s\n' "$shared_out"
if printf '%s' "$shared_out" | grep -q 'fixture absent, skipped'; then
    skip=$((skip+1))
elif [ "$shared_rc" -ne 0 ]; then
    fail=$((fail+1))
fi

# ── MiMo-V2 gate (mini fixture, vendored remote modeling oracle) ──
total=$((total+1))
mimo_dir=/tmp/onebit-mimo
if [ -f "$mimo_dir/logits_last.npy" ] && [ -f "$mimo_dir/model.safetensors" ]; then
    echo "5 7 9 11 3" > /tmp/onebit-mimo-ids.txt
    if ! "$CXX" $FLAGS Testing/cmp_mimo_v2.cpp src/mimo_v2.cpp src/safetensors_reader.cpp src/q4nx_reader.cpp \
        -o "$BIN/cmp_mimo" 2>/dev/null; then echo "✗ mimo_v2: COMPILE FAILED"; fail=$((fail+1));
    elif "$BIN/cmp_mimo" "$mimo_dir" /tmp/onebit-mimo-ids.txt "$mimo_dir/logits_last.npy" 20 18 >/dev/null 2>&1; then
        echo "✓ mimo_v2 engine (MoD hybrid: SWA+full GQA, sigmoid group-topk MoE)";
    else echo "✗ mimo_v2 engine: top-20 mismatch vs HF"; fail=$((fail+1)); fi
else
    echo "  - mimo_v2: fixture absent, skipped (python3 Testing/make_mini_mimo_v2.py /tmp/onebit-mimo)"; skip=$((skip+1))
fi

# ── Qwen3_5 text gate (mini fixture, HF oracle) ──
total=$((total+1))
q35_dir=/tmp/onebit-q35
if [ -f "$q35_dir/logits_last.npy" ] && [ -f "$q35_dir/model.safetensors" ]; then
    echo "5 7 9 11 3" > /tmp/onebit-q35-ids.txt
    if ! "$CXX" $FLAGS Testing/cmp_qwen3_5.cpp src/qwen3_5.cpp src/safetensors_reader.cpp src/q4nx_reader.cpp \
        -o "$BIN/cmp_q35" 2>/dev/null; then echo "✗ qwen3_5: COMPILE FAILED"; fail=$((fail+1));
    elif "$BIN/cmp_q35" "$q35_dir" /tmp/onebit-q35-ids.txt "$q35_dir/logits_last.npy" 20 18 >/dev/null 2>&1; then
        echo "✓ qwen3_5 text engine (GatedDeltaNet + gated GQA hybrid)";
    else echo "✗ qwen3_5 text engine: top-20 mismatch vs HF"; fail=$((fail+1)); fi
else
    echo "  - qwen3_5: fixture absent, skipped (python3 Testing/make_mini_qwen3_5.py /tmp/onebit-q35)"; skip=$((skip+1))
fi

# ── Mesh: self-aware network substrate (optional — needs the CMake build) ──
total=$((total+1))
if [ -x build/mesh_peer ]; then
    if bash Testing/mesh_smoke.sh build/mesh_peer >/dev/null 2>&1; then
        echo "✓ mesh (peer discovery + ask/answer)";
    else echo "✗ mesh (peer discovery + ask/answer)"; fail=$((fail+1)); fi
else
    echo "  - mesh: mesh_peer binary absent, skipped (cmake --build build --target mesh_peer)"; skip=$((skip+1))
fi

# ── JARVIS fleet dispatch (optional — needs build/1bit + build/mesh_peer) ──
total=$((total+1))
if [ -x build/1bit ] && [ -x build/mesh_peer ]; then
    if bash Testing/jarvis_mesh_smoke.sh build/1bit build/mesh_peer >/dev/null 2>&1; then
        echo "✓ jarvis fleet dispatch (mesh-aware, DSH brain path)";
    else echo "✗ jarvis fleet dispatch (mesh-aware, DSH brain path)"; fail=$((fail+1)); fi
else
    echo "  - jarvis fleet: binaries absent, skipped (cmake --build build --target onebin mesh_peer)"; skip=$((skip+1))
fi


run rni-bf16 Testing/aie2p_bf16_rni_selfcheck.cpp --

# ── HRX + LSE backend lifecycle (optional — need the fetched nlohmann include) ──
# Both of these selfchecks have existed for a while and NOTHING invoked either of
# them. That is how the HRX one's own documented compile line went stale unnoticed:
# it omitted src/hrx_inprocess.cpp, where hrx::Inprocess now lives, so the check
# would not have linked even if someone had wired it up. Both self-skip their live
# half without HRX_*/LSE_* set, so the lifecycle half runs anywhere.
for _spec in "hrx-backend|src/backend_hrx.cpp src/hrx_inprocess.cpp|Testing/hrx_backend_selfcheck.cpp" \
             "lse-backend|src/backend_lse.cpp|Testing/lse_backend_selfcheck.cpp"; do
    _name="${_spec%%|*}"; _rest="${_spec#*|}"; _srcs="${_rest%%|*}"; _chk="${_rest##*|}"
    total=$((total+1))
    if [ ! -f build/_deps/nlohmann_json-src/include/nlohmann/json.hpp ]; then
        echo "  - $_name: nlohmann include absent, skipped (needs a configured build tree)"
        skip=$((skip+1)); continue
    fi
    # Keep the compiler's own words, like run() above: a bare COMPILE FAILED names nothing.
    if ! _log=$("$CXX" $FLAGS -Ibuild/_deps/nlohmann_json-src/include \
                $_srcs "$_chk" -o "$BIN/$_name" 2>&1); then
        echo "✗ $_name: COMPILE FAILED"
        printf '%s\n' "$_log" | tail -4 | sed 's/^/    /'
        fail=$((fail+1)); continue
    fi
    if "$BIN/$_name" >/dev/null 2>&1; then
        echo "✓ $_name"
    else echo "✗ $_name: CHECK FAILED"; fail=$((fail+1)); fi
done

echo "======================================"
echo "$((total-fail-skip))/$total passed, $skip skipped"
[ "$fail" -eq 0 ] || { echo "$fail FAILURES"; exit 1; }

