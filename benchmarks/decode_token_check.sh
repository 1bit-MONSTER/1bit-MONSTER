#!/usr/bin/env bash
# decode_token_check.sh — is the native DECODE producing the same tokens as FLM's?
#
# WHY THIS EXISTS
# Every gate in this project is a PREFILL boot token, and the decode half of the
# six-model scorecard was measured as tok/s. Timing only. RESULTS-coverage-multifamily
# section 10 documents a latent RoPE bug that lives ONLY on the decode path
# (partial_rotary_factor is initialised to the Qwen3.6-35B value 0.25 and is only
# overwritten for MoE/GDN models, so ra2 rotates 32 of 128 head dims for every plain
# model). Nothing in the current gate set can see that: a model can pass every gate we
# have and still answer wrongly once it starts decoding. This script is the missing
# check.
#
# The reference already exists and is already wired in: NPU_FLM_PREFILL=1 NPU_FLM_DECODE=1
# continues past the prefill with FLM's own forward() and prints each token. So all this
# does is run both paths and diff the token id sequences.
#
# USAGE
#   benchmarks/decode_token_check.sh <engine> <model.q4nx> [n_tokens] [ids.txt]
# e.g.
#   export NPU_XCLBIN_DIR=$PWD/engine/npu/xclbins
#   benchmarks/decode_token_check.sh engine/npu/build/npu_engine_qwen3_0_6b \
#       ~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx 8 /tmp/ids_1024.txt
#
# EXIT: 0 = token sequences match, 1 = they differ, 2 = could not get both sequences.
set -uo pipefail

ENGINE=${1:?usage: decode_token_check.sh <engine> <model.q4nx> [n_tokens] [ids.txt]}
MODEL=${2:?usage: decode_token_check.sh <engine> <model.q4nx> [n_tokens] [ids.txt]}
NG=${3:-8}
IDS=${4:-/tmp/ids_1024.txt}
: "${NPU_XCLBIN_DIR:=$PWD/engine/npu/xclbins}"
export NPU_XCLBIN_DIR

# Tokens appear as "  [<step>] <id>" on the FLM-ref decode path and as
# "  [<step>] batch=<b> toks: <id> ..." on the native one.
#
# ALIGNMENT (this was wrong and produced a false MISMATCH): the FLM-ref path prints its
# prefill token as "  [0] boot=<id>" and the native path prints the same token as the first
# row of its decode ("  [1] <id>" / "  [1] batch=1 toks: <id>"). Excluding the FLM boot line
# therefore shifted the two sequences by one and reported a divergence where there was none.
# Both sides are now taken from their FIRST token onward, and the boot comparison is kept
# because it is the strongest single check.
grab() { sed -n 's/^[[:space:]]*\[[0-9]*\][^0-9]*\(.*\)$/\1/p' | tr -s ' ' '\n' | grep -E '^[0-9]+$' | tr '\n' ' '; }
grab_flm_boot() { sed -n 's/^[[:space:]]*\[0\][[:space:]]*boot=\([0-9]*\).*/\1/p' | head -1; }

echo "engine : $ENGINE"
echo "model  : $(basename "$(dirname "$MODEL")")"
echo "tokens : $NG   ids: $IDS"
echo

FLM_OUT=$(NPU_FLM_PREFILL=1 NPU_FLM_DECODE=1 timeout 900 "$ENGINE" "$MODEL" "$NG" "$IDS" 2>/dev/null)
NAT_OUT=$(NPU_RUNLIST=1 timeout 900 "$ENGINE" "$MODEL" "$NG" "$IDS" 2>/dev/null)

# Both sides from their FIRST token onward, so the sequences are aligned. The FLM boot line
# is kept deliberately: it is the prefill gate and the strongest single check.
FLM_TOKS=$(printf '%s\n' "$FLM_OUT" | grab)
NAT_TOKS=$(printf '%s\n' "$NAT_OUT" | grep -vE 'boot=|Prefill' | grab)
FLM_BOOT=$(printf '%s\n' "$FLM_OUT" | grab_flm_boot)
NAT_BOOT=$(printf '%s\n' $NAT_TOKS | head -1)

echo "FLM-ref decode tokens : ${FLM_TOKS:-<none>}"
echo "native  decode tokens : ${NAT_TOKS:-<none>}"
if [ -n "${FLM_BOOT:-}" ] && [ -n "${NAT_BOOT:-}" ]; then
    if [ "$FLM_BOOT" = "$NAT_BOOT" ]; then
        echo "prefill boot          : MATCH ($FLM_BOOT)"
    else
        echo "prefill boot          : MISMATCH (FLM $FLM_BOOT vs native $NAT_BOOT) -- the prefill is the gate; fix that first"
    fi
fi
echo

if [ -z "${FLM_TOKS// /}" ] || [ -z "${NAT_TOKS// /}" ]; then
    echo "RESULT: could not obtain both sequences (is the device free? did the run fail?)"
    echo "        FLM-ref output tail:"; printf '%s\n' "$FLM_OUT" | tail -4 | sed 's/^/          /'
    echo "        native output tail:";  printf '%s\n' "$NAT_OUT" | tail -4 | sed 's/^/          /'
    exit 2
fi

# Compare only the first NG tokens of each, so a differing length is not a false match.
F=$(printf '%s\n' $FLM_TOKS | head -n "$NG" | tr '\n' ' ')
N=$(printf '%s\n' $NAT_TOKS | head -n "$NG" | tr '\n' ' ')
if [ "$F" = "$N" ]; then
    echo "RESULT: MATCH — the native decode agrees with FLM on the first $NG tokens."
    exit 0
fi

echo "RESULT: MISMATCH — the native decode diverges from FLM."
echo "  FLM   : $F"
echo "  native: $N"
echo
echo "  If the first token already differs, the decode path is wrong from the start"
echo "  (see section 10: the ra2 rope_dim bug, partial_rotary_factor 0.25 -> 1.0f)."
echo "  If they agree for a while and then diverge, it is more likely accumulation"
echo "  (KV cache, norm) than a per-token constant."
exit 1
