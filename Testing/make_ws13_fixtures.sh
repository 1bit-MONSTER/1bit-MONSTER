#!/usr/bin/env bash
# make_ws13_fixtures.sh — generate the ws13 (DeepSeek V4/V4.1 architecture) fixtures.
#
# These are the tiny, seeded fixtures the ws13 gates compare against; they are NOT
# committed (they are ~1 MB of weights, and regenerating is deterministic). Each one
# carries the HF oracle's logits, per-layer hidden states, the reference's own
# compressor outputs and the indexer's score table + returned indices, all captured
# by forward hooks on a real forward (see Testing/make_mini_deepseek_v41.py).
#
# Requires: torch + transformers with native `DeepseekV4` (5.16.1 on this fleet) and
# numpy. Point PYTHON at an interpreter that has them, e.g.
#     PYTHON=~/ft-zaya/bin/python Testing/make_ws13_fixtures.sh
#
# Layout (override the root with WS13_FIXTURE_DIR):
#   ssl      sliding, window 32 / 5 tokens          — the existing-modules baseline
#   ssl64    sliding, window 4 / 64, rd 8           — truncation + visible rope theta
#   csa      sliding+CSA+HCA, window 4 / 64         — default index_topk (ties in play)
#   csa_nt   same, --index-topk 64                  — non-selective: the integration gate
#   odd_nt   same but H=320/5 heads/32 experts, nt  — shape-agnosticism gate
#   hca160   window 4 / 160, --index-topk 64        — HCA actually emits an entry
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
D="${WS13_FIXTURE_DIR:-/tmp/onebit-ws13}"
MK="$ROOT/Testing/make_mini_deepseek_v41.py"
mkdir -p "$D"

gen() { local name="$1"; shift; echo "== $name"; "$PY" "$MK" "$D/$name" "$@"; }

gen ssl      --profile sliding    --window 32 --prompt-len 5
gen ssl64    --profile sliding    --window 4  --prompt-len 64  --rope-frac 0.5
gen csa      --profile compressed --window 4  --prompt-len 64
gen csa_nt   --profile compressed --window 4  --prompt-len 64  --index-topk 64
gen odd_nt   --profile compressed --window 4  --prompt-len 64  --rope-frac 0.5 \
             --hidden 320 --heads 5 --head-dim 16 --q-lora 24 --o-lora 24 \
             --o-groups 4 --experts 32 --moe-int 96 --index-topk 64
gen hca160   --profile compressed --window 4  --prompt-len 160 --index-topk 64

echo "ws13 fixtures ready in $D"
