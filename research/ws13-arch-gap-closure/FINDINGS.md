# ws13 FINDINGS — Phase 0: the oracle exists, and the gate can fail

**Date:** 2026-09-12 · **Box:** ryzen · **Env:** `~/ft-zaya/bin/python` — transformers **5.16.1**, torch 2.9.0+rocm6.4
**Instrument:** `Testing/make_mini_deepseek_v41.py` (new) + the existing `Testing/cmp_deepseek_v4.cpp` comparator, **unmodified**.

## 1. What the reference can and cannot give us

`transformers` 5.16.1 ships native `DeepseekV4` — **including** `compressed_sparse_attention` (CSA) / `heavily_compressed_attention` (HCA) layer types, `compress_rates`, and `DeepseekV4CSACache`, which carries the `"indexer"` entries. So the executable reference **implements exactly the machinery the engine is missing** (P1 no longer needs to source an oracle).

It does **not** know V4.1: no `DeepseekV41*` class and no `deepseek_v41` in `AutoConfig`'s `CONFIG_MAPPING`. Consequences recorded for planning:

* V4.1-only modules (`engram`, `candidate_*`, `ffn.gate.bias_vl`, the MTP rewrite) have **no** transformers reference. Their authority is DeepSeek's own inference code — `inference/engram.py`, `inference/model.py`, `inference/vision.py` in `deepseek-ai/DeepSeek-V4.1-Flash` (fetched, not yet read → P0.2).
* The old `~/models/venv-zaya` interpreter used by `make_mini_deepseek_v4.py` is **gone from both boxes**; the instrument now runs on the surviving `ft-zaya` venv. The generator also no longer hardcodes a site-packages path.

## 2. Measured gates

Commands (from the repo root, `g++ -std=c++17 -Iinclude -Isrc -O2` — the `run_all.sh` line, unchanged):

```
python3 Testing/make_mini_deepseek_v41.py /tmp/onebit-dsv41-ssl    --profile sliding
python3 Testing/make_mini_deepseek_v41.py /tmp/onebit-dsv41-csa64  --profile compressed --prompt-len 64
python3 Testing/make_mini_deepseek_v41.py /tmp/onebit-dsv41-hca160 --profile compressed --prompt-len 160
/tmp/cmp_dsv4 <fixture-dir> <fixture-dir>/ids.txt <fixture-dir>/logits_last.npy 20 18
```

| fixture | layer types | window / prompt | compressor entries | engine vs reference | exit |
|---|---|---|---|---|---|
| `ssl` | 4 × sliding | 32 / 5 | 0 | top1 **342 = 342**, overlap **20/20** | 0 **PASS** |
| `csa64` | sliding, CSA, HCA, sliding | **4** / 64 | CSA 16 | top1 534 ≠ **685**, overlap **16/20** | 1 FAIL |
| `hca160` | sliding, CSA, HCA, sliding | **4** / 160 | CSA 40 + HCA 1 | top1 301 ≠ **207**, overlap **14/20** | 1 FAIL |

`(need 18)` is the comparator's threshold. `top1=342` on the sliding profile **reproduces the 2026-08-16 mini-gate exactly** ("mini-gate top1 342 == HF, 20/20"), so the generalized generator is faithful to the original fixture.

**The instrument has the property that matters: it is green on the modules that exist and red on the ones that do not.** The divergence grows with how much the compressor matters (0 entries → 20/20; 16 → 16/20; 40+1 → 14/20), which is what a sensitive gate looks like.

## 3. Traps found (each cost a wrong result before it was measured)

1. **A gate that cannot fail is not a gate.** The first compressed fixture used `sliding_window=32` with a 5-token prompt: the compressed entry was redundant with the sliding window and the **engine PASSED the compressed profile** (top1 390 = 390, 18/20). The fixture was blind to the very mechanism it existed to test. Fixed by making the compressed branch load-bearing (`--window 4`, prompt ≈ 16 × rate). *The reference emits one entry per window of `compress_rate` tokens, so the fixture must be longer than the rate and its window smaller than its prompt.*
2. **The compress rate is baked into the tensors.** `compressor.position_bias` is shaped `[compress_rate, dim]` (`[4,32]` CSA, `[128,16]` HCA here), so a same-weights ablation by changing the rate is **impossible** (`load_state_dict` shape mismatch). A real loader must read `compress_ratios` from the checkpoint/config rather than assume a default, and the fixture's rate is part of its identity.
3. **The reference and the published checkpoint disagree on names for the same maths.** transformers nests `self_attn.compressor.{kv_proj,gate_proj,kv_norm,position_bias}` and `self_attn.compressor.indexer.{kv_proj,gate_proj,kv_norm,q_b_proj,position_bias,scorer.weights_proj}`. The published V4-Flash/V4.1 **checkpoint** uses `attn.compressor.{wgate,wkv,norm,ape}` and a separate `attn.indexer.{wq_b,wk_b?,weights_proj,...}`. P1.1 needs an explicit name map, and the maths must be read against the checkpoint names while being validated against the executable one.
4. **HCA is barely covered.** With rate 128 it emits nothing until the prompt exceeds 128 tokens (1 entry at 160). A CSA-only fixture would silently leave the HCA branch untested — the same blindness as trap 1, one level up. `hca160` exists for this reason; a dedicated HCA-sensitive fixture (smaller HCA rate) is still owed.

## 4. Status against the ws13 plan

* **P0.1 — partially done.** The existing modules reproduce the reference on the sliding profile (20/20, top1 identical), and the reference now writes per-layer `hidden_states.npz`. The stated acceptance (≤1e-6 **per layer**) is *not yet measurable*: the engine side has no per-layer dump — `cmp_deepseek_v4.cpp` compares final logits only. Next instrument step: make the comparator (or a sibling) write engine hidden states so the bound can be taken layer by layer.
* **P0.2 — not started.** Read DeepSeek's `inference/model.py` + `inference/engram.py` for the maths of CSA/HCA/indexer and the V4.1-only modules (no transformers reference exists).
* **P0.3 — not started** (runtime format for real checkpoints: fp8 / GGUF / Q4NX / 1BP).
* **P0.4 — not started** (Mamba-3 oracle).
* **P1.1 — has its gate.** `hca160` (and `csa64`) are the fixtures the compressor + indexer implementation must flip to PASS, with `hidden_states.npz` to find the first diverging layer instead of guessing from logits.

## 5. Notes for whoever picks this up

* Fixtures are cheap and deterministic (`--seed 0`, ~4×10⁵ params, seconds to build) and live in `/tmp`; regenerate them on demand rather than committing model weights.
* `Testing/run_all.sh` currently *skips* the V4 gate when the fixture is absent — it must stay that way: `hca160`/`csa64` are **expected-fail** fixtures, so wiring them into the suite as pass/fail would make the suite red by design until P1.1 lands. Add them as a separate, explicitly-expected-red gate when P1.1 starts.
* The thin logits-overlap margin (16/20 against a threshold of 18) is a deliberate *fixture* property, not a robust metric — prefer the per-layer dumps for P1.1's acceptance and keep the logits overlap as a smoke signal.
