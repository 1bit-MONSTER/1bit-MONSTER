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

---

# Phase 0 continued — P0.1 measured (per-layer bound MET), and the gate localizes the gap

**Date:** 2026-09-12 · **Instrument:** `Testing/cmp_deepseek_v4.cpp` (extended with an optional
per-layer dump) + `Testing/cmp_deepseek_v4_layers.py` (new) + `Testing/make_mini_deepseek_v41.py`
(`--weights-dtype` added).

## 1. P0.1 acceptance: the existing modules reproduce the reference PER LAYER

The logits gate said "20/20 overlap"; that is not the stated acceptance. With the engine now
dumping its residual streams at every layer boundary and the reference's `hidden_states` compared
layer by layer (`Testing/cmp_deepseek_v4_layers.py`), on the **fp32** sliding fixture:

| state (input to layer) | max&#124;Δ&#124; | rel | &#124;ref&#124;max |
|---|---|---|---|
| 0 | **0.000e+00** (bit-exact) | 0.00e+00 | 0.052 |
| 1 | 7.451e-09 | 1.36e-07 | 0.055 |
| 2 | 9.313e-09 | 1.68e-07 | 0.055 |
| 3 | 1.118e-08 | 2.25e-07 | 0.050 |

**worst = 1.118e-08 against a 1e-6 tolerance → PASS**, ~90× inside the bound, and the layer-0 state
(the expanded embedding) is bit-exact. Modules covered: mHC (Sinkhorn-Knopp streams), shared-KV MQA
with per-head sinks and partial RoPE, hash routing (`tid2eid`) on layers 0–1, top-k sqrtsoftplus MoE
on layers 2–3, shared SwiGLU experts with `swiglu_limit`. **P0.1 is done for the modules that exist.**

**Mapping verified, not assumed:** transformers appends `hidden_states` *before* each decoder layer,
so `hidden_states[i]` is the input to layer i and there are `num_layers` stream-valued entries plus a
final collapsed+normed `[T, H]`. The engine records at the top of each layer iteration for the same
reason, and its extra post-last-layer state has no reference counterpart (the logits gate covers the
collapsed head).

**A bf16 fixture cannot carry this claim.** Re-loading the bf16-roundtripped weights perturbs the
layer-0 state by **1.13e-04** — 100× above the bound — so the fixture stores **fp32** by default now
(`--weights-dtype bfloat16` is still available for logits-level work), and the comparator *warns* when
the stored dtype and the tolerance are inconsistent. The 1e-6 bound is a statement about the
implementation only with fp32 weights.

## 2. The same instrument localizes the missing machinery

Same run on the **compressed** fixture (fp32, window 4, 64 tokens, layer types
`[sliding, CSA, HCA, sliding]`, 16 CSA entries):

| state (input to layer) | after which layer | max&#124;Δ&#124; | verdict |
|---|---|---|---|
| 0 | — (init) | **0.000e+00** | bit-exact |
| 1 | 0 (`sliding_attention`) | **7.451e-09** | matches |
| 2 | 1 (**`compressed_sparse_attention`**) | **1.171e-02** | **first divergence** |
| 3 | 2 (`heavily_compressed_attention`) | 1.798e-02 | error propagates |

The engines agrees exactly everywhere the machinery it has is exercised, and diverges at **exactly
the layer that carries the compressor + indexer** — 6 orders of magnitude above the bound, not a
marginal miss. That is the bisect instrument P1.1 needs: implement the compressor and the gate flips
state 2 (then the logits gate, 16/20 → ≥18/20) instead of leaving a "logits differ" mystery.

(For completeness, the logits gate on this fixture: engine top1 534 vs ref 685, overlap 16/20 → FAIL,
exit 1. The `csa64`/`hca160` fixtures remain expected-fail by design and stay out of `run_all.sh`.)

## 3. Status

* **P0.1 — DONE** (per-layer ≤1e-8 on the modules that exist; instrument committed).
* **P1.1 — gate + bisect instrument ready** (state 2 of the compressed fixture is the acceptance
  target; `hidden_states.npz` gives the per-layer reference).
* **P0.2 — next** (read DeepSeek's `inference/engram.py` + `model.py` for the V4.1-only maths, which
  transformers cannot provide: no `DeepseekV41` class exists).
* **P0.3, P0.4 — not started.**
