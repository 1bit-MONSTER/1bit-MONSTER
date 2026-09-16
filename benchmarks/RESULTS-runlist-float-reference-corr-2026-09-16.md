# RESULTS — the runlist arm against a CPU/float reference (the objective's corr gate)

Date: 2026-09-16. Goal `mtuhp2fy-c8yfgb`. This measures the gate the completion
auditor required and that answer-level accuracy had been (wrongly) substituted for:
**per-layer logits corr >= 0.998 vs a CPU/float reference, plus token parity.**

## Method

1. Real float reference: `hf download Qwen/Qwen3-0.6B --local-dir /tmp/qwen3-hf-0_6b`
   (float32 safetensors, 1.5 GB).
2. Same input on both sides: the Qwen chat-templated token ids for
   `The capital of France is` -> `151644 872 198 785 220 65063 220 1055 220 49000 220 285 151645 198 151644 77091 198`
   (specials emitted separately; see the `=`/`<|im_end|>` note in
   `RESULTS-runlist-true-native-dense-qwen3-2026-09-16.md`).
3. Native logits: `NPU_RUNLIST=1 NPU_DUMP_LOGITS=1 npu_engine_qwen3_0_6b ... 2 <ids>`
   -> `/tmp/runlist_logits.txt` (151936 `idx value` lines), written at the priming
   step, i.e. the last prompt position.
4. Float logits: `AutoModelForCausalLM.from_pretrained(..., dtype=torch.float32)`
   on CPU, logits at the same final position.
5. Pearson correlation over the logit vector; greedy tokens iterated on both sides.

## Results

| quantity | value |
|---|---:|
| **full-vocab Pearson corr (native vs CPU/float)** | **0.919048** |
| corr over top-10 / top-50 / top-100 / top-1000 | 0.939 / 0.832 / 0.733 / 0.539 |
| greedy argmax at step 1 | native `151667`, float `151667` — **SAME** |
| greedy token parity over 32 steps | **exact for the first 16**, first divergence at index 16 |

Full-VOCAB CORR: **0.919048**, against the gate's `>= 0.998`. **The corr clause is
NOT met.** The gap is not a tail artifact (the top-k bands are *lower*, not higher),
and it is not a vocabulary-mapping error (both sides rank `151667` and `151644` at
the top). It is error from the int8 weights + bf16 activations.

The greedy *token* still matches for 16 steps because only the argmax must agree;
the distribution difference is what eventually parts the two sequences at step 16.

## Consequence for the objective

- **Met**: decode >= 4 tok/s (67/37/18/11); launches 112 -> 1 `xrt::runlist`
  submit/token; Zaya1-8B no regression (0.998469 fused, 0.999342 non-fused);
  answer-level accuracy 20/20 (0.6B, 1.7B) and 19/20 (4B, 8B).
- **Not met**: `corr >= 0.998` vs a CPU/float reference (0.919 here). Token parity
  holds only for 16 of 32 greedy steps.

Closing the corr gap is a precision change to the quantized path (weights are int8,
activations bf16) — excluded by this goal's boundaries ("authoring new AIE kernels",
small-M/cascade xclbin work), so it is recorded as a **known limitation** rather
than attempted here.

Scope of this measurement: measured for Qwen3-0.6B only. The 1.7B/4B/8B reference
comparison needs those HF checkpoints downloaded; the 0.6B figure establishes that
the gate as written fails on this arm.
