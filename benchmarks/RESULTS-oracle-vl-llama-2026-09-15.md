# Oracle accuracy for Qwen3-VL-4B and Llama-3.1-8B — the two models with no prior tally

Goal `mu35shsg-i3hlyi`, task-acc-vl-llama. Date 2026-09-15 ~22:55 ADT.
Box: strixhalo. Working tree: `~/1bit-MONSTER-goal` (goal/runlist-decode-wire),
harness added as `benchmarks/oracle_accuracy_model.sh`.

The correctness lane (goal `mu34scbf-4ffm1o`) closed the scoreboard for
Qwen3-0.6B/1.7B/4B/8B. **Qwen3-VL-4B and Llama-3.1-8B appeared nowhere in its
records.** This is the first per-model tally for them, on the same 20-prompt
oracle set the 0.6B scoreboard uses
(`benchmarks/prompts/qwen3_0_6b_oracle_set.txt`).

## Method (same rules as `oracle_accuracy_0_6b.sh`)

- The model's **own chat template** is applied for the native arm, so native and
  FLM are asked the same question (FLM applies the same template internally).
- Native decode is greedy (`NPU_GREEDY=1`); each invocation is one process.
- Scoring is whole-output, case-insensitive substring match; every extraction is
  asserted non-empty before it is scored.
- **I1 note.** The engine's `tokenizer/tokenize` tool is *not* used. Measured on
  this box, it prints Qwen specials (`BOS=151643 EOS=151645`) for the Llama
  vocab, so native ids are built with HF `tokenizers` 0.23.1 /
  `transformers` 5.16.1 from each model's own `tokenizer.json`.

Harness:

```
bash benchmarks/oracle_accuracy_model.sh <Model-NPU2> <engine> <flm_tag> <ntok> <set> <out>
```

## Result: Qwen3-VL-4B — native 20/20, FLM 20/20

```
cd ~/1bit-MONSTER-goal
OUT=/tmp/oracle_acc_VL4B.tsv ENGINE_ENV="NPU_RUNLIST=1" \
  bash benchmarks/oracle_accuracy_model.sh \
    Qwen3-VL-4B-Instruct-NPU2 npu_engine_qwen3_vl_4b qwen3vl-it:4b 128 \
    benchmarks/prompts/qwen3_0_6b_oracle_set.txt /tmp/oracle_acc_VL4B.tsv
```

```
prompts scored        : 20  (extraction errors: 0)
FLM oracle self-check : 20/20
native arm (npu_engine_qwen3_vl_4b) : 20/20
```

Every prompt is `flm=Y native=Y`. The two arms are near-verbatim on this set,
which is also the strongest available evidence that the template matches: France
is native `"The capital of France is **Paris**.  Paris is not only the political
center of France (home to the Élysée Palace and the National Assembly) but also a
global cultural, historical, and economic hub."`, FLM `"… **Paris**.  Paris is
not only the political center of the country — housing the Élysée Palace…"`.

VL-4B is `NV=151936, NC=36, H=2560`, so it satisfies `dense_qwen3` in
`npu_engine_universal.cpp` and the runlist path is eligible without any extra
env. No Llama-style problem applies.

## Llama-3.1-8B — accuracy blocked on the runlist, and a shared ERT signature

`npu_engine_llama` on `Llama-3.1-8B-NPU2` does **not** take the runlist path by
default. In `engine/npu/src/npu_engine_universal.cpp`:

```
const bool dense_qwen3 = cfg.NV == 151936 && !cfg.has_moe && (...);
const bool runlist_eligible = dense_qwen3 ||
    (!cfg.has_moe && elf_env && elf_env[0]);
```

Llama is `NV=128256`, so `dense_qwen3` is false and a plain invocation falls
through to the 112-launch split path. Measured, 40-token templated prompt:

```
=== Prefill 40 [fallback] ===
Prefill: 16655ms (416 ms/tok)
  [0] boot=791 (24ms)
  [1] batch=1 toks: 6864  17190ms (17190 ms/tok)
  [2] batch=1 toks: 315   17045ms (17045 ms/tok)
```

**The split path is correct but ~16.7 s/token.** Its decoded ids are
`6864 315 9822 374 12366` → `" capital of France is Paris"` (boot 791 = `"The"`).
So accuracy *could* be scored through it — at ~95 min for 20×16 tokens — but the
fast path is the point of the goal.

Supplying a freshly generated Llama per-ctx ELF set makes the model *eligible*,
and then it fails at the first layer:

```
cd ~/1bit-MONSTER-goal
bash benchmarks/gen-layer-elfs.sh ~/.config/flm/models/Llama-3.1-8B-NPU2 \
     /tmp/llama-elfs 512 llama            # 512 ELFs, 4.3 s, MAX_L=8192

NPU_LAYER_ELF_DIR=/tmp/llama-elfs NPU_RUNLIST=1 NPU_GREEDY=1 \
  engine/npu/build/npu_engine_llama ~/.config/flm/models/Llama-3.1-8B-NPU2/model.q4nx \
  16 /tmp/ll_ids.txt
```

```
RuntimeLayer: layer kernel ctx=1 ready
RuntimeLayer: lm_head kernel ready (/tmp/llama-elfs/elf_0002_lmhead.bin)
RuntimeLayer: init OK (32 layers)
=== Prefill 40 [runlist] ===
RuntimeLayer: runlist wait FAILED: runlist failed execution (ERT_CMD_STATE_TIMEOUT)
txn_op_idx = 0xFFFFFFFF   ctx_pc = 0x28B06005
[runlist] prefill forward ctx=1 failed
[runlist] whole-layer path failed (rc=1); falling back to split path
```

Raw log: `/tmp/ll_rl.raw` on strixhalo.

**The generator is not the cause.** `gen-layer-elfs.sh Qwen3-4B-NPU2 …/tmp/q4b-chk 3 qwen3`
reproduces the shipped ELFs byte-for-byte:

```
ctx1 shipped=40c1ea5ad4e14931 new=40c1ea5ad4e14931 SAME
ctx2 shipped=1de57a85a3effa54 new=1de57a85a3effa54 SAME
ctx3 shipped=688ffcaf76f5fc2b new=688ffcaf76f5fc2b SAME
```

so the failure is Llama-specific weight/BO binding at runlist execute time, not
ELF generation. This is the **same signature the MoE lane is chasing for
Qwen3.6-35B-A3B** (`goal mtusoiy1-cfdhqr`, commits `535f84a14`, `d63271894`,
`829c74553`, `617401a75` on this branch: "v1.0.5 35B forward dies with
ERT_CMD_STATE_TIMEOUT", "no region-A/B BO bound before the ERT timeout"). A dense
non-Qwen3 model with the identical first-execute timeout is a second, smaller
reproducer for that investigation.

## Status

| arm | tally | note |
|---|---|---|
| Qwen3-VL-4B native runlist | **20/20** | FLM oracle 20/20; harness command above |
| Qwen3-VL-4B FLM | 20/20 | same set, same process run |
| Llama-3.1-8B native runlist | **blocked** | first-layer ERT_CMD_STATE_TIMEOUT |
| Llama-3.1-8B native split | correct, 16.7 s/tok | ids verified vs the France prompt; tally pending a serial window |
| Llama-3.1-8B FLM | not yet run for the full set | `flm run llama3.1:8b` works (France → "Paris.") |

Remaining for this task: the Llama per-model tally — either through the split
path in one ~75–95 min serial window, or through the runlist once the shared
ERT timeout is fixed.

## Addendum (2026-09-16 03:17Z): the Llama runlist ERT was environmental — the tally is 20/20

After the ERT root-cause and repair (commit `e719ed64d`; see
`RESULTS-ert-rootcause-and-repair-2026-09-16.md`): `ERT_CMD_STATE_TIMEOUT` is the
amdxdna driver TDR `timeout_in_sec=2`, raised to 15. It was **not** a packing/ELF
defect — with the device to itself the Llama-3.1-8B runlist completes at
**82.9 ms/tok (12 tok/s)**, matching the documented `gen-layer-elfs` figure, and
the full tally ran in ~6 minutes:

```
OUT=/tmp/oracle_acc_Llama31_8B_runlist.tsv \
ENGINE_ENV="NPU_LAYER_ELF_DIR=/tmp/llama-elfs NPU_RUNLIST=1" \
  bash benchmarks/oracle_accuracy_model.sh Llama-3.1-8B-NPU2 npu_engine_llama \
    llama3.1:8b 16 benchmarks/prompts/qwen3_0_6b_oracle_set.txt \
    /tmp/oracle_acc_Llama31_8B_runlist.tsv
```

```
prompts scored        : 20  (extraction errors: 0)
FLM oracle self-check : 20/20
native arm (npu_engine_llama, runlist) : 20/20
```

So the "Llama blocked on ERT" rows above are superseded. The split-path fallback
(16.7 s/token, correct but slow) is no longer needed. The remaining unexplained
first-layer ERT is scoped in the repair doc: reproducible neither alone nor in my
controlled 2–3 hwctx A/B, while `@agent-baaa57`'s 2-engine ERT vs quiet exit-0
through ctx 2096 is the strongest contention A/B.

### Final status for this task

| model | native arm | FLM oracle | set |
|---|---|---|---|
| Qwen3-VL-4B | **20/20** (runlist) | 20/20 | 20-prompt oracle set |
| Llama-3.1-8B | **20/20** (runlist) | 20/20 | same set |

Both are one process invocation per row, greedy, the model's own chat template,
HF tokenize/detokenize. Tallies: `RESULTS-oracle-vl-llama-tally.tsv` (VL) and
`RESULTS-oracle-vl-llama-tally-runlist.tsv` (Llama).
