# Concurrent engine slots: every slot's tokens are byte-identical to its serial reference — 2026-09-18

Step 5 of goal `mu7nwtjv`. The ceiling run proved the slots hold contexts; this proves the slots are
**correct** — each concurrent instance's greedy token stream is compared byte-for-byte with a serial
run of the same binary, model, prompt and ng. No "it started" claims, no averages over runs that were
not compared.

## Method

`benchmarks/engine_slot_identity_bench.sh`. The serial reference is run **twice** first; if the two
references differ the script refuses to use them as ground truth (they did not differ in any run
below). Concurrent slots run with `NPU_NO_DEVICE_LOCK=1` (the engine otherwise flocks the device and
serialises). `LAUNCH=stagger` starts each slot only after the previous one is serving — the real
"agents come online over time" slot pattern; `LAUNCH=simul` starts them all at once. Tokens are read
from the engine's own `  [n] <id>` lines. Raw outputs: `evidence-engine-slots/eng_identity_*.out`.

## Results

Serial reference determinism: **identical** across two serial runs in every configuration.

| config | identical | per-slot decode | aggregate | per-slot prefill |
|---|---|---|---|---|
| **N=8 stagger**, 1024-tok prompt, ng=64 | **8 / 8** | 71–74 tok/s | 580 tok/s* | 539–604 ms |
| **N=4 simult**, 1024-tok prompt, ng=64 | **4 / 4** | 16–21 tok/s | 72 tok/s | 5140–6010 ms |
| **N=8 simult**, tiny prompt, ng=200 | **8 / 8** | ~11 tok/s | 88 tok/s | 174–695 ms |
| serial 0.6B, 1024-tok prompt, ng=64 | — | 59–61 tok/s | — | 554–616 ms |
| serial 0.6B, tiny prompt, ng=200 | — | 98 tok/s | — | 95 ms |

\* The 580 tok/s in the staggered row is the sum of the eight per-slot rates, and with ng=64 the
slots only partly overlap (each finishes ~1.4 s after it starts, the next begins ~0.6 s later), so it
is **not** an eight-way simultaneous aggregate. The two `simul` rows are the honest concurrent
aggregates; they agree with the earlier independent result that the device shares rather than
serialises (`RESULTS-dual-engine-concurrency-2026-09-18.md`: 2×0.6B = 68 tok/s).

**No device error in any run** — no `ERT_CMD_STATE_TIMEOUT`, no `TDR`, no `timed out`, no
`CREATE_HWCTX` in the runs that were supposed to serve.

## What is validated, and what is not

* **Validated:** with staggered joins, 8 engine slots hold 8 contexts and every slot's 64-token
  stream is byte-identical to the serial run — correctness survives the slot layout.
* **Validated:** 4 simultaneous slots at the 1024-token prompt, and 8 simultaneous slots at a tiny
  prompt, are likewise 4/4 and 8/8 identical.
* **Not yet validated:** simultaneous launch of many slots at a long prompt. That case is where the
  bf16 prefill-init race lives (`RESULTS-engine-slot-ceiling-2026-09-18.md`), so the honest statement
  is that token identity is proven for the configurations above, and the long-prompt all-at-once case
  is blocked on that race, not on identity.

## Reproduce

```bash
# on strixhalo, from the repo root:
N=8 P=/tmp/p_1k.txt    NG=64  LAUNCH=stagger bash benchmarks/engine_slot_identity_bench.sh
N=4 P=/tmp/p_1k.txt    NG=64  LAUNCH=simul   bash benchmarks/engine_slot_identity_bench.sh
N=8 P=/tmp/tiny_ids.txt NG=200 LAUNCH=simul  bash benchmarks/engine_slot_identity_bench.sh
```
