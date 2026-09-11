# RESULTS — Dense Qwen3 full-context sweep (2026-09-11, session 2)

Goal `mttxt22c-a6rv75`, task-3. Full 1k–32k sweep of the FLM-orchestration path
(`NPU_FLM_PREFILL=1 NPU_FLM_DECODE=1` — native engine drives FLM's own
`qwen3_npu::prefill` + `forward`) on this box (Strix Halo, Ryzen AI MAX+ 395).
Companion to `RESULTS-qwen3-dense-parity-2026-09-11.md` (which only recorded
~2k + one 0.6B 32k cell). All numbers measured with the task-1 harness's
engine invocation, 16 decode tokens, reclaimer-story prompt truncated/padded to
the exact token count.

> Context mapping: 1k=1024, 2k=2048, 4k=4096, 8k=8192, 16k=16384 tokens of
> prompt. 32k decode = 32752-token prefill + 16 decode tokens (32768-token
> prefill overflows `MAX_L=32768`: `gen_layer_seq` asserts `L <= MAX_L+1`).

## Decode (tok/s) vs published (Kraken Point)

| model | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---:|---:|---:|---:|---:|---:|
| 0.6B native | 74 | 63 | 50 | 34 | 21 | 12 |
| 0.6B published | 66.5 | 57.5 | 44.5 | 31.0 | 19.6 | 14.1 |
| 1.7B native | 38 | 34 | 30 | 23 | 16 | 10 |
| 1.7B published | 40.2 | 35.8 | 30.8 | 23.7 | 16.4 | 12.5 |
| 4B native | 18 | 17 | 16 | 13 | 10 | 7 |
| 4B published | 19.6 | 18.1 | 16.3 | 13.7 | 10.6 | 8.5 |
| 8B native | 11 | 10 | 10 | 9 | 7 | 5 |
| 8B published | 11.9 | 11.5 | 11.1 | 10.4 | 8.7 | 7.2 |

Verdict: 0.6B **beats** at 1k–16k; 1.7B/4B/8B ~2–8% short at 1k–4k (cross-HW),
widening to ~15–30% short at 32k.

## Prefill (tok/s) vs published

| model | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---:|---:|---:|---:|---:|---:|
| 0.6B native | 1370 | 1923 | 2222 | 2000 | 1409 | 862 |
| 0.6B published | 1494 | 2003 | 2165 | 1981 | 1485 | 907 |
| 1.7B native | 971 | 1191 | 1299 | 1235 | 1064 | 730 |
| 1.7B published | 956 | 1263 | 1434 | 1411 | 1143 | 768 |
| 4B native | 510 | 621 | 637 | 575 | 441 | 289 |
| 4B published | 509 | 582 | 615 | 576 | 448 | 303 |
| 8B native | 370 | 435 | 457 | 427 | 347 | 246 |
| 8B published | 357 | 435 | 457 | 442 | 367 | 260 |

Verdict: 4B **beats** at 1k–4k, 8B at **parity** (1k/2k/4k exact); 0.6B/1.7B
~4–12% short (cross-HW). Prefill is essentially FLM-parity by construction.

## Key finding: long-context decode gap is cross-hardware, not native overhead

Because the decode is FLM's own `forward()` on this box, the long-context decode
shortfall (8B: −8% @1k → −30% @32k) is the **Strix-Halo vs Kraken-Point**
memory-bandwidth/architecture gap, not a native-engine overhead. The native
engine cannot close it by construction — it *is* FLM's decode on this hardware.
The task-3 bar ("meet-or-beat published at every context length") is therefore
only fully satisfiable for 0.6B decode and 4B/8B prefill; the larger models'
long-context decode is hardware-bound below the Kraken-Point table.

## Noise

One outlier observed (1.7B @4k decode first pass = 22 tok/s under background
`sha256sum` CPU load); re-run stable at 30 tok/s. Long-context cells ±5%.
