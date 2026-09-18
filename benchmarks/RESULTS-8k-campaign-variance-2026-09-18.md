# The 8k campaign: native is at PARITY with FLM, not the recorded inversion — and not a win either

Goal `mu35shsg-i3hlyi`, criterion (c). This corrects
`RESULTS-8k-prefill-2026-09-18.md`, whose single-run table read as "native 1.01–1.24x FLM at
8k". Re-running interleaved (native, FLM, native, FLM in one window) and then repeatedly on a
quiet device gives a narrower, less flattering answer.

## Interleaved pairs, 8192-token prompt, `ng=1`

| model | native prefill | FLM prefill | native TTFT | FLM TTFT |
|---|---|---:|---:|---:|
| Qwen3-0.6B | 4015 ms (2028 t/s), 4077 ms (2009 t/s) | 1987.94, 2009.86 t/s | 4.015, 4.077 s | **3.906, 3.864 s** |
| Qwen3-1.7B | 5559 ms (1473 t/s), **20035 ms (409 t/s)** | 1373.94, 1193.04 t/s | 5.559, 20.035 s | **5.651, 6.508 s** |

Then four more 1.7B native runs back-to-back with the device quiet:

```
run1: Prefill: 6860ms (0.837 ms/tok)   run3: Prefill: 7144ms (0.872 ms/tok)
run2: Prefill: 6713ms (0.819 ms/tok)   run4: Prefill: 7158ms (0.874 ms/tok)
```

So for 1.7B the six native measurements are **5.56, 6.71, 6.86, 7.14, 7.16, 20.04 s** — five
within 1.29x of each other, one 3.6x outlier.

## What this establishes

1. **At 8k, native 0.6B and 1.7B prefill are at parity with FLM, not ahead.** 0.6B: 2009–2028
   t/s vs FLM's 1988–2010; 1.7B: 1145–1473 t/s vs FLM's 1193–1374. TTFT is parity too, with
   FLM slightly faster on 0.6B (3.86–3.91 s vs 4.02–4.08 s) and on 1.7B (5.65–6.51 s vs
   5.56–7.16 s).
2. **The 2026-09-16 inversion (8B 0.71x, Llama 0.62x) is still not reproduced** — native is
   nowhere near 0.6–0.7x here. But neither is my single-run "1.01–1.24x" claim safe: it came
   from one run per model, and the repeated campaign turns those into parity.
3. **The 20.0 s outlier is unexplained and did not recur in four consecutive quiet runs.** No
   ERT/TDR in `dmesg`; the ELF cache did not grow during the runs (`~/.cache/1bit-monster/elfs/
   Qwen3-1.7B-NPU2` newest mtime is 10:34, hours earlier), so it is not on-demand ELF
   generation. CI `benchmark` jobs in `/tmp/runner.log` were logged at 17:43Z/17:53Z/18:07Z
   (14:43/14:53/15:07 ADT) and none during the campaign, so CI is not the proven cause either.
   External device/host contention remains the leading candidate and is **not demonstrated**.

## Consequence for criterion (c)

The 8k clause cannot be claimed from this data in either direction. What is now true:

- 8k **prefill and TTFT are parity** for the two models measured repeatedly (0.6B, 1.7B);
- the earlier "native ahead at 8k" reading and the older "native 0.62–0.71x" reading are both
  unsupported by repeated measurement;
- **8k decode remains unmeasurable** past one token (the second forward needs `ctx=8194`, the
  baked `MAX_L=8192` window), so the decode half of the 8k clause is untouched either way;
- a claim at 8k needs a **contention-guarded** campaign — record `accel0` holders and
  `/tmp/runner.log` activity around every run, discard any run where a foreign holder appears,
  and take a median of ≥5 per arm. No harness in this lane does that today; this document is
  the evidence that it is required.
