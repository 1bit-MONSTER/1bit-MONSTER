# Two FLM engines vs two native engines, and why neither gets 2x — 2026-09-18

Follow-on probe. Question: load **two FastFlowLM servers** and see how they aggregate — the
reference point for this lane's own two-instance numbers — and find out what bounds concurrency on
this NPU at the hardware level.

## Method

- FLM: `flm serve qwen3:0.6b --port 8099` and `flm serve qwen3:1.7b --port 8100` (the production
  MoE server on :8098 untouched), 1058-token text prompt, 64 decode tokens, `temperature=0`.
  **Two warm-up requests per server are discarded** — see the cold-start trap below. Rates are the
  server's own `usage.decoding_speed_tps`; the serial and concurrent phases are 2 reps each.
- Native: the numbers from `RESULTS-npu-concurrency-config-sweep-2026-09-18.md` (1024-token prompt,
  128 decode tokens, rates from the engine's own `ms/tok`).
- Box quiet for both (load < 2), no foreign `flm bench`/`c8k_guarded` process running.

## Result: FLM loses half of each instance when a second server is loaded

| phase | qwen3:0.6b | qwen3:1.7b | aggregate |
|---|---:|---:|---:|
| serial 0.6b (2 reps) | 76.34, 73.73 → **75.0 tok/s** | — | — |
| serial 1.7b (2 reps) | — | 38.02, 38.07 → **38.05** | — |
| **concurrent (2 pairs)** | 21.3, 21.2 | 21.3, 21.3 | **42.6 tok/s** |

- Sharing efficiency: 42.6 / (75.0 + 38.05) = **38%**.
- Gain over running just the bigger model: **1.12x**.
- Each instance keeps only 28% (0.6b) and 56% (1.7b) of its solo rate.

For comparison, the same experiment with **two native engines** on the two models
(`RESULTS-npu-concurrency-config-sweep-2026-09-18.md`, 3 reps + repeats):

| engine | serial 0.6B | serial 1.7B | concurrent aggregate | efficiency | gain vs bigger alone |
|---|---:|---:|---:|---:|---:|
| FastFlowLM (2 servers) | 75.0 | 38.05 | **42.6** | **38%** | **1.12x** |
| native (2 processes) | 67 (61–77) | 39 | **95** (91–113) | **82%** | **2.44x** |

**The native engine's two-instance efficiency is better than twice FLM's, and its aggregate is
2.2x FLM's — with the larger model keeping its full solo rate, which FLM's does not.** Per instance
and single-stream the two are at parity (75 vs 67 tok/s at ~1k context, consistent with the audited
decode ratio of 1.00–1.07x); the difference is entirely in how they behave when a second model is
loaded.

## Why neither gets 2x: the kernels ask for the whole device

`xclbinutil --dump-section AIE_PARTITION:JSON` on FLM's own shipped xclbins
(`/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/{layer,mm,attn}.xclbin`):

```json
"aie_partition": { "partition": { "column_width": "8", "start_columns": ["0"] },
                   "operations_per_cycle": "2048" }
```

**`column_width: 8` — every kernel in the stack requests all 8 NPU columns.** A hardware context
built from these xclbins therefore occupies the whole device, so two loaded models cannot run
*spatially* in parallel; they time-share by preemption. That is the mechanical reason both engines
plateau instead of doubling, and it explains the context data measured alongside it:

```
hwctx_limit   = 16    [Debug] Maximum number of hwctx (driver parameter)
context_limit = 64
xrt-smi examine -r platform → Total Columns: 8 ; aie-partitions → 2 partitions
```

Live contexts: production MoE server **9**, each FLM dense server **4**, each native engine **8**
(demand-driven: 4 on a short run, 8 on a long one). We ran **25 and 37 live contexts with zero
context-creation failures**, and aggregate throughput stayed flat at 68–70 tok/s for 2, 3 and 4
native instances — more contexts redistribute device time, they do not add throughput.

**The unlock, if it is wanted:** two engines can only be truly parallel if the kernels are built for
a *narrower* partition — e.g. two 4-column builds (`column_width: 4`, `start_columns: [0]` and
`[4]`), which the driver's `aie2_max_col` / `start_col_index` parameters exist to expose. That is a
kernel-rebuild + A/B project (the generator is in this tree and the toolchain is installed), not a
configuration change. On the evidence above it is the only route to ~2x; everything available by
configuration has been swept.

## The cold-start trap (why FLM server numbers must be warmed)

The same server, same request, first vs later:

| request | TTFT | prefill t/s | decode t/s |
|---|---:|---:|---:|
| 1st after load (0.6b, 536 tok) | 1.857 s | 288.6 | 41.67 |
| 2nd | 0.757 s | 708 | 83.15 |
| 3rd | 0.679 s | 790 | 83.68 |

The first inference after `flm serve` starts is **~2x slower** (decode 41.7 → 83.2 tok/s,
prefill 289 → 790 t/s). Any FLM server measurement that skips a warm-up understates it by half —
and an A/B that warms the native arm (cold process but excludes startup from its reported rate) and
not the FLM arm would be wrong in FLM's favour. The dual-engine run above discards two warm-ups per
server for exactly this reason. (At 1058 tokens the warm numbers are 1192–1257 t/s prefill /
75 tok/s decode for 0.6b, and 814–834 t/s / 38.05 tok/s for 1.7b.)

## Reproduce

```bash
python3 /tmp/flm_client.py <port> <model_tag> "$(cat /tmp/flm_prompt1k.txt)" 64   # one timed request
/opt/fastflowlm/bin/flm serve qwen3:0.6b --port 8099 &                            # server A
/opt/fastflowlm/bin/flm serve qwen3:1.7b --port 8100 &                            # server B
# then run both clients at once for the concurrent pair; discard two warm-ups per server first
xclbinutil --dump-section AIE_PARTITION:JSON:/tmp/ap.json --input \
    /opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2/layer.xclbin
modinfo -p amdxdna          # parameter descriptions, incl. hwctx_limit / aie2_max_col / start_col_index
```
