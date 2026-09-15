# Qwen3-0.6B dense decode: the wiring goal's target is already exceeded (2026-09-15)

Goal `mtuhp2fy-c8yfgb` asks to lift native Qwen3-0.6B dense decode *from 2 tok/s*
toward FLM-class, **target ~4–8 tok/s**, by wiring the pre-built small-M and
cascade-fused xclbins. Measured directly today:

```
engine/npu/build/npu_engine_qwen3_0_6b \
  ~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx 8 /tmp/ids.txt

baseline (M=128 path)         Prefill: 326ms (65.112 ms/tok) [GEMM 25ms, attn 134ms, conv+other 316ms]
                              === 11.4 ms/tok (88 tok/s) | tokens=8 ===
NPU_FUSED_USE=1               Prefill: 342ms (68.311 ms/tok) [GEMM 25ms, attn 137ms, conv+other 331ms]
                              === 11.3 ms/tok (88 tok/s) | tokens=8 ===
```

**88 tok/s against a target of 4–8 tok/s** — over 20× the bar, and the goal's own
`task-r1` had already recorded 79 tok/s @1k / 91 @256. FLM's published 0.6B decode
is 66.5 @1k, and `RESULTS-native-vs-flm-dense-qwen3-2k-2026-09-14.md` records
native ahead of it (+19.4 %). So the goal's success criterion ("≥ 4 tok/s") has
been satisfied for some time; the "2 tok/s" premise is stale.

## The prescribed mechanism is not the lever, and the code says so

Both prescribed artifacts exist and are wired; neither is what moved decode.

- **Small-M (`_m1/_m8/_m32`) — deliberately OFF**, with the reason in the source
  (`npu_engine_universal.cpp:1474-1477`):

  ```cpp
  int sm = 0;   // default OFF: the _m1 kernel's weight contract differs from the
                // M=128 path (garbage decode, no perf win — launch-bound)
  const char* e = getenv("NPU_SMALL_M"); if (e && *e) sm = atoi(e);
  if (sm != 1 && sm != 8 && sm != 32) sm = 0;
  ```

  So a prior pass on this same goal found the `_m1` kernel's **weight contract
  differs** from the M=128 path (garbage decode) *and* that decode is
  **launch-bound**, so a smaller-M GEMM cannot help. That is why the default is 0
  and all seven xclbins (`final_i8_{GU,D}_qwen3_0_6b_{m1,m8,m32}`) sit unused.

- **Cascade-fused (`NPU_FUSED_USE=1`) — wired and embedded, but ~neutral**:
  11.3 vs 11.4 ms/tok is noise. Both naming variants are present
  (`final_cascade_fused.xclbin` 64938 B / `insts_cascade_fused.txt` 210100 B, the
  names `npu_embedded.h` embeds, plus the model-tagged `…_qwen3_0_6b` pair).

## Where the prefill cost actually is

The profile line is informative for anyone still chasing 0.6B: prefill is
326 ms with **attn 134 ms** — attention is ~41 % of prefill, GEMM only 25 ms. So
the remaining 0.6B headroom is in attention, not in the FFN/cascade the goal
points at; and decode at 88 tok/s is not the constraint.

## Recommendation

Close `mtuhp2fy-c8yfgb` as **achieved by a different means than prescribed**:
- "≥ 4 tok/s" ✅ (88 measured; 79.4 @1k recorded, ahead of FLM's published 66.5)
- token parity + corr ✅ (recorded in `RESULTS-native-vs-flm-dense-qwen3-2k-2026-09-14.md`)
- "small-M and fused xclbins actually wired" ✅ (both are in the code paths;
  small-M is intentionally disabled with a documented falsification)
- launch-count 112 → ~28 ❌ **not** demonstrated, and now not needed for the speed
  target — the fused path is measured at ~neutral here, so this criterion should be
  re-scoped or dropped rather than chased.
- `task-4` (extend to 1.7B/4B/8B + record) is satisfiable from the existing
  `RESULTS-native-vs-flm-dense-qwen3-2k-2026-09-14.md`, which already carries
  decode @1k/@2k for 0.6B/1.7B/4B/8B.
