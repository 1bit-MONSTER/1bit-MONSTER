# Eight models on the NPU, and what hwctx counts really mean — 2026-09-18

Two questions answered with measurements: can the box host 8 small models as a multi-agent fleet,
and was the >16 hwctx observation an overclock?

## 1. Eight models loaded: yes — but the array does not multiply

Measured with **8 × `flm serve qwen3:0.6b`** resident (ports 8099–8106) plus the production MoE
server, 1058-token prompt, 64 decode tokens:

```
NPU contexts while all 8 are resident: 37      (hwctx_limit = 16)
servers ready: 8/8 ; every process ~0.5-1 GB RSS
solo (one server, peers idle)              76.3 tok/s
one request each to :8099/:8100/:8101      76.2 / 71.6 / 71.7 tok/s   <- idle peers are ~free
2-way concurrent   28.5 + 28.5  =  57 tok/s   (0.75x solo)  per-request wall 4.25 s
4-way concurrent   14.1 x4      =  56 tok/s   (0.74x solo)  per-request wall  8.48 s
8-way concurrent    7.1 x8      =  57 tok/s   (0.75x solo)  per-request wall 16.91 s
```

**Aggregate is flat at ~57 tok/s from 2 to 8 concurrent requests, and it is *below* the solo rate
of 76.** Each extra concurrent request divides the same device time; latency scales linearly with
the number of concurrent requests.

So for multi-agent orchestration on this box:

* **Resident ≠ parallel.** Eight loaded models are eight *addressable* agents but one compute
  resource. The honest capacity is **~57 tok/s total while more than one is working**, and **~76
  tok/s if requests are serialised through one model**.
* **Idle agents are cheap** (76.2/71.6/71.7 tok/s with seven idle peers), so a fleet where only one
  agent thinks at a time is fine. A fleet where several think at once pays ~25% aggregate for the
  privilege and accepts linear latency growth.
* The only measured way to get *more* than one array's worth is **disjoint column partitions with
  sliced kernels**: 2.06x for two contexts on 4-column halves
  (`RESULTS-4col-spatial-partitioning-2026-09-18.md`), i.e. two real agents at full speed, not
  eight.

## 2. The 37 hwctx reading was not an overclock

* `hwctx_limit = 16` is documented by the kernel itself as **`[Debug] Maximum number of hwctx`** and
  it is **per partition**; `xrt-smi examine -r aie-partitions` reports **2 partitions**, so the
  nominal accounting is 2 × 16 = 32 — and we have measured 25 and 37 live contexts with zero
  creation failures. The count is real; the *number* comes from partition accounting, not from
  pushing the silicon.
* **No clock or power setting was touched at any point.** `xrt-smi examine -r platform` reports
  `Power Mode: Default`, unchanged all session; `aie2_max_col`, `start_col_index` and
  `time_quantum_ms` are all empty (toolchain/driver defaults). FLM's CLI does default to
  `--pmode performance`, which is a **power-state request through the driver**, not an overclock —
  and the device still reports Default.
* Contexts are a **scheduling** resource: they are preempted, and the flat ~57 tok/s aggregate is
  the proof that more contexts add no compute. The knob that *would* change behaviour under
  concurrency is `time_quantum_ms` (module parameter → needs an amdxdna reload, i.e. a maintenance
  window), not the context count.
* Caveat found while counting: `aie-partitions` per-PID attribution **garbles past ~35 live
  entries** (nonsense PIDs like 1388285088 appear). The total is usable; the per-process breakdown
  is not at that depth.

## 3. The branch decision — executed, and it surfaced a real difference

Done as decided: **`npu/col-slice` cut fresh from `origin/main`** (5ffbaf43f) instead of merging
331 upstream commits into the audited branch — that merge had **47 conflicts** and would have
produced a hybrid history nobody could audit. The audited `goal/runlist-decode-wire` is untouched.

On the new branch (`0b4e6f4f0`, pushed, builds clean rc=0):

* ported upstream's knob to the two generators our path uses — `n1_core_bf16_v1.py`,
  `n1_core_attn.py`: `-x/--col-offset` → `tile(col_off + col, row)`;
* re-applied the 512-bit engine TU (`-mavx512f … -mavx512vbmi -ffp-contract=off`) to the new build
  script;
* carried over this lane's harnesses and results docs.

**But the smoke test says the branch swap is not behaviour-neutral.** Same prompt, same model, same
tiles, identical first token, then divergence:

```
goal branch  (bf16, 1k, ng=16): [576, 3840, 315, 24231, 44295, 22148, 5812, 2973, …]   coherent
npu/col-slice(bf16, 1k)        : [576, 1112, 1112, 1112, 11, 2987, 2987, 2987, …]      repeats
npu/col-slice(int8 runlist, 1k): [23718, 30536, 32984, 63788, 63788, 63788, …]        repeats
```

Not the tiles (both sets are byte-identical at 162080 B) and not `ng` (4/8/16 all repeat). The
decode is where they differ, and upstream rewrote it: between the merge base and main,
`npu_engine_universal.cpp` +2097 lines, `npu_runlist_bridge.cpp` +641, `runtime_layer.cpp` +1123.

**So: build on upstream's *tooling* — generators, artifact-family gate, restored tiles, build
discipline — but do not put the col-slice engine work on upstream's decode blind.** Two ways
forward, both bounded: keep engine work on the goal branch and copy in upstream's build/gate files,
or bisect main's decode to find where the 1k stream degrades (the first bad commit, then decide).

## Reproduce

```bash
N=8 bash ~/eight_models.sh        # load 8 servers (sequential readiness is slow; measure separately)
MAXTOK=64 bash ~/measure8.sh      # the numbers above, against servers already resident
# cleanup: kill the extra servers by PID (a pkill -f pattern matches your own ssh command line)
```
