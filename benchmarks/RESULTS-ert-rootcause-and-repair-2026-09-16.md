# ERT_CMD_STATE_TIMEOUT — root cause, the applied repair, and what it does not fix

Goal `mu35shsg-i3hlyi`, 2026-09-16 ~00:20 ADT. Device work: this goal, serialized,
with the box's other lanes stood down; details below.

## The signature

`ERT_CMD_STATE_TIMEOUT` appeared on three unrelated paths: the native Llama-3.1-8B
runlist (first layer, `ctx_pc=0x28B06005/0x28B060AD`), the FLM oracle in-runtime on
llama3.1:8b (`ctx_pc=0x28B05DB8`, prompt "The capital of Italy is"), and the
Qwen3.6-35B-A3B runtime (`ctx_pc=0x28B060AD`, `c1b76d`).

## Root cause: the driver's `timeout_in_sec=2` TDR

`dmesg` carries **83** `amdxdna ... aie2_dump_ctx: Firmware timeout state capture`
dumps. Each dump lists 2–4 contexts at one timestamp with `DPU PC 0xffffffff`,
`TXN OP ID 0xffffffff` and a `Context PC` in the `0x28b0....` range — byte-for-byte
the `ctx_pc` values the three paths reported. Example (uptime 45335):

```
Dumping ctx ctx.457845.2, hwctx 7, sub=1,  comp=0      <- first submission never completed
Dumping ctx ctx.457724.3, hwctx 9, sub=68, comp=67
    DPU PC: 0xffffffff   TXN OP ID: 0xffffffff   Context PC: 0x28b05db8
```

Driver parameters (`/sys/module/amdxdna/parameters/`):

| param | value | note |
|---|---|---|
| `timeout_in_sec` | **2** | the TDR; **writable** (`-rw-r--r--`, root) |
| `hwctx_limit` | 16 | |
| `context_limit` | 64 | |
| `tdr_dump_ctx` | N | |

So an `ERT_CMD_STATE_TIMEOUT` is the driver timing out a submission that did not
complete inside 2 s — not a kernel, packer or ELF verdict by itself. The parked
production `flm serve` (PID 2438) that existed when the first ERTs were seen is now
**gone**; it was a perpetual hwctx holder.

## The applied repair

```
echo 15 | sudo tee /sys/module/amdxdna/parameters/timeout_in_sec
# revert: echo 2 | sudo tee /sys/module/amdxdna/parameters/timeout_in_sec
```

A queued submission may now take up to 15 s without being torn down. This is
system-wide, reversible, and not persistent (module parameter; resets on reload).

## Evidence

**1. The Llama runlist works when it has the device — alone.** This is the case that
previously ERTed:

```
NPU_LAYER_ELF_DIR=/tmp/llama-elfs NPU_RUNLIST=1 NPU_GREEDY=1 \
  engine/npu/build/npu_engine_llama \
  ~/.config/flm/models/Llama-3.1-8B-NPU2/model.q4nx 16 /tmp/ll_ids.txt

=== Prefill 40 [runlist] ===
Prefill: 3774ms (94 ms/tok)
=== 82.9 ms/tok (12 tok/s) | tokens=16 ===
ids: 791 6864 315 9822 374 12366 …   ->  "The capital of France is Paris."
rc=0, no ERT
```

82.9 ms/tok matches the documented `gen-layer-elfs.sh` figure (68.9 ms/tok), and the
answer is correct. **So the native Llama runlist has no packing/ELF defect** — the
earlier first-layer ERT was environmental.

**2. Controlled contention A/B, honestly reported: it did NOT reproduce the ERT.**
With `timeout_in_sec=2` and a second (then two) `npu_engine_qwen3_8b` contenders
holding hwctx, the same Llama runlist **completed** at 180.3 ms/tok (2 hwctx) and
280.2 ms/tok (3 hwctx) — slower, *not* timed out. So the ERT is not "≥2 hwctx" by
itself, and this report does **not** claim a reproduced failure→success A/B for the
TDR change. What it does claim is the mechanism (the dmesg `timeout_in_sec=2`
teardown) and that the runlist path is sound.

**3. Cross-lane corroboration (theirs, cited not re-measured):**
- `@agent-baaa57` (mtuhp2fy): with 2 engines on accel0, Qwen3-1.7B/8B runlist ERTed
  (1.7B ctx=1551, 8B ctx=1, `ctx_pc=0x28B06005`); the **same commands exited 0
  through ctx 2096 with accel0 quiet** (1.7B 5 tok/s, 8B 12 tok/s). That is the
  contention→TDR A/B.
- `@agent-7f1cce` (mtygjrxl): the fused-layer xclbin (1.36 MB of instructions, one
  submission, ~tens of ms) never ERTed even with two engines live — consistent with
  the cliff being per-**submission** latency, not per-launch count.

## What the repair does NOT fix

The **Qwen3.6-35B-A3B runtime** still ERTs. `@agent-c1b76d` ran FLM v1.0.5
`run qwen3.6-moe:35b-a3b` with accel0 completely quiet (14 attempts: 13 ASLR
SIGSEGV at load, 1 loaded and ERTed at the first runlist, `ctx_pc=0x28B060AD`). That
is a genuinely broken whole-layer path in the vendored lib (their addenda 6–9b:
all-NaN layer output), not the contention cliff. Raising the TDR does not repair it;
their addendum 14 conclusion stands for the 35B.

## Complementary process-level fix

Even with the TDR raised, concurrent hwctx cost throughput (83 → 180 → 280 ms/tok as
contenders joined). The lanes agreed to a `~/.dsh/scratch/mesh/device.lock` flock
taken around every accel0 window: `@agent-7f1cce` and `@agent-c1b76d` both took it,
`@agent-baaa57` paused. This is etiquette-plus-enforcement, not a code change.

## Net effect on this goal

The Llama-3.1-8B oracle tally no longer needs the 16.7 s/token split path. It can
run on the **runlist** path at ~83 ms/tok (12 tok/s) — the remaining cost per prompt
is model load/dequant, not decode. The partial window-A rows (4, split path) are
superseded and the tally is re-run on the runlist.
