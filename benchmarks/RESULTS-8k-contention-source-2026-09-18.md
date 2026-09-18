# Found the 8k variance: a foreign process holding `/dev/accel/accel0` — 2026-09-18

Goal `mu35shsg-i3hlyi`. Follows `RESULTS-8k-campaign-variance-2026-09-18.md`, which could not
explain a 3.6x native outlier (20.0 s vs 5.6–7.2 s for an identical 1.7B 8k prefill) and
listed external contention as the leading but undemonstrated candidate. It is now demonstrated,
and the instrument that catches it is committed with this document:
`benchmarks/c8k_guarded.sh`.

## The evidence

Running the campaign through the new guard (which records every PID with `accel0` open before
and after each run, excluding the production `flm serve`) discarded **3/3** runs:

```
run 1 [DISCARD(post-holder: 90145:/tmp/attrib ; 98853:/tmp/attrib)] native  4121ms (0.503 ms/tok)
run 2 [DISCARD(post-holder: 90145:/tmp/attrib ; 98853:/tmp/attrib)] native 17971ms (2.194 ms/tok)
run 3 [DISCARD(post-holder: 90145:/tmp/attrib ; 98853:/tmp/attrib)] native 13551ms (1.654 ms/tok)
```

The holder is:

```
PID    ELAPSED  %CPU  CMD
90145    49:45  98.2  /tmp/attrib
98853    34:44  97.6  /tmp/attrib        # /tmp/attrib, 291744 B, compiled 14:53
```

Two long-lived `/tmp/attrib` processes (~98% CPU each, resident for ~35–50 minutes) hold
`/dev/accel/accel0` — alongside the expected `flm serve` — and the native prefill in those runs
is inflated **3.3x and 4.4x** relative to the 4.12 s quiet run.

That is the outlier class. It is the same mechanism the lane's own
`benchmarks/npu-device-preflight.sh` warns about: the engine takes
`/tmp/1bit-npu-device.lock` itself, but other tools do not, so they overlap silently.

## What this changes

1. **The 8k variance is explained and is not a property of the engine.** The numbers in
   `RESULTS-8k-campaign-variance-2026-09-18.md` that were taken while a foreign holder was
   present (the 20.0 s run, and by the same mechanism the earlier wide spreads) are
   contaminated, not informative.
2. **The earlier "cross-window variance" story in the decode work needs the same suspicion.**
   The decode penalty that appeared (15.5–15.9 ms exec) and vanished (12.5 ms) across windows
   was attributed to BO allocation state; a foreign `accel0` holder is the simpler and now
   demonstrated explanation for that class of shift. That does not change the withdrawal of the
   penalty claim — it strengthens it — but it does mean the *variance mechanism* recorded there
   is probably wrong and should be read as "unidentified, most likely foreign device
   contention".
3. **Every number in this lane needs the guard.** No harness here recorded device holders around
   a measurement before now. `c8k_guarded.sh` does, discards contaminated runs, records
   `/tmp/runner.log` growth (CI activity) per run, and prints medians of the accepted runs only.

## Status of the 8k clause

Unchanged in substance and now cleaner to state: 8k is **parity, not a win**, for the models
measured repeatedly on a quiet device (0.6B 2009–2028 t/s vs FLM 1988–2010; 1.7B 1145–1473 vs
1193–1374), the old 0.62–0.71x inversion is not reproduced, and 8k decode is unmeasurable past
one token (`ctx=8194` vs the baked `MAX_L=8192` window). What was missing is now present: an
instrument that can tell a measurement from a contended one.

## For the device's other users

`/tmp/attrib` (PIDs 90145 and 98853) is not a process of this lane. It holds `accel0` for its
whole lifetime and does not take `/tmp/1bit-npu-device.lock`, so it silently invalidates any
NPU measurement taken beside it. The lane's own rule is to leave other lanes' work alone, so it
was not touched; the room has been told, and the guard now makes the contamination visible in
every campaign's output rather than in a post-hoc explanation.

## The guard is now IN THE ENGINE, and verified with a controlled holder

The warning above is fine for a human running a campaign, but the class of error is silent by
construction: a tool that does not take the lock cannot be fixed by asking its author to look
at a document. So the same check is now in `engine/npu/src/npu_runlist_bridge.cpp`
(`check_foreign_accel_holders()`), called at both runlist entry points —
`npu_runlist_session_init` (the unified path) and `npu_runlist_decode` (the whole-layer path):

- default: prints `[contention] WARNING: foreign holder(s) on /dev/accel/accel0: <pid>:<cmd>; …`
  naming each non-`flm serve` holder, and continues;
- `NPU_STRICT_DEVICE=1`: refuses to run (`exit(3)`) instead of producing a number;
- `NPU_ALLOW_CONTENDED=1`: silences it;
- the production `flm serve` is excluded (it is expected to hold the device).

**Verified with a controlled holder**, not with whatever happened to be on the box: a dummy
`python3` process holding `/dev/accel/accel0` was started, and the three modes behave as
specified —

```
default                  -> rc=0, "[contention] WARNING: foreign holder(s) …: 172049:python3 -c …"
NPU_STRICT_DEVICE=1      -> rc=3, WARNING then "[contention] refusing to run (NPU_STRICT_DEVICE=1)"
NPU_ALLOW_CONTENDED=1    -> rc=0, 0 contention lines
```

That is a better test than the live `/tmp/attrib` case, because the live case disappeared
between two checks (the two processes had exited by the time the rebuilt binary was ready) —
which is itself the point: the contamination is transient and unattended, so the instrument has
to be deterministic.

## Second contamination axis: host CPU load (the "conv+other" term is host work)

The device guard above is necessary and not sufficient. The 8k native prefill of a **small**
model is dominated by host work, not by the array: for Qwen3-0.6B the engine reports

```
Prefill:  4078ms (0.498 ms/tok) [GEMM  ..., attn 2434ms, conv+other  ~4000ms]
```

so ~2.4 s of attention on the device and ~4 s of host `conv+other`. When the host is busy that
term is what moves. Measured on 2026-09-18 with a foreign `pf` process at ~3000–3100% CPU
(~31 of 32 cores) plus rising `pi` activity — **all device-holder checks clean**:

| 1-min load before | native 0.6B 8k prefill | ms/prompt-token |
|---:|---:|---:|
| 16.83 | 4078 ms | 0.498 |
| 20.89 | 7268 ms | 0.887 (1.8x) |
| 23.23 → 27.64 | 11238 ms | 1.372 (2.8x, still rising during the run) |

The same command, the same binary, the same prompt, no foreign device holder. The engine's own
breakdown shows the split: `attn` stays flat at ~2.4 s while `conv+other` tracks the load.

**Consequences:**

- **Every 8k native number in this lane needs the host load recorded beside it**, not just the
  device holders. The "parity at 8k" reading in `RESULTS-8k-campaign-variance-2026-09-18.md` was
  taken with the load unrecorded, so it is a measurement of the environment as much as of the
  engine: at load ~17 the 0.6B prefill is 4078 ms; at load ~23 it is 11238 ms.
- The earlier "native ahead at 8k" table and the older "0.62–0.71x inversion" table are both
  compatible with load differences of this size — for the host-bound models (0.6B, and to a
  lesser degree 1.7B) the load term can move the result by 2.8x on its own.
- FLM is much less host-bound (its own numbers moved only 1628 → 1953 t/s across the same
  runs), so **load bias favours FLM** on the small models: the native side degrades first.
- **Rule:** `benchmarks/c8k_guarded.sh` now gates on both axes — foreign `accel0` holders
  (transient `flm serve` instances included; only the production `qwen3.6-moe:35b-a3b` server
  is excluded) and 1-min load (default ceiling 18, calibrated above), plus a load-rise check
  across the run, and a settle loop that waits for a transient holder to release the device
  before measuring.

## Pre-flight: catch a holder that is already there

The around-run checks catch a holder that arrives mid-campaign; a campaign that starts with one
already resident just discards every run. `benchmarks/c8k_guarded.sh` now refuses to start in
that case:

```
ABORT: foreign holder(s) on /dev/accel/accel0 are already present before the campaign:
  pid=271948 age=03:20 read_bytes: 0 write_bytes: 0 cmd=python3 -c import time; f=open("/dev/accel/accel0","rb") …
  A long-lived holder with read_bytes=0 write_bytes=0 is spinning, not measuring.
  Wait for it to exit, or set C8K_IGNORE_PREEXISTING=1 to run anyway …
```

The detail line (age + `read_bytes`/`write_bytes`) is @agent-dddf9e's diagnostic, and they
proved it on the live case: `/tmp/attrib` was **their lane's** leaked attribution harness
(prism kernel symbols, cwd `~/1bit-MONSTER-dddf9e`), it had been running with zero device I/O —
i.e. spinning, not measuring — and after they killed it their box triad recovered from
180.9–192.4 GB/s to 202.5–217.1 GB/s, confirming the shared-LPDDR mechanism and explaining a
drift they had already tagged `strixhalo-busy` in their own 15:13 window. No recorded number of
theirs was invalidated; the two lanes independently caught the same leak from opposite ends
(their quiet-threshold rule refused the tag; this guard refused the runs).

Also fixed in the same rewrite: `foreign_holders()` emitted holders separated by `"; "`, which
broke on any command line containing a semicolon (a `python3 -c "…; …"` holder produced
phantom "holders"). It now emits one `pid:cmd` per line.

**Still blocked:** a six-model 8k campaign needs a window with no `pf`-style CPU hog and no
foreign `accel0` job. Two attempts today (15:35 and 15:45 ADT) were rejected wholesale —
loads 26.7 → 59.3, then 38.8 → 41.5 with `pf` at ~3000% CPU — so 0.6B is the only model with a
guard-accepted 8k figure (parity, 0.97x median).

## The second contaminant is attributed too: the Prism lane's own test suite

The `pf` hog that kept the load at 26–48 and inflated every 8k campaign attempt is not a
mystery process. `c8k_guarded.sh` now prints the top consumer's pid **and its parent**, which
identifies it in one line:

```
topcpu=2367 pf pid=301752 parent=301751:python3 /home/bcloud/1bit-MONSTER-dddf9e/tests/prism/check_oracle_agreement.py …
```

Full chain, from `/proc`:

```
294542  /bin/bash ./tests/prism/run_prism_tests.sh
294543  python3 /home/bcloud/1bit-MONSTER-dddf9e/tests/prism/check_oracle_agreement.py /tmp/tmp.ukJdcbVkTy/p…
294544  /tmp/tmp.ukJdcbVkTy/pf /home/bcloud/models/prism/1bp/Ternary-Bonsai-27B-PQ2_0.1bp 760 6511 314 9338 369
        exe=/tmp/tmp.ukJdcbVkTy/pf   cwd=/home/bcloud/1bit-MONSTER-dddf9e
```

So both contaminants of this session came from the same lane's tooling — `/tmp/attrib` (their
attribution harness) and now `pf` (their oracle-agreement test's forward pass) — and neither
takes the device lock or any CPU-slot convention. The lane has been told; this is a scheduling
fact for the box, not a defect in the engine.

The guard's `topcpu=` field now carries `parent=<ppid>:<cmd>` for exactly this reason: a bare
`pf` at 2600% CPU is unattributable from the output, and "which lane is doing this" is the first
question any reader of a rejected run will ask.
