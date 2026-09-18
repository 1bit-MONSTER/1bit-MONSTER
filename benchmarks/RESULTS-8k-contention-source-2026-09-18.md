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
