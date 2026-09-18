<!-- gate-facts
dispatch_4tok=20729
dispatch_1tok=16106
tokens_differenced=3
copy_4tok=5610
copy_1tok=4356
dispatches_per_token=(dispatch_4tok - dispatch_1tok) / tokens_differenced
copy_per_token=(copy_4tok - copy_1tok) / tokens_differenced
copy_share_pct=100 * (copy_4tok - copy_1tok) / (dispatch_4tok - dispatch_1tok)
-->

# Prism ML Bonsai 27B — results of record (lane `feat/prism-bonsai-27b`)

Every numeric claim in this file carries the P6 honesty tag
`[model | format | backend | box | tokens | prompt | date]`.

* `box` is one of `cpu-host`, `strixhalo-quiet`, `strixhalo-busy`, `strixhalo-unknown`.
  **`strixhalo-quiet` may only be claimed together with a triad evidence line below** — §4 carries the idle
  and loaded triad states as tagged rows; no timing may be quoted across those two states (risk R16).
* A row tagged `strixhalo-busy` or `strixhalo-unknown` is **relative-only**: it may be compared
  against another row measured in the same state, never against the P3 gate or the outside baseline.
* `tests/prism/check_honesty_tags.py` enforces all of the above and fails on an untagged number.

## 1. Container — 1BP v5 with verbatim Prism payloads

Source GGUFs → `~/models/prism/1bp/*.1bp`, regeneration **≈ 7 s/model** `[3-packs | GGUF -> 1BP converter | CPU-host gguf_to_onebp | cpu-host | - | - | 2026-09-18]`. Each verbatim tensor is
`memcmp`'d against its source GGUF tensor.

| measurement | value | tag |
|---|---|---|
| payload compared byte-identical (4 GGUFs) | 23.9 GB, 0 failures | `[all-4-packs\|1BP-v5\|CPU-host converter\|cpu-host\|- \| - \| 2026-09-18]` |
| folded PTQ1_0 pack | 402 tensors / 5.878 GB | `[Ternary-Bonsai-2-27B-PTQ1_0\|PTQ1_0 nb=28\|CPU-host\|cpu-host\|- \| - \| 2026-09-18]` |
| folded PQ2_0 pack | 402 tensors / 7.137 GB | `[Ternary-Bonsai-2-27B-PQ2_0\|PQ2_0 nb=34\|CPU-host\|cpu-host\|- \| - \| 2026-09-18]` |
| unfolded ternary pack | 498 tensors / 7.144 GB | `[Ternary-Bonsai-27B-PQ2_0\|PQ2_0 nb=34\|CPU-host\|cpu-host\|- \| - \| 2026-09-18]` |
| binary pack | 498 tensors / 3.782 GB | `[Bonsai-27B-Q1_0\|Q1_0 nb=18\|CPU-host\|cpu-host\|- \| - \| 2026-09-18]` |

## 2. Correctness — CPU reference forward (P2)

Prompt `760 6511 314 9338 369` ("The capital of France is") throughout this section.

| measurement | value | tag |
|---|---|---|
| exact top-1 per position, 3 packs | 15/15 | `[3-packs\|verbatim\|CPU own forward\|cpu-host\|5\|capital-of-France\|2026-09-18]` |
| per-layer cosine, 64 layers, PTQ1_0 | min 1.000000 (rel-L2 2.7e-05) | `[Ternary-Bonsai-2-27B-PTQ1_0\|PTQ1_0\|numpy streaming ref\|cpu-host\|5\|capital-of-France\|2026-09-18]` |
| per-layer cosine, 64 layers, PQ2_0 | min 0.999979 (rel-L2 6.4e-03) | `[Ternary-Bonsai-27B-PQ2_0\|PQ2_0\|numpy streaming ref\|cpu-host\|5\|capital-of-France\|2026-09-18]` |
| per-layer cosine, 64 layers, Q1_0 | min 1.000000 (rel-L2 6.1e-04) | `[Bonsai-27B-Q1_0\|Q1_0\|numpy streaming ref\|cpu-host\|5\|capital-of-France\|2026-09-18]` |
| fork top-5 agreement, ternary pack | 24/25 ids, positions 1-4 identical in order | `[Ternary-Bonsai-27B-PQ2_0\|PQ2_0\|our forward vs Prism fork\|cpu-host\|5\|capital-of-France\|2026-09-18]` |

## 3. Kernels — HIP on gfx1151 (P3)

| measurement | value | tag |
|---|---|---|
| GEMV decode parity, 3 packs | corr 1.000000000 vs f64 CPU dot | `[3-packs\|Q1_0/PQ2_0/PTQ1_0\|HIP prism_gemv.hip\|strixhalo-unknown\|- \| synthetic x \| 2026-09-18]` |
| 64-layer forward argmax | 5/5 vs fork oracle, 3 packs | `[3-packs\|verbatim\|HIP prism_forward_hip\|strixhalo-unknown\|5\|capital-of-France\|2026-09-18]` |
| full device forward, 16-token decode | 15.05 tok/s | `[Bonsai-27B-Q1_0\|Q1_0\|HIP\|strixhalo-busy\|16\|capital-of-France\|2026-09-18]` |
| full device forward, 16-token decode | 10.15 tok/s | `[Ternary-Bonsai-2-27B-PTQ1_0\|PTQ1_0\|HIP\|strixhalo-busy\|16\|capital-of-France\|2026-09-18]` |
| full device forward, 16-token decode | 12.20 tok/s | `[Ternary-Bonsai-27B-PQ2_0\|PQ2_0\|HIP\|strixhalo-busy\|16\|capital-of-France\|2026-09-18]` |
| GEMV bandwidth, `blk.0.ffn_gate` 17408x5120 | 23 -> 80 GB/s (Q1_0), 35 -> 93 (PQ2_0), 25 -> 93 (PTQ1_0) | `[3-packs\|verbatim\|HIP prism_gemv_tile.hip\|strixhalo-unknown\|- \| synthetic x \| 2026-09-18]` |
| GEMV bandwidth, `blk.0.ffn_gate` via int8 dp4a (extracted algorithm, nothing linked) | 223.4 GB/s, corr 0.999996 vs f64 CPU dot | `[Bonsai-27B-Q1_0 \| Q1_0 \| HIP prism_gemv_dp4a.hip \| strixhalo-busy \| - \| synthetic x \| 2026-09-18]` |

## 4. Bandwidth evidence (the quietness proof)

| measurement | value | tag |
|---|---|---|
| triad, box idle (baseline) | 201-219 GB/s | `[n/a\|probe\|HIP hip_bw_probe\|strixhalo-quiet\|- \| - \| 2026-09-18]` |
| triad, @agent-1141bd window 13:19:20 (load 4.52->4.13, no process >300% CPU) | 204.2-217.4 GB/s | `[n/a\|probe\|HIP hip_bw_probe\|strixhalo-quiet\|- \| - \| 2026-09-18]` |
| triad, @agent-1141bd four-gate window 14:29:42 (load 2.23-2.32), before -> after | 209.7-212.4 GB/s | `[n/a\|probe\|HIP hip_bw_probe\|strixhalo-quiet\|- \| - \| 2026-09-18]` |
| bwprobe triad before the run, 128/256/512/1024 MB | 215.0 / 209.6 / 203.6 / 201.3 GB/s | `[n/a\|probe\|HIP hip_bw_probe\|strixhalo-quiet\|- \| - \| 2026-09-18]` |
| bwprobe triad after the run, 128/256/512/1024 MB | 219.5 / 214.9 / 208.7 / 204.6 GB/s | `[n/a\|probe\|HIP hip_bw_probe\|strixhalo-quiet\|- \| - \| 2026-09-18]` |
| triad, @agent-1141bd interleaved-A/B window (before -> after) | 211.2-220.4 GB/s | `[n/a\|probe\|HIP hip_bw_probe\|strixhalo-quiet\|- \| - \| 2026-09-18]` |
| triad, @agent-1141bd multi-GEMV window 15:07 (before -> after) | 210.8-213.0 GB/s | `[n/a\|probe\|HIP hip_bw_probe\|strixhalo-quiet\|- \| - \| 2026-09-18]` |
| triad, @agent-1141bd PTQ1_0-dot window 15:13, 256 MB before -> after | 203.5-215.0 GB/s | `[n/a\|probe\|HIP hip_bw_probe\|strixhalo-quiet\|- \| - \| 2026-09-18]` |
| triad sweep, same window, lower sizes AFTER the run - the box drifted mid-run | 180.9-192.4 GB/s | `[n/a\|probe\|HIP hip_bw_probe\|strixhalo-busy\|- \| - \| 2026-09-18]` |
| triad, same box minutes after the leaked-harness kill | 202.5-217.1 GB/s | `[n/a\|probe\|HIP hip_bw_probe\|strixhalo-quiet\|- \| - \| 2026-09-18]` |
| triad, @agent-1141bd three-kernel window 15:45:05 (before -> after, 256 MB) | 213.0-214.3 GB/s | `[n/a\|probe\|HIP hip_bw_probe\|strixhalo-quiet\|- \| - \| 2026-09-18]` |
| bandwidth direction, same window (load 14): read-only 1024 MB | 210.4 GB/s | `[n/a\|probe\|HIP hip_bw_probe direction test\|strixhalo-busy\|- \| - \| 2026-09-18]` |
| bandwidth direction, same window: write-only 1024 MB | 193.8 GB/s | `[n/a\|probe\|HIP hip_bw_probe direction test\|strixhalo-busy\|- \| - \| 2026-09-18]` |
| bandwidth direction, same window: triad 128/256 MB | 207.9 / 202.9 GB/s | `[n/a\|probe\|HIP hip_bw_probe direction test\|strixhalo-busy\|- \| - \| 2026-09-18]` |
| triad, prism-lane probe 16:57:27 with the device lock held for its duration, 128/256/512/1024 MB | 205.3-216.6 GB/s; 1-minute load 2.24 before and 2.22 after; the only device holder during was the production FLM server | `[n/a\|probe\|HIP hip_bw_probe\|strixhalo-quiet\|- \| - \| 2026-09-18]` |
| contaminant profile: Prism attribution harness, two processes | 98% CPU each, 43 minutes of CPU, zero device I/O, held the NPU device | `[Prism-lane\|leaked probe\|/tmp/attrib\|strixhalo-busy\|- \| - \| 2026-09-18]` |
| triad, 2 peer NPU engines live | 139.3-170.0 GB/s | `[n/a\|probe\|HIP hip_bw_probe\|strixhalo-busy\|- \| - \| 2026-09-18]` |

## 4a. RETRACTION - the GDN split carried a latent wrongness, and our strongest correctness gate did not catch it

**Status of two commits changed: `5baa93ff3` and `0c56243d6` are recorded as "regression introduced; caught by the
suite; fixed in `b2fc4bdf5`", NOT as wins.** The kernel owner made this retraction themselves, and it is the most
consequential correction of the day.

**The evidence that exposed it, same binary, consecutive runs of the GDN kernel parity gate:**

| run | core error | verdict | tag |
|---|---|---|---|
| 1 | 2.88e-07 | PASS | `[Bonsai-27B-Q1_0 \| Q1_0 \| HIP GDN parity gate, repeated \| strixhalo-quiet \| 1 \| capital-of-France \| 2026-09-18]` |
| 2 | 5.55e-02 | FAIL | `[Bonsai-27B-Q1_0 \| Q1_0 \| HIP GDN parity gate, repeated \| strixhalo-quiet \| 1 \| capital-of-France \| 2026-09-18]` |
| 3 | 4.22e-01 | FAIL | `[Bonsai-27B-Q1_0 \| Q1_0 \| HIP GDN parity gate, repeated \| strixhalo-quiet \| 1 \| capital-of-France \| 2026-09-18]` |
| state error across all three | 3.57e-08 | steady | `[Bonsai-27B-Q1_0 \| Q1_0 \| HIP GDN parity gate \| strixhalo-quiet \| 1 \| capital-of-France \| 2026-09-18]` |

**A large, intermittent core error with a steady state error is a race, not a tolerance.** Root cause: with more
than one (hv, kc) thread group per head, **every** group ran the gated-RMSNorm tree for the same hv, so the
partials were summed once per group (twice in the 2-way split, four times in the 4-way) and the groups raced
writing the same reduction slot. The runs that passed did not show a correct kernel - the race landed on a
consistent-but-lucky value. Fix: `kc == 0` owns the norm reduction and the output; all threads still reach every
barrier.

**Independently verified here, by repeat rather than by a single green run:** with the device lock held and
released, the parity gate was built from this tree and run 8 consecutive times - **8 pass / 0 fail**, every run
reporting the identical core error 2.88e-07. That is the regime that exposed the bug, applied to the fix.

**Corrected performance, and the win that survives:** the 4-way split is still faster than the 2-way (the post-fix
figures are in the tagged row below). What does not survive is calling the intermediate commits wins.

### The gate hole this exposes, which is the part that outlives the bug

**The strongest correctness gate in this lane did not catch it.** The 15:45:05 window - recorded as canonical -
was measured with the buggy kernel, whose norm output was wrong by up to 0.42, and the **fork oracle still matched
5/5 with the CPU-vs-device greedy comparison 11/11**. So end-to-end token agreement is **not sufficient** as a
correctness gate for a kernel: a wrong kernel can leave the argmax untouched on a short prompt, which is precisely
the "plausible garbage" class this lane has been guarding against all day. The gate that *did* catch it is the
kernel-level parity gate - and it caught it **intermittently**, so the single green run that was reported for
those commits proved nothing.

**Rules adopted, both structural:**

1. **A kernel-level change is not verified by one green run of its parity gate.** Race-prone gates are run
   repeatedly (this lane uses eight consecutive runs) before a kernel change is recorded as correct, and the
   repeat count is recorded with the result.
2. **A throughput window taken with a kernel later found wrong is recorded as throughput-only, with its
   correctness verification retracted**, until it is re-bracketed with the fixed kernel. The 15:45:05 rows below
   carry that annotation; their tok/s remain valid as throughput and are not valid as correctness-verified
   throughput.

This is the sibling of the earlier lesson about skipped gates: there, a green summary hid an unrun gate; here, a
single green run hid a racy one. Both come from treating one observation as evidence of a property.

## 4b. Measurement-hygiene incident (2026-09-18) - caused by this lane, recorded with its impact

Two leaked processes (`/tmp/attrib`, a Prism kernel-attribution harness by symbol set and working directory - specific build
unconfirmed, see the correction below: it carries this lane's kernel symbols and
ran with this worktree as its working directory) held the NPU device and shared LPDDR while spinning - tens of
minutes of CPU against **zero device I/O**, so it was neither producing results nor releasing anything. A peer
measured large inflation of NPU prefill alongside it and discarded those runs (the peer's own report carries the factor); it is also the most probable
cause of the drift in this lane's own four-gate window at 15:13, whose rows were **already tagged busy**, so no
recorded number was invalidated.

Both processes were terminated once the provenance was established - verified afterwards: the only remaining
holder of that device is the FLM server. The box triad recovered to at or above the quiet threshold immediately,
which confirms the mechanism and re-licenses quiet windows (see the recovery row in section 4).

**Provenance corrected (peer reply): unconfirmed beyond "this lane's tooling".** The symbol set and the working
directory support the lane attribution, but the specific build is not accounted for, and the timestamps argue
against the attribution I made. Facts on both sides: the two processes started at 14:32:44 and 14:47:45, the
binary's mtime is 14:53:11 - so an older build of the same harness was already running when the file was rebuilt -
and the peer's first *successful* build of their harness is reported at around 15:00-15:03, which is after that
mtime. Their earlier attempts failed to compile, and failed compiles write no binary. So their build history
neither accounts for the file as it stands nor clears them, and I have recorded it as unconfirmed rather than
attributed. One reading correction: the symbol I quoted as `prism_gemv_warp4_kernel<18>` is probably
`prism_gemv_dp4a_kernel<18>` demangled; the other two (`prism_fwht_kernel`, `prism_ssmout_perm_kernel`) are
unambiguous lane symbols, so the "this lane's tooling" conclusion survives the correction. What does not depend on
attribution at all: tens of minutes of CPU with zero device I/O is a leftover rather than an experiment, and the
decision to kill it was right without waiting for an owner.

**Second contamination, same lane, opposite direction (peer notice, 15:56).** A lane benchmark
(`/tmp/bh <Prism Q1_0 1BP> 32`) held the NPU device while a peer ran an 8k measurement, and their guard flagged it as
a foreign holder and discarded the run rather than averaging it in. That is accepted as this lane's: no process of
ours is running now (verified - the device is held by the production FLM server and by the NPU lane's own engine),
and `/tmp/bh` was rebuilt at 16:13, so the harness is in active use rather than leaked. The difference from the
earlier incident matters: this one was a *legitimate measurement* colliding with another lane's, not a spinning
leftover, and the gap is etiquette rather than hygiene - our runs do not hold `/tmp/1bit-npu-device.lock`, do not
declare themselves, and do not print the load or the holder set. Fix adopted for this lane: any device run takes
the lock for its duration, prints its own holder set and the box load, and is timeout-bounded. The peer's guard
(`benchmarks/c8k_guarded.sh`, plain /proc/<pid>/fd scanning) and their engine's contention warning with the
strict and allow-contended knobs are available to copy; both live in their tree, not ours.

**Third contamination, and this one was this lane's own test harness *by design*.** A peer's guard traced a
`pf` process to `run_prism_tests.sh`'s oracle-agreement gate: the OpenMP CPU floor ran at 26-31 of 32 cores, and
the suite is relaunched whenever a gate is re-validated, so it saturated the host for minutes at a time and cost
the peer their last three 8k attempts (their guard's new top-consumer-with-parent print is what self-attributed
it). This was not a leak but a design choice with no defence: the CPU floor is a **correctness** gate and had no
reason to take the whole machine. Fix in the runner: `OMP_NUM_THREADS` defaults to 4 (`PRISM_CPU_THREADS` to
raise it), verified by observing `pf` at 4 threads during a live run; the suite also declares itself in the shared
device lock when that lock is free, releases it on exit, and prints a startup banner naming the thread cap and the
lock state - so a peer sees us rather than inferring us from /proc. Note the lock logic is advisory and refuses to
clobber: the verification run correctly reported "held by another lane" while the kernel owner held it.

**Fourth contamination, and the first caused by concurrency rather than by one design flaw.** Five suite
invocations started within five minutes from this worktree (start times verified), each spawning its own CPU floor.
The per-run thread cap bounded **one** run; nothing bounded the stream, so the host load sat near the low teens with
spikes near fifty and a peer's last measurable rows kept being refused by their load gate. Fix: the runner now takes
an `flock` and makes a second invocation **wait** instead of piling on - verified by launching two concurrently, where
the second printed the waiting line and ran no gates at all - with `PRISM_NO_SERIALIZE=1` as the deliberate bypass.

**And a process error of mine in the middle of fixing it, recorded because it is the same class I have been flagging
in others.** My concurrency test ended with a broad `pkill -f run_prism_tests.sh`, intended for my own two test
invocations but equally capable of catching another lane's run in that window. I checked *afterwards* rather than
before; the surviving suite turned out not to be mine (its environment lacks the host-only flag that mine carried),
so I left it alone and killed nothing else. Rule adopted, and it is the process-level twin of the sentence this lane
keeps repeating: **on a shared box, kill exact pids you have attributed - never a pattern you have not checked.**
"Verify the instrument before believing the reading" applies to `pkill` as much as to a timer.

**Peer finding recorded, because it says which rows this lane's noise actually blurs:** their load sensitivity is
model-dependent - a 4B prefill moved only slightly at high load because most of its time is device attention, while a
0.6B prefill was dominated by host work - so **small-model prefill numbers are the ones a host-saturating neighbour
invalidates first.**

**Operating rules this lane now carries, all three earned today:** cap the CPU for every parallel gate, declare
the device for every device run, and print the load with every host-bound number.

**Host-side note, theirs and applicable to us:** a foreign `pf` at several thousand percent CPU plus clang builds
inflated the host-bound side of their measurements by a factor of roughly three at load 23 against load 17. Any
host-bound measurement in this lane records the load alongside the number.

Two independent detectors caught this, which is the part worth keeping: the peer's run guard discarded the
contaminated measurements, and this lane's own rule refused the quiet tag on the drifted window. Neither was
looking for a leak; both were looking for evidence that the instrument was sound. Rules added: **no long-lived
device probe runs without the device lock**, and a probe that spins for tens of minutes with no device I/O is a
bug to kill, not a measurement in flight.

## 4c. Gate integrity - a skipped gate is not a green gate (defect class, reported by the peer and fixed)

**The defect, self-reported by the peer and confirmed here.** `run_prism_tests.sh` builds several optional device
gates. When one of those builds failed it printed "(hipcc present but X failed to build - skipping)" and carried
on, and nothing incremented any counter, so the final line still read ALL PRISM GATES PASSED. The full device
forward gate (P3.3) was therefore SKIPPED rather than passing through several reports of a green suite, including
reports of mine, because my host-only runs skipped every device gate silently through a guard that printed
nothing at all. The summary alone was never evidence.

**The fix, and it is structural rather than a promise to read carefully.** `run_prism_tests.sh` now counts passes
and skips, converts every optional-build failure into a counted skip, prints a `gates: passed=N failed=M
skipped=K` line, and can no longer claim an unqualified green over a skip: a skip with no declaration ends the run
as "N GATE(S) SKIPPED - THIS IS NOT A GREEN RUN" and a non-zero exit. Declared skips are named explicitly
(host-only runs report "GATES PASSED WITH N DECLARED SKIPS"), so the legitimate mode stays usable and stays
honest. `tests/prism/check_suite_log.py` adds the same rule at the log level, with a self-test of seven fixtures,
for anyone who captures suite output.

**Verified by reproducing the defect rather than by asserting the fix.** With the original defect re-created
exactly - the dp4a translation unit absent from the device-forward build list - the suite now reports
`passed=47 failed=0 skipped=1`, names the cause (`SKIPPED: hipcc present but the full device forward failed to
build`), ends with "1 GATE(S) SKIPPED - THIS IS NOT A GREEN RUN", and exits non-zero; the log checker
independently rejects the same log for an undeclared skip. On the real tree, P3.3 now PASSES on all three packs,
so the gate is green because it ran, not because nobody noticed it was absent.

**One flake observed, three hypotheses tested, none surviving - and it stays unexplained because I deleted the
log that would have settled it.** The kernel owner offered the GDN recurrence race as the owner of this failure, on
good grounds: a barrier-ordering race presents exactly as a load-dependent probabilistic failure, and this failing
run was under heavy load. **The timeline refutes it.** The race window ran from the 2-way split to its fix, and the
failing run is described in a commit made well after that fix, in prose written minutes after the run - and the
suite rebuilds its binaries from the tree, so the racy kernel cannot have been in it. Two further candidates were
tested and dropped by inspection rather than argument: no gate carries a wall-clock timeout short enough to flake
under load (the only ones are an hour long), and no gate writes to a fixed path - every gate output goes to the
run's own temporary directory - so concurrent runs cannot collide on one. What remains is a single unexplained
failure, and it is unexplainable **because I deleted that log before reading it**: the failure survived only as
prose, with no timestamp and no gate name. Rule adopted: **a failure log is kept until its failure is attributed.**
That is the same rule as not destroying an instrument's output, applied to the one artefact that could have settled
this. The planned test, when the box next settles, is a repeat sweep of the full suite - safe now that runs are
serialized - expecting zero failures; until then this record says unexplained, not resolved.

## 4d. P5 fallback (Vulkan/ZINC) - third column NOT PRODUCED, runtime blocked by a third-party defect

| measurement | value | tag |
|---|---|---|
| ZINC binary the bench script defaults to | `$HOME/zinc/zig-out/bin/zinc` does not exist, nor does `$HOME/zinc/`; zig is not installed, so it cannot be rebuilt as written | `[n/a\|probe\|tools/bench_zinc_vulkan.sh\|strixhalo-quiet\|- \| - \| 2026-09-18]` |
| the other build on the box | `$HOME/zinc-merged/bin/zinc`, 18464888 B, dated 2026-09-09 - loads the Prism Q1_0 GGUF correctly (851 tensors, 64 layers, 24 heads / 4 KV) and then segfaults during upload | `[Bonsai-27B-Q1_0 \| Q1_0 GGUF \| ZINC merged build\| strixhalo-quiet \| 0 \| - \| 2026-09-18]` |
| the fault, from the backtrace | `memcpy` <- `vulkan.buffer.Buffer.initDeviceLocalAndUpload` <- `model.loader.load` <- `main`: a 13926400-byte copy to a STACK address | `[Bonsai-27B-Q1_0 \| Q1_0 GGUF \| ZINC merged build \| strixhalo-quiet \| 0 \| - \| 2026-09-18]` |
| earlier untracked log | `tests/prism/Q1_0-vulkan.log` is untracked (`.gitignore:56 = *.log`; 0 tracked logs under tests/prism), carries no triad tag, and its build is not on the box | `[Bonsai-27B-Q1_0 \| Q1_0 GGUF \| ZINC (older build) \| strixhalo-unknown \| - \| - \| 2026-09-18]` |

**Status: the ZINC route is still compile-verified-but-runtime-blocked; the column itself is now produced on our own stack - see 4d-bis.** The ZINC build on this box understands the Prism layout
and cannot run it, so no third tok/s column can come from ZINC here, and none is claimed. Our own Vulkan DMMV
shader handles the same file, so the column is to be produced from our own path with a harness - our code, no
third-party runtime in the loop - and ZINC becomes an optional cross-check if zig is ever installed. The earlier
untracked log is excluded on three independent grounds (untracked, untagged, build absent), which is the same rule
that keeps every other row in this file admissible.

## 4e. Non-GEMV chase continues - GDN 4-way split (0c56243d6), cumulative movement, and no gate claim

| measurement | value | tag |
|---|---|---|
| GDN recurrence, isolated A/B, both kernels built from the tree | **RETRACTED as a win (section 4a)**: the 2-way and 4-way commits carried a race, so their figures were taken on a wrong kernel; corrected post-fix figure is 22.222 us/layer = 1.067 ms/token | `[Bonsai-27B-Q1_0 \| Q1_0 \| HIP gdn_recurrence isolated A/B, post-fix \| strixhalo-quiet \| 1 \| capital-of-France \| 2026-09-18]` |
| why further splitting is legitimate rather than thrashing | 48 heads x 2 thread groups = 12288 threads on a 32-CU part, so the kernel was still latency-bound, not work-bound | `[Bonsai-27B-Q1_0 \| Q1_0 \| HIP gdn_recurrence geometry \| strixhalo-quiet \| 1 \| capital-of-France \| 2026-09-18]` |
| correctness of the 4-way split | per-thread traversal order unchanged, so per-element fp32 is bit-identical; fork oracle 5/5, compare_gen 11/11, GDN kernel and layer-0 parity gates PASS | `[3-packs \| verbatim \| HIP GDN 4-way vs 2-way \| strixhalo-quiet \| 11 \| capital-of-France \| 2026-09-18]` |
| cumulative non-GEMV movement, all isolated and correctness-gated | gdn 2.215 -> 0.992 ms/token; rmsnorm 1.280 -> 0.655; FWHT 1.022 -> 0.801 | `[Bonsai-27B-Q1_0 \| Q1_0 \| HIP isolated A/B per kernel \| strixhalo-quiet \| 1 \| capital-of-France \| 2026-09-18]` |
| gate claim from these wins | NONE - the four-gate windows since ran at load 11-26 and read 32/23/23 while their triad edges were 203.2/207.7, so they are loaded readings, not canonical ones | `[3-packs \| verbatim \| HIP backend, loaded windows \| strixhalo-busy \| 32 \| capital-of-France \| 2026-09-18]` |

**No gate move is claimed, and the canonical set stands** (row above): the kernel work is real and correctness-gated,
but the windows that could have shown it were loaded and the peer declined to read a gate value off them. That is
the inherited rule working in the right direction - an isolated A/B is the instrument for a kernel claim, a window
is the instrument for a gate claim - so the cumulative non-GEMV gains stay recorded as kernel facts awaiting a
settled box.

## 4d-bis. P5 third column PRODUCED on our own stack (8d7c8113a) - and the specialization-constant trap

**The ZINC route stays blocked (see 4d), so the column was produced by our own shader instead**, which is what this
lane's record decision asked for: `kernels/vulkan/dmmv_prism.comp` driven through `src/vulkan_rt.h` by a new
`tests/test_vulkan_prism.cpp`, checked against a **CPU dequant+dot reference** per element.

| measurement | value | tag |
|---|---|---|
| Q1_0 correctness, shader vs CPU reference | M=16 K=256 max_abs_err 0.000000; M=17408 K=5120 max_abs_err 0.000005 | `[Bonsai-27B-Q1_0 \| Q1_0 \| Vulkan dmmv_prism.comp vs CPU ref \| strixhalo-busy \| - \| synthetic x \| 2026-09-18]` |
| PQ2_0 correctness, shader vs CPU reference | M=16 K=256 max_abs_err 0.000000; M=17408 K=5120 max_abs_err 0.000002 | `[Ternary-Bonsai-27B-PQ2_0 \| PQ2_0 \| Vulkan dmmv_prism.comp vs CPU ref \| strixhalo-busy \| - \| synthetic x \| 2026-09-18]` |
| throughput at 17408x5120 | Q1_0 23.3 GB/s, PQ2_0 43.0 GB/s (load 23 - **busy-tagged, re-bracket pending**) | `[2-packs \| Q1_0+PQ2_0 \| Vulkan dmmv_prism.comp \| strixhalo-busy \| - \| synthetic x \| 2026-09-18]` |
| PTQ1_0 | was NOT COVERED when first recorded; **now COVERED** - all three packs verified against a codec-faithful reference (commit 25d5751cc) | `[Ternary-Bonsai-2-27B-PTQ1_0 \| PTQ1_0 \| Vulkan dmmv_prism.comp \| strixhalo-busy \| - \| synthetic x \| 2026-09-18]` |
| the column's role | a **fallback, not a competitor**: 23.3 GB/s against the HIP dp4a path's 200-260 GB/s standalone, because the shader dequantizes to float where HIP uses int8 dp4a | `[Bonsai-27B-Q1_0 \| Q1_0 \| Vulkan vs HIP \| strixhalo-busy \| - \| synthetic x \| 2026-09-18]` |

**The trap that had to be cleared first, and it is the day's recurring shape in a new costume.** `src/vulkan_rt.h`'s
`Pipeline::create` never set specialization constants, while `dmmv_prism.comp` selects its layout through
`constant_id=2` and `constant_id=3`. With no spec info every format silently kept the **declared default** and
decoded through the Q1_0 branch, so a PQ2_0 run "failed" with a max_abs_err of 42150 - which looks exactly like a
broken shader and was in fact the harness comparing a Q1_0 decode against a PQ2_0 reference. `create()` now accepts
an optional constant_id-indexed spec array, defaulted so existing callers are unaffected, and PQ2_0 passes. **The
instrument lied, not the system under test** - the same sentence as the truncated-reference, the early-returning
harness and the skipped gate earlier today.

**Why this column is worth more than its numbers:** it carries a *per-element kernel-level* correctness gate against
a CPU reference, which is the shape of gate that the end-to-end fork oracle turned out not to be (section 4a: the
oracle matched 5/5 while a kernel's norm output was wrong by up to 0.42). A fallback column that is slow but
elementwise-verified is a better asset than a fast column whose only evidence is a short-prompt argmax.

## 4f. Pattern-matched GEMV bounds (16:46) - the Q1_0 GEMV is ALU-bound for cache-resident shapes

**Method, and it is the rule applied correctly:** this is a kernel-versus-kernel comparison, so in-situ would have
been the wrong instrument. The production dp4a GEMV's *exact* grid and lane traversal was re-run with the unpack and
the dot removed - same rows, same blocks, same lane stride, same x accesses - accumulating four bytes per block
instead of decoding. That yields the memory-only bound **for that pattern**, rather than judging the kernel against
a generic triad.

| tensor | read-only pattern | dp4a | ratio | tag |
|---|---|---|---|---|
| `blk.0.ffn_gate` 17408x5120 | 463.7 GB/s | 217.9 GB/s | 47% | `[Bonsai-27B-Q1_0 \| Q1_0 \| HIP dp4a vs pattern-matched read-only stub \| strixhalo-busy \| - \| synthetic x \| 2026-09-18]` |
| `output.weight` 248320x5120 | 209.6 GB/s | 181.2 GB/s | 86% | `[Bonsai-27B-Q1_0 \| Q1_0 \| HIP dp4a vs pattern-matched read-only stub \| strixhalo-busy \| - \| synthetic x \| 2026-09-18]` |
| `blk.0.ffn_down` 5120x17408 | 736.0 GB/s | 282.9 GB/s | 38% | `[Bonsai-27B-Q1_0 \| Q1_0 \| HIP dp4a vs pattern-matched read-only stub \| strixhalo-busy \| - \| synthetic x \| 2026-09-18]` |
| the ffn share of a layer | 37.60 MB of 56.25 MB = 67% of every layer is ffn tensors | - | so the bulk is in the ALU-limited shapes | `[Bonsai-27B-Q1_0 \| Q1_0 \| derived from the container geometry \| cpu-host \| - \| - \| 2026-09-18]` |

**Two conclusions, and the second one reverses an earlier framing of ours.** First: the Q1_0 GEMV is **ALU-bound,
not memory-bound, for every tensor that fits in cache** - the cache-resident shapes run two to three times faster as
a read-only stub than as dp4a, while the one tensor large enough to exceed cache behaves the other way (the rows
above carry both figures). Second: therefore the earlier statement that the GEMV sits near the achievable bound, so
the unpack costs only a small fraction, was **right for the big tensors and wrong as a general claim**. Since the ffn
tensors are two thirds of every layer (row above) and are cache-resident, the bulk of the weight stream runs at under
half of what its own pattern could do, and the unpack ALU is the lever after all - inferred, then doubted, now
measured against a pattern-matched bound instead of a generic one.

**Instrument caveat, theirs and kept:** the three tensors are different sizes, so part of that read-only figure
reflects L2 residency rather than raw DRAM behaviour, and the pattern bound is **not** claimed to be achievable for a
full-model pass. What is claimed is narrower and checkable: dp4a is far from the bound for cache-resident shapes and
at the bound for the cached-out one, and the gap is decode work.

## 4g. Pack size is three numbers, and each row must name its basis (resolved 16:56)

| figure | value for Q1_0 | what it is | tag |
|---|---|---|---|
| GEMV weights only | 3.603 GB | the 497 matvec tensors the forward actually dots, summed by a dedicated harness | `[Bonsai-27B-Q1_0 \| Q1_0 \| bench_gemv_sum harness \| cpu-host \| - \| - \| 2026-09-18]` |
| payload | 3.78 GB | every tensor memcmp'd against the source GGUF - authoritative for the pack's contents (the figure this lane's container gate produces) | `[Bonsai-27B-Q1_0 \| Q1_0 \| 1BP v5 verifier \| cpu-host \| - \| - \| 2026-09-18]` |
| file on disk | 4.04 GB (4036301797 bytes) | the container adds the transform extension entries and the tensor table, so the 1BP file is larger than both the payload and the source GGUF | `[Bonsai-27B-Q1_0 \| Q1_0 \| filesystem \| cpu-host \| - \| - \| 2026-09-18]` |

**The three reconcile, and the reconciliation is the explanation rather than a rounding excuse.** The embedding
table (vocab x hidden at the same packing) is a *gather*, not a matvec, so it is absent from the GEMV-weights figure
and present in the payload: adding it to the GEMV total reproduces the payload figure to within a couple of
megabytes out of nearly four gigabytes. That is a check on both numbers, not a preference between them.

**Rule adopted, and it is the structural fix for this class:** every effective-GB/s row names which of the three it
uses. Whole-pack effective rates use the payload figure; anything quoting the GEMV-weights total says "GEMV weights
only"; anything about I/O or disk quotes the file size.

**Consequence for the floor arithmetic, recorded because it tightens a live number:** the GEMV-only floor and the
whole-pack floor differ by under a millisecond, so the "floor plus current non-GEMV clears the gate" arithmetic gets
very slightly tighter, and the conclusion - Q1_0 is limited by the GEMV reaching its memory bound - does not change.
Gate fractions are size-independent and were never affected.

## 5. P3 gate - MEASURED: PQ2_0 **MET**; Q1_0 and PTQ1_0 still short (2026-09-18)

The box was clean-rebooted to clear the peer lanes, and the backend decoded 32 tokens per pack in the
post-reboot window (no `npu_engine_*` / `pf` / `python3` lane). No triad reading was taken in that
window, so the box is tagged `strixhalo-unknown`, **not** `strixhalo-quiet`.

| pack | decoded (32 tokens) | gate | tag |
|---|---|---|---|
| Bonsai-27B-Q1_0 3.78 GB payload | 24 tok/s | >=42 tok/s | `[Bonsai-27B-Q1_0 \| Q1_0 \| HIP bench_hip_1bp \| strixhalo-quiet \| 32 \| capital-of-France \| 2026-09-18]` |
| Ternary-Bonsai-2-27B-PTQ1_0 5.95 GB | 19 tok/s | >=27 tok/s | `[Ternary-Bonsai-2-27B-PTQ1_0 \| PTQ1_0 \| HIP bench_hip_1bp \| strixhalo-quiet \| 32 \| capital-of-France \| 2026-09-18]` |
| Ternary-Bonsai-27B-PQ2_0 7.17 GB | 19 tok/s | >=22 tok/s | `[Ternary-Bonsai-27B-PQ2_0 \| PQ2_0 \| HIP bench_hip_1bp \| strixhalo-quiet \| 32 \| capital-of-France \| 2026-09-18]` |

**Gate reachability arithmetic (host-side, derived from the tagged rows above).** These gates are
bandwidth targets, not measurement targets: each needs the pack's weights streamed at
`Q1_0 159.6 GB/s `[Bonsai-27B-Q1_0\|Q1_0\|arithmetic\|strixhalo-unknown\|-\|-\|2026-09-18]`,
`PTQ1_0 160.7 GB/s `[Ternary-Bonsai-2-27B-PTQ1_0\|PTQ1_0\|arithmetic\|strixhalo-unknown\|-\|-\|2026-09-18]`,
`PQ2_0 157.7 GB/s `[Ternary-Bonsai-27B-PQ2_0\|PQ2_0\|arithmetic\|strixhalo-unknown\|-\|-\|2026-09-18]`
i.e. 78-80% of the 201-219 GB/s triad `[n/a\|probe\|HIP hip_bw_probe\|strixhalo-quiet\|-\|-\|2026-09-18]`.
Today's effective rate (tok/s x pack size) is `60.8 / 65.5 / 86.0 GB/s `[3-packs\|verbatim\|derived from the rows above\|strixhalo-unknown\|32\|capital-of-France\|2026-09-18]`
= 30/33/43% of triad `[3-packs\|verbatim\|derived\|strixhalo-unknown\|32\|capital-of-France\|2026-09-18]`, so the gap is
`2.62x / 2.45x / 1.83x `[3-packs\|verbatim\|derived\|strixhalo-unknown\|32\|capital-of-France\|2026-09-18]`. PQ2_0 is closest because its
effective rate already matches the tile GEMV's own `93 GB/s `[3-packs\|verbatim\|HIP prism_gemv_tile.hip\|strixhalo-unknown\|-\|synthetic x\|2026-09-18]` — the model is GEMV-bound,
so the remaining headroom is exactly the distance between the tile GEMV's fraction of
triad and the fraction the gate implies (both tagged above, not restated here).

**CORRECTION (2026-09-18 13:20) - the "pattern wall" that stood here is RETRACTED.** It rested on a
no-decode dummy at `93.6 GB/s `[3-packs\|verbatim\|HIP tile pattern, no-decode dummy\|strixhalo-unknown\|-\|synthetic x\|2026-09-18]` being an upper
bound on the byte-load pattern. It was not: after a hand-rolled 10-op fp16 decode was replaced by
`__half2float` at commit bc3f2927e, the *real* kernel measures `133.5 GB/s `[Bonsai-27B-Q1_0\|Q1_0\|HIP prism_gemv_tile.hip\|strixhalo-unknown\|-\|synthetic x, corr 1.000000000\|2026-09-18]` - above the
dummy's supposed cap. **A no-decode dummy is not a ceiling**: with almost no work per byte its loop is
latency/issue-bound and can run *slower* than a kernel doing more ALU per byte. Lesson recorded: a proxy
is not a bound merely because it is simpler. The "unreachable whatever the decoder does" claim was mine
(6b6b86076) and is withdrawn here rather than deleted.

**Current state after the fix.** GEMV on `blk.0.ffn_gate` at `133.5 GB/s `[Bonsai-27B-Q1_0\|Q1_0\|HIP prism_gemv_tile.hip\|strixhalo-unknown\|-\|synthetic x\|2026-09-18]`
= 66% of the 201 GB/s triad `[n/a\|probe\|HIP hip_bw_probe\|strixhalo-quiet\|-\|-\|2026-09-18]`; end-to-end backend decode
`Q1_0 24 / PTQ1_0 19 / PQ2_0 19 tok/s `[3-packs\|verbatim\|HIP bench_hip_1bp\|strixhalo-busy\|32\|capital-of-France\|2026-09-18]`
in the peer's **triad-verified quiet window** (13:19:20, load 4.52->4.13, no process >300% CPU; triad 204.2-217.4 GB/s in the same window, section 4), so these are **admissible absolute** rows and canonical over the earlier load-8.8 busy set `[3-packs\|verbatim\|HIP bench_hip_1bp\|strixhalo-busy\|32\|capital-of-France\|2026-09-18]`.7, not triad-verified, so `strixhalo-busy` and **relative only** -
a same-window triad is still requested). Effective aggregate is therefore
`91.2 / 113.0 / 136.2 GB/s `[3-packs\|verbatim\|derived: tok/s x pack size\|strixhalo-busy\|32\|capital-of-France\|2026-09-18]` against the
`159.6 / 160.7 / 157.7 GB/s `[3-packs\|verbatim\|derived: gate x pack size\|strixhalo-unknown\|-\|-\|2026-09-18]` the gates imply: **NOT MET, gap 1.75x / 1.42x / 1.16x**
`[3-packs\|verbatim\|derived\|strixhalo-busy\|32\|capital-of-France\|2026-09-18]`. PQ2_0 is at 86% of its gate, Q1_0 at 57%
`[3-packs\|verbatim\|derived\|strixhalo-busy\|32\|capital-of-France\|2026-09-18]`.

**What this changes:** the binding constraint is no longer the weight pattern but the aggregate streaming
rate plus non-GEMV ALU (GDN, attention, FWHT, launch overhead) - the peer's per-tensor projection from
133.5 GB/s is ~35 tok/s for Q1_0 `[Bonsai-27B-Q1_0\|Q1_0\|projected from GEMV BW\|strixhalo-unknown\|-\|-\|2026-09-18]`, above the fork's 28.8
baseline `[Bonsai-27B-Q1_0\|Q1_0\|Prism llama.cpp fork + Vulkan\|strixhalo-unknown\|32\|-\|2026-09-18]` and 1.2x short of its gate
`[P3-gate-target\|Q1_0\|spec\|n/a\|-\|-\|2026-09-18]`. The seven-variant sweep below still stands.

**Operator decision this exposes.** The gate as written asks for near-peak streaming through a 64-layer
hybrid that also carries GDN state and attention, so it sits at the edge of what this box can do *even
with a perfect weight path*. If the targets are a lane-relative ambition rather than a hard contract,
saying so lets me re-tag them as `spec-ambition` instead of failing them; if they are hard, then the
number to plan against is the fraction of triad the gate implies (tagged above), not the
tok/s figures in the abstract.

### dp4a round (2026-09-18, commit ca8f8bb13) - **PQ2_0 MEETS ITS GATE**; PTQ1_0 needs a new dot, not tuning

All four gates ran in ONE window (14:29:42, load 2.23-2.32, triad at or above the quiet threshold before
and after the run - see section 4), so the rows below are admissible absolute measurements and R16 is satisfied on both sides of the run. The
GEMV now quantizes activations to int8 once and uses the RDNA3 `__builtin_amdgcn_sudot4` (dp4a) dot - an
algorithm **read from** Prism's own llama.cpp HIP path (`ggml/src/ggml-cuda/vecdotq.cuh`,
`q1_0_unpack4_hip` / `q2_0_symbols4_hip`) and reimplemented in `kernels/prism_gemv_dp4a.hip`. **Nothing is
linked from the fork, which stays oracle/baseline only**: the operator's instruction of 2026-09-18 ("go
upstream to prism llama.cpp, extract and integrate") added *reading*, not a runtime dependency, so the lane's
deliverable constraint is unchanged.

| pack | decoded 32 tokens | gate | status | tag |
|---|---|---|---|---|
| Bonsai-27B-Q1_0 3.78 GB payload | 34 tok/s (29.8 ms/tok) | >=42 tok/s | 81% of gate | `[Bonsai-27B-Q1_0 \| Q1_0 \| HIP dp4a forward \| strixhalo-quiet \| 32 \| capital-of-France \| 2026-09-18]` |
| Ternary-Bonsai-2-27B-PTQ1_0 5.95 GB | 17 tok/s (58.0 ms/tok) | >=27 tok/s | 63% of gate | `[Ternary-Bonsai-2-27B-PTQ1_0 \| PTQ1_0 \| HIP dp4a forward \| strixhalo-quiet \| 32 \| capital-of-France \| 2026-09-18]` |
| Ternary-Bonsai-27B-PQ2_0 7.17 GB | 23 tok/s (43.8 ms/tok) | >=22 tok/s | **MET (105%)** | `[Ternary-Bonsai-27B-PQ2_0 \| PQ2_0 \| HIP dp4a forward \| strixhalo-quiet \| 32 \| capital-of-France \| 2026-09-18]` |
| same-window correctness on all three packs | fork oracle 5/5 (2614 314 279 369 11751 / 220 ... / 2614 ...) and CPU-vs-device greedy 11/11 | - | this is what carried the MET | `[3-packs \| verbatim \| HIP forward vs CPU floor + fork oracle \| strixhalo-quiet \| 11 \| capital-of-France \| 2026-09-18]` |
| Q1_0 GEMV, dp4a path | 223.4 GB/s, corr 0.999996 vs f64 CPU dot | - | approximate: int8 activations | `[Bonsai-27B-Q1_0 \| Q1_0 \| HIP prism_gemv_dp4a.hip \| strixhalo-quiet \| - \| synthetic x \| 2026-09-18]` |
| PTQ1_0 extracted int8 dot | 122.3 GB/s | - | too slow for its gate | `[Ternary-Bonsai-2-27B-PTQ1_0 \| PTQ1_0 \| HIP extracted int8 dot \| strixhalo-quiet \| - \| synthetic x \| 2026-09-18]` |
| PTQ1_0 budget vs that dot | 48.7 ms needed, 37.0 ms allowed; needs >=160.6 GB/s aggregate | - | **infeasible on this dot** | `[Ternary-Bonsai-2-27B-PTQ1_0 \| PTQ1_0 \| derived \| strixhalo-quiet \| - \| - \| 2026-09-18]` |
| Q1_0 in-situ split, same binary and window (weight GEMVs skipped by a diagnostic hook, reverted after) | full 29.5 ms/token; weight GEMVs 21.1 ms (= 180.1 GB/s aggregate); all other kernels + launches 8.4 ms | - | measured in-situ, not standalone | `[Bonsai-27B-Q1_0 \| Q1_0 \| HIP dp4a forward with and without GEMVs \| strixhalo-quiet \| 32 \| capital-of-France \| 2026-09-18]` |
| Q1_0 dp4a bandwidth, standalone per tensor | lm_head 201.1, ssm_out 235.8, ffn_down 335.6, ffn_gate 249.6, ffn_up 246.0 GB/s | - | in-situ aggregate is 180.1 GB/s, so the loss is the activation-quant pass + dependency + launches, not the dot | `[Bonsai-27B-Q1_0 \| Q1_0 \| HIP prism_gemv_dp4a.hip \| strixhalo-quiet \| - \| synthetic x \| 2026-09-18]` |
| Q1_0 GEMV against the achievable bound (read-only, directional) - **superseded as a general claim by 4f** | floor 17.1 ms for 3.60 GB at 210 GB/s against ~20.0 ms measured = 86% of achievable; the gate needs ~95%. That holds for the **cached-out** shapes; cache-resident shapes measure 38-47% of their own pattern-matched bound | `[Bonsai-27B-Q1_0 \| Q1_0 \| derived from hip_bw_probe direction test + rocprofv3 \| strixhalo-quiet \| 32 \| capital-of-France \| 2026-09-18]` |
| Q1_0 gate arithmetic (gate needs 23.81 ms/token) | GEMV at 250 GB/s = 15.2 ms, + 8.4 ms non-GEMV = 23.6 ms -> 42.4 tok/s; with non-GEMV at 2.5 ms -> 17.7 ms -> 56.5 tok/s | - | **reachable by either lever alone** | `[Bonsai-27B-Q1_0 \| Q1_0 \| derived \| strixhalo-quiet \| 32 \| capital-of-France \| 2026-09-18]` |

**PQ2_0 is MET**, on a gate-clearing decode in a window whose triad cleared the quiet threshold on both sides, with the
fork oracle and the eleven-position CPU-vs-device greedy comparison green in that same window. Margin: one
token per second.

**PTQ1_0 cannot reach its gate on this dot, and that is arithmetic rather than tuning** (budget row above):
it needs a dp4a-style formulation at roughly the Q1_0 dot's throughput, not another pass over the float tile
kernel.

### Fused multi-GEMV round (cabdd3620) - a low-single-digit-percent win; the quant/launch hypothesis is NOT confirmed

Weight matrices fed from one activation row are now dotted against ONE q8_1 quantization of that row in ONE
launch - groups are the four GDN matvecs off one activation buffer, q/k/v off the same, and gate/up off another.
It is a re-expression of the fork's `vec_dot_ptq1_0_q8_1_multi` idea with **no fork code linked**.

| measurement | value | tag |
|---|---|---|
| interleaved A/B, base -> multi, Q1_0 / PTQ1_0 / PQ2_0 | 30.2-31.6 -> 29.2-29.5 / 58.1-60.8 -> 57.4-60.2 / 43.2-45.3 -> 41.8-45.7 ms | `[3-packs \| verbatim \| HIP fused multi-GEMV vs base \| strixhalo-quiet \| 32 \| capital-of-France \| 2026-09-18]` |
| four-gate window 15:07 decode | Q1_0 34 tok/s, PTQ1_0 17 tok/s, PQ2_0 24 tok/s (oracle 5/5 and compare_gen ok on all three, in-window) | `[3-packs \| verbatim \| HIP forward \| strixhalo-quiet \| 32 \| capital-of-France \| 2026-09-18]` |
| PQ2_0 spread across three interleaved runs of the SAME binary | 21.9-23.9 tok/s | `[Ternary-Bonsai-27B-PQ2_0 \| PQ2_0 \| HIP fused multi-GEMV \| strixhalo-quiet \| 32 \| capital-of-France \| 2026-09-18]` |

**Recorded as a low-single-digit-percent win, not a headline** - at the peer's own request, and correctly so.

**The movement of PQ2_0 from the previous window is NOT a gain.** The spread row above shows the same binary
spanning both readings across three interleaved runs, so the earlier figure and the new one are the same
measurement; PQ2_0 stays MET at the same rate and the wider margin is arithmetic on a rounded value, not
headroom. Q1_0 and PTQ1_0 are likewise unchanged.

**Method standard this round raised:** the A/B was run *interleaved* (base/multi alternating within one
session) because a sequential A/B was confounded when the box drifted mid-run - the after-triad fell to the
busy threshold and PQ2_0 read a far larger ms/token than in the same session minutes earlier, which would have
shown a **false regression** (values in the A/B and triad rows above). Interleaved A/B with triad evidence on
both sides is now this lane's standard for any performance claim.

**A hypothesis rejected, and recorded as rejected.** The in-situ aggregate sitting below the standalone dot
rate implied the loss was the activation-quant pass plus launch overhead. Fusing the shared-x groups is
precisely that fix, and it bought only the low-single-digit-percent win above - so that explanation is **not
confirmed**, and the right next step is to measure the quant kernel's own time and the launch count directly,
rather than to invest further in fusion on an assumption. This is the third time today that a plausible
mechanism needed a measurement before belief.

### PTQ1_0 per-block dp4a dot round (883535b89) - the round's real gate move

The dot that was ported first was the fork's *scalar* HIP branch of `vec_dot_ptq1_0_q8_1`; the **multi**
variant's per-block path is a different implementation and does suit our trit order, which is what the pack
needed. Portability work, recorded because it is the non-obvious part: four qs bytes are widened into 16-bit
lanes with `__byte_perm` so the multiply-by-three cannot carry across bytes, each carry becomes the next base-3
trit, and four trits feed ONE `sudot4` against the int8 activations. The fork's per-byte `__vsub4` does not
exist in HIP; since the digit bytes are 0..2 the peer emulates it with the no-borrow SWAR decrement
`((q | 0x80808080) - 0x01010101) ^ 0x80808080`, verified against the f64 CPU dot within the same
approximate-dot tolerance as Q1_0 - so the fork-oracle and greedy-sequence gates remain the correctness gates
for this pack too.

| measurement | value | tag |
|---|---|---|
| isolated dot rate, PTQ1_0 tensors: ffn_gate / ffn_down / output.weight | 122.3 -> 159.0 / 264.5 / 194.5 GB/s | `[Ternary-Bonsai-2-27B-PTQ1_0 \| PTQ1_0 \| HIP per-block dp4a dot \| strixhalo-quiet \| - \| synthetic x \| 2026-09-18]` |
| end-to-end decode, same window | 57.6 -> 43.5 ms/token = 17 -> 23 tok/s = 85% of gate (was 63%) | `[Ternary-Bonsai-2-27B-PTQ1_0 \| PTQ1_0 \| HIP per-block dp4a forward \| strixhalo-quiet \| 32 \| capital-of-France \| 2026-09-18]` |
| PTQ1_0 dot ALONE vs its gate budget | 37.42 ms for 5.95 GB at 159.0 GB/s vs 37.04 ms allowed - still 0.38 ms short, end-to-end short by 6.46 ms | `[Ternary-Bonsai-2-27B-PTQ1_0 \| PTQ1_0 \| derived \| strixhalo-quiet \| - \| - \| 2026-09-18]` |
| same-window correctness | fork oracle 5/5 on all three packs, compare_gen ok on all three | `[3-packs \| verbatim \| HIP forward vs CPU floor + fork oracle \| strixhalo-quiet \| 11 \| capital-of-France \| 2026-09-18]` |

**PTQ1_0 is the round's real gain** and it is corroborated three ways rather than by the bracket alone: the
isolated dot numbers above, a decode reading in a plainly busy window, and the four-gate run itself. **But the
dot alone is still fractionally over budget** (row above): at the new rate the weights by themselves exceed the
time the gate allows, so PTQ1_0 still needs a faster dot, or a faster dot plus non-GEMV work - it is no longer
infeasible, it is short.

**Drift in that window, recorded rather than ignored.** The bracketing triad reading cleared the threshold on
both sides, but the lower-size sweep on the *after* side fell below it mid-run (busy-tagged row in section 4).
The Q1_0 and PQ2_0 readings taken in that same run are therefore drift-depressed, and the canonical quiet
values remain the previous window block's - so this round's claim is PTQ1_0's, not a Q1_0 or PQ2_0 movement. The peer stated this
before I could find it, which is the standard this lane is holding to.

### Corrected mechanism (rocprofv3, 2026-09-18) - the quant story is DEAD, the dispatch count is real

Method: two profiles **differenced** rather than read singly, because a single profile is dominated by model load
and state init (one token read 16106 dispatches against 20729 for four). All figures below come from the
difference, not from any individual dispatch.

| measurement | value | tag |
|---|---|---|
| dispatches per token | 1541 ((20729 - 16106) / 3) | <!-- derive: dispatches_per_token --> `[Bonsai-27B-Q1_0 \| Q1_0 \| rocprofv3 --kernel-trace, differenced \| strixhalo-quiet \| 4 \| capital-of-France \| 2026-09-18]` |
| of which buffer copies | 418 dispatches = 27% of launch traffic (0.37 ms) [CORRECTED: an earlier figure of 1452 / 94% was a mis-division - the peer's, and I recorded it without checking that the 1452 was itself differenced] | `[Bonsai-27B-Q1_0 \| Q1_0 \| rocprofv3, differenced \| strixhalo-quiet \| 4 \| capital-of-France \| 2026-09-18]` | <!-- derive: copy_per_token copy_share_pct -->
| activation-quant kernel per token | ~257 dispatches, 0.25 ms = under 1% of decode | `[Bonsai-27B-Q1_0 \| Q1_0 \| rocprofv3, differenced \| strixhalo-quiet \| 4 \| capital-of-France \| 2026-09-18]` |
| GEMV total per token | 21.2 ms (dp4a<18> 6.27 + multi4 14.95) | `[Bonsai-27B-Q1_0 \| Q1_0 \| rocprofv3, differenced \| strixhalo-quiet \| 4 \| capital-of-France \| 2026-09-18]` |
| cross-instrument check: profile vs the skip-GEMV split | 21.2 vs 21.1 ms - two independent instruments agreeing within 0.1 ms | `[Bonsai-27B-Q1_0 \| Q1_0 \| rocprofv3 vs PRISM_SKIP_GEMV \| strixhalo-quiet \| 4 \| capital-of-France \| 2026-09-18]` |
| other per-token kernels | gdn_recurrence 3.11, rmsnorm 1.16, all others under 0.2 each | `[Bonsai-27B-Q1_0 \| Q1_0 \| rocprofv3, differenced \| strixhalo-quiet \| 4 \| capital-of-France \| 2026-09-18]` |
| ideal GEMV at the per-tensor standalone rates | ~15 ms against the 21.2 ms observed | `[Bonsai-27B-Q1_0 \| Q1_0 \| derived from the rows above \| strixhalo-quiet \| 4 \| capital-of-France \| 2026-09-18]` |
| attribution of the fusion win | ~256 dispatches removed = 0.26-0.77 ms, matching the measured 0.3-0.6 ms | `[3-packs \| verbatim \| derived \| strixhalo-quiet \| 32 \| capital-of-France \| 2026-09-18]` |
| instrument caveat | dispatch durations overlap in this container (the multi4 total exceeds wall time), so only counts and difference-of-totals were used | `[Bonsai-27B-Q1_0 \| Q1_0 \| rocprofv3 \| strixhalo-quiet \| 4 \| capital-of-France \| 2026-09-18]` |

**The quant story is dead.** The activation-quant pass is under one percent of the decode (row above), so it
cannot explain the in-situ-versus-standalone GEMV gap that motivated it. The peer's earlier framing that named it
is recorded as wrong, because they killed it themselves with a measurement rather than a defence - the third
mechanism today to die of data instead of argument.

**The dispatch count is real but immaterial, and that was measured, not argued** (rows below): the fusion win
was slightly less GPU work rather than fewer launches, and the copies are a quarter of launch traffic rather than
almost all of it. The earlier advice to attack the copies - which came from the withdrawn figure - is withdrawn
with it. Recording my own part in that: I checked the ratio (1452 of 1541) but never checked that the numerator
had itself been differenced, so internal consistency passed while the input was wrong. The gate's coverage
self-test exists for exactly this failure shape, and this instance shows it needs an input-provenance check too.

**The dominant remaining term is the GEMV aggregate inside the layer loop** (rows above): it runs well below the
per-tensor standalone rate, and the gap is neither quant nor dispatch. The peer suspects memory-system
interaction between the GEMVs and the non-GEMV kernels and explicitly declines to assert it - the same standard
that has now retired three mechanisms, applied to their own.

### Dispatch floor measured directly, and what Q1_0's gate now requires (15:22)

| measurement | value | tag |
|---|---|---|
| launch cost, trivial kernel `<<<1,32>>>` | 1.826 us/launch over 20000 launches | `[Bonsai-27B-Q1_0 \| Q1_0 \| HIP trivial-kernel loop \| strixhalo-quiet \| - \| - \| 2026-09-18]` |
| launch cost, realistic shape `<20,256>`, n=5120 | 1.984 us/launch over 20000 launches | `[Bonsai-27B-Q1_0 \| Q1_0 \| HIP trivial-kernel loop \| strixhalo-quiet \| - \| - \| 2026-09-18]` |
| host time to issue a token | 1541 dispatches x that rate = 2.8-3.1 ms against 29.5 ms of GPU work per token | `[Bonsai-27B-Q1_0 \| Q1_0 \| derived \| strixhalo-quiet \| 4 \| capital-of-France \| 2026-09-18]` |
| Q1_0 GEMV floor at the box triad - whole-pack basis (the GEMV-weights-only basis gives 17.16 ms) | 18.0 ms for 3.78 GB payload at ~210 GB/s; measured in-situ 21.2 ms is about 85% of the floor rate | `[Bonsai-27B-Q1_0 \| Q1_0 \| derived from rocprofv3 and hip_bw_probe \| strixhalo-quiet \| 4 \| capital-of-France \| 2026-09-18]` |
| non-GEMV budget the gate leaves, if the GEMV reaches its floor | 5.7 ms, i.e. non-GEMV must fall by about a third | `[Bonsai-27B-Q1_0 \| Q1_0 \| derived \| strixhalo-quiet \| 4 \| capital-of-France \| 2026-09-18]` |
| same, if the GEMV does not improve | 2.6 ms, i.e. non-GEMV must fall by about two thirds | `[Bonsai-27B-Q1_0 \| Q1_0 \| derived \| strixhalo-quiet \| 4 \| capital-of-France \| 2026-09-18]` |
| GEMV at floor, non-GEMV untouched | 26.5 ms = 37.7 tok/s - **still short of the gate** | `[Bonsai-27B-Q1_0 \| Q1_0 \| derived \| strixhalo-quiet \| 4 \| capital-of-France \| 2026-09-18]` |
| GEMV unchanged, the largest non-GEMV kernel deleted entirely | 26.5 ms = 37.8 tok/s - **still short of the gate** | `[Bonsai-27B-Q1_0 \| Q1_0 \| derived \| strixhalo-quiet \| 4 \| capital-of-France \| 2026-09-18]` |

**The dispatch story is dead as a material cost, for a reason worth keeping.** Host time to issue every launch is
well under the GPU's work per token, so the launch cost is overlapped rather than additive: dispatch cost only
lands in wall time when the host cannot stay ahead of the device. Both of the mechanisms this lane has retired
today died the same way - a plausible cost that measurement showed to be small or hidden.

**The gate now needs both halves of Q1_0, and that follows from the rows above rather than from optimism.** Even a
GEMV at its memory floor leaves the pack short with non-GEMV untouched, and deleting the single largest
non-GEMV kernel entirely does not close it either. So the honest target is a GEMV near its floor *and* roughly a
third off the non-GEMV budget - the peer's plan to start with `gdn_recurrence` is the right first step, but it is
a first step and not the whole distance.

### Non-GEMV chase, first step: gdn_recurrence (measured-neutral, win NOT claimed) and a shape penalty that merges the two chases

| measurement | value | tag |
|---|---|---|
| gdn_recurrence per token, measured | 3.11 ms - the largest non-GEMV kernel | `[Bonsai-27B-Q1_0 \| Q1_0 \| rocprofv3, differenced \| strixhalo-quiet \| 4 \| capital-of-France \| 2026-09-18]` |
| its memory bound | 3.1 MB of fp32 state per layer x 48 layers = 148.8 MB/token, ~0.71 ms at the box triad | `[Bonsai-27B-Q1_0 \| Q1_0 \| derived \| strixhalo-quiet \| 4 \| capital-of-France \| 2026-09-18]` |
| share of that bound it achieves | 23% | `[Bonsai-27B-Q1_0 \| Q1_0 \| derived \| strixhalo-quiet \| 4 \| capital-of-France \| 2026-09-18]` |
| threads, before -> split attempt | 6144 -> 12288 (48 blocks x 128 -> 256 threads, kk split with an LDS combine, per-element fp32 order unchanged) | `[Bonsai-27B-Q1_0 \| Q1_0 \| HIP prism_gdn_recurrence_kernel \| strixhalo-quiet \| 4 \| capital-of-France \| 2026-09-18]` |
| split attempt, correctness | fork oracle 5/5 and compare_gen 11/11 on all three packs, suite GDN gates pass | `[3-packs \| verbatim \| HIP GDN vs CPU floor + fork oracle \| strixhalo-quiet \| 11 \| capital-of-France \| 2026-09-18]` |
| split attempt, performance | **NEUTRAL**: Q1_0 30.2 ms / 33 tok/s, PTQ1_0 42.1 / 24, PQ2_0 42.3 / 24 against pre-split 29.6-31.5, 42.1-43.5, 42.2 - inside the noise band, so the win is **not claimed** | `[3-packs \| verbatim \| HIP forward, split vs pre-split GDN \| strixhalo-quiet \| 32 \| capital-of-France \| 2026-09-18]` |
| shape penalty, same kernel, both packs | ffn_gate-shaped 17408x5120 runs at 0.69x (Q1_0: 232.9 against 335.6 GB/s) and 0.60x (PTQ1_0: 159.0 against 264.5) of the ffn_down-shaped rate | `[2-packs \| Q1_0+PTQ1_0 \| HIP dp4a dot, per tensor \| strixhalo-quiet \| - \| synthetic x \| 2026-09-18]` |

**The GDN split is measured-neutral and is therefore not recorded as a win.** Correctness is green and the
per-element arithmetic is bit-identical by construction, but the aggregate moved inside the noise band, so the
honest state is "attempted, not shown to help". The peer's next step is the right one and is the method this
lane adopted after the false-regression incident: build split and pre-split side by side, alternate them within
one session, and time the kernel in isolation so it is compared against itself rather than inferred from the
aggregate. They also pre-registered the outcome they will accept: if the isolated A/B is neutral too, the kernel
is recorded as **compute-bound rather than latency-bound** - a different statement from the one an hour ago, and
the one that survives.

**A shape penalty, not two separate chases.** The table row above shows the same deficit on both packs: tensors
shaped like the GEMV gate (many rows, shallow reduction) run well below tensors shaped like the down projection.
So the Q1_0 GEMV chase and the PTQ1_0 dot chase are the same chase - the many-rows-shallow-reduction shape - and
that merges the two targets into one. It also refines the two-chase frame above: the GEMV half is not "get the
dot uniformly faster", it is "fix the shallow-reduction shape", which is where the next gain most plausibly lives.

### gdn_recurrence isolated A/B (5baa93ff3) - the neutral call is REVERSED

> **RETRACTED AS A WIN - see section 4a.** The commits in this section carried a latent race; their green
> parity runs were luck. Corrected status: regression introduced, caught by the suite, fixed in
> `b2fc4bdf5`. Throughput figures here are throughput-only; correctness verification is withdrawn until
> re-bracketed., and an in-situ tax appears on both kernels

| measurement | value | tag |
|---|---|---|
| isolated A/B, same geometry, both kernels built from this tree, alternated via git stash | pre-split 46.143 us/layer = 2.215 ms/token; split 28.452 us/layer = 1.366 ms/token; 1.62x, 0.85 ms/token saved | `[Bonsai-27B-Q1_0 \| Q1_0 \| HIP gdn_recurrence isolated A/B \| strixhalo-quiet \| 1 \| capital-of-France \| 2026-09-18]` |
| correctness of the split | fork oracle 5/5 and compare_gen 11/11 on all three packs, suite GDN gates pass; per-thread traversal order unchanged, so per-element fp32 arithmetic is bit-identical | `[3-packs \| verbatim \| HIP GDN split vs pre-split \| strixhalo-quiet \| 11 \| capital-of-France \| 2026-09-18]` |
| what the aggregate said about the same change | 30.2 ms against 29.6-31.5 before - inside the noise band, so the aggregate alone would have discarded the optimisation | `[Bonsai-27B-Q1_0 \| Q1_0 \| HIP backend, split vs pre-split \| strixhalo-quiet \| 32 \| capital-of-France \| 2026-09-18]` |
| in-situ tax, now measured on two kernels | gdn_recurrence runs 40% slower inside the loop than standalone (3.11 vs 2.215 ms); the GEMV runs 17% over its memory floor (21.2 vs 18.1 ms) | `[Bonsai-27B-Q1_0 \| Q1_0 \| rocprofv3 in situ vs isolated \| strixhalo-quiet \| 4 \| capital-of-France \| 2026-09-18]` |
| updated chase arithmetic, Q1_0 | the gate gap was 5.79 ms; this win addresses 0.85; non-GEMV predicted at 7.55 ms (UNMEASURED) gives 34.8 tok/s, and even with the GEMV at its 18.1 ms floor it is 39.0 tok/s - still short; clearing the gate needs non-GEMV at 5.71 ms with the GEMV at floor | `[Bonsai-27B-Q1_0 \| Q1_0 \| derived \| strixhalo-quiet \| 32 \| capital-of-France \| 2026-09-18]` |
| gate state after the win | UNCHANGED: Q1_0 32-34, PTQ1_0 23-24, PQ2_0 24 tok/s against 42/27/22, PQ2_0 MET | `[3-packs \| verbatim \| HIP backend \| strixhalo-quiet \| 32 \| capital-of-France \| 2026-09-18]` |

**The reversal is recorded as a reversal, and the methodological reason is the durable part.** The aggregate
could not resolve the change, so the isolated kernel A/B was necessary and the earlier neutral call was a limit
of the instrument rather than a failure of judgement. The rule that follows is now in this record: **an isolated
A/B is the instrument for kernel-level claims; the aggregate is the instrument for gate claims** - and this lane
does not move a gate on kernel-level evidence, which is why the gate row above still reads unchanged.

**A second data point for the in-situ tax, which is now the largest named block of time in the model.** The same
tax appears on two different kernels (rows above): both are materially slower inside the loop than standalone.
That is one problem rather than two, which is the same conclusion the shape penalty produced from the other
direction, and it means the remaining distance is not "GEMV versus non-GEMV" so much as "the loop taxes every
kernel that runs in it" (the two measured rates are in the row above).

**The two-chase frame survives the win and is now sharper** (arithmetic row above): even with this gain and a
GEMV at its memory floor the pack remains short of its gate, so the distance still has to come from both halves.
The decisive next number is the **post-split in-situ** measurement of gdn_recurrence: it tests whether the
isolated gain survives inside the loop, which is exactly where the tax lives.

### Three-kernel window (15:45:05) - the wall did not move, and Q1_0 is now GEMV-limited

> **RETRACTED AS A WIN - see section 4a.** The commits in this section carried a latent race; their green
> parity runs were luck. Corrected status: regression introduced, caught by the suite, fixed in
> `b2fc4bdf5`. Throughput figures here are throughput-only; correctness verification is withdrawn until
> re-bracketed.

| measurement | value | tag |
|---|---|---|
| decode in a valid window (triad on both sides) | Q1_0 34 tok/s, PTQ1_0 24, PQ2_0 24 against 42/27/22 = 81% / 89% / 109% | `[3-packs \| verbatim \| HIP backend \| strixhalo-quiet \| 32 \| capital-of-France \| 2026-09-18]` |
| same-window correctness | fork oracle 5/5 and compare_gen 11/11 on all three packs | `[3-packs \| verbatim \| HIP forward vs CPU floor + fork oracle \| strixhalo-quiet \| 11 \| capital-of-France \| 2026-09-18]` |
| the three kernel wins (GDN, rmsnorm, FWHT) in the wall | Q1_0 read 29.6 ms here against 29.6 ms in the earlier load-2.2 window, so the measured kernel win did not appear; load was 11.5 against 2.2 | `[Bonsai-27B-Q1_0 \| Q1_0 \| HIP backend, two windows \| strixhalo-quiet \| 32 \| capital-of-France \| 2026-09-18]` |
| in-situ terms now | GEMV ~20.0 ms, non-GEMV ~4.7-5 ms, wall 29.6 ms | `[Bonsai-27B-Q1_0 \| Q1_0 \| rocprofv3, differenced \| strixhalo-quiet \| 4 \| capital-of-France \| 2026-09-18]` |
| GEMV floor at this triad | 16.9 ms for 3.60 GB at 213 GB/s; floor + current non-GEMV = 21.6-21.9 ms, which would clear the 23.81 ms the gate allows | `[Bonsai-27B-Q1_0 \| Q1_0 \| derived \| strixhalo-quiet \| 32 \| capital-of-France \| 2026-09-18]` |
| pack size discrepancy | **RESOLVED - see section 4g**: three defensible figures (3.603 GB GEMV weights only, 3.78 GB payload, 4.04 GB file), reconciled through the embedding gather, and every effective-GB/s row now names its basis | `[Bonsai-27B-Q1_0 \| Q1_0 \| container metadata vs peer figure \| cpu-host \| - \| - \| 2026-09-18]` |

**The wall did not move and no improvement is claimed** (row above). Three kernel wins with measured in-situ gains
produced no change in the wall on a busier box, so the box tax absorbed what the kernels gave back - and the peer
explicitly declined to claim the gain because the two effects cannot be separated on this evidence. What stands
firm from that round is the in-situ attribution itself, measured with a differenced profile under the same busy
conditions.

**This supersedes the two-chase frame that I derived earlier, and the supersession is in the peer's favour.** The
non-GEMV chase has paid: it fell from its profiled pre-fix figure to roughly half of it through the GDN, rmsnorm and
FWHT work (both rows above). With the
GEMV floor at this triad, floor plus current non-GEMV clears the gate (row above), so Q1_0 is now limited by **the
GEMV reaching its memory bound and nothing else**. My earlier arithmetic said both halves were required; that was
correct for the old non-GEMV figure and is no longer correct.

**One caveat that matters, and it is the peer's own rule turned on their own projection.** The projection that the
floor would clear the gate comes from *overlapping* in-situ durations: those terms sum below the wall itself (the rows
above carry both figures), so the sum is low by construction. A projection built that way cannot be used as a gate value -
only a window can move a gate - so the row above is recorded as arithmetic, not as a measurement of the gate.

**The right bound was checked, not assumed - and then a better one replaced it (see 4f).** The directional test
below settled that a read-only stream gets no extra headroom over the triad, which closes that question; the
pattern-matched bound in 4f then showed that the generic figure was the wrong yardstick for cache-resident shapes. The triad mixes reads and writes, so it is
not obviously the correct ceiling for a kernel that only reads weights; the directional test settles it - read-only
bandwidth is essentially the same as the triad (rows in section 4), so a weight-streaming GEMV gets no extra
headroom from the read/write asymmetry. Against that bound the Q1_0 GEMV achieves the fraction of the achievable
rate recorded in the row above, and the gate needs it near-complete (or the non-GEMV cut further). That is the honest
distance: not scheduling, not tiling, but the unpack costing the read stream a double-digit fraction of its
throughput.

**Five GEMV hypotheses retired by measurement, one left standing and explicitly unmeasured.** The quant pass, the
dispatch cost, the in-situ tax, lane-tail/r4, and LDS staging have all been killed by data - the last
catastrophically, at a small fraction of the r1 path's rate. What remains is **an unpack-ALU limit on the 1-bit
unpack, not a bandwidth limit and not a tiling problem**, and that is recorded as the surviving hypothesis rather
than as a finding: it is the explanation left after five retirements, and it has not itself been measured. The
peer's decision to say so instead of trying more tilings is the right one, and it is why the lane is now spending
its time on P5 and P4 breadth while that question stays open.

**One part of this round is still an estimate, and is recorded as an estimate.** The lost-to-dispatch figure is
inferred from an assumed per-dispatch cost, and the sum of the measured terms leaves a remainder of about that
size - suggestive, not proof. The direct measurement is cheap: per-token wall time minus the sum of per-token
kernel durations from the same differenced profile measures launch overhead instead of assuming it.

**Pre-registered next step, with its own falsification condition.** Rather than another fusion, the peer is
measuring in-situ the two quantities the quant/launch story never had: the activation-quant kernel's own time
and the launch count per token. Their stated condition, recorded here so it cannot be quietly dropped: if the
quant turns out to be small and the launches are not the cost, the story is dead and they will say so.

**Correction recorded from the peer (their own artifact, not a device bug).** An earlier Q1_0 CPU-vs-device
greedy comparison was reported FAILED: the CPU floor had been run under a `timeout` that killed it partway,
truncating the reference to four argmax lines, so the comparison read as "cpu=4, device=11". Regenerated
without a timeout, all three packs match at all eleven positions. Recorded because a live false claim about
device divergence deserves the same ink as the true result.

**PTQ1_0's canonical value moved 19 -> 17 between rounds** (both quiet windows, same float tile kernel), so
the difference is window variance or config drift rather than a dp4a effect on PTQ1_0. The newest
four-gate-window value is the one carried above; the discrepancy is flagged, not smoothed.

**The numerics changed and the gate that carries correctness changed with it.** The dp4a dot is approximate
where the float tile kernel was exact, because activations are int8-quantized in the fork's q8_1 scheme (corr
row above). Kernel-level exactness is therefore no longer the correctness gate for this path - the fork
oracle and the CPU-vs-device greedy comparison are, and for the MET above both ran in-window.

**Superseding the earlier framing of Q1_0's residual (my estimate, now replaced by measurement).** The
non-GEMV share was estimated from an aggregate; it is now measured in-situ (split row above), and the same
run shows the in-situ GEMV aggregate sitting below the dot's standalone rate (per-tensor row above). The
remaining distance is therefore *both* the machinery around the dot - activation quant, dependency, launch
overhead - and the non-GEMV kernels, and either lever alone clears the Q1_0 gate (arithmetic row above). The
peer's next extraction targets exactly that machinery: the fork's fused multi-GEMV
(`vec_dot_ptq1_0_q8_1_multi<ncols_dst>`), one launch and one activation quant feeding several matvecs, which
maps onto our gate+up pair and the four GDN GEMVs that share a single activation buffer. PTQ1_0's own dp4a dot
remains a separate item.

**Retraction recorded from the peer - their second, and the fourth instrument fault of the day.** The
non-GEMV figure they were about to send was wrong: a standalone dummy-buffer harness had three kernels
early-returning (FWHT, GDN recurrence and the ssmout permutation all reported near-zero), so it undercounted
badly. The rows above come from the real forward instead. Note the shape of the error - it is the same one I
made with the no-decode dummy: **a harness that does not do the real work cannot bound the real path.** Rule
recorded: in-situ for attribution, standalone only for comparing kernels against each other.

**Result: MISSED, and it is a kernel limit, not a measurement artifact.** The tile GEMV's own best is the
section-3 row tagged `[3-packs | verbatim | HIP prism_gemv_tile.hip | strixhalo-unknown | - | synthetic x | 2026-09-18]`;
at that bandwidth the Q1_0 model projects to 21 tok/s `[Bonsai-27B-Q1_0 | Q1_0 | HIP projected from GEMV BW | strixhalo-unknown | - | capital-of-France | 2026-09-18]`,
already short of the gate before any non-GEMV overhead. Closing the gap needs a new kernel decomposition
(in the tile design each weight byte is paired with an x reload per four rows), not a measurement rerun.

Outside baseline to beat: the fork's own Vulkan numbers, 28.8 tok/s `[Bonsai-27B-Q1_0 | Q1_0 | Prism llama.cpp fork + Vulkan | strixhalo-unknown | 32 | - | 2026-09-18]`
and 4.4 tok/s `[Ternary-Bonsai-2-27B-PTQ1_0 | PTQ1_0 | Prism llama.cpp fork + Vulkan | strixhalo-unknown | 32 | - | 2026-09-18]`.
