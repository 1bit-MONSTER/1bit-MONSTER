# L1 attention kernel — integration audit of the delivered evidence

Goal `mu35shsg-i3hlyi`, tasks `attn-localise` and `attn-fix`. Date 2026-09-15
~23:55 ADT. **No NPU run by this goal.** This closes the two tasks by auditing
the L1 lane's delivered, committed evidence (goal `mttxt22c-a6rv75`, agent
`ea691a`, now offline; worktree `~/1bit-MONSTER-goal`, branch
`goal/runlist-decode-wire`). Every number below is read from the cited file and
commit, not re-measured.

## task-attn-localise — contract: phase-timing table (launch 1 vs 2, A2/C2 progress markers) + `main_aie_partit` comparison; each hypothesis listed with the measurement that killed it

Delivered in `benchmarks/LEVERS-register-2026-09-15.md` (commits `a55155f60`,
`07e1a62b5`, `c210d7179`) and resolved in
`benchmarks/RESULTS-attention-c2-regression-2026-09-15.md` (commit `07e1a62b5`,
`4c5a327cd`).

**Phase-timing table (one N=512 launch; `NPU_ATTN_POLL=1`; the `bV` control BO
returns in 0.003 ms mid-launch, so `sync(FROM_DEVICE)` does not block and the
readings are real):**

| N=512, one launch | control `bV` sync | A2 first non-zero | C2 first non-zero | `wait()` total |
|---|---|---|---|---|
| shipped | 0.003 ms | **0.52 ms** | **0.66 ms** | 0.99 ms |
| ours | 0.003 ms | **1.49 ms** | **never (≥20 s)** | 20000 ms (loop cap) |

A2 — through QK^T *and* the softmax contract — is on the device at 1.5 ms. The
6 s is after A2, in the PV or C2 writeback.

**`main_aie_partit` comparison** (register §"the 6 s", extracted from the
mirror-JSON `Offset`/`Size` because `xclbinutil` refuses the section name):
`88632 B` (shipped) vs `76472 B` (ours); both raw array config, zero ELF headers,
first eight words identical — so the difference is in the body and not a
by-eye-diffable embedded core ELF.

**Every hypothesis, with the measurement that killed it** (register §"the 6 s"):

| hypothesis | measurement that killed it |
|---|---|
| host dispatch / XRT | `submit` 0.487 ms vs `wait` 6047.2 ms (shipped: 0.014 / 0.719) |
| kernel throughput / payload | seq 1 / 8 / 512 all 6046–6048 ms — workload-independent |
| partition complexity / tile count / work | N=256 and N=512 agree to 1 ms (6046 vs 6047) across different partitions; chunked N=1024 is 1170 ms |
| the softmax's 8×MAX_SEQ loop | A2 is on the device at 1.49 ms |
| the never-terminating core loop (`range_(0xFFFFFFFF)`) | rebuilt N=512 with the loop bound `range_(1)` — **6049 ms** (falsified 2026-09-15) |
| the C2 circular-wait reading | `c210d7179` |

**Surviving mechanism, then resolved.** `RESULTS-attention-c2-regression-2026-09-15.md`
(commit `07e1a62b5`): commit `b2cfb080f` split the core body into `if n_grp == 1:
… else: …` and left the PV/C2 writeback block only in the `else` branch. For
N ≤ 512 the core produced A2 but never consumed the PV feed and never wrote C2,
while the sequence still fed the `n_k_pv` (A,B) pairs and awaited the C2 read task
— **the host waiting on a C2 task no core ever satisfied. That is the 6 s: the
regression's driver timeout, not kernel cost.** The `n_grp == 1` block is
restored, and the N=512 MLIR from the fixed generator is byte-identical (2674
lines) to the pre-`b2cfb080f` output. A third, separate defect in the chunked
path (the `A2O` FIFO element was the whole `(8,N)` = 8192 B while each group's
`a2t` transferred a strided 4096 B, so the second element was never produced) was
fixed in `4c5a327cd` by making the element the group slice `(8,512)` and setting
`params[3]` to 0.

**Verdict: satisfied.** The phase-timing table, the `main_aie_partit` sizes, the
falsified-hypothesis table and the surviving mechanism are all present and cited.

## task-attn-fix — contract: ms/launch within ~1.5x of the shipped capture on the same shape and prompt, or a doc naming the surviving mechanism and its next instrument

Delivered in `benchmarks/RESULTS-attention-c2-regression-2026-09-15.md`
(commits `07e1a62b5`, `4c5a327cd`, `7e53cfa08`).

| build (N=512, same shape and prompt) | C2 | max_abs_err | ms/call |
|---|---|---|---|
| shipped reference | 2/2 non-zero | 4.564293e-02 | **2.161** |
| ours, before fix | 0/2 (all zeros) | 1.205004e+01 | 6048 |
| **ours, after fix** | **2/2 non-zero** | **4.564293e-02** | **2.515** |

`2.515 / 2.161 = 1.16×` — **within the ~1.5× bound**, and the output matches the
shipped reference *exactly* (same `max_abs_err`), not merely within tolerance.
`attn_insts.txt` for N=512 is still `f3d0a132bde24a60`, so the `n_grp == 1` path
is untouched.

The chunked path is also fixed and verified NPU == EMU to the digit:

| build | C2 | NPU max_abs_err | EMU max_abs_err | ms/call |
|---|---|---|---|---|
| N=256 | 2/2 | 4.853413e-02 | **4.853413e-02** | 1.007 |
| N=512 | 2/2 | 4.564293e-02 | **4.564293e-02** | 2.456 |
| N=1024 chunked | 2/2 | **1.026515e-01** | **1.026515e-01** | 4.314 |

**Verdict: satisfied** on the primary limb (≤1.5× on the same shape and prompt).
The `N<=512` softmax contract of the phase-2 objective is likewise implemented and
gated: the `n_grp == 1` path is byte-identical in the sequence and NPU == EMU at
N=256 and N=512, which exercise different `n_n` branches.

## What is *not* closed: task-attn-engine

`task-attn-engine` asks for an **engine run past 1024 keys** using the generated
kernel with gated output. The delivered engine runs are:

- `824823e63` — the chunked kernel driven by `npu_engine_zr1`
  (`NPU_ATTN_MAX_SEQ=1024`) on a **600-token** prompt: `AttnCtx` loads and
  initialises, `[MoE L1 dbg] corr=0.999342`, 178.8 ms/tok. 600 keys < 1024.
- the decisive drop-in test: generated **N=512** == the embedded **captured**
  kernel, engine-driven, byte-identical output
  (`132187 41195 98398 22969 98398 68020 6496 4508`). 16-token prompt.

So the generated kernel *is* engine-driven and gated, but at ≤600 keys. A run
**>1024 keys** (e.g. a ≥2048-token prompt on the chunked kernel) is the remaining
step; it is the next device action for this goal.
