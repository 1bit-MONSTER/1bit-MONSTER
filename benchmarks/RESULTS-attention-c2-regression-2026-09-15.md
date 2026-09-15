# The attention kernel was never slow — it was regressed (2026-09-15)

Corrects the "zero-output defect / C2 handshake / 2000× slower" section of
`LEVERS-register-2026-09-15.md`. Two independent generator bugs, both introduced
by `b2cfb080f` ("break the 512-score ceiling — N=1024 builds, chunked path
wired"), are the whole of it. The C2 handshake was never broken.

## Bug 1 — the `n_grp == 1` core lost its PV/C2 block (N ≤ 512)

`b2cfb080f` split the flat core body into `if n_grp == 1: … else: …`. The
PV/C2 writeback block that used to run unconditionally was left at its old
indentation, so git matched it as *context* and it ended up **only in the
`else` branch**:

    A2o = A2o_c[c].acquire(Produce,1); softmax(…); A2o_c[c].release(Produce,1)
    Cb = C2_c[c].acquire(Produce,1)   ← this block, for N ≤ 512, was gone
    zero(Cb); for ki: matmul(A,B,Cb); …

The commit's own check ("the shipped N=512 instruction stream is byte-identical,
so the `n_grp == 1` branch is provably the original path") validates the
*sequence* (`attn_insts.txt`, still `f3d0a132bde24a60`), not the *core ELF*. The
regression was invisible to it.

Consequence for N ≤ 512: the core produced A2 but never consumed the PV feed and
never wrote C2, while the sequence still fed `n_k_pv` (A,B) pairs and awaited the
C2 read task. That is the 6 s: the host waiting on a C2 task no core ever
satisfied. `max_abs_out == 0` and `max_abs_err == max_abs_ref` follow directly.

Fixed: the block is restored in the `n_grp == 1` branch.

**Verification.** MLIR for N=512 from the fixed generator is byte-identical
(2674 lines) to the pre-`b2cfb080f` generator's output.

| build (N=512) | C2 | max_abs_err | ms/call |
|---|---|---|---|
| shipped reference | 2/2 non-zero | 4.564293e-02 | **2.161** |
| ours, before fix | 0/2 (all zeros) | 1.205004e+01 | 6048 |
| **ours, after fix** | **2/2 non-zero** | **4.564293e-02** | **2.515** |

The generated kernel now matches the shipped reference's output *exactly* and is
within 16 % on time. **There is no 2000× gap and no L1 timing gate to open** —
the number was the regression's driver timeout, not kernel cost.

## Bug 2 — the chunked `seq()` never fed the PV nor read C2 (N > 512)

The same commit added the chunked core branch *with* a Cb/PV block, but the
chunked `seq()` branch ends at the A2 writeback: no PV A/B feed and no C2 read
task. The chunked core therefore blocks on A/B acquires that never arrive, and C2
is never read.

Fixed: the chunked branch now feeds `n_k_pv` (A,B) pairs (A = A2 read back from
bo4 at row stride N, B = V[kv]) and emits the C2 writeback, matching the
`n_grp == 1` geometry; the per-group params tasks are now awaited/freed with the
A2 writeback instead of being orphaned.

**Verification (N=1024, `attn_insts.txt` 46992 → 90544 B):**

| | C2 | ms/call |
|---|---|---|
| before | 0/2 (all zeros) | 3.100 |
| **after** | **2/2 non-zero** | **4.690** |

But the values are **still wrong** (`max_abs_err` 2.5–3.9e-01 vs `max_abs_ref`
1.67e-01), and the A2 dump localises it exactly: head 0 row 0 block map is
`[0]=107 [1]=114 [2]=100 [3]=114 [4..7]=0` — **group 0's A2 (cols 0–511) lands,
group 1's (cols 512–1023) does not**. So the chunked path has a third, still-open
defect: the second group's A2 never reaches SCR. The result is not a handshake
problem — C2 is written now — it is that half the softmax weights are missing.

## What this means for the register

- "**the sequence is FLM's byte-for-byte, so the defect is our core's C2
  protocol**" — the byte-identity is real but it only covers the sequence; the
  core regression was invisible to it. There is no C2 protocol defect to find.
- "**our kernel outputs ALL ZEROS, always**" — true only of the regressed
  `n_grp == 1` build (N=512), which was the only build the repro used.
- "**our generated kernel is ~2000× slower than FLM's**" — the 6048 ms was a
  host wait on an unsatisfied FIFO, removed by Bug 1's fix. The real N=512 cost
  is 2.5 ms against FLM's 2.2 ms.
- "**bisected the zero-output defect to the C2 writeback handshake**" — the
  "C2 produce only" row (7055 ms) removed the *consumes* but left the sequence's
  PV *feeds*; the host still waited on the C2 task. The bisect localised the
  wait, not a handshake.
- The ruled-out table's rows all still hold as measurements; none of them was
  the cause.

## Still open

1. **Chunked group-1 A2** (above) — the one remaining correctness bug in the
   generated attention. Likely in the per-group A2 writeback / `A2o` FIFO
   element accounting (the FIFO element is the full `(8,N)`, while each group's
   `a2t` transfers a strided 4096 B of it).
2. The stale-`dist` engines need a rebuild only if the generated xclbin is
   promoted over the captured ELF; the bench drives the artifact directly.
