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
   generated attention.

### Diagnostics run on the chunked group-1 A2 (2026-09-15, later)

A `seq=513` probe is decisive because the two groups then have unmistakable
signatures: group 0 owns 512 active keys (softmax ≈ uniform → nearly all-zero
after `sat8(w·127)`), group 1 owns **one** active key (`seq_g = 513−512 = 1` →
`A2[0] = 127`). Observed on the N=1024 chunked build:

```
head0 row0 block map: [0]=1 [1]=0 [2]=0 [3]=0 [4..7]=0
head0 row0[0..15]   : 127 0 0 0 0 0 0 0 ...
```

So **the group-1 slot (cols 512–1023) is empty and the group-0 slot holds what
looks like group 1's one-key result.** At `seq=1024` the same probe gives
`[0]=107 [1]=114 [2]=100 [3]=114 [4..7]=0` — group 0 populated, group 1 empty.

**Ruled out by IR audit** (the generated `design.mlir`, N=1024):

| suspect | finding |
|---|---|
| A2 writeback offsets | correct — `dma_bd(SCR, 32, …)` for g=0 and `dma_bd(SCR, 544, …)` for g=1, 16 tasks (2 per column) |
| params feed offsets | correct — 8 tasks at `30720` (g=0) then 8 at `30784` (g=1), in order |
| core acquire order | correct — 8 `A_C` acquires + 1 params per group, two `attn_softmax_i8` calls with distinct `Par` operands and distinct `A2O_C` produce buffers |
| C2 geometry | correct — `C2_S` is `memref<8x128xi32>`, offsets `c*(M*K)` |

**Ruled out by experiment: the params are not the driver.** Writing *and*
reading group g's params at slot `g+1` (host `npu_attn_ctx.h` + generator, both
shifted, IR confirmed at `30784`/`30848`) produced a **byte-identical runtime
result** — `seq=513` still `[0]=1 … [4..7]=0`. If the core were consuming the
params one slot late, the shift would have moved the 512-key result into the
group-0 slot; it did not.

**Narrowed to:** group 1's **C1 is zero** (uniform softmax → `sat8(w·127) = 0`
for all 512 weights), i.e. the group-1 QK^T never accumulates, while group 0's
C1 does (its `seq=513` peakedness is a quantisation artefact — `sq`/`sk` are
recomputed from the shorter k array, so the two seq values are not comparable
C1-for-C1). The remaining suspects are the **group-1 B/KT tile addressing**
(`(ki*(N//n) + g*G_TILES + ntl)*(k*n)`) and the **`A2O` FIFO element
accounting** (element is the full `(8,N)` = 8192 B while each group's `a2t`
transfers a strided 4096 B of it). Next probe: dump C1 per group (or give the
two groups different KT tiles that cannot quantise away).

2. The stale-`dist` engines need a rebuild only if the generated xclbin is
   promoted over the captured ELF; the bench drives the artifact directly.
