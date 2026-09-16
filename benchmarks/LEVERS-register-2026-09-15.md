# Levers register — matching FastFlowLM, broken into small wins (2026-09-15)

> **Status: the objective is met for the supported set, and this file says exactly
> where the edge of that set is.** Six models — Qwen3-0.6B / 1.7B / 4B / 8B,
> Qwen3-VL-4B, Llama-3.1-8B — meet or beat FastFlowLM on prefill, TTFT and decode
> at every length from 1 to 8191 tokens, verified against FLM's own runtime
> (`NPU_FLM_PREFILL=1 NPU_FLM_DECODE=1`) on this box with the same tokens, and the
> fast path is the **default** — no environment required.
>
> **Final verification, default invocation, npt=1024, no flags:**
>
> | model | native prefill | FLM prefill | native / FLM |
> |---|---:|---:|---:|
> | Qwen3-0.6B | **550 ms** | 802 ms | 1.46× |
> | Qwen3-1.7B | **803 ms** | 1064 ms | 1.32× |
> | Qwen3-4B | **1617 ms** | 2042 ms | 1.26× |
> | Qwen3-8B | **2268 ms** | 2806 ms | 1.24× |
> | Qwen3-VL-4B | **1616 ms** | 2032 ms | 1.26× |
>
> Boot tokens match everywhere; the continuations diverge only at the documented
> bf16 tie positions (`RESULTS-token-disagreements-are-ties`). Long context is
> verified separately: 0.6B / 4B / 8B at npt=8191 all return 59277, FLM's own value.
>
> **What is outside the set, stated plainly:** contexts **beyond 8192** (the layer
> ELFs bake `MAX_L=8192` and the generator asserts `L ≤ MAX_L+1`, so 8192 tokens is
> the window) and the **families the engine does not route** (Nanbeige, Phi4, the
> Gemma3/4 pair, LFM2, Qwen3.5). Those are levers L1–L5 below, each with the gate
> that decides it — not failures of this objective.


One page to answer "how do we get to FLM parity, and what is left". Every number
here is measured on this box and reproducible from the command in the row; a
number without a command is a claim, and this file does not carry those.

## 1. Where the objective stands

Supported set: **Qwen3-0.6B / 1.7B / 4B / 8B, Qwen3-VL-4B, Llama-3.1-8B**. Within
the KV window (1…8191 tokens) native meets-or-beats FLM on prefill, TTFT and
decode; beyond 8192 neither the layer ELFs nor the attention captures exist.

| model | native prefill | FLM prefill | native dec | FLM dec | tokens |
|---|---:|---:|---:|---:|---|
| 0.6B @1k | **591 ms** | 744 ms | 13.3 ms/tok | 13.4 | 25 220 16 13 |
| 0.6B @4095 | **1700 ms** | 1891 ms | **19.1** | 20.3 | = FLM |
| 4B @4095 | **5556 ms** | 6464 ms | **58.6** | 64.2 | 7/8 (step 8 a tie) |
| 8B @4095 | **8070 ms** | 9029 ms | **94.2** | 104.3 | 8/8 |
| VL-4B @4095 | **5566 ms** | 6528 ms | **58.5** | 64.2 | 8/8 |

Run any row with the default invocation — no environment — against
`NPU_FLM_PREFILL=1 NPU_FLM_DECODE=1` for FLM's own compute.

## 2. Small wins already landed

Each was a small, independently verified change. Ordered as they happened.

| # | win | measured effect | command / where |
|---|---|---|---|
| 1 | 4096-context attention captures (nh16 + nh32) | 2049–4095 served by the NPU instead of a **240614 ms** CPU attention pass: **142×** at 0.6B/4095 | `RESULTS-native-4096-context-2026-09-15.md` |
| 2 | the ctx-2200 cliff | a prompt past the shipped ELF set ran at **2 tok/s**; now **111 tok/s** (28×) | `RESULTS-ctx2200-cliff` |
| 3 | `NPU_UNIFIED` was unreachable (the runlist block returned first) | makes the fast-prefill+fast-decode combination reachable at all: prefill 25×/40× | `RESULTS-native-4k-decode-and-unified` |
| 4 | the unified KV handoff used the session's region stride | nh32 decode was wrong from the first step; now 8/8 tokens | same doc |
| 5 | `RT_ARGMAX_MARGIN` | turned "mixed fidelity" into "these are bf16 ties" — 6 disagreements, every gap ≤ 0.25 logits vs 0.875–4.375 where they agree | `RESULTS-token-disagreements-are-ties` |
| 6 | the default path picks the fast prefill | TTFT **13353 → 591 ms** at 1k (22.6×) and **69109 → 1700 ms** at 4k (40.6×); the default now beats FLM on prefill/TTFT | `RESULTS-default-path` |
| 7 | two 4096-sized host structures (RoPE tables, host KV caches) | all of >4096 was broken — wrong token *and* heap corruption; 4200 now correct and **252 s → 3.3 s** (77×) | `RESULTS-ctx8192-UNBLOCKED` |
| 8 | nh32 8192 capture + region stride + cap | **all six models reach 8191**; 4B/8B/VL-4B match FLM at 4200…8191 | same doc |
| 9 | generated ELFs go to `~/.cache/1bit-monster/elfs/` | long prompts stopped writing 512 untracked files per run into the checkout | `5d0b84535` |

The pattern worth keeping: **seven of the nine were not optimisations.** They were
a missing artifact, a shadowed code path, a wrong constant, or a silent fallback.
The engine's ceiling was already high; what it lacked was *being driven*.

## 3. Levers still on the table

Ranked by (value to the objective) / (effort). Each has the gate that decides
whether it is worth continuing — no lever here should be started without its gate.

### L1 — fix the softmax arity and measure the generated attention *(route past 8192)*
`n1_core_attn.py:134` hard-wires **four** C1 tiles; `n_n = N // 128`, so **N=512 is
the largest N the generator can emit** and the `-N` flag is not a parameter.
**Fix:** generalise the operand list to `n_n` tiles, confirm the AIE softmax kernel
accepts that arity.
**The build was fixed (one stale path, not a blocker).** `build_attn.sh` took its
python bindings from `install_tmp` but its compiler from `build_tmp/bin/aiecc` — a
later rebuild whose parser rejects the older dialect syntax. Pointing it at
`install_tmp/bin/aiecc` (the matching install) makes N=512 compile and reproduce
the shipped `attn_insts.txt` **byte-for-byte**. My earlier "the local mlir-aie WIP
patch broke it" was a misdiagnosis, corrected in the survey note.

**And the 512 ceiling is real — there are two of them, and both are in the kernel
contract, not in a flag.** With the compiler matched, N=1024 parses and then fails
at resource allocation, the same way under both schemes:

```
--alloc-scheme=basic-sequential -> 'aie.tile' op allocated buffers exceeded available memory
--alloc-scheme=bank-aware       -> 'aie.tile' op Bank-aware allocation failed
```

The two causes, both visible in the source:

1. **the softmax contract is 512 scores.** `attn_softmax_i8` is declared with four
   `C_ty` half-tiles, and `n1_core_attn.py` says why in a comment: *"takes 4 C1
   half-tiles + params + a2 (extra halves unused for N < 512 — **the contract reads
   only `c1[t>>7]`**)"*. `t>>7` is `t/128`, i.e. the tile index for `t < 512`. At
   N=1024 there are eight tiles and the kernel would read the first four — so a
   design that *did* fit would be **silently wrong**, not merely slow. That is the
   clamp, and it is in `attn_kernel_reference.cc`, not in the generator.
2. **C1 is held resident on the core tile, one per N-tile.** `C1_ty` is a (8,128)
   int32 = 4 KB tile, and `C1` allocates `n_n` of them per column plus `A2_ty` of
   (8,N) int8: **N=512 → 4×4 KB + 4 KB = 20 KB; N=1024 → 8×4 KB + 8 KB = 40 KB**,
   past the core tile's data memory. Hence the allocation failure.

**Both are smaller than they look, and the second reading shrank them further.**
`attn_softmax_contract` is *already* general — its signature is
`(const int32_t* const c1[], const float* params, int8_t* a2)` and it indexes
`c1[t >> 7]`, with a comment saying so: *"the caller supplies n_half = max_seq/128
pointers (2 for N=256, 4 for N=512)"*. The four-pointer
`attn_softmax_i8` wrapper is only a fixed-arity **shim**. And the generator's
`params` are `{scale, seq, MAX_SEQ, 0.0f}` — **`params[3]` is free**.

So the plan is:

1. **`attn_quant.h`**: take the A2 **row stride** from `params[3]`, defaulting to
   `max_seq` when 0 — two lines, changing `a2[r*max_seq + …]` to `a2[r*row_stride + …]`.
   The host caller (`npu_attn_ctx.h`) passes 0, so its behaviour is unchanged.
2. **`n1_core_attn.py`**: chunk the N dimension into groups of **four** tiles
   (512 scores), which is exactly the wrapper's arity — so **no kernel-arity change
   at all**. Each group zeroes `C1[0..3]`, accumulates its QK^T, and calls softmax
   with `params = {scale, seq_slice, 512, N}` and an `a2` offset of `512·g`
   (the slice per row is contiguous: `(t/8)*8 + t%8 = t` for `t < 512`).
3. **Result: four `C1` tiles resident instead of `n_n`** — 16 KB + 8 KB A2 = 24 KB
   at N=1024, against 40 KB today. That is what makes it fit.

Then build `NPU_ATTN_N=1024` and measure. The delicate part is not the kernel but
the **sequence feed**: the FIFO order is hand-matched to the unrolled QK^T loop
("matches the seq feed" in the source), so the group loop has to reorder the B
tiles consistently with it. The regression test is already in hand — N=512 must
keep reproducing the shipped `attn_insts.txt` byte-for-byte
(`f3d0a132bde24a60`).
**Gate:** generalise the operand list, then build `NPU_ATTN_N=1024` and time it
against the captured `attn_mha_1024_nh16.elf` on the same shape and prompt. Within
~1.5× of the capture, N=16384 is the route past 8192; another 1200× (this repo has
that precedent for a generated attention ELF) closes the lever.

#### L1 status — the BUILD gate is passed; the timing gate is open *(2026-09-15)*

`NPU_ATTN_N=1024` now builds: **104784 B xclbin, aiecc rc=0**. The regression the
gate named holds exactly — `NPU_ATTN_N=512` still reproduces the shipped
`attn_insts.txt` **byte-for-byte** (`f3d0a132bde24a60`), which is what proves the
group loop's single-group branch *is* the original path rather than merely
agreeing with it.

Getting there turned up two things the plan did not anticipate. Both are in the
kernel/verifier contract, not in the generator:

1. **The strided A2 writeback is rejected in its obvious encoding.** Group *g*'s
   slice is 8 rows of 512 bytes at row stride N, i.e. `sizes=[1,1,8,512]`. With
   `strides=[1,1,N,1]` aiecc refuses it:
   `'aie.dma_bd' op Stride 2 is 1 elements * 1 bytes = 1 bytes, which is not
   divisible by 4`. The degenerate leading dims are **size 1**, so their stride is
   never applied — but `verifyStridesWraps` (`lib/Dialect/AIEX/IR/AIEXDialect.cpp`)
   checks *every* stride ≥ 1 for 4-byte divisibility **regardless of that
   dimension's size**, and it is not the branch that `skipTransformationChecks`
   guards. `strides=[4,4,N,1]` is semantically identical and legal.
2. **A row stride alone is not enough — the causal mask needs per-group params.**
   Group *g* owns the keys `[512g, 512g+512)`, so its mask must fire at the
   group-local `t_local ≥ seq − 512g`. The plan's `params = {scale, seq_slice,
   512, N}` only works if `seq_slice` can *differ per group*, which means the feed
   has to read a different params tile per group. It now reads group *g*'s at
   `15·K_FRAME + g·64`, and the host writes one 8-float set per group with its own
   clamped key count. Feeding every group the global `seq` **compiles, runs, and is
   silently wrong for every group past the first** — precisely the failure mode the
   "silently wrong, not merely slow" line above was warning about, arrived at from
   the other direction.

Landed: `n1_core_attn.py` (group loop), `attn_quant.h` (A2 row stride from
`params[3]`, 0 = packed = previous behaviour), `npu_attn_ctx.h` (per-group params;
`NPU_ATTN_EMU` emulation mirrors the chunked core so the contract stays checkable
host-side), and `engine/npu/tools/attn_kernel_bench.cpp` — a standalone driver for
`AttnCtx` that needs no engine, no Zaya and no model on disk:

```
g++ -std=c++17 -O2 -mavx2 -I src -I generators -o ck_test tools/attn_kernel_bench.cpp \
    -lxrt_coreutil -lxrt_core -laiebu -luuid -ldl
NPU_ATTN_MAX_SEQ=1024 NPU_ATTN_EMU=1 ./ck_test XCLBIN INSTS 1024   # contract, host-side
NPU_ATTN_MAX_SEQ=1024             ./ck_test XCLBIN INSTS 1024 20  # NPU + ms/call
```

It checks the kernel against a double-precision causal-GQA reference and times it.
First results, back-to-back on an otherwise idle device, same shape (nq8/nkv2/hd128):

| kernel | N | insts | max abs err vs float | ms/call |
|---|---|---|---|---|
| shipped `attn.xclbin` (FLM's) | 512 | shipped | 3.32e-01 | **3.0** |
| ours, generator, same build | 512 | **byte-identical to shipped** | 3.32e-01 | **6047** |
| ours, chunked generator | 1024 | new | 1.67e-01 | **1170** |

**Read that table carefully, because the first two rows are the finding.** Row 2
runs a byte-identical instruction stream to row 1 — verified, not assumed: the
`sha256sum` of the shipped `xclbins/attn_insts.txt` and of the fresh build's agree
at `f3d0a132bde24a60` — and returns *bit-identical* output (3.321927e-01 both, to
7 digits), and is **~2000× slower**. `attn_insts.txt` being byte-identical was
never evidence that the *kernel* matched; it was evidence that the *schedule* did.

**Correction to this note's own first reading (same day).** The sentence that stood
here first said the deficit "is in our core kernel code". That is **not
established, and the follow-up measurements point away from it**:

| ours, N=512 | seq | ms/call |
|---|---|---|
| | 1 | 6048 |
| | 8 | 6048 |
| | 512 | 6047 |

The cost is **independent of the workload** — 1 key or 512 keys, the same time to
the millisecond. This is a **fixed per-launch cost**, so it is not the arithmetic
scaling with the sequence.

**And it is device-side, not host dispatch** — a split of one launch (the bench
re-issues the identical call, `AttnCtx`'s members being public):

| | submit | wait |
|---|---|---|
| ours, N=512 | 0.487 ms | **6047.2 ms** |
| shipped, N=512 | 0.014 ms | **0.719 ms** |

Submission is instant in both. The six seconds are spent *waiting for the NPU to
finish*, so the host path, the XRT call and the instruction stream are all
exonerated — the device takes 6 s to complete a single invocation.

**But do not read the seq-invariance as "the kernel's work is fine".** That was the
next over-correction, and it is wrong for a specific reason: the core's per-
invocation work is fixed by `MAX_SEQ`, **not** by `seq`. The softmax contract loops
`t` over `max_seq` (= `params[2]` = 512) and merely *masks* `t >= seq`; the mmul
runs its full tile set either way. So the core does the same work at seq=1 and
seq=512 — the table above is consistent with a kernel that is simply very slow, and
does not distinguish it from one that stalls. What the numbers do rule out is any
explanation involving the *amount of data transferred*, which is what "dispatch
overhead" would predict.

So the open question is narrowed to: **why does one invocation of our partition
take 6 s of device time to complete, when the same instruction stream driving FLM's
partition takes 0.7 ms?** The suspect is the core program's synchronisation against
the instruction stream — our schedule is FLM's (byte-identical) but our *core code*
is our own reimplementation (`attn_kernel_reference.cc`), so a FIFO/token contract
that is satisfied only after a wait/timeout would look exactly like this: correct
results (the DMAs landed), fixed cost, independent of payload.

Next, concretely: `xclbinutil --dump-section main_aie_partit` on both containers and
compare the core ELFs; and instrument the core's loop (a cycle counter written to a
BO) to see whether it is spinning or blocked. Still not kernel *optimisation*.

**Two things tried since, and what they rule out.** `xclbinutil` will not name the
section (`Section 'main_aie_partit' isn't a valid section name`); extracting it from
the mirror-JSON `Offset`/`Size` with `dd`-equivalent works fine and gives 88632 B
(shipped) vs 76472 B (ours). Both blobs are raw array config, not ELF — **zero**
`\x7fELF` headers in either, and their first eight words are *identical*
(`0 0xb8 0x800 0 0x5b7f 0 0x3039 0`), so the difference is in the body and is not a
naively-embedded core ELF anyone can diff by eye. That route needs the amdxdna
partition format, not more time with `xclbinutil`.

What the N=1024 datapoint now rules out is the *other* easy story. The chunked build
has **more** N-tiles and **more** groups than the N=512 build and is **5× faster**
(1170 vs 6047 ms) — so the cost does not track partition complexity, tile count, or
work. What it does track is the **order of the object-FIFO feed**: the chunked
generator changed that order; the non-chunked path is the original one. A stall that
changes by 5× when only the feed order changes, with byte-identical instructions,
points at the core code desynchronising from the stream and being released by a
timeout — not at anything the host or the container controls. It also means this
cost has been present in the generated attention all along, including the N=512
build that was believed to be the working one.

The cheapest decisive next test is therefore **not** more container forensics: build
the N=256 variant (same kernel code, different partition and feed) and see where its
fixed cost lands. If it lands on its own value again, the constant is a property of
each generated design's synchronisation and the fix belongs in `n1_core_attn.py`'s
feed/FIFO contract.

**Done, and it is a timeout.** N=256, a different partition with a different tile
count, the same kernel code and the *original* feed order:

| build | feed order | N | ms/launch |
|---|---|---|---|
| non-chunked | original | 256 | **6046** |
| non-chunked | original | 512 | **6047** |
| chunked | per-group | 1024 | **1170** |

N=256 and N=512 agree to **one millisecond** (6046 vs 6047) across different
partitions and different amounts of work, while the chunked design sits at a
different constant again. Two designs landing on two exact constants, with the work
between them varying by 4× and the results bit-correct, is the signature of a
**fixed timeout being waited out**, not of execution time. The most likely mechanism
follows directly from the generator: the core body is `for _ in range_(0xFFFFFFFF)`
— the tile never terminates, so nothing tells the driver the invocation is finished
and it falls off a cliff-edge timer instead. The data is already in the BOs, which
is why every result is correct.

That hypothesis is cheap to falsify and is the next action: give the core loop a
finite bound (or a completion signal) and re-measure. If the constant collapses to
milliseconds, the entire 2000× was never a kernel-speed problem at all — it was the
core never saying "done", and the lever's timing gate reopens completely.

**Falsified, same day.** Built N=512 with the core body's outer loop changed from
`range_(0xFFFFFFFF)` to `range_(1)` — one line, nothing else — and it measures
**6049 ms**, i.e. the same constant, with the same bit-identical output (3.321927e-01).
The never-terminating core loop is **not** the cause, and the "the tile never says
done" story is dead.

Where that leaves it, stated as narrowly as the evidence allows:

- The cost is device-side (`submit` 0.5 ms, `wait` 6 s), per launch, and does not
  respond to N (256/512 identical to 1 ms), to the amount of work (seq 1/8/512
  identical), or to the core's loop bound.
- It *does* differ between two generators of ours (non-chunked ≈ 6046–6049 ms;
  chunked ≈ 1170 ms), so it is a property of the built partition, not of the driver
  or of XRT — but not one that scales with anything in the design that has been
  varied so far.
- It is therefore most likely a **fixed wait inside the kernel's own
  synchronisation** (a FIFO/token acquire that is satisfied only on a timer) that
  the non-chunked and chunked feeds happen to hit a different number of times.

The two falsified stories are worth more than the surviving one: *kernel
throughput* and *the infinite core loop* are both ruled out, so whoever picks this
up should instrument the core (cycle counter into a BO, or a marker write per
phase) rather than guess a third time. The bench is the instrument and it is
committed.

**And then the instrument that does not need a generator change at all: watch the
device from outside the launch.** `run()` zeroes the A2 scratch and C2 before each
call, so a non-zero byte in either is a progress marker; the poll mode
(`NPU_ATTN_POLL=1`) launches, then spins syncing those BOs and timestamping the
first non-zero, then waits. The validity question — does `sync(FROM_DEVICE)` block
while a launch is active, which would make every reading a lie — is answered by
polling **`bV`, a BO the kernel never writes**: it returns in **0.003 ms** mid-launch,
so sync does not block and the readings are real.

| N=512, one launch | control `bV` sync | A2 first non-zero | C2 first non-zero | `wait()` total |
|---|---|---|---|---|
| shipped | 0.003 ms | **0.52 ms** | **0.66 ms** | 0.99 ms |
| ours | 0.003 ms | **1.49 ms** | never (≥20 s) | 20000 ms (loop cap) |

**The softmax is not the bottleneck, and neither is QK^T.** Our A2 — the softmax
output, i.e. everything through the QK^T scan *and* the softmax contract — is on
the device at **1.5 ms**, versus the shipped kernel's 0.52 ms. Three times slower
and 4000× off the 6 s. This also kills the "the softmax's 8×MAX_SEQ serial loop is
the fixed cost" reading of the seq-invariance, which was the best remaining
arithmetic story.

So the 6 s is **after A2**: in the PV phase or the C2 writeback — the last two
stages and nothing else.

One loose end that must be closed before that is treated as final: in the poll
launch C2 never went non-zero, yet the kernel still completed (the `wait()` after
the 20 s loop returned immediately, so it had finished at some earlier point). A
kernel that produces A2 but not C2 and then completes is not what `run()` does —
`run()`'s output is correct — so either the direct launch differs from `run()`'s in
some way not yet identified, or the C2 writeback is itself the thing that stalls and
is only completed by whatever `run()` does differently. **Next action is to
instrument the PV phase specifically** (and to reconcile the direct launch against
`run()`), not to touch the softmax.

**That loose end was then attacked directly, and it survived.** The probe launch was
changed to match `run()`'s pre-launch syncs exactly (`bQ`, `bKT`, `bV` pushed
TO_DEVICE before the launch — the one place the two launches differed): identical
result, `a2_first = 1.504 ms`, `c2_first = never`. So the difference between the
probe and `run()` is **not** the pre-launch syncs.

Two explanations remain and they are not equivalent:

1. **The kernel is pipelined and each launch advances it one iteration**, so the
   outputs lag: the probe's A2 at 1.5 ms could belong to an iteration whose C2 is
   simply the *next* launch's business. If so, "6 s per launch" is the pipeline's
   iteration time, the 1.5 ms A2 is the previous iteration's tail, and the whole
   picture above — including "the 6 s is after A2" — has to be re-read rather than
   built on.
2. **The C2 writeback stalls**, and the probe's kernel completing without writing C2
   is the same defect seen from the other side.

These are distinguished cheaply and that is the next action, before any further
conclusion: run the **probe launch twice in a row** and see whether C2 appears on
the second one. If it does, explanation 1 is right and the phase timing needs
re-doing against a warmed pipeline. If it does not, explanation 2 is right and the
C2 writeback is the target. Nothing above should be quoted as final until that is
settled — the A2 timing in particular is only meaningful under explanation 2.

**Settled, and it is neither — it is worse and more useful than both.** Two
consecutive probes, both zeroed first, both launched with `run()`'s exact syncs:

```
poll#1: control_bV_sync=0.003 ms  a2_first=1.538 ms  c2_first=never  waited=20000 ms
poll#2: control_bV_sync=0.003 ms  a2_first=1.500 ms  c2_first=never  waited=20000 ms
```

So explanation 1 (pipelined, C2 on the next launch) is **falsified** — the second
probe is no different from the first. And the shipped kernel on the *same probe
path* produces both A2 and C2 in under a millisecond. The two kernels therefore
differ **qualitatively**, not by a constant:

| | first launch (`run()`) | later bare launch |
|---|---|---|
| shipped | correct output, ~1 ms | A2 + C2, <1 ms |
| ours | correct output, ~6 s | A2 at ~1.5 ms, **C2 never** |

Two things follow, and the second is the one that matters:

1. **The QK^T + softmax stages are genuinely fast in our kernel** (~1.5 ms to
   produce A2, against the shipped kernel's 0.5 ms). That part of the earlier
   reading survives.
2. **Our kernel does not re-arm like FLM's.** The shipped kernel produces a complete
   result on every launch; ours produces A2 on later launches, never C2, and still
   reports completion. A design that behaves differently on launch #2 than on
   launch #1 is desynchronised somewhere, and that is now the leading and
   best-evidenced explanation for the 6 s as well: the first launch is the one that
   pays the full desync cost.

**Therefore the earlier sentence "the 6 s is after A2" is not safe to build on** —
it was read from a launch that never finishes its PV phase, so it constrains the
QK^T/softmax stages and says nothing about the PV's cost. Retracted as a conclusion
about the whole design; kept only as the statement about the first two stages.

The next action is no longer a timing question. It is: **why does our kernel produce
a complete result once and then stop producing C2?** The shipped kernel, byte-identical
instruction stream and all, does not. That is a correctness-shaped question about our
core's synchronisation, it is answerable from the generator, and it should be settled
before any further performance work on this kernel — including before the L1 timing
gate is attempted again.

**RETRACTED the next day, by a check that should have been there from the start.**
The claim that stood here was "our kernel is good for exactly one call per process",
with "the first call is real — it matches the shipped kernel's output to the digit".
Both halves are wrong, and the error is instructive.

The bench now prints `max_abs_out` (the largest output magnitude) alongside the
error. That single number separates "produced a real result" from "produced zeros":

| N=512 | max_abs_err | max_abs_ref | **max_abs_out** |
|---|---|---|---|
| ours | 3.321927e-01 | 3.321927e-01 | **0.000000e+00** |
| EMU (host contract) | 4.564293e-02 | 3.321927e-01 | 3.778357e-01 |
| shipped | 4.564293e-02 (loop) | 3.321927e-01 | 1.060043e+01 |

**`max_abs_err == max_abs_ref` is the signature of an all-zero output**, and our
kernel has `max_abs_out = 0` on *every* call — first call included. The "real first
result" I celebrated was that arithmetic coincidence: an all-zero output has error
exactly equal to `max|ref|`, which happens to be 3.321927e-01. And the *shipped*
kernel reported the same 3.32e-01 in the very first run for the same reason — its
first launch after `init` returns garbage too (10.33 here, 5.60 under the newer XRT,
0.332 when its output happened to be zeros); only its **loop** iterations are
trustworthy, and those sit at 4.564293e-02 — **identical to the host emulation's**,
which is what a working kernel looks like.

So the correct finding is simpler and stronger than the one it replaces:

**Our generated attention kernel produces all-zero output. It has never produced a
result in this harness, at any N, on any call.**

That also unifies everything else, which is the real reason to prefer it:

- A2 **is** produced, 1.5 ms in (measured, non-zero in SCR). QK^T and softmax work.
- C2 is never written. The PV phase is the failure.
- The 6 s is the host waiting on a kernel that never finishes — the PV stalls, the
  driver times out, C2 stays zero. Hence the data-independence, the N-invariance,
  and the two different constants (a stall, not work).
- The instruction stream being byte-identical to FLM's is consistent: the schedule
  is right, and our core fails to complete the phase that schedule sets up.

Consequences, replacing the retracted ones:

- **Every timing number for our kernel in this document is the cost of a launch that
  produced nothing.** 6047 ms buys zeros. That does not invalidate the *cost*
  measurements (the device really did take that long) but the "2000× slower" framing
  is wrong in kind: it is not a slow kernel, it is a kernel whose PV phase stalls
  until a timeout.
- **The L1 timing gate cannot be evaluated at all** until this is fixed, at any N.
  There is no performance question here yet; there is a correctness one.
- The lesson worth keeping: `max_abs_err` alone cannot distinguish a correct kernel
  from one that outputs zeros whenever the reference is non-zero — you must look at
  `max_abs_out`. I read four separate confirmations out of that ambiguity before
  noticing.

Next action, corrected: **find why the PV phase never completes in
`n1_core_attn.py`** — the core stalls after the A2 writeback, so the suspect is the
PV's consume of the A-tap read back from SCR (the `A_s`/`A_c` FIFO being reused for
QK^T tiles, the params tile, and then the A2 read-back) or the C2 produce. The
shipped kernel, on the identical instruction stream, does not stall.

#### The zero-output defect: exact repro, and what has been ruled out

> **Superseded (2026-09-15) — read this first.** The defect was **not** a C2
> handshake. Two `b2cfb080f` generator bugs explain everything below: the
> `n_grp == 1` core lost its PV/C2 block (N ≤ 512 output all zeros and waited
> 6 s on a C2 task nothing produced), and the chunked `seq()` never fed the PV
> nor read C2 (N > 512). With Bug 1 fixed, **N=512 is correct and FLM-class —
> 2/2 non-zero C2, error 4.564293e-02 exactly matching the shipped kernel,
> 2.515 ms vs FLM's 2.161 ms.** There was no 2000× gap. The bisect and
> ruled-out tables below stand as measurements but none of them was the cause.
> See [RESULTS-attention-c2-regression-2026-09-15.md](RESULTS-attention-c2-regression-2026-09-15.md).
*(Was titled "the re-arm defect" until the `max_abs_out` check above showed there is
no re-arm to speak of: the output is zeros on every call. The repro and the
ruled-out list below are unaffected — the C2 check does correctly report that C2 is
never written — but the name was wrong.)*

**Repro** (in-tree, no engine, no model):

```
g++ -std=c++17 -O2 -mavx2 -I src -I generators -o ck tools/attn_kernel_bench.cpp \
    -lxrt_coreutil -lxrt_core -laiebu -luuid -ldl
NPU_ATTN_MAX_SEQ=512 ./ck <ours>.xclbin <ours>_insts.txt 512 2   # -> run(): 0/2 wrote a non-zero C2
NPU_ATTN_MAX_SEQ=512 ./ck xclbins/attn.xclbin xclbins/attn_insts.txt 512 2  # -> 2/2
```

The check is in the timing loop and it is the whole reason this was findable: it
zeroes C2 on the host before each `run()` and verifies the device wrote it back. A
timing harness without that line reports a confident number for a pipeline that
stopped after its first call — which is what this one did for several rounds.

**Localised:** the QK^T and softmax phases run — A2 is on the device 1.5 ms into
every launch — while C2 never appears at all. So the break is specifically the PV
phase's consume/produce handshake or the C2 writeback, not the front half of the
kernel and not the launch path. (This was written as "DO re-arm"; nothing re-arms,
because nothing ever completes. The stage localisation itself stands.)

Ruled out by measurement, so nobody repeats them:

| story | test | result |
|---|---|---|
| kernel throughput | seq 1 vs 8 vs 512 | identical, 6048 ms |
| infinite core loop | `range_(0xFFFFFFFF)` → `range_(1)` | 6049 ms, same output |
| amount of work | N=256 vs N=512 vs chunked N=1024 | 6046 / 6047 / 1170 ms |
| host dispatch | submit vs wait split | 0.5 ms submit, 6 s wait |
| XRT version | relink against `/usr/local/xrt-runlist` | 6049 ms either way |
| container metadata | partition blobs, headers, topology | structurally identical |
| core FIFO depth | `C2_c` 1 → 2 (deeper blows tile memory) | still 0/2, still 6048 ms |
| pre-launch syncs | probe with `run()`'s exact `bQ`/`bKT`/`bV` pushes | no change |
| pipelined lag | two consecutive zeroed probes | probe #2 same as #1 |
| C2 drain position | arm the C2 read before the PV loop instead of after | still 0/2, still 6048 ms |

Still untested and the best next leads, in order: (1) the core's per-iteration
lock/token accounting around the C2 produce — A2 re-arms and C2 does not, which
narrows it to a specific acquire/release pair; (2) a **minimal** two-FIFO design
built through the same `build_attn.sh` flow, to establish whether this toolchain's
`--unified --dynamic-objFifos` output re-arms at all, separating "our attention
generator is wrong" from "our build flow cannot re-arm". Do (2) first if time is
short: it is the cheaper fork and it decides where the work belongs.

#### Bisected: the stall is the C2 writeback handshake

The zero-output defect is now localised by cutting pieces out of the design and
timing what is left. All at N=512, `ms/call`, same bench:

| build | QK^T+softmax+A2 | PV consumes | C2 produce | ms/launch |
|---|---|---|---|---|
| full | yes | yes | yes | **6048** |
| A-tap from Q instead of SCR | yes | yes | yes | **8064** |
| front half only | yes | no | no | **4.6** |
| C2 produce only | yes | no | yes | **7055** |

Reading it:

- **The front half is fine and fast.** QK^T + softmax + the A2 writeback complete in
  4.6 ms — the same order as the shipped kernel's 2 ms. Everything through the
  softmax is correct and quick.
- **The A2 read-back is not the stall.** Pointing the PV's A-tap at `Q` instead of
  the scratch buffer changes nothing except the timeout constant (8064 vs 6048), so
  the round-trip through DDR is not what hangs.
- **The PV's matmul consumes are not the stall either.** Removing them and keeping
  everything else still hangs (7055 ms).
- **What is left is the C2 produce itself** — `C2_c[c].acquire(Produce,1)` /
  `zero(Cb)` / `release(Produce,1)` plus the shim's `C2_s[c]` read task. That alone,
  with no PV arithmetic at all, reproduces the hang.

This is consistent with a suspicion already in the tree: the dump path in
`npu_attn_ctx.h` carries a "Race check: re-scan after a delay to see if the S2MM is
draining" — the C2 writeback is an S2MM, and someone previously suspected exactly
this drain. That comment now has a measurement behind it.

Next action is therefore narrow and specific: **why does the C2 FIFO's
produce→drain handshake never complete?** The A2o path uses the identical FIFO
pattern (core→mem→shim, depth 2/1) and works, so the difference is in the C2
wiring or in what the core does across the acquire — not in the PV and not in the
generator's feed order. Note that `C2_c` is depth 1 where `A2o_c` is depth 2, and
that raising it to 2 did *not* fix the hang, so depth alone is not the answer.

#### The sequence is provably FLM's — so the defect is the core's C2 protocol

This is the sharpest constraint available and it took too long to apply. The
instruction stream is not just "similar to" FLM's: `attn_insts.txt` for this build
is **byte-identical** to the shipped file (verified by `sha256sum`,
`f3d0a132bde24a60`). That stream *is* the sequence — every
`shim_dma_single_bd_task`, every `dma_start_task`, every `dma_await_task`, every
token. **Our sequence is a working sequence.** It cannot be the bug, and neither
can the feed order, the task counts, the C2 drain's position, or anything else on
that side of the design.

Which leaves exactly one artifact that differs: **the core ELF**. FLM's core
satisfies the C2 handshake in that stream; ours does not. The failure is in what
our core does across `C2_c[c].acquire(ObjectFifoPort.Produce, 1)`.

That also re-reads the bisect honestly. "Front half only" was fast (4.6 ms) — but
it removed the C2 task *from the sequence as well*, so it was not a core-only
comparison. The clean statement is the one above: with FLM's sequence intact, our
core hangs; with the C2 task deleted, nothing waits for our core and the measure
falls to its true value.

**And the circular-wait reading of it is falsified too.** The obvious mechanism — the
core blocks on the C2 acquire, so it never consumes the PV feed, so the sequence
never reaches the C2 task that would unblock it — predicts that moving the acquire
after the consumes fixes it. Built exactly that (PV accumulates into a resident
`C1[c][0]`, then `C2_c[c].acquire`), token counts unchanged, sequence untouched:
**6048 ms, unchanged.** So the wait is not an ordering deadlock between the core's
C2 acquire and the PV feed.

That leaves the C2 read task itself never completing, for a reason that is neither
ordering, nor depth, nor the A-tap source, nor the PV arithmetic. The next
experiment should be on the *token accounting* of that one task: the core's
`acquire(Produce)`/`release(Produce)` pair on `C2_c` versus what that task expects,
given that `A2o` uses the same pair, the same two-arg link and the same 4096-byte
transfer and satisfies its handshake. Something about the C2 pair differs from the
A2o pair in a way not yet named — that is the whole remaining question.

Consequence for the next attempt: **do not touch the sequence** — it is the one
part of this design known to be correct, and changing it (as the C2-drain-position
experiment did) invalidates the byte-identity that makes the rest diagnosable. The
work is in the core: how the C2 buffer is acquired and released relative to the A2o
produce and the PV consumes. The A2o path uses the same core→mem→shim pattern with
the same two-arg `object_fifo_link` and *does* satisfy its handshake, so the
difference is in the core's use of the C2 buffer, not in the wiring.

**Fork (2) is already answered, from the repo rather than from a new design.** The
i8 decode GEMMs (`final_i8_D_*`) are built by the *same* generators with the *same*
flags — `grep` over `build_*.sh` shows every one of them using
`--unified --dynamic-objFifos` — and `npu_engine_i8ctx_inc.h` drives them through
the identical launch pattern this kernel uses: `hw_context`, then
`xrt::kernel(*hc, "MLIR_AIE")`, then `(*k)((unsigned)3, ...)` and `r.wait()`, once
per token per layer, thousands of times in a single process. Those kernels
demonstrably work repeatedly — that path is the byte-exact reference the rest of
the work is measured against.

So **our build flow re-arms, and the one-call defect is specific to
`n1_core_attn.py` and its design.** That eliminates the toolchain, the flags, the
XRT version and the launch pattern in one step, and it makes lead (1) — the core's
token accounting around the C2 produce — the only remaining direction. No minimal
design needs to be written.

What the two containers differ in (from `xclbinutil`; topology, connectivity and
kernel name are structurally identical):

| | shipped | ours |
|---|---|---|
| xclbin `Version` | 2.15.75 | 2.13.0 |
| `main_aie_partit` size | 88632 B | 76472 B |
| ms/launch | 2.5–3.0 | 6047 |

The cost also survives an XRT swap: relinking the bench against the engine's
`/usr/local/xrt-runlist/lib` gives ours 6049 ms, the shipped one 2.5 ms. That run
also showed the shipped container returning a *different and wrong* answer under
the newer XRT (5.60 vs 0.33), so it is version-coupled to the XRT it was built
against — worth knowing before anyone tries to "fix" this by swapping XRT.

It also settles what the N=1024 number is not: 1170 ms is **5× faster than our own
N=512 build**, from the *same* compiled kernel object, differing only in the
partition graph. Nothing about chunking has been shown to cost anything, and work
volume is not what drives these numbers at all.

Still open, in order:

1. **Localise the fixed per-launch cost.** Start by extracting and diffing the two
   `main_aie_partit` sections and by timing `xrt::hw_context` creation separately
   from a launch. The instrument is the bench above; the question is why a launch
   in our container costs seconds when the same instruction stream in FLM's costs
   milliseconds. Do **not** begin by optimizing the kernel — the evidence does not
   support that yet.
2. **The chunked path has no *engine* run.** It executes on the NPU (that is what
   row 3 is) and its error is in the same range as its own emulation, but it has
   never been driven through `zaya_decode.cpp` end to end. The gate as written
   asked for "the same shape and prompt", which means the engine — the bench is a
   kernel instrument, not that gate.
3. The absolute-vs-float error (0.33 at N=512, on white-noise input) is a property
   of the int8 design and is identical for FLM's kernel and ours, so it is not a
   defect to chase here. It also means the bench is not a clean oracle: NPU 0.33
   against its own host emulation's 0.046 on the same buffers is an unexplained
   gap in the *emulation comparison*, not a demonstrated kernel defect.

### L2 — a Nanbeige nh20 capture at 4096 *(MIS-SCOPED 2026-09-15 — see note)*

> **Superseded (2026-09-15): L2 as written was mis-scoped.** A correct nh20
> attention ELF cannot matter for Nanbeige's bf16 arm, because that arm is
> **architecturally excluded**: the block that loads the attention ELFs and calls
> `bf16mm_init` is gated at `npu_engine_universal.cpp:4433` by
> `if (getenv("NPU_PREFILL_BF16") && !has_moe)`, and Nanbeige is MoE
> (`[ModelConfig] … experts=16 top_k=2`). So `!has_moe` is false, the block never
> runs, no attention ELF of any shape is ever loaded, and the run falls through to
> the generic `=== Prefill NNNN [fallback] ===`. Every 4096-candidate experiment
> was therefore vacuous. Two claims made while chasing this are withdrawn: the
> mm/dequant xclbins were never missing (they live in
> `/home/bcloud/amd-oss/fastflowlm/src/xclbins/Nanbeige4.1-3B-NPU2/`, the dir
> `Bf16Mm` actually uses), and `load_attn_elf` was never unreachable for a naming
> reason. The real work item, if bf16-MoE prefill is wanted, is the `!has_moe`
> guard itself — a **feature**, not a capture. Full trail:
> [RESULTS-attention-c2-regression-2026-09-15.md](RESULTS-attention-c2-regression-2026-09-15.md).
> Nanbeige's geometry was also settled from `config.json`: nh20 / hd128 / nkv4,
> qout 2560, QKV N = 3584 (the generator's shape table had a stale nh32/hd80
> reading on every Nanbeige entry; corrected).
Nanbeige's default i8 path matches FLM exactly (`1033 @1024`, `5938 @256`); only
its **bf16 arm** falls to the broken nh20 kernel. A correct nh20 capture is a
drop-in file by the existing shape naming — the same one-file change as L1's
neighbours, for a family that is otherwise already at parity.

### L3 — `kv_quant.h`: INT4 KV *(storage term, not a quick win)*
Declared, never implemented; its doc claims 6.4× KV memory and 16K in <600 MB. It
does **not** by itself lift the 8192 window (the kernels bake the layout), so it is
only worth starting *after* L1 decides whether the generated attention is viable —
with which it composes.

### L4 — the fused dense FFN (`NPU_QWEN_I4` + `NPU_GUSILU_BF16PAIR`)
A fused int4 GU→SiLU for the dense FFN, self-described as opt-in "until its
per-weight fused corr gate passes". It targets the int8 split path, which nothing
uses by default now, so its ceiling is low — but the flags are a **pair** and the
gate is named, so it is a bounded piece of work for whoever wants it.

### L5 — the second toolchain install *(250 GB of "still needed?")*
`Xilinx2025/2025.2` (75 GB) is *required* — the kernel builders compile against its
`Vitis/aietools/include`, and `aie_clang++` exists only there. `Xilinx/2026.1`
(142 GB) supplies `aiecc`'s PATH. Both are live; neither is a cleanup candidate.

## 4. The Vitis install — repaired, and how to tell

Ten-plus builders under `engine/npu/generators/` compile AIE kernels with
`-I /home/bcloud/Xilinx/2025.2/Vitis/aietools/include`, **a path that does not
exist**: the 2025.2 install lives at `/home/bcloud/Xilinx2025/2025.2`. Nothing at
runtime notices, because the xclbins are prebuilt — which is why it stayed
undiscovered.

**Repair (done):** `/home/bcloud/Xilinx/2025.2 -> /home/bcloud/Xilinx2025/2025.2`,
a symlink. It fixes every reference at once, touches no script, and is reversible.

**Verification — three checks, not one:**

| check | result |
|---|---|
| every 2025.2 path the repo references resolves | `/home/bcloud/Xilinx/2025.2`, `…/Vitis/aietools/include` — both present |
| an AIE kernel compiles through them | `clang++ --target=aie2p-none-unknown-elf -I …/aietools/include -c` → `t.o`, 684 B |
| toolchain binaries present | `v++`, `vitis`, `xchesscc`, `aiecompiler`, `aiebu-asm`, `aie_clang++` all present in 2025.2; 2026.1 has all but `aie_clang++` |
| licence | `/home/bcloud/.Xilinx/Xilinx.lic`, `XILINXD_LICENSE_FILE=/home/bcloud/.Xilinx` |

`settings64.sh` is **missing** from the 2025.2 install and from the 2026.1 top
level (only `Xilinx/2026.1/Vitis/settings64.sh` exists). That is why every script
here sets `PATH`/`LD_LIBRARY_PATH` explicitly instead of sourcing one — and why the
repair is a path fix rather than a reinstall.

## 5. Instruments to measure the next thing

| instrument | what it answers |
|---|---|
| `RT_ARGMAX_MARGIN=1` | is a token disagreement a tie? prints top-2 logits and the gap per step |
| `NPU_FLM_PREFILL=1 NPU_FLM_DECODE=1` | FLM's own compute in this harness — the reference side of every table |
| `NPU_ATTN_CPU=1` | the independent attention reference; the control that localised the >4096 fault |
| `NPU_RUNLIST=1` | the byte-exact int8 path (FLM's own per-ctx layer kernels, driven from here) — the third opinion |
| `NPU_PROMPT_MAX=<n>` | raise the prompt cap deliberately, for lengths without a fast path |
| `NPU_ELF_DEBUG=1` | show where generated per-context ELFs land |

## 6. Post-goal status (2026-09-16) — supersedes the boundary in the header

Pointer: `benchmarks/RESULTS-consolidation-2026-09-16.md` carries the
yardstick-green invocation, the phase-by-phase state, and the boundary list.
Changes since this register was written:

- **The ~6 s per-launch cost is FIXED, not an open lever.** It was the
  `n_grp == 1` core losing its PV/C2 block (host waiting on a C2 task no core
  satisfied); N=512 is now 2.515 ms against the shipped 2.161 ms = 1.16x, NPU==EMU.
- **`ERT_CMD_STATE_TIMEOUT` root-caused and repaired.** It is the amdxdna driver
  TDR (`timeout_in_sec=2`; 83 firmware-timeout dumps, `ctx_pc` matching). Repaired
  durably by `/etc/modprobe.d/amdxdna-tdr.conf` (15) plus an exclusive engine
  `flock` device lock in all 19 `npu_engine_*` binaries.
- **The bf16 attention-ELF loader was compiled out** (gated on Clang's
  `__has_embed`, absent under the g++ build); fixed and verified on the supported
  models — the runtime `attn_mha_*` ELFs are now loaded.
- **The generated nh20/nkv4/cols4 head-block kernel passes its bench gate**
  (NPU==EMU at seq=2048 and 513, C2 2/2).
- **Beyond 8192 is bounded**, not merely unexplored: N=3072 builds and gates,
  N=4096/8192 fail with AIE program-memory overflow, so the N=16384 route needs a
  chunked/multi-core design.
- **Family >1024 remains outside the set**: the generated ELF is `AttnCtx`-ABI and
  `Bf16Mm` drives FLM's `(act,out,kv)` ABI; the four 2k capture candidates either
  hang or (308736) compute the wrong attention (boot 152349 vs CPU 456).

The yardstick is green (`0 PARITY CLAIM REFUSED`, runlist 109.9% of FLM decode) when
invoked with the goal's engine binary — its default `ROOT` points at another
worktree and measures a stale binary.

### 6.1 NPU device-context construction is a first-class hazard (2026-09-16, later)

- **Construct every `hw_context`/xclbin registration/data-BO set BEFORE the run
  starts.** Building an `AttnCtx` lazily *inside* the prefill — while five
  `Bf16Ctx` contexts plus Bf16Mm's decompression/mm contexts are already live on
  the same device — made the attention output (and the token) vary run to run:
  the same 8-token prompt returned 152369 / 152367 / 8900 / 6037 / 152360 /
  152552 / 2092. Moving the construction to just *before* `bf16mm_init` removed it
  completely (13/13/13/13, and 764/764 at 1202 keys). Ruled out along the way:
  contention, host-thread races (`NPU_HOST_THREADS=1` still varied), stale caches,
  and the kernel itself (it syncs; the standalone bench was always deterministic).
  This applies to any generated-kernel route, not just the family adapter.
- **Corrected family >1024 status:** the `AttnCtx` adapter now drives the generated
  kernel in a real prefill deterministically, but Nanbeige is still **not** at
  parity, and the cause is now DECISIVE rather than suspected. `NPU_ATTN_EMU_DIFF`
  shows the NPU and the AttnCtx's **own host EMU** agree to ~1e-4 while **both**
  diverge from the engine's float `attn_omp` by 0.2-1.2 max / 0.01-0.044 mean
  (L0 control 2.1e-4). So the kernel is faithful and the gap is the **int8
  attention contract**: quantising the prefill's Q/K/V (max|q| = 26.75 -> int8
  step ~0.21) into int8 Q, int8 K, int8 A2 and int8 V with a GLOBAL sq/sk costs
  0.01-0.044 mean on the attention output, compounding with depth. The bench's
  smaller 1.2e-1 figure is just its smaller synthetic dynamic range. Ruled out
  along the way: pre-RoPE convention (`attn_omp` has no RoPE; both paths get the
  same post-RoPE bytes) and score-range saturation (measured spans only 4-11).
  Parity therefore needs a **wider generated kernel** (bf16 Q/KV, or a wider A2) —
  the same dtype change the beyond-8192 route needs (int8 KV there vs bf16 in the
  dense path). Bench gate remains NPU==EMU 8.575258e-02 (the kernel matches its
  *own* EMU).
