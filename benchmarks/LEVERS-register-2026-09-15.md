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

### L2 — a Nanbeige nh20 capture at 4096 *(the one non-dense family already close)*
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
