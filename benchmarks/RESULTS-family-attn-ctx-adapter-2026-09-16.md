# Family attention via the AttnCtx adapter (Nanbeige), 2026-09-16

## The ABI gap

The generated family attention kernel (`n1_core_attn.py`, nh20/nkv4/hd128, N=2048)
has the **AttnCtx** signature — `attn(k(3, instrBO, instrSize, Q, KT, C2, V, SCR))`,
the same ABI `zaya_decode.cpp` drives. `Bf16Mm::run_attn` drives a *different*
kernel: FLM's captured `(act, out, kv)` ABI. So pointing Nanbeige's bf16 prefill at
the generated ELF cannot work by construction, and the prefill fell through to the
captured short-context ELF / the CPU fallback (the doc-level "not through that ELF").

## The adapter

`engine/npu/src/npu_engine_universal.cpp` now carries an `NPU_ATTN_CTX=1` path in
the bf16 prefill. When the shape is nh20/nkv4/hd128 it constructs the same `AttnCtx`
`zaya_decode.cpp` uses (`NPU_ATTN_XCLBIN` / `NPU_ATTN_INSTS` / `NPU_ATTN_MAX_SEQ`)
and, because the generated kernel is decode-shaped (M=8), issues one query row per
call over the prefill block, writing the float result back to `bA` as bf16. It is a
**correctness** path (much slower than the captured short-context ELF), gated off by
default.

The float Q/K/V are already present (`bqo` after host q_norm/k_norm + RoPE, and
`kv_caches[l][0].k/v`), so no new packing was needed.

## Measurements (device lock held; NPU_ATTN_KV_REGION=4194304)

```
NPU_ATTN_CTX=1 NPU_PREFILL_BF16=1 NPU_BF16=1 NPU_ATTN_KV_REGION=4194304 \
NPU_PREFILL_MAX=2048 NPU_ATTN_COLS=4 \
NPU_ATTN_XCLBIN=.../attn_gen_2048_nh20_hd128.xclbin \
NPU_ATTN_INSTS=.../attn_gen_2048_nh20_hd128_insts.txt NPU_ATTN_MAX_SEQ=2048 \
  ./engine/npu/build/npu_engine_nanbeige4_1_3b <model.q4nx> 4 <ids.txt>
```

| prompt | keys | prefill | boot | notes |
|---|---:|---:|---:|---|
| repeated passage (raw ids) | 1172 | 340.4 s | **13** | 32/32 layers `[NPU_ATTN_CTX]`, attn 338 s, 0 crash |
| `/tmp/nb_ids1500.txt` | 1500 | 457.8 s | 152343 | 32/32 layers, attn 456 s, 0 crash |

No hang and no ERT at either length — the AttnCtx ABI drives the generated kernel
end to end, which is the ABI-adapter result the family lane needed.

## Identity vs `flm run nanbeige4.1:3b`

Same passage, same model dir, `flm run nanbeige4.1:3b -c 4096`:

- FLM reports `Prefill chunk 1/1 with 1202 tokens` and its first generated token is
  `<n` (its markup for a newline) = **token 13**. Our path's boot is **13**. First
  token matches.
- **Caveat (why this is not yet a strict identity):** FLM applies the model chat
  template (1202 tokens) while the engine consumed the raw 1172 ids. A strict
  token identity needs the template-tokenised ids on both sides. Also the earlier
  1500-key run was not compared against FLM (only the 1172-key one was).

## Boundary

- With the CPU / captured-ELF fallback the same prompts give context-free-looking
  boots (13 and 51752 across runs) — the signature the family doc recorded. The
  AttnCtx path is the first family path that runs the **generated** attention.
- The `NPU_ATTN_CTX` default xclbin path points at the goal tree's
  `engine/npu/xclbins/`, which does not carry the nh20 build yet — pass
  `NPU_ATTN_XCLBIN`/`NPU_ATTN_INSTS` (the family worktree's artifacts) until it is
  vendored.

## Addendum: the strict (template-matched) identity FAILS

The 1172-token raw-id run's "boot 13 == FLM's first token 13" was a **coincidence**
(newline is a common opener); the strict test resolves it.

Building the prompt with the model's own chat template (`tokenizer_config.json`'s
`chat_template`, rendered with jinja2) yields **1202 tokens** — exactly the count
`flm run nanbeige4.1:3b` reported for the same passage. Running the engine on those
ids:

```
NPU_ATTN_CTX=1 ... /tmp/nb_ids_tmpl.txt   (1202 ids, template-matched)
-> Prefill 1202 [bf16], 32/32 layers [NPU_ATTN_CTX], attn 334 s
-> boot = 764
```

FLM's first generated token on the same passage is **13**. So with the tokenisation
matched, the engine's first token is **764 vs FLM 13 — NOT identical.** The earlier
"first token matches" line must be read as raw-id-only and is superseded by this.

Conclusion for the family status: the AttnCtx adapter is a working **vehicle** (the
generated kernel drives a real prefill, 32/32 layers, no hang/ERT, and the kernel
itself gates NPU==EMU on the bench), but Nanbeige is **not at token parity**: at
>1024 keys the end-to-end prefill diverges from FLM (764 vs 13). The divergence is
downstream of the attention kernel — candidates are the bf16 prefill's Q/K/V
normalisation/RoPE ordering relative to the kernel's pre-RoPE contract, the GDN
layers in the 32-layer stack, and the chat-template system prefix (the template is
Nanbeige's tool-calling template, whose rendered 1202 tokens include a system
section that may not be what FLM actually feeds).

So: cited **partial** for Nanbeige — attention ABI route established and driven;
token parity at >1024 keys NOT established, with the mismatch pinned to the prefill
stack rather than the kernel (the kernel is separately gated).

## Addendum 2: the AttnCtx path is NON-DETERMINISTIC in the engine (blocker)

Chasing the 764-vs-166103-vs-13 discrepancy exposed something more basic: **the
`NPU_ATTN_CTX` path does not return the same answer twice.**

Same 8-token prompt, byte-identical command, repeated:

| path | runs |
|---|---|
| `NPU_ATTN_CTX=1` (generated kernel via AttnCtx) | 152369, 152367, 8900, 6037, 152360, 152552, **2092** |
| captured-ELF bf16 prefill (no AttnCtx) | 152343, 152343 — **stable** |
| plain fallback (no bf16 prefill) | 152367, 152367 — **stable** |

Ruled out:

- **Device contention** — no other process on `/dev/accel/accel0`, no new
  `Firmware timeout state capture` in dmesg during these runs, `device.lock` held.
- **Host-thread race on the KV cache** — `NPU_HOST_THREADS=1` still varies
  (152360 / 152552 / 2092).
- **A stale-KV cache in `AttnCtx::run`** — `repack` is true on every call here
  (`seq` strictly increases within a layer, and `kv_caches[l]` moves per layer).
- The kernel itself syncs (`r.wait(); bC2->sync(FROM_DEVICE)`), and on the standalone
  bench the same AttnCtx reproduces NPU==EMU to 8.575258e-02 across runs.

So the variance is introduced by running AttnCtx **interleaved with the engine's
other NPU contexts** (the bf16 GEMM contexts on the same device) — a condition the
standalone bench never exercised. It is not a property of the generated kernel.

**Consequence: no token identity is claimable for Nanbeige.** The adapter is a
working *vehicle* (32/32 layers, no hang, no ERT, and the kernel gates NPU==EMU on
the bench), but this path cannot support a parity claim until the variance is
root-caused. The earlier "boot 13 == FLM 13" line is fully retracted; the 1202-token
template-matched run (764) and the CPU-attention run (166103) are both unreliable
**for the AttnCtx side**, though the CPU-attention run is itself deterministic.

Next diagnostic (if resumed): dump the AttnCtx's layer-0 output inside the engine
and compare against the host packing for the same row, and check the engine's
hw_context count at the time AttnCtx launches (`hwctx_limit=16`) — if the engine
already holds its full complement of contexts, the AttnCtx launch may be silently
degraded. Also worth testing: create the AttnCtx **before** the bf16 contexts, and
force an explicit device-level sync between the last GEMM and the AttnCtx launch.

## Addendum 3: the non-determinism is ROOT-CAUSED and FIXED (construct AttnCtx early)

**Cause.** The `AttnCtx` (its own `hw_context`, xclbin registration and data BOs)
was being constructed **lazily inside the prefill's attention block**, i.e. in the
middle of a run that is already driving five `Bf16Ctx` contexts plus Bf16Mm's
decompression/mm contexts on the same device. Registering a new context and
allocating its BOs at that point perturbs the in-flight work, and the delivered
attention (and hence the token) varies run to run.

**Fix.** Construct it **before** the bf16 contexts: `ac_ctx_init(dev, NH, NKV, HD)`
is called just before `bf16mm_init(...) && npu_bf16_prefill_init(...)`, and the
attention block only *uses* the pre-built context.

**Result — deterministic.**

| case | before | after (early init) |
|---|---|---|
| 8-token prompt | 152369 / 152367 / 8900 / 6037 / 152360 / 152552 / 2092 | **13, 13, 13, 13** |
| 1202-token templated ids | (not reliable) | **764, 764** (prefill 335 s) |

Also confirms the two reference paths were never the problem (captured-ELF bf16
= 152343 x2, plain fallback = 152367 x2).

**So the generated kernel now drives a real prefill reproducibly.** The adapter is
no longer merely a "vehicle"; its output is stable. Note the `[NPU_ATTN_CTX] EARLY
init OK` line is printed once, before `bf16 contexts ready`.

## But Nanbeige is still NOT at parity — and now the reason is measurable

With determinism restored, the 1202-template-matched prompt gives **764**, where
`flm run nanbeige4.1:3b` gives **13**. Two separate divergences are now visible:

1. **AttnCtx vs CPU attention, same in-situ Q/K/V** — the deterministic CPU
   (`NPU_ATTN_CPU=1`) prefill gives **166103**, the AttnCtx path gives **764**. The
   bench already showed the generated int8 attention carries max abs error
   ~8.5e-2 against its own EMU on an output whose mean magnitude is ~3e-2 — large
   enough to move the final argmax. So the int8 generated attention is too lossy
   for this bf16 prefill at long context, independent of any race.
2. **Prefill stack vs FLM** — the CPU-attention path (float, the trusted reference
   for attention) gives 166103, which also differs from FLM's 13. So something
   upstream of attention (Q/K/V norm+RoPE ordering vs the kernel's pre-RoPE
   contract, the GDN layers in the 32-layer stack, or the rendered template's
   system section vs what FLM actually feeds) diverges from FLM.

Cited conclusion for Nanbeige: **adapter fixed and deterministic; parity not
established**, with the two divergences above explicitly localised.

## Addendum 4: the AttnCtx attention does NOT match the engine's own attention reference

With the path now deterministic, `NPU_ATTN_DIFF=1` compares, in-situ, the AttnCtx
result (`bA`) against the engine's float attention (`attn_omp -> bat`) on the SAME
Q/K/V, per layer. 8-token prompt, nh20/nkv4/hd128:

```
[ATTN-DIFF L0] npt=8 max|npu-host|=0.152134 at (tok 1, dim 689) | max|bActQ|=26.75 max|bKv|=19.5 | npu[0][0]=0.234375 host[0][0]=0.234375
[ATTN-DIFF L1] max|npu-host|=0.197524  (tok 7, dim 1003)
[ATTN-DIFF L2] max|npu-host|=0.784974  (tok 7, dim 1445)
[ATTN-DIFF L3] max|npu-host|=0.349967  (tok 7, dim 2018)
[ATTN-DIFF L4] max|npu-host|=0.818040  (tok 4, dim 1428)
[ATTN-DIFF L5] max|npu-host|=0.690879  (tok 7, dim 1132)
[ATTN-DIFF L6] max|npu-host|=0.591314  (tok 5, dim 2243)
[ATTN-DIFF-H0] per head (max|npu-host| / max|npu| / max|host|):
  h5: 0.5323/0.3203/0.2415  h6-h9: 0.4827/0.2412/0.2415  h10-h14: ~0.295/0.154/0.154
```

Readings:

- **`npu[0][0] == host[0][0]` exactly on every layer** — token 0 agrees; the
  divergence is at later tokens (1, 4, 5, 7) and concentrates in specific heads
  (h5-h9 worst). That is a *content* disagreement, not a uniform rounding offset.
- **Magnitude:** 0.15-0.82 against head outputs scaled ~0.15-0.32 — far larger than
  the bench's `NPU==EMU 8.575258e-02` (which is max abs error on an output whose
  mean magnitude is ~3e-2). So the kernel matches **its own** host EMU, but the
  EMU's contract does **not** match the engine's `attn_omp` on the same data.

- Prime suspect: the **pre-RoPE / scale contract**. The bf16 prefill's own comment
  says the captured `attn.xclbin` "expects PRE-RoPE'd Q and K + raw V — the host
  applies q_norm/k_norm + RoPE, the kernel does NOT". The AttnCtx (written for
  Zaya's decode) computes its own global `sq`/`sk` and per-dim `sv` and quantises
  Q/K/V to int8 — a different contract from the engine's float `attn_omp`. Feeding
  the AttnCtx the bf16 path's post-RoPE `bqo` therefore does not reproduce
  `attn_omp`.

This is now the sharpest localisation available for Nanbeige: the divergence is in
the **AttnCtx's Q/K/V contract vs the engine's attention**, not in the kernel and
not in a race. Next step would be to make the two agree on one layer (same Q/K/V,
same scale/RoPE convention) before any parity claim.

## Addendum 5: the pre-RoPE hypothesis is RULED OUT; the gap is bigger than the bench's int8 error

Two checks on addendum 4's prime suspect:

1. **`attn_omp` applies no RoPE.** Its body is `scores[p] = (q·k)/sqrt(HD)`, softmax,
   then `sum_p softmax * v` — no rotation anywhere (npu_engine_universal.cpp:556).
   It consumes `bqo` (already q_norm/k_norm'd + RoPE'd) and the KV cache. The AttnCtx
   adapter passes the **same** `bqo` row and the **same** `kv_caches[l][0].k/v`. So
   both sides already receive post-RoPE Q/K — feeding AttnCtx "pre-RoPE" Q/K would
   make it *differ* from `attn_omp`, not match it. The pre-RoPE hypothesis is dead.

2. **The in-situ gap is larger than the kernel's own int8-vs-float error.** The
   standalone bench (same nh20 kernel, CK_NQ=20/CK_NKV=4/CK_HD=128/COLS=4) reports,
   for the EMU against the bench's float reference:

   | seq | max_abs_err | mean_abs_ref | max_abs_ref |
   |---:|---:|---:|---:|
   | 8 | 1.202951e-01 | 3.627920e-01 | 2.008213e+00 |
   | 32 | 4.889638e-02 | 2.039903e-01 | 1.053947e+00 |
   | 128 | 1.115882e-01 | 1.169874e-01 | 1.101512e+00 |

   So the int8 contract's own error against float at seq=8 is ~1.2e-1 — whereas the
   in-situ `NPU_ATTN_DIFF` at 8 tokens is up to **8.2e-1**. A 7x gap is not explained
   by quantization alone (though the magnitudes differ, so this is indicative, not
   exact).

So the disagreement is *not* a RoPE convention and *not* pure int8 rounding. What
remains: the AttnCtx's per-call contract differs from `attn_omp` in a way the bench
cannot see, because the bench feeds the AttnCtx from **its own** reference buffers
while the engine feeds it the prefill's `bqo`/KV cache. Concrete next probe: dump the
AttnCtx's layer-0 `C2` **inside the engine** (`NPU_ATTN_DUMP=1` with
`NPU_ATTN_DUMP_SEQ` set to the first prefill call) and recompute the expected C2 from
the same `bqo`/KV-cache bytes on the host — is the difference (a) the Q/K/V bytes the
engine hands over, or (b) the AttnCtx's internal scale/quantisation? That separates
"wrong input" from "wrong contract" in one run.

## Addendum 6: in-engine dump — the AttnCtx softmax output is SATURATED (wrong input)

`NPU_ATTN_CTX=1 NPU_ATTN_DUMP=1 NPU_ATTN_DUMP_SEQ=8` on the 8-token prompt:

```
[attnDump] params0=5.198035e-03 seq=8
[attnDump] pass0 C2 head0 nonzero=128
[attnDump] pass1 C2 head0 nonzero=128
[attnDump] C2 head0 nonzero idx: 0 1 2 3 4 5 6 7 64 65 ... 448 449 450 451  (mmul C row-0 pattern)
[attnDump] A2 head0 t=0..3: 127 127 127 127 | t=128..131: 0 0 0 0
Segmentation fault (core dumped)
```

**The A2 (the kernel's int8 softmax output) is saturated at 127 for every token
shown**, where the standalone bench delivered `2/8/8/8/3/27/3/5/9` and the host A2
expectation agreed with it exactly. A saturated softmax output means the scores the
kernel sees are far too large (or the per-call params/scale is wrong) — i.e. the
engine is feeding the AttnCtx something that does not match the contract the bench
exercises. That is the **"wrong input"** branch, not merely a contract/precision
disagreement.

This also explains the in-situ `NPU_ATTN_DIFF` magnitude (0.15-0.82) being ~7x the
bench's int8-vs-float error (1.2e-1 at seq=8): a saturated A2 is not quantisation
noise, it is a different computation.

Also note: **the DUMP path itself segfaults** (core dumped after the A2 line) when
run in-engine — the diagnostic code assumes the bench's buffer sizes/state. That
must be fixed before the C2raw-vs-expv comparison can complete; treat the printed A2
as the usable signal for now.

Concrete next step: instrument the AttnCtx's per-call params inside the engine
(print `sq`, `sk`, `sv[0..3]`, and the packed Q/K extremes for the seq=8 call) and
compare with the same quantities the bench computes for its own buffers — the one
that is off identifies the input conversion (most likely the Q/K scale, since
`max|bActQ|=26.75` in-engine).

## Addendum 7: the input conversion is CORRECT — the AttnCtx softmax saturates on large scores

`NPU_ATTN_DBG=1` prints the AttnCtx per-call scale in-engine (`npu_attn_ctx.h`, gated,
first three calls only):

```
[ACTX-DBG] seq=1 nq=20 nkv=4 hd=128 sq=14.2098 sk=17.0042 maxq=8.9375  maxk=7.46875 maxv=0.234375 p0=0.000365807 sv0..3=0.001845 9.948e-05 0.0003153 0.0002922
[ACTX-DBG] seq=2 nq=20 nkv=4 hd=128 sq=4.74767 sk=17.0042 maxq=26.75   maxk=19.5    maxv=0.4375   p0=0.00109486  sv0..3=0.001845 0.0003499 0.0007382 0.001146
[ACTX-DBG] seq=3 nq=20 nkv=4 hd=128 sq=8.19356 sk=17.0042 maxq=15.5    maxk=19.5    maxv=0.4375   p0=0.000634405 sv0..3=0.001845 0.0003499 0.0007382 0.001146
```

The conversion arithmetic is **correct**: `sq = 127/max|q|`, `sk = 127/max|k|`,
`p0 = 1/(sq·sk·sqrt(hd))`, and the per-dim `sv` is positive and stable. So the
in-engine **input conversion is not the defect** (it is the same formula the bench
uses and the same one `zaya_decode.cpp` uses).

What the numbers do reveal is the **score magnitude**: with `max|q|=26.75` and
`max|k|=19.5`, the raw `q·k/sqrt(hd)` is of order `128 · 26.75 · 19.5 / 11.3 ≈ 6e3`
(int8-scaled: `127·127·128 · p0 ≈ 2.3e3`). The softmax must therefore subtract a
large max to stay in range — and addendum 6 shows the delivered **A2 saturated at
127**. So the working hypothesis is now:

> The AttnCtx's softmax/A2 quantisation does not handle the large raw score range the
> bf16 prefill produces (post-RoPE Q/K with max ~20-27), while Zaya's decode — the
> path AttnCtx was written for — feeds bounded, normally-distributed Q/K, as does the
> standalone bench.

That is a **contract range limit**, not a wiring bug. It also matches the shape of the
evidence: `npu[0][0] == host[0][0]` exactly (single-key case is just V, no softmax
pressure), while the error grows with the number of keys and concentrates in heads
whose scores are largest.

Options if this line is continued: (a) scale Q/K into the AttnCtx's expected range
before the call and undo the scale after (the params already carry a scale, so a
bounded pre-normalisation is cheap); (b) add a max-subtraction/max-aware A2 to the
kernel; (c) accept that the generated int8 kernel serves bounded-Q/K families only,
and cite the range limit as the exclusion for the bf16 prefill. The `NPU_ATTN_DBG`
probe stays in `npu_attn_ctx.h` (env-gated, first three calls only) for the next
round.

## Addendum 8: the score-range estimate was WRONG — scores are small (range hypothesis refuted)

Addendum 7 estimated the raw score magnitude at ~6e3 from `max|q|·max|k|·hd`. That
estimate was an upper bound assuming every term aligns; it is not what the data does.
Measured directly in-engine (`NPU_ATTN_DBG=1 NPU_ATTN_DBG_SEQ=8`, head 0, `q·k/sqrt(hd)`
over the 8 keys), per layer:

```
L0: min=0        max=0        span=0        (first call, KV not yet populated)
L1: min=-0.679   max=5.059    span=5.738
L2: min=-6.637   max=3.987    span=10.624
L3: min=-1.133   max=3.026    span=4.159
L4: min=-4.172   max=4.115    span=8.287
L5: min=-4.776   max=3.610    span=8.386
```

**Spans of 4-11 are ordinary softmax territory**, so the AttnCtx softmax has no range
problem here and the "large-Q/K score range" explanation is **refuted**. That also
removes the reason to trust addendum 6's `A2 = 127` reading: with normal scores the
softmax cannot saturate, and the DUMP path is the one that *segfaults* in-engine, so
`A2 = 127` (0x7F — a fill pattern) is most likely a bad SCR read in the diagnostic,
not the kernel's output.

Where this leaves the Nanbeige gap — the surviving explanation is **precision**, which
addendum 5's ruling-out does not actually cover:

- The bench's int8-vs-float error was measured on the **bench's own** random buffers.
  In-engine the data is larger (`max|q| = 26.75`, `sq = 4.75` -> int8 step ~0.21 in
  q-units), so the same int8 contract is materially coarser on the real prefill than
  the bench numbers suggest — which is exactly why the bench's 1.2e-1 does not bound
  the in-situ 0.82.
- The generated kernel is **int8 KV + int8 Q**, while the dense bf16 prefill is bf16
  throughout — the dtype mismatch @agent-baaa57 flagged in the beyond-8192 handoff.

So the honest status for Nanbeige: the adapter is fixed, deterministic, and its input
conversion is verified correct; the residual divergence is a **precision/contract
mismatch (int8 attention vs the bf16 prefill)** compounded by the prefill stack itself
differing from FLM even with float attention (CPU path 166103 vs FLM 13). Closing it
needs either a bf16/higher-precision variant of the generated kernel or a
quantisation scheme matched to the prefill's value range.

## Addendum 9: DECISIVE — it is the int8 contract, not the kernel (npu == its own EMU)

`NPU_ATTN_EMU_DIFF=1` runs the AttnCtx's **own host EMU** (`run_emu`, the
float-dequantised int8 contract) and `attn_omp` (pure float) on the same in-situ bytes,
then compares both to `attn_omp`. Last row of the 8-token block, per layer:

| layer | max\|emu-float\| | max\|npu-float\| | mean\|emu-float\| | mean\|npu-float\| |
|---|---:|---:|---:|---:|
| L0 | 0.000214 | 0.000290 | 3.11e-05 | 3.76e-05 |
| L1 | 0.197073 | 0.197524 | 0.0110877 | 0.0110882 |
| L2 | 0.784570 | 0.784974 | 0.0229454 | 0.0229443 |
| L3 | 0.351938 | 0.349967 | 0.0285645 | 0.0285644 |
| L4 | 0.635414 | 0.634633 | 0.0409032 | 0.0408971 |
| L5 | 0.691580 | 0.690879 | 0.0433834 | 0.0433841 |
| L7 | 1.231850 | 1.233040 | 0.0441776 | 0.0441805 |

**`npu` and `emu` agree to ~1e-4 while BOTH diverge from float by 0.2-1.2 max /
0.01-0.044 mean.** That settles it:

1. **The kernel is faithful** — it reproduces its own host contract essentially
   exactly (and this is the same kernel that gates NPU==EMU 8.575258e-02 on the bench).
2. **The disagreement is the int8 contract itself**: quantising the prefill's
   Q/K/V (max|q| = 26.75 -> int8 step ~0.21) costs 0.01-0.044 mean on the attention
   output, and up to 1.23 worst-case. The standalone bench's 1.2e-1 figure is smaller
   only because the bench's synthetic buffers have a smaller dynamic range.
3. **L0 is the control**: at layer 0 the same contract costs 2.1e-4 — the contract is
   not "broken", it is *precision-limited on this data*, and the error compounds with
   depth (L1 -> L7).

So the residual Nanbeige gap needs a **higher-precision generated attention** (a
bf16-KV/bf16-Q variant, or a quantisation scheme matched to the prefill's range), not
a wiring or kernel fix. This is the same dtype mismatch @agent-baaa57 flagged for the
beyond-8192 route (int8 KV there vs bf16 KV in the dense bf16 path).

Cited status for Nanbeige: the `AttnCtx` adapter is implemented, deterministic, and
verified correct at the interface; end-to-end parity is **blocked by the int8
attention contract** (quantified above) and, independently, by the bf16 prefill stack
differing from FLM even with float attention (CPU 166103 vs FLM 13).

## Addendum 10: term attribution — int8 Q/K dominate; A2 and V are negligible

Env-gated EMU variants in `npu_attn_ctx.h` (`NPU_ATTN_EMU_FLOAT_A2`, `NPU_ATTN_EMU_FLOAT_V`)
keep the int8 path for every term except the one under test, then re-run
`NPU_ATTN_EMU_DIFF` (8-token prompt, last row, per layer):

| variant | L1 max\|emu-float\| | L2 max\|emu-float\| | L7 max\|emu-float\| |
|---|---:|---:|---:|
| int8 everywhere (baseline) | 0.197073 | 0.784570 | 1.231850 |
| **float A2** (int8 Q/K/V) | 0.192873 | 0.781741 | 1.205480 |
| **float V** (int8 Q/K/A2) | 0.195907 | 0.781921 | 1.233460 |

Making the softmax output exact changes the divergence by ~2%; making V exact
changes it by ~1%. Both are noise next to the 0.197-1.232 baseline. **So the
residual Nanbeige gap is the int8 Q/K quantisation** (the QK^T), which is the one
term neither variant touches — and it is the term a per-dim/per-head/row scale cannot
rescue (a per-dim K scale lives inside the mmul sum; a per-head/row Q scale is a
temperature).

**Consequence for the fix:** a targeted change is worth doing here — a
**bf16/mixed-precision QK^T** (wider score accumulation) rather than a full bf16
rewrite of the whole kernel, since the PV/A2 sides are already effectively exact at
int8. That is also the term the beyond-8192 route needs addressed (int8 KV there vs
bf16 in the dense path).

The two variants are committed env-gated and OFF by default (`emu_vo` is only set by
the `NPU_ATTN_EMU_DIFF` probe). Both were corrected during this work: the first cut
applied a spurious extra `/127` to the float-V dequantisation (`sv` already carries
`max|V_d|/127`), which made float-V look 127x wrong; the fixed numbers are above.

Also reconfirmed while toggling: `npu-float` is invariant across the variants
(0.197524 / 0.784974 / 1.23304), i.e. the NPU output does not depend on the EMU
variant — the earlier apparent dependence was a line-selection artefact in the
paired-loop shell, not a real effect.
