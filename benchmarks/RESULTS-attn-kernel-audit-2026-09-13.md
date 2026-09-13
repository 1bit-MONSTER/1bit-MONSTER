# RESULTS — independent audit of the captured attention kernel (2026-09-13)

Independent recomputation, from scratch, of the claim in
`benchmarks/RESULTS-bf16-prefill-CORRECTION-2026-09-12.md` § *The captured
attention kernel's numerics are not the reference's* — the claim that the
embedded attention ELF "does not use the reference softmax scale" (row-1 implied
mixing weight 0.727 where `softmax(q·k/√128)` gives 0.826; "scale 1/16 gives
0.750").

Trigger: `research/TRACKING.md` recorded that claim as verified. It is the kind
of claim that gets quoted for a year, so it was worth attacking directly.

## Method

One engine run, correct configuration, on this box (Qwen3-0.6B):

```
NPU_RUNLIST=0 NPU_PREFILL_BF16=1 NPU_PREFILL_MAX=256 NPU_DUMP_ATTNIO=1 \
  npu_engine_qwen3_0_6b <model.q4nx> 1 ids256.txt
  -> Prefill: 352ms (1.375 ms/tok) [GEMM 30ms, attn 132ms, ...]   [0] boot=1614
```

`boot=1614` is the token the two trusted paths also give at 256 tokens, so this
is a *correct* configuration (unlike the 512-capped run whose dumps were the
only ones left in `/tmp`). The run dumps layer-0 attention I/O:

- `eng_act.bin` — Q, 256 rows x 2048 bf16
- `eng_kv.bin` — K/V, 4 regions x 4194304 bf16 (r0/r1 = K heads 0-3 / 4-7,
  r2/r3 = V heads 0-3 / 4-7, 512 values per token)
- `eng_out.bin` — the kernel's attention output, same shape as Q

Dumps kept at `~/npu-build/parity/dump256_correct/`, engine log at
`~/npu-build/parity/audit_256.log`. The audit code is a from-scratch numpy
reimplementation (`attn_audit.py`), not a re-run of the branch's `check_attn.py`.

**Q and K are post-RoPE.** `qk_norm_pi` applies the norm weights and then
`ra(...)` (RoPE) to both, and the KV buffer is built from those same values
("build bKv directly from the norm'd+RoPE'd ks/vs"). So a host recomputation
over the dumped Q/K/V is comparing attention *arithmetic only* — the comparison
is structurally sound. (`check_attn.py`'s docstring says Q is "PRE-RoPE'd",
which is misleading but does not change what it computes.)

## What reproduced

- The branch's `check_attn.py` and my independent implementation agree to the
  digit: **17/256 query rows match within 0.05**, row 1 `max|d|` = 0.1891,
  row 2 = 0.6253, row 5 = 0.0844. The measurement is reproducible.
- The head -> KV-head mapping is **correct**: for all 16 heads at row 1, the
  kernel's output lies in the span of *its* expected kv-head's V vectors
  (residuals 0.8 %–2.7 %), and no other kv-head fits. So the mismatch is not a
  layout artifact of the checker.

## What did NOT reproduce, and why the stated mechanism is wrong

1. **"Row 255 matches to ~4 bf16 ULP (0.008)"** — not reproducible. On this
   dump row 255 gives `max|d|` = **0.194**, and the pattern is not
   "good long rows / bad short rows". Row 191 is worse than row 63. The original
   figure came from a dump that no longer exists (the `/tmp` dumps left over were
   from a 512-token run that was capped to 256 rows — `boot=132352`, a *wrong*
   token — so they were never a sound basis).
2. **Row 0 matching is degenerate.** Row 0 has one key, so its attention output
   *is* V0 by construction, whatever the kernel does with scores. It cannot be
   cited as agreement.
3. **The deviations are not a single scale error.** Under a wrong softmax scale,
   every head would be off by the *same* factor. Measured per-head implied scales
   on row 1: 0.46x … 1.85x of `1/√128`, plus one head whose implied weight
   (0.178) is far below the reference (0.548) and yields a *negative* implied
   scale — impossible for a convex mixing weight. Head-averaged implied w1 is
   0.7463 against a reference 0.7630 (not 0.727 against 0.826).

   The heads that do agree are the peaked ones (implied w1 ≈ 0.998 vs reference
   0.998–1.000), i.e. the rows where softmax is nearly 0/1 and carries almost no
   information about the scale.

## What survives

- The kernel's attention output **does** deviate from reference causal softmax
  on a substantial fraction of rows (239/256 above the 0.05 threshold), and the
  deviation is real: not a layout error, not a RoPE mismatch, not row-0
  degeneracy.
- Therefore an **argmax boot-token gate is a weak correctness test** for any path
  that composes these kernels differently — that conclusion stands, and it is
  the one the correction drew.
- The **256-row cap** that forced the @1k prefill withdrawal is independent of
  all this and was verified separately (engine code + boot-token re-gate), so the
  withdrawal itself is unaffected.

## What is now unverified rather than established

The *mechanism* — "the kernel uses the wrong softmax scale, ≈1/16" — is **not
supported**. The honest description is "the kernel's attention weights differ
from reference softmax in a head-dependent way", which is consistent with
several causes and is not yet attributed:

- bf16 rounding on short rows (the original note itself hedged this),
- per-head scaling or a different exp path inside the captured kernel,
- the possibility that the kernel is not computing textbook causal softmax at
  all.

Discriminating needs a dump of the kernel's *scores* (not just inputs/outputs),
or a float reference run of the same model — neither exists today.

## Reproduce

```
NPU_RUNLIST=0 NPU_PREFILL_BF16=1 NPU_PREFILL_MAX=256 NPU_DUMP_ATTNIO=1 \
  engine/npu/build/npu_engine_qwen3_0_6b ~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx \
  1 ~/npu-build/parity/ids256.txt          # expect boot=1614, dumps in /tmp
mkdir -p /tmp/audit && cp /tmp/eng_*.bin /tmp/audit/   # /tmp is hardcoded; copy first
python3 benchmarks/tools/attn_audit.py /tmp/audit --map
```

`benchmarks/tools/attn_audit.py` is the from-scratch checker used here (row-level
agreement, per-head implied scales, and the head→kv-head mapping test). Expected
output on the dump this doc describes: `17/256` rows matching, row 0 = 0.0177,
row 255 = 0.1940, implied scales spanning 0.46×–1.85× with one negative, and
`0/16` heads failing the mapping test.
