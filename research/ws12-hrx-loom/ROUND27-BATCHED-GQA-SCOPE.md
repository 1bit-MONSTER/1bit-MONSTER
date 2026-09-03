# Round 27 — scope: land prefill GQA-batched attention mms on HRX2 (close the 25i pp32 regression)

**Date:** 2026-09-02 (scoping, read-only — no fork edits made)
**Live fork:** `~/hrx-ws/hrx-v2-src` (HEAD `8df3330`, round-25t era). NOTE:
`~/hrx-gfx1151/llama-src` is the stale 95b1a89 snapshot (round-4 base) — the
rounds run out of `hrx-ws`. Harnesses link it via `harnesses/build.sh`
(`FORK=/home/bcloud/hrx-ws/hrx-v2-src`).
**Goal:** reverse the round-25i prefill regression (3B pp32 121.5 → 27.8
t/s) by giving the KQ^T/kqv attention mms an HRX2 route at prefill shapes —
the bounded move; NOT a fused attention kernel (see "Non-goals").

## Why this is the bounded move (recap)

The fused-attention direction has the workstream's worst failure mode —
silent numerical wrongness (R20/21 JIT-folded tables, R25b bug-replicating
reference, R25d silent fallback, R12 transpose) — and, for decode shapes,
requires solving loom's runtime-variant-scalar question (loom command
programs: "a scalar used by varying device arithmetic must live in
device-visible storage"; the type checker rejects non-buffer scalar launch
bindings — `loom/docs/src/guide/command-programs.md:42-50,79-86`). The
existing ggml-hrx2 catalog only ever carries JIT-time shape config + one
4-byte constant (softmax scale). Prefill attention is FIXED-shape at graph
build, so it never touches that unknown.

## What already exists (AMD's, dormant for prefill)

All paths relative to `~/hrx-ws/hrx-v2-src/ggml/src/ggml-hrx2/`:

- `kernels/mul_mat_f16_f32_batched.loom` — 4 exports (`..._batched`,
  `_cols8`, `_rows2_cols8`, `_cols8_contiguous`). Every dimension incl. the
  GQA broadcast (`src0_i02 = i12 / (dst_ne2/src0_ne2)`) is a JIT-config
  value → shape-agnostic kernel. Dtype-independent except: src0 view element
  type (`xf16`) and byte→elem divides (`/2` for f16; src0 count derivation
  divides by 2 too).
- `catalog/routes/mul_mat_f16_f32_batched.json` — 4 routes
  (family `mul_mat_f16_f32_batched` op MUL_MAT prio 100/140/150; family
  `..._cont` op CONT prio 160 = the mul_mat+permute+cont fusion writing the
  softmax-ready contiguous layout). **Shape domains cap at `rows 128–512`**
  (+ `k_pow2`, `cols_multiple_of_8` variants).
- `ggml-hrx2.cpp`:
  - supports contract for the f16 batched path (~2182): rejects
    `src0->type != GGML_TYPE_F16`, requires `op->ne[2] % src0->ne[2] == 0`
    and `op->ne[3] % src0->ne[3] == 0` (GQA divisibility), allows STRIDED
    nb2/nb3 on src0 (KV-cache views) → the zero-copy-friendly contract
    already exists.
  - attention claim in `supports_op` (~11794): *"attention KQ^T/kqv mms:
    F16 src0 (KV cache) x F32 src1"* → `supports_mul_mat_f16_f32_route`.
  - scheduler-level graph fusion (~11262):
    `{MUL_MAT, SOFT_MAX, MUL_MAT, PERMUTE, CONT}` → `_cont_route` dispatch.

## The regression mechanism (two stacked gates)

1. **Shape (binds for every dtype):** route domains `rows 128–512` only
   match the decode window at KV ≥ 128. Prefill (pp32 → KV=32, cols=32)
   never matches a batched route → those mms fall to CPU regardless of
   dtype. Decode at KV < 128 falls too. This is the 25i regression.
2. **Dtype:** F32-src0 is rejected even in-domain. The live fork defaults
   `type_k/type_v = GGML_TYPE_F16` (`src/llama-context.cpp:2905-2906`), so
   whether the roster's "F32 mms" are genuinely F32-KV depends on the bench
   flags — pin empirically before choosing a fix path.

## Checkpoint 0 (5 min, no edits)

On the live fork: one pp32 run of Qwen2.5-3B-Q4NX with
`GGML_HRX2_TRACE_ROUTES=1` (or a node dump). Record for each attention
MUL_MAT: src0 dtype, rows/cols/ne2/ne3, and whether it fell to CPU. This
decides C1 vs C2.

## Fix paths

### C1 — route-only (if the runs use F16 KV, i.e. defaults)
Pure catalog + claim work over the EXISTING kernel: new route entries with
prefill shape domains (rows domain incl. 32; cols incl. 32; same
`k_pow2`/cols8 structure). No kernel edits, no ABI change. Smallest
possible change.

### C2 — f32 sibling (if the runs use F32 KV)
- Mechanical kernel copy: src0 view `xf16`→`xf32`, element byte divides
  `/2`→`/4` (incl. the `src0_count` derivation — the exact class of site
  that silently corrupts: R25h off-by-8, R25b per-expert columns).
- Relax the supports predicate to accept `GGML_TYPE_F32` src0 with
  `nb[0]==4` (mirror of ~2182).
- Widen route domains as in C1.
- Extend the graph-fusion gate (~11262) + `_cont` extraction to the f32
  family.

### C3 — decode side (NOT this round)
KV < 128 or > 512, growing per token → runtime-variant shapes → the
loom-internals/constants question. Stays CPU. Record as the deferred piece.

## Verification gates (the workstream's rules)

1. Route coverage trace: count NPU-vs-CPU attention mms — route-domain
   mismatch is a SILENT fallback (R25d class); corr alone cannot catch it.
2. Same-fork-build CPU F32 reference regenerated (stale-reference trap,
   R21–R26 addendum).
3. 32-token dump harness: corr at the established noise floor, top1 == ref.
4. Roster re-verify ("Paris." + top tokens).
5. pp32 3B ≥ 121.5 (regression reversed) and tg32 ≥ 25i numbers — the
   zero-copy constraint: kernel reads the GTT KV view directly (strided
   nb2/3 contract), no staging round-trip, or the 25i decode win evaporates.

## Non-goals (explicit)

- Fused attention kernel (multi-session; silent-wrongness surface + needs a
  trustworthy reference that is NOT the decomposed chain).
- Runtime-variant shapes / constants-ABI spelunking (decode-side problem;
  loom's own verifier rejects scalar launch bindings for varying scalars).
- Softmax, mask, rope: routes already claimed for the prefill shapes
  (25i added `soft_max_f32_mask_generic_wg256`, r32 masked).

## Sources

- Round log: `README.md` (this dir), rounds 18/22/25d/25i + Verification
  addendum + Round 26.
- Fork evidence: paths above under `~/hrx-ws/hrx-v2-src`.
- Loom ABI semantics: `loom/docs/src/guide/command-programs.md` (scalar vs
  specialization vs device storage).
