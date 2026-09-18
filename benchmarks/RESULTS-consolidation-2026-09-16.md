# Consolidation: goal state, yardstick, and the boundary list (2026-09-16)

Goal `mu35shsg-i3hlyi`. This is the pointer document `task-consolidate` asks for:
the yardstick invocation that is green, the post-goal state, and a plain statement
of what is still outside the set. It does not restate every measurement; each row
names the doc/commit that carries it.

## Yardstick — green, with the invocation that is actually green

`~/npu-ab/npu_ab.sh` defaults its `ROOT` to `/home/bcloud/1bit-MONSTER` (a
different worktree) and therefore measures a stale binary; the same is true of its
`NPU_LAYER_ELF_DIR` default (`txn-elfs`, a mismatched per-model set). With the
goal's binary and the 8193-context ELF set:

```
cd ~/1bit-MONSTER-goal
bash ~/npu-ab/npu_ab.sh --ctx-k 1 --decode-tokens 16 --reps 1 \
  --engine "$PWD/engine/npu/build/npu_engine_qwen3_0_6b" \
  --layer-elf-dir "$HOME/npu-ab/elfs-4k"
```

```
REFUSED: 0
FLM (on-box)   decode 73.72 t/s   prefill 1250.88 t/s
native runlist decode 81.0  t/s   prefill   71.4  t/s   gate NATIVE-TOKENS-PLAUSIBLE
native split   decode  2.0  t/s
-> runlist vs FLM: 109.9% of FLM decode
```

**No `PARITY CLAIM REFUSED`.** The decode parity claim holds; prefill parity on the
runlist path does not (the runlist prefill is one whole-layer forward per prompt
token — the fast prefill is the bf16/unified path, measured in
`RESULTS-default-path-2026-09-15.md`).

## Post-goal state

**Phase 1 — the six supported models.**
- 0.6B: native runlist 20/20 easy, 13/15 hard (correctness lane, `mu34scbf`).
- 1.7B / 4B / 8B: 3/3 / 3/3 / 2/3 (8B's miss a stated budget truncation).
- Qwen3-VL-4B: native runlist **20/20** vs FLM 20/20 (`e060acc4e`).
- Llama-3.1-8B: native runlist **20/20** vs FLM 20/20 (`dac5417f4`), 0 extraction
  errors; the old first-layer ERT was environmental.
- The residual "content errors" (Kyoto/Barcelona/Neptune/helium) are classified as
  template/scoring faults (`74b17357c`), not engine faults.

**Phase 2 — the window.**
- The ~6 s per-launch cost is explained (the `n_grp == 1` core lost its PV/C2
  block; the host waited out a driver timeout) and fixed: N=512 is 2.515 ms against
  the shipped 2.161 ms = **1.16x**, NPU==EMU (`a42100b8d`, L1 lane).
- The `N<=512` softmax contract is implemented; NPU==EMU at N=256/512/1024.
- The generated int8 attention is **engine-driven past 1024 keys**: generated
  N=2048, `NPU attention ready MAX_SEQ=2048`, attention at seq≈2100
  **corr=1.000000** (`96db25fb8`).
- Beyond 8192 is **bounded**: N=2048 and N=3072 build and gate; N=4096 and N=8192
  fail with AIE **program-memory overflow**, so the N=16384 route needs a
  chunked/multi-core design (`0c3feded4`). The six models reach 8191 (register
  entry 8); the cap is `MAX_L=8192` in the per-context layer ELFs.

**Phase 2b — the ERT (`ERT_CMD_STATE_TIMEOUT`).**
- Root-caused to the amdxdna driver TDR `timeout_in_sec=2` (83 firmware-timeout
  dumps; the reported `ctx_pc` values match byte-for-byte).
- Repaired durably: `/etc/modprobe.d/amdxdna-tdr.conf` sets 15, and all 19
  `npu_engine_*` binaries take an exclusive `flock` device lock at startup
  (`c066f1621` lineage; docs `RESULTS-ert-rootcause-and-repair-2026-09-16.md`).
- Not fixed by this: the Qwen3.6-35B-A3B runtime ERT (a broken vendored lib).

**Phase 3 — the un-routed families.**
- Per-family verdict table landed (`b75dcc0c7`).
- The generated **nh20/nkv4/cols4 head-block kernel passes its bench gate**:
  NPU==EMU at seq=2048 (`8.575258e-02`) and seq=513 (`2.377548e-01`), C2 2/2
  (`349260e17`).
- The bf16 **attention-ELF loader was compiled out under g++** (it was gated on
  Clang's `__has_embed`); fixed, and verified on the supported models (0.6B bf16
  600-key tokens identical to `NPU_RUNLIST=1`) (`c066f1621`).
- **Family >1024 is not achieved.** The generated ELF is an `AttnCtx`-ABI kernel
  (Q/KT/C2/V/SCR); `Bf16Mm` drives FLM's `(act, out, kv)` ABI. Of the four
  `cap2048`-not-`cap1024` capture candidates, three hang the device and the fourth
  (`308736`) runs but is numerically wrong (boot 152349 vs the byte-exact CPU
  attention's 456) (`27094a854`, `52e051825`).

## Boundary — what is still outside the set

Stated plainly, with the source that bounds it:

| outside the set | why / source |
|---|---|
| contexts **beyond 8192** | layer ELFs bake `MAX_L=8192`; generated int8 attention caps at ~3072 (AIE program memory, `0c3feded4`); N=16384 needs a chunked/multi-core design |
| family bf16 attention **above 1024 keys** | no correct nh20/nh24/hd256 attention ELF for the `Bf16Mm` `(act,out,kv)` ABI; the generated kernel is `AttnCtx`-ABI (`52e051825`) |
| **LFM2-1.2B / 2.6B** | no engine binary exists (`engine/npu/build/`) |
| the ~6 s cost | **not** in this list — it is fixed (1.16x of the shipped capture) |
| the 35B MoE runtime | vendored lib's whole-layer path ERTs / NaNs (`c1b76d` addenda) |

The un-routed families' ≤1024 path remains correct-but-slow (CPU attention), which
is the honest verdict `RESULTS-family-attention-shape-2026-09-14.md` records.

---

## Addendum 2026-09-18 — criterion (c)'s current state, and the boundary refresh

This addendum supersedes the (c) rows above; the boundary table below it is still right but
now has one more member.

**1k: the clause set is met for all six supported models.** One window, 32 decode tokens,
same-window FLM pairs at the same prompt/token counts, native = the default unified path with
the decode-overlap fix (`efb35df4c`):

| model | native prefill / TTFT / decode | FLM |
|---|---|---|
| Qwen3-0.6B | 1828–1894 t/s / 0.541–0.560 s / 78–80 tok/s | 1323.11 / 0.743 s / 72.33 |
| Qwen3-1.7B | 1316–1346 / 0.761–0.778 s / 39.8–40.0 | 980.09 / 1.002 s / 39.56 |
| Qwen3-4B | 646–649 / 1.577–1.586 s / 19.1–19.2 | 509.70 / 1.926 s / 18.80 |
| Qwen3-VL-4B | 648–650 / 1.576–1.580 s / 19.0–19.1 | 530.72 / 1.840 s / 18.80 |
| Qwen3-8B | 456–459 / 2.228–2.248 s / 10.95 | 362.63 / 2.706 s / 10.71 |
| Llama-3.1-8B | 453–455 / 2.200–2.213 s / 11.4 | 316.02 / 3.178 s / 10.84 |

Authority: `RESULTS-1k-six-model-2026-09-18.md` (and `RESULTS-unified-decode-overlap-2026-09-18.md`
for the scheduling change that made the decode column). Llama's row requires a **warm**
per-context ELF cache and `NPU_LAYER_ELF_DIR`; its cold-cache prefill is ~2.7x slower.

**8k: parity on prefill/TTFT, and decode still outside the set.** Interleaved repeated runs
put 0.6B at 2009–2028 t/s vs FLM 1988–2010 and 1.7B at 1145–1473 vs 1193–1374 — **parity**, not
the 0.62–0.71x inversion the 2026-09-16 yardstick recorded, and not the 1.01–1.24x win a
single-run table briefly claimed. Authority: `RESULTS-8k-campaign-variance-2026-09-18.md`,
with the CORRECTION banner on `RESULTS-8k-prefill-2026-09-18.md`. The 8k **decode** row cannot
be taken at all: the second decode forward needs `ctx=8194`, and the per-ctx ELFs are baked at
`MAX_L=8192`.

**Criterion (c) is therefore NOT met as written.** It asks for prefill/TTFT/decode >= FLM at
1k..8191; the 1k end is met for all six, the 8k prefill/TTFT end is at parity (neither ahead
nor the recorded deficit), and the decode end above ~8193 contexts does not exist yet.

**The boundary table gains one row, and one existing row gets a sharper reason:**

| outside the set | why / source |
|---|---|
| (existing) contexts beyond 8192 | layer ELFs bake `MAX_L=8192` — now demonstrated *at the decode step*, not just as a capability note: the 8k decode's second forward (`ctx=8194`) fails to build, and the split-path fallback then dies on the missing i8 `G_K2560_N9728` tile (`RESULTS-8k-prefill-2026-09-18.md`) |
| (new) **any number taken while a foreign process holds `accel0`** | the engine takes `/tmp/1bit-npu-device.lock`; other tools do not. Two long-lived `/tmp/attrib` processes inflated identical 8k prefills 3.3–4.4x on 2026-09-18. Instrument: `benchmarks/c8k_guarded.sh`; register entry: `LEVERS` §6.7 |

The un-routed families' verdicts and the ≤1024-key family boundary above are unchanged.

### Addendum 2026-09-18 (second) — (c) at 8k for all six, and the one cell that falls short

- `RESULTS-8k-six-model-2026-09-18.md` — guard-accepted 8k prefill/TTFT for all six models;
  native ahead 1.04–1.10x prefill, TTFT parity-or-faster for five.
- `RESULTS-8k-decode-top-of-window-2026-09-18.md` — the decode clause **is** measurable at the top
  of the window (prompt `8192-ng` keeps `ctx<=8192`); all six at parity or ahead with `ng=32`
  (the `ng=8` reading of 0.93x for 0.6B was warm-up).
- `RESULTS-0_6b-prefill-scaling-2026-09-18.md` + `RESULTS-0_6b-prefill-hostbound-2026-09-18.md` —
  the single non-passing cell: 0.6B TTFT at 8k, 1–3% behind, localised to the host `conv+other`
  prefill term (4007–4071 ms of a 4030–4094 ms total; the device attention is ~96% hidden, so
  prefill pipelining has a ~25 ms ceiling and the fast int8 device prefill is the path that fails
  the accuracy gate). Remaining lever: the host prefill path itself (~0.45 ms/token ≈ 3 GFLOPS
  effective, so memory/format-bound). Raised with the user — hours of work, past the 2 h threshold.
- **Criterion (c) as measured: 1k met for all six; 8k 17 of 18 cells met; the 18th classified.**
