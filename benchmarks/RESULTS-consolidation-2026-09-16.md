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
