# Upstream check: the column-slice chain is already on `origin/main` (and it is xchesscc-free) — 2026-09-18

Asked to go upstream and check all commits. My branch is **331 commits behind `origin/main`**, and
the thing this thread has been rebuilding by hand **already exists there** — built with Peano, no
Chess involved.

## What upstream has (the fix, and the RE chain around it)

| upstream artifact | what it is |
|---|---|
| `engine/npu/generators/n1_core_fused_gu_silu_d.py` | the **only** generator carrying the column-offset knob: `tiles = [[tile(col_off + col, row) …]]`, exposed as `-c/--cols` **and `-x/--col-offset`** |
| `engine/npu/generators/build_zaya_fused_cols.sh` | "column-sliced FUSED GU→SiLU→D xclbins … needs **4-column half-array kernels at col offsets 0 and 4**" — builds `…_h0` / `…_h1` |
| `tools/two_stream_decode.sh` | two concurrent streams, *"one per 4-column half (cols 0-3 \| 4-7), **the issue-#2128 column-sliced co-schedule**"*, with a numeric corr gate — *"a gate that cannot fail is not a gate"* (it previously printed success on a corr=-0.0016 run) |
| `docs/guides/npu-modes.md` | documents `NPU_BF16=1`'s artifact family (`final_bf16_<T>_K<K>_N<N>.xclbin` / `insts_bf16_…`, `T ∈ {QKV,O,G,U,GU,D}`), that **there is no fallback**, and where they come from |
| `engine/npu/tests/check_artifact_families.py` + `Testing/artifact_family_coverage_selfcheck.sh` | discovers the families the engine *constructs* from source and refuses a family with zero tracked members — the check that was missing when 54 bf16 tiles were deleted |
| `5ff650347` | restores those 54 bf16 xclbins/insts and extends `PROVENANCE.json` |
| `0d2900215` | `build_npu.sh` rebuilds on header edits; a failed compile fails the build |
| `5ffbaf43f` | wires `MoERuntimeLayerEngine` into the engine build (+ `benchmarks/RESULTS-moe35b-decode-gap-2026-09-18.md`) |
| `16e7cc124` | the NPU_BF16 doc + artifact-family gate + the non-`.q4nx` container guard (a `.gguf` under `NPU_BF16=1` used to exit 139) |

Upstream's half-builds prove the whole chain is **Peano-only** (`--no-xchesscc --no-xbridge` with
`mlir-aie/install_tmp` + `llvm-aie`), i.e. exactly the xchesscc-free path: the column slice is a
*placement shift in the generator*, not a compiler feature.

## The port, done here

Mirrored upstream's knob into the two generators our dense-Qwen3 path uses — a 10-line change, same
shape as upstream's:

```
engine/npu/generators/n1_core_bf16_v1.py   -x/--col-offset → tile(col_off + col, row)
engine/npu/generators/n1_core_attn.py      -x/--col-offset → tile(col_off + col, row)
```

Built both halves of the Qwen3-0.6B bf16 set (`~/xclbins-h0`, `~/xclbins-h1`, via
`benchmarks/flm_half_build.sh`):

```
QKV 1024/4096 · O 2048/1024 · GU 1024/6144 · D 3072/1024
offset 0 → columns {0,1,2,3}      offset 4 → columns {4,5,6,7}
```

## Measured in the engine: the halves are correct, but they don't buy throughput *yet*

One instance on h0, two instances at once with the second on h1 (1k prompt, 256 tokens):

| | single (h0) | pair (h0 + h1) | aggregate vs solo |
|---|---:|---|---:|
| decode | 77 tok/s | A=35, B=35 | **0.92x** (46% each) |
| tokens vs single | — | both **IDENTICAL** | |

So: the half kernels load, run, and are numerically correct in the engine — and the pair is *no
better than the full-width control* (0.88x). The diagnosis from the previous step is confirmed by
construction: **the halves only cover the prefill GEMMs.** At this mix the wall time is dominated by
the decode, which still runs FastFlowLM's **8-column `layer.xclbin`**, and the prefill's attention
rides an **8-column `attn_mha_*.elf`** — the two produce the sharing signature no matter how narrow
our GEMMs are (prefills even inflate 4-5x under host contention: 2605/2249 ms vs 534 solo).

**The one remaining piece is a 4-column whole-layer decode kernel.** Upstream shows the shape of
that work — `n1_core_fused_gu_silu_d.py` is a *fused whole-layer* design and
`build_zaya_fused_cols.sh` slices it — but for dense Qwen3 there is no such kernel in either tree
(FLM's `layer.xclbin` is the fused one, and it has no source). Everything else on the list is done:
the bf16 GEMMs (here), the attention generator (knob ported; build is a `-x 4` run of
`build_attn.sh`), and the co-scheduling harness.

## Reproduce

```bash
# ported knob (already applied): engine/npu/generators/{n1_core_bf16_v1,n1_core_attn}.py -x/--col-offset
bash ~/build_half.sh QKV 1024 4096 4 0        # → ~/xclbins-h0  (also O/GU/D, and offset 4 → h4)
bash ~/engine_halves.sh                        # single (h0) vs pair (h0 + h1)
# upstream references
git log --oneline goal/runlist-decode-wire..origin/main | head -30
git show origin/main:engine/npu/generators/build_zaya_fused_cols.sh | head -40
git show origin/main:tools/two_stream_decode.sh | head -20
```
