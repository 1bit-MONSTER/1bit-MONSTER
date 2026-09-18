# Rebuilding FastFlowLM's kernels from source: it is all in `~/amd-oss/iron` — 2026-09-18

Asked to rebuild FLM's kernels. They have **source on this box**, and this note records where,
what rebuilds today, and the one piece of plumbing between here and a 4-column build.

## Where the source is

`~/amd-oss/iron/` (the IRON repo, `iron.operators.*`) contains AMD's port of the FastFlowLM
overlays:

| path | what it is |
|---|---|
| `iron/operators/flm/gemm/` | **"the IRON port of FastFlowLM's `mm` overlay"** — "built from source in this repo". Fixed 64/512/128 tiling + fused epilogue (silu/gelu/sigmoid), runtime `M,K,N`, and `Rounding.FLOOR` **"reproduces the shipped FastFlowLM overlay bit-for-bit on NPU2"** |
| `iron/operators/flm/mm_prebuilt/` | runs FLM's shipped `mm.xclbin` **unmodified** — a `RemoteFileArtifact` pinned by SHA-256 to an immutable FastFlowLM commit, so the shipped binary can be A/B'd against the port |
| `iron/operators/flm/packing.py` | the B pre-pack the overlay expects |
| `iron/operators/{dequant,mha,rms_norm,rope,silu,swiglu_prefill,swiglu_decode,gemm,softmax,gemv,transpose,…}` | the rest of a transformer's op set, each a buildable design |

So "rebuild them for FastFlowLM" is not a reverse-engineering task: the `mm` overlay has a public
IRON implementation, and the shipped xclbin is fetchable for comparison.

## What rebuilds today (measured on this box)

```
cd ~/amd-oss/iron && ~/amd-oss/iron-venv/bin/python -m pytest \
    iron/operators/flm/gemm/test.py -k "test_gemm and iter0" -x -q
→ 33 passed, 152 deselected in 231.10s
```

33 correctness cases of the rebuilt `flm.GEMM` compile and run against the golden reference on the
Strix Halo NPU, through the Peano toolchain in `~/amd-oss/iron-venv`. The suite also covers
`one_xclbin_serves_every_shape`, `every_clamp_bound`, split-leg bounds and artifact-stem keying —
i.e. the runtime-parameter design is tested, not just one shape.

## Narrow devices exist — the toolchain can express 4 columns

`aie.dialects._aie_enum_gen.AIEDevice` members:

```
npu1, npu1_1col, npu1_2col, npu1_3col,
npu2, npu2_1col, npu2_2col, npu2_3col, npu2_4col, npu2_5col, npu2_6col, npu2_7col
```

and `Device(DEVS["npu2_4col"])` resolves to **`cols=4, rows=6, arch=AIE2p`** (vs `cols=8` for
`npu2`). That is the capability the earlier note said was missing: a narrower NPU2 target is a
first-class device in the toolchain, even though `aiecc` still stamps `column_width: 8` on every
xclbin it emits.

## The one gap, precisely

The operator path gets its device from the *runtime*, not from the caller:

- `flm.GEMM.__post_init__` → `dev = aie_utils.get_current_device()`, then
  `get_target_model(dev.resolve())`;
- `AIEContext(build_dir, mlir_verbose, compiler)` — **no device argument**;
- IRON's pytest fixture creates `AIEContext(...)` and lets
  `DefaultNPURuntime.device()` (= the physical 8-column NPU2) supply the device. There is no
  `IRON_DEVICE`-style environment override.

Calling `aie_utils.set_current_device(Device(DEVS["npu2_4col"]))` and then compiling fails at
`get_target_model(None)` — `Device.resolve()` hands back nothing outside the runtime's own binding.
So bridging a 4-column device into the design path is a **small patch** (bind the resolver to the
caller's `Device`, or make the device accessor consult a explicit override first), not new
research. After it: `flm.GEMM` for `npu2_4col`, correctness via the same 33-case suite, partition
inspection with `xclbinutil`, then the two-context co-scheduling measurement that this whole thread
is for.

## What this means for the two-engine question

- FLM's `mm` can be **rebuilt from source at any width**, and matched against their shipped binary.
- FLM's `layer.xclbin` (the whole-layer decode kernel) has no IRON equivalent yet — the operators
  for its pieces exist (`dequant`, `mha`, `rms_norm`, `rope`, `silu`), so that one is a design
  project rather than a port.
- Everything still points at the same experiment: two contexts, each on a narrower partition, to
  see whether aggregate throughput doubles (spatial) or stays ~1.4–1.6x (preemption).

## Reproduce

```bash
ls ~/amd-oss/iron/iron/operators/flm/{gemm,mm_prebuilt}      # the port and the pinned overlay
sed -n '1,40p' ~/amd-oss/iron/iron/operators/flm/gemm/README.md
cd ~/amd-oss/iron && ~/amd-oss/iron-venv/bin/python -m pytest iron/operators/flm/gemm/test.py -k "test_gemm and iter0" -x -q
~/amd-oss/iron-venv/bin/python /tmp/flm_gemm_cols.py 256 512 1024   # device grid probe (cols=8 / cols=4)
```
