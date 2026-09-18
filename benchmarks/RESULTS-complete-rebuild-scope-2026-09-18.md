# The complete rebuild: what can be rebuilt, what cannot, and the correction it forced — 2026-09-18

Asked for a complete rebuild after the column finding. Outcome: the toolchain and **our** kernels
rebuild fine, and so does a 4-column variant — but the rebuild also **refuted part of my own
earlier explanation**, and a *complete* 4-column stack turns out to be blocked by kernels that have
no sources in this tree.

## 1. Our attention kernel rebuilds, and reproduces the design

`bash generators/build_attn.sh` (mlir-aie `install_tmp` aiecc + the venv's Peano clang for aie2p):

| artifact | shipped | rebuilt | |
|---|---|---|---|
| `xclbins/attn_insts.txt` | `f3d0a132bde24a60…` | `f3d0a132bde24a60…` | **byte-identical** |
| `xclbins/attn.xclbin` | `3e9a276a4db695ef…` (94672 B) | `92ec30252f4049ac…` (95440 B) | differs |

The instruction stream — the part that determines what the device executes — is reproduced
exactly; the container differs, which is the documented Peano-vs-Chess PDI nondeterminism
(`engine/npu/README.md`: "xclbins compiled with Peano vs Chess produce different PDI binaries even
from the same MLIR source"). The pre-rebuild artifacts were restored afterwards, so the tree still
carries the validated files.

## 2. The same design at 4 columns builds, and really is 4 columns

`n1_core_attn.py … -c 4` (the generator takes `-c/--cols` = columns = q-heads per launch) via the
same recipe:

| | 8 columns (shipped) | 4 columns (built here) |
|---|---|---|
| instruction stream | 45,936 B | **22,976 B** |
| xclbin | 94,672 B | 51,130 B |
| tiles referenced in the MLIR | columns 0–7 | **columns 0,1,2,3 only** (12 distinct tiles) |
| `xclbinutil` AIE_PARTITION | `column_width=8, start_columns=[0]` | `column_width=8, start_columns=[0]` |

**The last row is the correction.** The 4-column design declares `column_width: 8` — because that
field is a **default stamped by this aiecc flow on every xclbin**, not a measurement of the design.
The generated `main_aie_partition.json` (`name: QoS`, `operations_per_cycle: 2048`,
`column_width: 8`, `start_columns: [0]`) is identical for a design that touches half the device, and
aiecc exposes no partition or column-range flag to change it (only placement options:
`--cores-per-col`, column-major, sequential_placer).

So the earlier statement in `RESULTS-two-flm-engines-and-device-columns-2026-09-18.md` — "every
kernel requests all 8 columns, therefore contexts time-share" — **does not follow from that
metadata**. What the shipped stack actually occupies is not visible in the partition descriptor at
all. The concurrency ceiling is set by the driver's *scheduler* (contexts are preempted), not by a
declared partition claim. The rest of that note stands (FLM's two servers aggregate 42.6 tok/s at
38% sharing efficiency against the native pair's 95 tok/s at 82%); only its explanation is
withdrawn.

## 3. Why the rebuild cannot be completed here

| kernel | who builds it | can we rebuild it 4-wide? |
|---|---|---|
| `attn.xclbin` / `attn_mha_*.elf` | this tree (`generators/n1_core_attn.py`, `build_attn.sh`) | **yes** — done above |
| `final_bf16_*.xclbin` (bf16 GEMM) | this tree (`n1_core_bf16_v1.py`, `build_bf16_xclbins.sh`) | yes (not on the default path) |
| int8 GEMM `n1_core_i8_*` | this tree | yes (only the slow split path) |
| **`mm.xclbin`** (bf16 prefill GEMM) | **FastFlowLM, prebuilt** | **no — no sources** |
| **`layer.xclbin`** (whole-layer decode) | **FastFlowLM, prebuilt** | **no — no sources** |
| **`dequant.xclbin`** | **FastFlowLM, prebuilt** | **no — no sources** |

The engine's default path loads FastFlowLM's `mm.xclbin` (24 references), `layer.xclbin` (11) and
`dequant.xclbin` (3); `~/amd-oss/fastflowlm` contains their runtime sources but **no AIE kernel
sources and no xclbin build flow** — they ship binaries. A complete 4-column stack would therefore
require builds we cannot produce, and a partial stack (ours only) would still have FLM's 8-column
kernels on the critical path of every engine instance.

## 4. What is left, and what it costs

**The driver scheduler is the real lever**, and every knob for it is a module parameter:

```
hwctx_limit = 16 [Debug]   context_limit = 64        (context counting)
time_quantum_ms            (empty; undocumented)     <- preemption quantum
disable_fine_preemption    (empty)                   <- fine-grain preemption on
aie2_max_col               (empty)                   <- max columns the device exposes
start_col_index            (empty)                   <- force partition start column
sys_eff_factor = 2, force_cmdlist (empty), mailbox_polling = 0
```

`aie2_max_col` + `start_col_index` are the documented way to expose and place a *narrower*
partition; `time_quantum_ms` / `disable_fine_preemption` change how contexts share. All of them
take effect only by **reloading amdxdna**, which drops the production MoE server's 9 hardware
contexts until it restarts. That is why this note stops here rather than trying them: it is a
production-touching change that needs a window and an explicit go-ahead, not an experiment to run
beside a live server.

Two routes if you want them:
- **Cheap, no reload:** patch the *partition descriptor* of the 4-column xclbin we just built
  (`xclbinutil --add-section AIE_PARTITION:JSON:…`) to `column_width: 4` / `start_columns: [0]`,
  build a second copy at `start_columns: [4]`, and see whether the driver registers two contexts
  from them and runs them concurrently. Answers the co-scheduling question directly, in user space.
- **Expensive, needs a window:** the module parameters above, with the production server restarted
  afterwards.

## Reproduce

```bash
bash engine/npu/generators/build_attn.sh            # 8-column rebuild; insts byte-identical
bash ~/build_attn4.sh 4 /tmp/attn4build             # 4-column build (direct recipe, no wrapper)
xclbinutil --dump-section AIE_PARTITION:JSON:/tmp/ap.json --input /tmp/attn4build/attn4.xclbin
grep -oE 'aie\.tile\([0-9]+, [0-9]+\)' /tmp/attn4build/design.mlir | sort -u   # columns 0-3 only
modinfo -p amdxdna                                  # parameter descriptions
```
