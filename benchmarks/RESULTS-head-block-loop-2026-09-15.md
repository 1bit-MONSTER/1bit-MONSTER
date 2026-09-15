# Head-block loop implemented — nh > cols and hd > 128 now build (2026-09-15)

Task `task-families` of goal `mu35shsg-i3hlyi`. Branch `family/head-block-loop`,
worktree `~/wt/family-head-block`, on top of the PV N-split commit `10a766324`.

**Status: BUILD-verified on CPU. NOT device-verified** — accel0 was held by the
correctness lane (`mu34scbf-4ffm0o`'s dense 20-prompt run) for the whole window.
No parity number is claimed anywhere in this document.

## What was blocking (the last structural gap)

The design is one core column per q head, fed from q row `c*K_FRAME`. A model with
more heads than columns therefore got a kernel for only the first `cols` heads:
the xclbin built, loaded and ran, and the host silently received the wrong heads.
Affected: Nanbeige nh20, Phi4 nh24, Qwen3.5-4B nh16.

## What changed (`engine/npu/generators/n1_core_attn.py`)

| piece | change |
|---|---|
| head blocks | `n_hpass = H / cols` with `assert H % cols == 0` (a partial block would read past the q BO). `--heads` is now the real head count, not an alias for `--cols`. |
| GQA | `--nkv` added; `gqa = cols / nkv` asserted to divide; every `cc // 4` becomes `(hp*n_aie_cols + cc) // gqa` |
| host sequence | the whole feed is wrapped in `for hp in range(n_hpass)`; q-tile offsets become `(hp*cols + c)*K_FRAME + ki*k`; the C2 writeback goes to `(hp*cols + c)*(M*K)` |
| params row | `PARAM_ROW = 15 if H <= 15 else H` — row 15 is the padding row the params ride in today, so `H <= 15` keeps the old offset and `H > 15` moves the params out of the head range |
| BO sizes | q frame `max(16, PARAM_ROW+1) * K_FRAME`; C2 `n_heads * M * K` (was `n_aie_cols * M * K`) |

`n_hpass == 1` is the original single-pass path and stays byte-identical: the
Python-level loops unroll and every offset/expression collapses to the old form.

## Verification done (CPU only)

| case | shape | result |
|---|---|---|
| **single-pass guard** | hd128, nh8, cols8, nkv2, N512 | xclbin 95440 B; `attn_insts.txt` **byte-identical** to the shipped file (`cmp` clean, `f3d0a132bde24a60`, 45936 B) |
| Nanbeige | hd128, **nh20**, cols4, nkv4, N512 | builds, 51130 B xclbin, insts `a82c3cb21cdb7b7d` |
| Phi4-mini | hd128, **nh24**, cols8, nkv8, N512 | builds, 95440 B xclbin, insts `c025631a75051df8` |
| Qwen3.5-4B | **hd256**, **nh16**, cols8, nkv4, N512 | builds (both changes at once), 104528 B xclbin, insts `c2ec8b63b7670a28` |
| Nanbeige @2k | hd128, nh20, cols4, nkv4, **N1024** (chunked + head blocks) | builds, 55802 B xclbin, insts `d84841898ad30d94` |

## A harness bug worth recording (caught before it produced a wrong claim)

The first verification pass reported the guard as DIFFERS while printing the correct
hash. Cause: `NPU_ATTN_INSTS=/tmp/hb_$name_insts.txt` — bash parses `$name_insts`
as one identifier, so every build wrote the same `/tmp/hb_.txt` and the `cmp` was
comparing a file that did not exist. Fixed with `${name}` in braces and a fresh
prefix. This is the same failure class the correctness-lane goal lists as a binding
rule ("assert the extraction belongs to the run you think it does") — a stale or
shared output path is the cheapest way to manufacture a false result.

## Host side (`AttnCtx` + the bench) — made here, compile-verified

`src/npu_attn_ctx.h` now checks shapes against the kernel's *structure* instead of
pinning them to the one built configuration:

| was | now |
|---|---|
| `if (nq != 8 \|\| nkv != 2 \|\| hd != 128) unsupported` | `nq % cols == 0` (one column = one head), `nkv` divides `cols`, `hd` a multiple of 128 and `<= K_FRAME` |
| `cols` implicit (8) | `cols` from `NPU_ATTN_COLS`, and `scrsz` scales with `cols` (the A2 scratch is per column, reused per pass) |
| params hard-wired to row 15 | `PARAM_ROW = 15` for `nq <= 15`, else row `nq`; `qsz = max(16, PARAM_ROW+1) * K_FRAME` |
| C2 `nq*8*hd` | unchanged — it already matches the generator's `n_heads*M*K` |

For the shipped shapes (`nq=8, nkv=2, hd=128, cols=8`) every one of these
collapses to the previous value, so the existing lane is behaviourally identical.
`AttnCtx` is included only by `zaya_decode.cpp` and the bench, **not** by the
dense `Bf16Mm` path.

`tools/attn_kernel_bench.cpp` takes `CK_NQ` / `CK_NKV` / `CK_HD` from the
environment (its ground truth was already shape-agnostic), and the whole thing
compiles: `g++ -std=c++17 -O2 -mavx2 -I src -I generators -o /tmp/ck2
tools/attn_kernel_bench.cpp -lxrt_coreutil -lxrt_core -laiebu -luuid -ldl`.

**Still open, and now the only engine-side gap:** the dense path that the family
measurements go through (`Bf16Mm` in `npu_engine_bf16_mm.h`, whose attention
selection is the thing the L1 lane patched for nh20/nh24) must pass the real
`nq`/`nkv`/`hd`/`cols` and size its q and C2 BOs the same way before a family can
be driven end to end.

## Open (device, in this order)

1. Standalone bench for one new shape: `NPU_ATTN_MAX_SEQ=<N> /tmp/ck <xclbin> <insts> <N> 2`
   → 2/2 non-zero C2 and NPU `max_abs_err` == EMU `max_abs_err` to the digit. The
   `ck` bench hardcodes NQ/NKV/HD, so it needs its own rebuild for nh20/nh24/hd256.
2. The engine-side changes above, then the family token-identity check against each
   family's FLM reference (Nanbeige `nanbeige4.1:3b`, Phi4 `phi4-mini-it:4b`).
3. Only then can the family verdict table carry numbers rather than "builds".

## Integration map: what the engine path actually needs (found 2026-09-15)

Two different attention kernels exist in the tree and they consume **different
artifacts**. Conflating them is why "the generated attention" and "the engine's
attention" have looked like one item:

| path | artifact | loaded by | shapes today |
|---|---|---|---|
| standalone kernel bench / Zaya decode | `attn.xclbin` + `attn_insts.txt` (the generator's output, driven through `AttnCtx`) | `tools/attn_kernel_bench.cpp`, `zaya_decode.cpp` | any the generator builds — **now includes nh20/nh24/hd256** |
| the bf16 family prefill path | `attn_mha_<tokens>_nh<NH>_hd<HD>.elf` (prebuilt `xrt::elf`, per context bucket: 256 / 1024 / 2048 / 4096 / 8192) | `Bf16Mm` in `npu_engine_bf16_mm.h`, `load_attn_elf(...)` | nh16 (qout 2048) and nh32 (qout 4096) only, hd128 |

So the families' >1024 attention does **not** come from this generator change by
itself: `Bf16Mm` wants an **ELF**, and there is no `attn_mha_2048_nh20_hd128.elf`
in the tree. That is exactly the artifact the L1 lane was capturing from FLM with
`cap_interposer`, and the one the withdrawn L2 note was chasing for the wrong
reason (the bf16 arm of a MoE model is gated out by `!has_moe` at
`npu_engine_universal.cpp:4433`, so a capture there would never be loaded).

The route that closes it, in order:

1. **Generator** — done: the shapes build (this doc).
2. **Host packing** — done for `AttnCtx` (this doc); still needed for `Bf16Mm`'s
   own packing if it is to drive them.
3. **ELF assembly per shape** — the missing piece. The toolchain is already in
   tree for the GEMM path: `Bf16Mm`'s `mm_app_cache` does "generate_seq + aiebu
   ELF assembly + kernel" (`npu_engine_bf16_mm.h:158-161`), so assembling an ELF
   from the generated design is the same mechanism, not a new one. The output must
   be named `attn_mha_<tokens>_nh<NH>_hd<HD>.elf` for the loader to find it.
4. **Verification** — bench first (NPU-vs-EMU equality, the L1 rule), then the
   family token identity at >1024.

Until step 3 exists, the honest family verdict at >1024 is "CPU reference
fallback", which is correct-but-slow and is what
`RESULTS-family-attention-shape-2026-09-14.md` already records.

Note that a verdict at **≤1024** does not depend on any of this: those buckets use
the shape-specific ≤1024 ELF or the embedded nh16 kernel, which is why the armed
family A/B runs at 1024 tokens.

### The ELF route: tools, targets, and the acceptance test (found 2026-09-15)

The missing artifacts are **preemptible ELFs** (8 sections: `.ctrltext` = the
instruction stream, `.rela.dyn`, `.dynamic`, `.note.xrt.UID`), which is what
`xrt::elf` loads. `readelf -h engine/npu/xclbins/attn_mha_2048_nh16.elf` shows
`ELF 32-bit LSB, machine AT&T WE32100, OS/ABI unknown: 45, ABI Version: 2` with
`.ctrltext` = 178512 B — a container format, not the xclbin.

The assembler is present, in two installs:

```
~/Xilinx2025/2025.2/Vitis/aietools/bin/aiebu-asm     # and aiebu-dump
~/Xilinx/2026.1/Vitis/aietools/bin/aiebu-asm
```

`aiebu-asm -t <target>` accepts `aie2ps | aie2asm | aie2txn | aie2dpu |
aie2_config | aie4 | aie2ps_config | aie4_config`. The API headers that name the
input forms are on the box at `~/.local/flm-v0946/include/aiebu/aiebu_assembler.h`
and `/usr/local/include/aiebu/` (also `amd-oss/fastflowlm/src/include/aiebu/`).
No script in the tree invokes it — the evidence is that the previous lane drove it
ad hoc, which is worth turning into a script as part of this work.

**The acceptance test already exists and is cheap.** A prior lane validated its
generated layer streams exactly this way: `aiebu-dump` the `.ctrltext` of the
generated ELF and of FLM's ELF for the same shape, then byte-diff
(`RESULTS-coverage-multifamily-2026-09-13.md:2551-2634` records 0 differing bytes
for `layer_ctx1025` vs FLM `elf_0016`, and for a generated Phi4 `layer_ctx1` vs
FLM `elf_0001`; `RESULTS-task-n2-…-2026-09-11.md:144` decodes nh16 = 6084 txns vs
nh32 = 10628). So an assembled attention ELF can be checked for *identity with
FLM's own kernel* before it is ever run, and where no FLM counterpart exists
(nh20/nh24/hd256) the check degrades to the bench gate instead.

Order for the next session, therefore:

1. `aiecc` the shape (done — builds for nh20/nh24/hd256).
2. Assemble the insts into `attn_mha_<tokens>_nh<NH>_hd<HD>.elf` with `aiebu-asm`,
   wrapping the invocation in a script.
3. Compare `.ctrltext` with an FLM capture of the same shape if one exists.
4. Bench gate (NPU vs EMU), then the family token identity at >1024.

Route (B) competes with route (A) — `cap_interposer` captures of `flm bench` for
each family (`device-claim-goal-lane-2026-09-15.txt` shows the recipe and three
capture dirs at `/tmp/nbcap/cap{1024,2048,4096}`) — but (A) needs the device for
every shape and (B) does not need it until verification, and (B) is the only route
that works for a shape FLM never ships.

### CLOSED: `aiecc --aie-generate-elf` emits the ELF directly (same session)

The ELF step needed no aiebu research — `aiecc` already wraps it:

```
--aie-generate-elf            Generate ELF for AIE control/configuration (via aiebu)
--elf-name=<string>           Output ELF filename for instruction ELF
--aie-generate-txn            Generate transaction binary MLIR for configuration
--generate-full-elf, --ctrlpkt-elf-name=...   (the ctrlpkt/full-ELF variants)
```

`build_attn.sh` now passes `--aie-generate-elf --elf-name=$NPU_ATTN_ELF` when
`NPU_ATTN_ELF` is set. Measured, CPU-only, one build:

| artifact | size | `.ctrltext` | `.note.xrt.UID` |
|---|---:|---:|---:|
| **ours**: `attn_mha_1024_nh20_hd128_ours.elf` (built from this generator) | 243600 B | 226336 B | 32 B |
| the tree's `attn_mha_1024_nh20_hd128.elf` (capture) | 177728 B | 163088 B | 32 B |
| FLM's `attn_mha_1024_nh16.elf` | 98848 B | 90384 B | 32 B |

Same container shape (`xrt::elf`-loadable, `.ctrltext` + `.note.xrt.UID`), so the
loader in `npu_engine_bf16_mm.h` can consume a generated ELF with no change beyond
the filename it already looks for. Ours is larger because the design carries the
chunked/multi-pass structure rather than FLM's single-pass kernel — sizes are not
expected to match, and the identity check against FLM's `.ctrltext` therefore
applies only where our design is meant to reproduce theirs (the layer streams), not
here. For attention the gate is the bench (NPU vs EMU) plus the family token
identity.

So the recipe is complete and scripted:

```bash
NPU_ATTN_K=128 NPU_ATTN_N=1024 NPU_ATTN_HEADS=20 NPU_ATTN_COLS=4 NPU_ATTN_NKV=4 \
  NPU_ATTN_ELF=engine/npu/xclbins/attn_mha_1024_nh20_hd128.elf \
  bash engine/npu/generators/build_attn.sh
```

Remaining for family >1024 parity: build the ELF per (shape, bucket), pass the real
`nq/nkv/hd/cols` from `Bf16Mm`, and run the device gate. No missing artifact, no
missing tool, no unknown format.

### Shape-ELF matrix: 1024/2048 build, 4096 overflows program memory

The selector in `npu_engine_bf16_mm.h` was the last code gap: it allowed only
nh16/nh32 to take an NPU kernel above 1024 keys, so a correct
`attn_mha_2048_nh20_hd128.elf` would have loaded and then been discarded. Each
long-context slot now carries its own `attn_shaped*` flag (set when the ELF that
actually loaded came from the `attn_mha_<tok>_nh<NH>_hd<HD>.elf` name), and the
2k/4k branches accept a shape-matched ELF for any family. The nh16/nh32 branches,
and everything at <=1024, are untouched; the TU compiles
(`g++ -fsyntax-only … src/npu_engine_bf16_mm_bridge.cpp` → rc 0, warnings only).

Built from this generator, CPU-only:

| family | shape | bucket | result |
|---|---|---|---|
| Nanbeige | nh20 hd128 nkv4, cols4 | 2048 | **OK** — 482960 B ELF, `.ctrltext` 0x06db60 |
| Nanbeige | nh20 hd128 nkv4, cols4 | 4096 | **FAILS**: `_XAie_LoadProgMemSection():231: Overflow of program memory` → `XAie_LoadElf failed with XAIE_INVALID_ELF` |
| Phi4-mini | nh24 hd128 nkv8, cols8 | 1024 | **OK** — 292176 B ELF, `.ctrltext` 0x0424f0 |
| Phi4-mini | nh24 hd128 nkv8, cols8 | 2048 | **OK** — 579408 B ELF, `.ctrltext` 0x083a70 |
| Phi4-mini | nh24 hd128 nkv8, cols8 | 4096 | **FAILS** — same program-memory overflow |

Two things follow, and the second is new work rather than a formality:

1. **A failed CDO build still leaves a file on disk** (961680 B / 1153872 B for the
   4096 attempts). Those are invalid and must not be installed — check the build
   exit status (and the `Compilation failed` line), never the file's existence.
2. **4096 does not fit.** The transaction count scales with
   `N` × head-block passes (and × head-dim tiles), so the unrolled stream for
   nh20×5 passes at N=4096 exceeds AIE program memory — a *design* limit, not a
   flag. Options, in the order I would try them: reduce the unrolled stream (the
   generator currently unrolls the QK^T and PV loops entirely); or keep the
   per-context-bucket approach and accept that the 4096 bucket needs the capture
   route; or make the chunked path's group loop a real AIE loop rather than
   Python-unrolled. Until one lands, the family verdict for >2048 is the CPU
   reference, which is correct but slow.

Bucket ELFs live in `/tmp/elfmat_attn_mha_*.elf` for now, deliberately **not** in
the tree: they are unverified, and the working models' ELFs share that directory.
They move into `engine/npu/xclbins/` only after the bench gate and a family token
identity pass.

### Measured boundary: the unrolled stream fits to N=3072, overflows at 4096

Two probes, because the first pass got a false reading from my own script (below):

| shape | 1024 | 2048 | 3072 | 4096 |
|---|---|---|---|---|
| nh8 hd128 cols8 nkv2 | OK | OK | **OK** | **OVERFLOW** |
| nh20 hd128 cols4 nkv4 | OK | OK | **OK** | **OVERFLOW** |
| nh24 hd128 cols8 nkv8 | OK | OK | — | **OVERFLOW** |

`Overflow of program memory` (`_XAie_LoadProgMemSection` → `Error generating CDO
files`) is therefore a **shape-independent** ceiling at the 4096 bucket (8 chunks
of 512), not a property of the head-block or N-split changes: nh8 — which has
neither — fails identically. N=3072 (6 chunks) still builds for both.

What that means for the verdict table: a family's `(2048, 4096]` bucket cannot be
served by an ELF from this generator today, and a 3072-context ELF must **not** be
installed in its place — a kernel used past its length is wrong, not merely slow
(the rule the nh32-2k guard already encodes). So >2048 stays the CPU reference
until the stream shrinks.

Both places that scale with the chunk count are Python-unrolled: the core's
per-group body (`n1_core_attn.py`, `for g in range(n_grp)`, unrolled because `C1`
is a Python list of buffers) and the sequence's DMA task list. The next diagnostic
is cheap and does not need the device: build 3072 and 4096 and diff the containers
(`xclbinutil`/`aiebu-dump`) to see which section actually grows into the limit,
since the fix differs — a real AIE loop over groups needs the C1 list replaced by a
rotating buffer pair, whereas a shim/BD-slot limit needs the feed restructured
instead.

### Harness bug: the build script reported failure on every xclbin-only build

`build_attn.sh` ended with `[ -n "${NPU_ATTN_ELF:-}" ] && echo …`. As the last
command of a `set -e` script, the `&&` form returns 1 whenever `NPU_ATTN_ELF` is
unset — so builds that succeeded (identical "Compilation completed successfully"
lines) exited non-zero, and my first boundary pass reported "FAIL" for nh8 at
1024, 2048 and 3072. Fixed to an `if` statement, and the table above was re-read
from the build logs rather than the exit status. Two false-failure classes in one
session (this, and the `${name}` substitution collision) is the argument for
reading the *log* for the success line and never trusting a wrapper's exit code
alone.

#### Which side grows into the limit (no device needed)

The generator emits the design directly, so the scaling can be counted without
aiecc at all (`n1_core_attn.py -M 8 -K 128 -m 8 -k 64 -n 128 -c 8 -b 2`):

| N | mlir lines | `aie.dma_bd` (sequence) | `func.call` (core program) |
|---|---:|---:|---:|
| 1024 | 4930 | 552 | 160 |
| 2048 | 9442 | 1096 | 304 |
| 3072 | 13954 | 1640 | 448 |
| 4096 | 18466 | 2184 | **592** |

Both sides grow linearly in N, and the failure names program memory
(`_XAie_LoadProgMemSection`), i.e. the **core** program: 448 kernel calls fit,
592 do not. The unrolled group body in the core is therefore the binding
constraint, which matches the generator's structure — `for g in range(n_grp)` is
Python-unrolled because `C1` is a Python list of buffers, so each 512-key group
adds its own matmul/zero/softmax calls to the tile's program.

That fixes the direction of the fix: replacing the C1 buffer list with a rotating
pair (so the group loop can be a real AIE loop) removes the per-group growth from
the core program, and the sequence's BD count is a separate, larger ceiling that
does not bind first.
