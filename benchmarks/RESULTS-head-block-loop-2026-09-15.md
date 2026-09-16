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

### The ceiling is lifted: the chunked group loop is now a real AIE loop

The counting above pointed at the core program, and the core body never uses the
group index `g` — all four `C1` tiles are reused per group, and the per-group
differences (the params tile, the A2 slice) are host-fed. So unrolling it bought
nothing and cost program memory. `for g in range(n_grp)` → `for g in range_(n_grp)`.

| build | before | after |
|---|---|---|
| N=512 (single group, the guard) | `f3d0a132bde24a60` | **`f3d0a132bde24a60` — unchanged** |
| N=3072, nh8/cols8 | OK | OK (90448 B xclbin) |
| N=4096, nh8/cols8 | **Overflow of program memory** | **OK** (90448 B xclbin — same as 3072, i.e. no longer growing with the chunk count) |

The full shape-ELF matrix now builds end to end:

| family | shape | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| Nanbeige | nh20 hd128 nkv4 cols4 | 243600 B | 482960 B | 961680 B |
| Phi4-mini | nh24 hd128 nkv8 cols8 | 292176 B | 579408 B | 1153872 B |

**Two things this does NOT mean.** (1) It is not verified: the emitted stream for
every chunked build has changed, and a loop that alters FIFO acquisition order is
exactly the class of change that produced the silent C2 failure earlier in this
file's history — the bench gate (NPU vs EMU) and a family token identity are still
required, and the chunked kernels measured before today no longer correspond to
what the generator now emits. (2) The ELF byte size did **not** change for the
previously "failed" 4096 attempts (961680 / 1153872 B before and after): the ELF
carries the *transaction stream*, which is still unrolled on the host side — what
shrank is the core program. That also explains why a failed build left a
plausible-looking ELF on disk: the file is written before the CDO step that
overflows.

The six bucket ELFs sit in `/tmp/elf2_attn_mha_*.elf`, still deliberately outside
`engine/npu/xclbins/` until the device gate passes.

### The host KV layout is hardcoded to 4 kv heads — which reorders the family work

`npu_engine_universal.cpp:4495-4527` sizes the KV regions as

```
region = MAX_L x 4 heads x HD x 2 bytes          # the "4" is literal
kv_region = 4194304 (8 MB = 8192 tokens) by default
           2097152 (4 MB = 4096 tokens) when H == 2560 or H == 4096
           4194304 again when npt > 4096 (the capture's own stride wins)
NPU_ATTN_KV_REGION overrides it; NPU_ATTN_V_REGION_ADD (default 2) is the
K-in-regions-0/1, V-in-regions-2/3 split that FLM's own nanbeige sequence uses.
```

So the *host* packs K and V as **4 kv heads per token** and the ELF is expected to
match that. Measured against the shapes this lane has just made buildable:

| family | nkv | hd | host layout matches? |
|---|---:|---:|---|
| Nanbeige nh20 | 4 | 128 | **yes** — nkv=4 is exactly the host convention, and the H=2560 row gives the 4 MB/4096-token region |
| Qwen3.5-4B nh16 | 4 | **256** | nkv yes, but the region's `HD` term must become 256 (MAX_L × 4 × 256 × 2) |
| Gemma3-1B/4B, Gemma4 | 2 | **256** | nkv 2 of 4 slots is a layout question as well as the HD term |
| Phi4-mini nh24 | **8** | 128 | **no** — 8 kv heads need a different region size and a different `(kvh < 4 ? 0 : 1)` split in `bKv` |

That is a *host* gap, not a kernel gap, and it is invisible to the selector change:
an nh20 or nh24 ELF can now be loaded and selected, and the nh24 one would be fed
the wrong KV layout. Two consequences:

1. **Nanbeige is the first family to attempt**, because it is the only one whose
   (nkv, hd) already matches the host convention. Its bucket ELFs (1024/2048/4096,
   nh20 hd128) are built and waiting.
2. **Phi4 needs `bKv` work before its ELF means anything**, and the hd256 families
   need the region's `HD` term generalised. Both are host changes with their own
   verification, not artifacts.

This also gives the `attn_kv_region` comment's own warning teeth: the table keys on
`H` as a proxy for nh16-vs-nh32, so a family inherits another model's region stride
unless it is overridden — which is why `NPU_ATTN_KV_REGION` exists. For a generated
ELF the stride should be derived from the ELF's own `-N` (its MAX_SEQ) rather than
from the model's H, because that is the only thing the kernel actually baked in.

### Artifact pre-flight: the assembled ELF carries the design exactly

Cheap, offline, and it removes one whole class of doubt before the device window.
Count the design's `aie.dma_bd` ops in the generated MLIR, then count the control
code's `XAIE_IO_WRITE`s in the assembled ELF with `aiebu-dump -m aie2ps -p`:

| shape | design `aie.dma_bd` | ELF `XAIE_IO_WRITE` | verdict |
|---|---:|---:|---|
| nh20 hd128 cols4 N=1024 | 1380 | 1380 | MATCH (6900 ops) |
| nh20 hd128 cols4 N=2048 | 2740 | 2740 | MATCH (13700 ops) |
| nh20 hd128 cols4 N=4096 | 5460 | 5460 | MATCH (27300 ops) |
| nh24 hd128 cols8 N=4096 | 6552 | 6552 | MATCH (32760 ops) |

Two things this establishes. **The ELF is a faithful carrier** — nothing dropped,
nothing duplicated, and it holds for the 4096 builds that used to overflow, so the
real group loop fixed the build without mangling the stream. And the stream's size
is exactly `NH x N` in transactions: `.ctrltext` grows 226336 -> 449376 -> 895456 B
for nh20 across 1024/2048/4096 (x1.985, x1.993 per doubling) and nh24/nh20 = 24/20
= 1.20, which is the 1.20 observed — i.e. the head-block passes and columns
multiply the way the design says they do, rather than one of them being silently
applied twice.

It does **not** say the design is correct: a wrong offset, a mismatched KV layout or
a FIFO desync all survive this check untouched. It says the artifact is worth
spending device time on. `aiebu-dump` lives at `/usr/local/bin/aiebu-dump` (not in
the Vitis trees, which is where I looked first) and takes `-m aie2ps`, not `-t`.

### Bug found by review: the head->kv mapping used the columns, not the model

`gqa` was derived as `cols / nkv`. It must be the **model's** `nh / nkv`: head `h`
reads kv head `h / gqa_model`, and the column count has nothing to do with it.

| family | nh | nkv | cols | old gqa (cols/nkv) | correct gqa (nh/nkv) | effect of the old value |
|---|---:|---:|---:|---:|---:|---|
| guard (working shape) | 8 | 2 | 8 | 4 | 4 | none — that is why it went unnoticed |
| Nanbeige | 20 | 4 | 4 | **1** | **5** | heads 0..3 read four *distinct* kv heads instead of all reading kv0: wrong K/V for 16 of 20 heads |
| Phi4-mini | 24 | 8 | 8 | **1** | **3** | every head read its own kv head |
| Qwen3.5-4B | 16 | 4 | 8 | **2** | **4** | half the heads read the wrong kv head |

The working shape cannot see the difference (cols/nkv == nh/nkv there), which is
exactly why a byte-identity guard on it proves nothing about this parameter — the
guard held while the family mapping was wrong. Fixed to
`gqa = n_heads // nkv` with `assert n_heads % nkv == 0`; the guard build is still
byte-identical (`f3d0a132bde24a60`), and the family streams changed as they should
(nb20 `6b269c9b396e637a`, ph24 `a4df122b20e4c801`, q35 `4d155ad4b67a2049`).

**Every artifact built before this fix is stale and must not be installed** — the
`/tmp/elf2_*` bucket ELFs and the `/tmp/hb2_*` bench kernels both predate it. They
are rebuilt before the device window opens, and the runner's paths are updated to
the rebuilt set so the gate cannot pass judgement on a superseded kernel.

### Post-gqa-fix pre-flight, and a KV-region undersize the host table would cause

The rebuilt bucket ELFs pass the same fidelity check as before the fix (aiebu-dump
on the ELF, not the xclbin — the xclbin has no `.ctrltext` to dump):

| shape | design `aie.dma_bd` | ELF `XAIE_IO_WRITE` | verdict |
|---|---:|---:|---|
| nh20 N=1024 | 1380 | 1380 | MATCH |
| nh20 N=2048 | 2740 | 2740 | MATCH |
| nh20 N=4096 | 5460 | 5460 | MATCH |
| nh24 N=1024 | 1656 | 1656 | MATCH |
| nh24 N=4096 | 6552 | 6552 | MATCH |

So the gqa fix changed the offsets it should have changed and nothing else.

The KV-region arithmetic, checked against the bucket ELFs:

| bucket | kernel needs `N x 4 heads x 128 dims x 2 B` | host (H=2560 row) |
|---|---:|---:|
| 1024 | 1048576 | 2097152 — larger, fine |
| 2048 | 2097152 | 2097152 — exact |
| 4096 | **4194304** | 2097152 — **UNDERSIZED** |

The host only widens to 4 MB above `npt > 4096`, so a Nanbeige run at 3073..4096
keys would hand an N=4096 kernel a 2 MB region: the kernel reads past it, which is
*wrong*, not merely slow — the same class the nh32-2k guard already encodes. Two
ways to fix, both host-side: set `NPU_ATTN_KV_REGION=4194304` for that bucket, or
derive the stride from the *ELF's own* `-N` instead of the model's `H` (which is
what the H table is doing, badly, for families). The second is the real fix; it is
also the one that needs the device to validate, since a wrong region stride and a
correct one differ only in the numbers the kernel returns.

Corollary worth keeping: `NPU_ATTN_KV_REGION` is not a debugging knob for families —
for an N=4096 bucket it is *required*.

### Engine binaries built from this branch (needed before any family run)

The family A/B in the queued runner uses `~/1bit-MONSTER-goal/engine/npu/build/*`,
and those binaries are **the other session's** — they predate the selector change,
so they cannot run a shaped ELF above 1024 keys. This branch's own binaries now
exist:

```
~/wt/family-head-block/engine/npu/build/npu_engine_<variant>   # 18 variants, 19:19-19:25
```

Built with `bash engine/npu/build_npu.sh` after `mkdir -p engine/npu/build` (a fresh
worktree has no build dir, and the script fails with "can't create …: No such file
or directory" rather than creating it). Verified as built from this source, not
stale: the nanbeige binary contains the new `AttnCtx` gate string ("is not a multiple
of cols"), while the generator-side asserts ("gqa = heads/nkv") are absent because
they are Python, not compiled in — which is the expected split, not a missing piece.

So two different binaries are in play and they are not interchangeable:

| purpose | binary | why |
|---|---|---|
| family A/B at <=1024 keys | `~/1bit-MONSTER-goal/engine/npu/build/…` (the other session's) | measures the current default path, which is what that table is for; the selector is irrelevant at <=1024 |
| Nanbeige above 1024 keys | `~/wt/family-head-block/engine/npu/build/…` (this branch) | only these carry the per-slot shaped-ELF selector, and the shaped `attn_mha_*_nh20_hd128.elf` must be installed where the loader searches |

### Gate result #1 (2026-09-15 22:27Z): control PASSES, three new shapes FAIL

First device window for this lane. The gate ran the bench for four shapes, EMU
(host reference, kv-aware: `kv = h/gqa`) versus NPU, on the same packed buffers.

| shape | EMU `max_abs_err` | NPU steady-state | verdict |
|---|---:|---:|---|
| **guard** nh8/nkv2/hd128/cols8 (byte-identical shipped kernel) | 4.564293e-02 | **4.564293e-02** | **MATCH** |
| nb20 nh20/nkv4/hd128/cols4 | 4.646571e-02 | 1.233142e+01 | FAIL |
| ph24 nh24/nkv8/hd128/cols8 | 2.403474e-02 | 1.064977e+01 | FAIL |
| q35 nh16/nkv4/hd256/cols8 | 4.791975e-01 | 4.027662e-01 | FAIL (close, not equal) |

Read the control first: the shipped kernel matches to seven digits on this bench, so
the harness, the packing and the EMU are sound at the shape they were validated on.
The three failures are therefore specific to this lane's changes, and every one of
them is real — nb20's output is stable across iterations at 1.23e+01 where an
all-zero output would score 6.17e-02, i.e. wrong values, not zeros.

**Nothing gets installed, and no parity claim is made.** The bucket ELFs stay in
`/tmp`; the shaped-ELF selector stays unexercised.

Two caveats on the window itself, both of which mean these numbers need a clean
re-run before they are used for diagnosis:

1. **The window was not exclusive.** The runner's idle check caught an inter-row gap
   in the other session's dense-20 loop, so the bench ran with 7 engine processes
   resident (its 19th/20th row in flight). The register documents concurrency as
   this box's largest variance source. The control passing argues the *verdicts* are
   not contention artefacts, but the magnitudes should be re-measured idle.
2. **Part 2 started in the same gap** and is now measuring the family A/B while the
   other session's last row runs, so those numbers need a re-run too.

Also fixed here: the runner reported the *first* `max_abs_err` in the NPU log, which
is a pre-iteration stale-buffer read, so the guard was printed as a 7.19 failure
whose own iterations said 4.564293e-02. The parser now takes the steady-state
iteration value. That is the third false-reading of this session produced by my own
harness (after the `${name}` collision and the `set -e` exit code), and the same rule
covers all three: read the line that is the measurement, not the first line that
looks like one.

### Isolation: the head-block loop is the broken change; nkv/gqa is fine

Two shapes chosen so that one variable moves at a time, both at N=512 so neither the
PV N-split (n_hd=1) nor the group loop (n_grp=1) is exercised:

| shape | nh | cols | nkv | n_hpass | gqa | EMU | NPU | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| B | 16 | 8 | 2 | **2** | 8 | 1.601470e-01 | 8.882848e+00 | **FAIL** |
| C | 8 | 8 | 4 | 1 | **2** | 2.549498e-02 | 2.549498e-02 | **MATCH** |

C proves the nkv>2 / gqa path works (nkv=4, gqa=2, seven-digit agreement), and B
fails purely because its head count exceeds the 8 columns. Every passing case this
session — the guard (nh8/cols8) and C (nh8/cols8) — has `n_hpass == 1`; every
failing one (B, nb20, ph24, q35) has `n_hpass > 1`. Together with result #1, that
narrows the bug to exactly one of this lane's changes: the multi-pass head feed.

Not yet localised *within* that change. The candidates, all mine, in the order I
would test them:

1. the A-tile Q-row base `(hp*cols + c)*K_FRAME` — my read of the "one column per
   head" contract;
2. the pass's KV base `(hp*cols + cc)/gqa`;
3. the pass's C2 writeback base `(hp*cols + c)*(M*K)`;
4. `PARAM_ROW` moving to row `n_heads` once `nh > 15` (B is the first shape to use it:
   nh8 and C keep row 15).

The decisive localiser is a per-head error dump rather than a max: if heads 0..7 are
exact and 8..15 wrong, the second pass's data path is at fault; if the error is spread
across all heads, the feed corrupted the first pass too. `ck2` prints only the max
today, so that dump is the next instrument to add.

**Caveat, unchanged and important:** this run also failed to get an exclusive device.
The wait loop capped at 600 s and then proceeded with 5 processes resident (the other
session's dense-20 loop is still going), so B's magnitude and C's agreement both need
a clean re-run before the *numbers* are quoted. The pass/fail split is unlikely to be
a contention artefact — C agreeing to seven digits is not something contention
produces — but that is an argument, not a measurement.

My runner is stopped; it will not fire again on its own.

### Per-head localisation: pass 1 is grossly wrong, pass 0 partially corrupted

Instrument: `CK_PER_HEAD=1` now prints the per-head max error **of the iteration**
(not of the pre-iteration stale-buffer call — the first placement printed the
pre-loop call, whose `max_abs_out` was 0.000000e+00, i.e. it was reporting |ref| for
an all-zero output; that is the fourth harness mis-placement of the session and the
same lesson each time: print the line that is the measurement).

B = nh16 hd128 cols8 nkv2, N=512, `n_hpass = 2`:

```
head:   0      1      2      3      4      5      6      7      8      9     10     11     12     13     14     15
err:  3.9e-1 9.4e-2 2.5e-2 8.7e-3 9.3e-2 2.4e0  2.0e-2 8.8e-3 3.3e0  8.0e0  7.0e0  8.0e0  7.7e0  3.2e0  2.1e-1 8.0e-1
```

Control (guard, nh8 cols8, `n_hpass = 1`): every head 8.0e-3 … 4.6e-2.

So the failure is **not** a wrong constant on every head: pass 0 (heads 0..7) is
mostly at control level with a few corrupted heads (0, 1, 4, 5), and pass 1
(heads 8..15) is grossly wrong with two exceptions (14, 15 at 2e-1 / 8e-1). That is
the signature of per-pass state being reused or clobbered, which is what this
change introduced: the A2 scratch region is per column and now written **twice per
launch** (once per pass) at the same address, and the C2 FIFO is per column with
depth `n_hd` (1), now produced/consumed twice per launch.

The four candidate bases (A-tile row, KV, C2, PARAM_ROW) all read correctly *on
inspection* — pass 1's A rows are (8+c), its KV base is (8+cc)/8 = 1 for every
column with gqa=8, its C2 base is (8+c)*M*K — which is why the next step is a
controlled layout experiment rather than another reading of the same code: pack the
same head into all 16 A-frame rows and observe which outputs follow it. That pins
the tap's row semantics empirically (the one thing the source text has not settled
for me), and it distinguishes "pass 1 reads the wrong rows" from "pass 1's results
are clobbered after being computed".

Consequence unchanged and absolute: **the bucket ELFs stay in `/tmp`, the shaped-ELF
selector stays unexercised, and no family verdict can be written yet.** The gate did
its job — this is what it is for.

### FIX: the A2 scratch must be per HEAD, not per column (multi-pass shapes verified)

The localisation pointed at shared per-pass state, and this file's own history had
the mechanism: the chunked path already learned that **a second write to the same A2
scratch slice is dropped silently** ("group 1's A2 came out all-zero",
`RESULTS-attention-c2-regression-2026-09-15.md`). My head-block change had every pass
writing its A2 to `32 + c*(M*N)` — the same slice — so pass 1's PV read pass 0's A2.

Changed, all on the host side of the design (no core change):

| item | before | after |
|---|---|---|
| A2 writeback (both branches) | `32 + c*(M*N)` | `32 + (hp*cols + c)*(M*N)` |
| PV A2 read (both branches) | `32 + c*(M*N) + ki*k` | `32 + (hp*cols + c)*(M*N) + ki*k` |
| SCR size | `32 + cols*M*N` | `32 + n_hpass*cols*M*N` (= per head) |
| `AttnCtx` scrsz | `32 + cols*8*MAX_SEQ` | `32 + nq*8*MAX_SEQ` (per head) |

Verification, EMU (host reference) against NPU, same packed buffers, one run each:

| shape | passes | groups | EMU | NPU | verdict |
|---|---:|---:|---:|---:|---|
| guard nh8/cols8/nkv2 | 1 | 1 | 4.564293e-02 | 4.564293e-02 | MATCH |
| nb20 nh20/cols4/nkv4 | 5 | 1 | 4.646571e-02 | 4.646571e-02 | MATCH (was 12.33) |
| ph24 nh24/cols8/nkv8 | 3 | 1 | 2.403474e-02 | 2.403474e-02 | MATCH (was 10.65) |
| **nb20 N=2048** | 5 | 4 | 4.646571e-02 | 4.646571e-02 | MATCH |
| **ph24 N=2048** | 3 | 4 | 2.403474e-02 | 2.403474e-02 | MATCH |
| q35 nh16/nkv4/hd256 | 2 | 1 | 4.791975e-01 | 4.654262e-01 | residual ~3% |

Two things this settles and one it does not:

1. The multi-pass head feed was the *only* thing wrong with the head-block change:
   with the A2 slice per head, every head-block shape agrees with the EMU to seven
   digits, at one group *and* at four groups (`n_hpass=5` and `n_grp=4` together).
   The earlier per-head pattern (pass 0 partly wrong, pass 1 grossly wrong) is
   exactly what "pass 1 overwrites pass 0's A2" predicts, and it is gone.
2. The guard's stream is still byte-identical (`f3d0a132bde24a60`), so the fix
   cannot have altered the single-pass path — the numbers above are not a
   reinterpretation of a changed control.
3. **hd256 is still not exact** (0.479 vs 0.465, ~3%). That is the PV N-split
   (`n_hd=2`), a different change, and the bench's own note says the absolute error
   at these magnitudes is partly inherent to the int8 design — but EMU and NPU are
   both quantised paths that should agree exactly, so a 3% gap is a real residual,
   not a tolerance to wave through.

Caveat, again: the device was never exclusive during any of this (the other
session's dense loop has been running throughout), so these magnitudes deserve a
clean re-run. The pattern — control plus five shapes agreeing to seven digits, with
the previously-failing ones now exact — is not something contention produces.

### The family >1024 blocker is an OPERAND-CONTRACT mismatch, not BO sizing

Ran the Nanbeige >1024 test end to end with the (now EMU-verified) nh20 hd128 2k ELF
installed in this worktree's `engine/npu/xclbins/`:

```
FLM reference arm : Prefill 2048 [flm-ref]   boot=7753
native arm        : Prefill 2048 [bf16]      boot=152981
                    Bf16Mm: attention ELF loaded (177728 B): attn_mha_1024_nh20_hd128.elf
```

So the selector works (the shaped ELF is found and loaded — the 177728 B line is the
*1k* slot's shape-matched capture, and my 2k file takes the 2k slot) and the native
arm runs — but it returns a different boot token. Reading `run_attn` explains it: the
bf16 path does **not** size or pack an A-frame itself. It hands the kernel

* `act` = the Q GEMM output, `attn_rows x qout`, laid out `[token][head][dim]`, and
* a KV region sized `MAX_L x 4 heads x HD x 2 bytes`,

and lets **the ELF's own taps** decide how to read them. The earlier greps for a
`K_FRAME` / params-row in that path came back empty because there is none to find:
the layout lives inside the ELF.

That is a different contract from the one my generator emits. `n1_core_attn.py`
builds for the `AttnCtx`/Zaya convention — head `h` at A-frame row `h*K_FRAME`, params
at row 15 (or row `n_heads` once `nh > 15`), KT/V as `nkv`-major `K*N` / `N*K` slices.
An ELF built that way, fed by a host that provides the bf16 layout, reads the wrong
places; the boot-token disagreement is that, not a tuning error.

So `task-family-host` is bigger than "size the BOs like AttnCtx": one of these has to
happen, and they are different projects —

1. **Emit for the bf16 contract** — teach the generator the `[token][head][dim]` A
   layout, the 4-kv-head KV region and the bf16 params convention, i.e. a second
   addressing mode in `n1_core_attn.py`; or
2. **Give the family path the AttnCtx contract** — have `Bf16Mm` drive the generated
   kernel through the same packed A-frame the bench and Zaya use, which means
   building that frame from the Q GEMM output.

Either way the *generator work is done*: the shapes build, their streams are faithful
(`aie.dma_bd` == `XAIE_IO_WRITE`), and five of six shapes now agree with the EMU to
seven digits. What is missing is the interface between that kernel and the family
path's operand layout.

Two smaller notes from the same run:

* the 1k slot's shape-matched ELF is the *capture* (`attn_mha_1024_nh20_hd128.elf`,
  177728 B), not a generated one — so at <=1024 Nanbeige is still running FLM's
  kernel, which is why the family A/B at 1024 tokens does not depend on any of this.
* `NPU_ATTN_KV_REGION=4194304` is not needed at 2048 (the H=2560 row already gives
  2 MB = 2048 tokens exactly); it is the **4096** bucket that is undersized.

### Which contract to bridge, and why I would pick the host side

The two options are not symmetric, and the asymmetry is that the kernel side is
already **verified** on one of them.

**What is verified today** (EMU vs NPU, one run each, same packed buffers):

| shape | passes | groups | verdict |
|---|---:|---:|---|
| guard nh8/cols8/nkv2 | 1 | 1 | MATCH to 7 digits |
| nb20 nh20/cols4/nkv4 | 5 | 1 | MATCH |
| ph24 nh24/cols8/nkv8 | 3 | 1 | MATCH |
| nb20 N=2048 | 5 | 4 | MATCH |
| ph24 N=2048 | 3 | 4 | MATCH |
| q35 hd256 | 2 | 1 | ~3% residual (PV N-split) |

All of that verification is against the **`AttnCtx` contract**: the packed A-frame
(head `h` at row `h*K_FRAME`, params at row 15 or `n_heads`), the int8 quantization
of `attn_quant.h`, and `nkv`-major `K*N` / `N*K` KT/V slices. Change the generator's
addressing to the bf16 layout and every one of those results has to be re-established
against a reference that does not exist yet for that layout (the bench computes its
reference from the AttnCtx-packed buffers, so it would have to be reworked too).

**Option 2 — give the family path the `AttnCtx` contract — keeps the verified side
verified.** The work is host-side and bounded, and the reference implementation
already exists in the same tree:

1. **A-frame + quantization**: port `AttnCtx::run`'s packing (q8 = sat8(round(q*sq)),
   `sq = 127/max|q|` over all heads; head `h` at row `h*K_FRAME`; params tile at
   `PARAM_ROW*K_FRAME`, one 8-float set per group when chunked) to build the frame
   from the bf16 path's Q GEMM output before loading the generated ELF.
2. **KV slices**: the same `nkv`-major slices the bench packs, from the bf16 path's
   KV region — with the region stride taken from the **ELF's own `-N`**, not the
   model's `H` (the H table is what undersizes the 4096 bucket).
3. **BO sizing**: q frame `max(16, PARAM_ROW+1)*K_FRAME`, C2 `n_heads*M*K`, SCR
   `32 + n_heads*8*MAX_SEQ` — the sizes `AttnCtx` now uses.
4. **Verification path**: bench parity for the shape, then the 2048-key family boot
   token against `NPU_FLM_PREFILL=1` (the test that failed with 152981 vs 7753).

**Option 1 — emit for the bf16 layout instead** — is the mirror image: the host stays
untouched but the generator gains a second addressing mode *and* every verification
above must be redone against a reworked bench. Unless there is a reason the family
path cannot build the AttnCtx frame (I have not found one — it already holds both the
Q GEMM output and the KV cache in memory), option 1 buys nothing for more risk.

**Second, independent reason to prefer option 2**: the family path's `act` is bf16,
and the generated kernel is **int8 x int8 → int32**. Feeding it is a *quantization*
step either way, and `attn_quant.h`'s contract (dual-compiled, bench-checked) is the
one with a host-side reference. Option 1 would leave that quantization implicit in
tap arithmetic, which is exactly how the "silently wrong" class of bug in this file
happened.

Unless you would rather I wrote the second addressing mode, I will take option 2 and
start with step 1 (the A-frame/quantization port), verifying each step on the bench
before it goes near the family path.

### Correction to the plan: do not port *into* the bf16 ELF interface — bypass it

The previous entry said to port the `AttnCtx` packing into the bf16 path. Reading
`run_attn` in full shows that understates the problem, and in a way that changes the
plan rather than just its size:

| the bf16 path does | what this file says about it |
|---|---|
| `run.set_arg(0..2)` | "the kernel's three SCALAR arguments, which are the only part of this call with **no stated meaning** in this file" |
| `memcpy(attn_kv->data() + r*reg, kv + r*reg, used*2)` for `r` in 0..3 | the stream transfers 4.50 MB while this fills 4.00 MB, so "0.50 MB of what the kernel reads is never written" (`BF16MM_ATTN_KV_PT` sweep exists because of it) |
| `attn_act` as the input, `attn_out` as the output | `BF16MM_ATTN_SWAP_IO`, because FLM handed the kernel the opposite pairing to what this engine assumes |
| a captured ELF with one 2k/4k/8k kernel per slot | Nanbeige's native boot token was **nondeterministic** (1214 / 131718 / 145029 from the same command) before the KV BO was zeroed |

So the family path's attention interface is **partially reverse-engineered**, not
merely a different operand layout. Porting the verified `AttnCtx` packing *into* it
would mean satisfying that machinery — unknowns included — and then trusting a
result produced through it.

**Revised plan, and it is the smaller one:** the generated kernel does not need that
interface at all. `AttnCtx` (header-only, in this same tree, already used by
`zaya_decode.cpp` and the bench) takes `init(xclbin, insts)` and
`run(q, k, v, seq, ao)` with the packing that is verified above, and it launches the
five BOs the generator declares (`q`, `KT`, `C2`, `V`, `SCR`) directly. The change is
therefore:

1. when a shape-matched generated attention kernel exists for the family's shape, the
   bf16 prefill path uses an `AttnCtx` instance for attention **instead of** loading a
   captured ELF and calling it through the scalar/region machinery;
2. the KV and Q data flow from the engine's existing buffers into `AttnCtx::run`,
   which is the only new code;
3. the existing ELF path stays exactly as it is for the six working models, so nothing
   verified today is put at risk;
4. verification is the pair already in hand: bench parity for the shape (done for
   nb20/ph24), then the 2048-key family boot token against `NPU_FLM_PREFILL=1`.

That also removes two of the residuals at once: the KV region is sized from the ELF's
own `-N` (so the 4096-bucket undersize cannot recur), and the ELF's scalar arguments
stop being part of the family path entirely.

Nothing about this changes the recommendation to bridge on the host side — it changes
*where*: a new, verified call path, not an extension of the unreversed one.

### The KV layout, decoded — and a correction to my own earlier claim

Filling it is one loop in `npu_engine_universal.cpp` (`qk_norm_pi`):

```cpp
int region = kvh < 4 ? 0 : 1, lh = kvh & 3;
for (int d = 0; d < HD; d++) {
    bKv[ region       * kv_region + pi * 512 + lh * HD + d] = bf16(ks[d]);   // K
    bKv[(region + v_add) * kv_region + pi * 512 + lh * HD + d] = bf16(vs[d]); // V
}
```

with `v_add = 2`. So:

* each token owns a **512-element slot**, which is **4 heads x 128 dims**;
* **K** lives in regions 0 (kvh 0..3) and 1 (kvh 4..7); **V** in regions 2 and 3;
* the iteration is `for kvh < NKV`, so NKV up to 8 is filled correctly.

**Correction:** the earlier audit claimed "the host KV layout is hardcoded to 4 kv
heads, so Phi4 (nkv8) needs a `bKv` change". That is **wrong**, and the `khv < 4 ? 0 :
1` line is what refutes it — Phi4's 8 kv heads already have a home (4 in region 0, 4
in region 1). Withdrawing it here, as the register's own rule requires. The consequence
for the plan is favourable: no `bKv` work is needed for any of these families.

What the same line does show is the **hd256 blocker**, precisely: the slot is a literal
`512`, i.e. `4 heads x 128 dims`. With HD=256 the write index `pi*512 + lh*256 + d`
reaches 1023 inside a 512-element slot, so an hd256 family would overwrite the next
token's K with its own V — a real defect, and one that no amount of region resizing
fixes without changing this arithmetic to `4 * HD`.

**The bridge mapping is therefore fully known**, with no unknowns left:

| `AttnCtx` wants | take from | formula |
|---|---|---|
| `k[t][kvh][d]` (int8, `nkv`-major) | `bKv` | `bKv[(kvh<4?0:1)*kv_region + t*512 + (kvh&3)*HD + d]` |
| `v[t][kvh][d]` | `bKv` | `bKv[((kvh<4?0:1)+2)*kv_region + t*512 + (kvh&3)*HD + d]` |
| `q[head][d]` | `act` (bf16, `rows x qout`) | direct bf16->float, then the `attn_quant.h` q8 contract |

That is what step 1 of the revised plan needs, and it is now a transcription rather
than a reverse-engineering exercise. The hd256 slot arithmetic belongs to
`task-family-host` as its own item.

### The generated kernel is per-QUERY-TOKEN — which reframes the family decision

`AttnCtx::run(const float* qo, const float* ko, const float* vo, int seq, float* ao)`
takes `qd = NQ*HD` floats of query and no row dimension: **one query token's heads
against `seq` keys**. `zaya_decode.cpp` calls it exactly that way, one token at a
time (line 784), which is what a decode step is. The bench does the same.

That is not an accident of the wrapper; it is the kernel's shape. The A-frame holds
head `h` at row `h*K_FRAME`, and the M=8 tile is the **eight columns' head rows**
(i.e. eight heads of one token), not eight tokens. So one launch = one query token.

Where that leaves each use:

| use | shape needed | verdict |
|---|---|---|
| decode (one token at a time) | one token | **the kernel's shape** — and per the register it is at decode parity |
| Zaya path | one token | already wired (line 784) |
| family **prefill** (the bf16 path) | up to 256 rows per call | **2048 launches for a 2048-key prefill**, ~2-5 ms each = ~4-10 s |
| the six working models | block | use the captures they already use — untouched |

So "give the family path the AttnCtx contract" means **one launch per query token**,
and for prefill that is 4-10 s where FLM takes ~2 s and the current CPU reference
takes ~240 s. Correct, and a large improvement on the fallback — but it is a
fallback-shaped path, not parity.

This is why the register's own L1 note says the chunked path "has never been driven
through `zaya_decode.cpp` end to end" and calls the bench "a kernel instrument, not
that gate": the gate wanted an *engine* run, and the engine's prefill is block-shaped.

**Three honest options for family >1024, and they differ in kind:**

1. **Per-token loop in the family prefill path** — bounded work, uses the verified
   kernel, gives correct output at ~2-5x FLM's prefill time instead of ~100x. This is
   what the bridge mapping I just decoded enables.
2. **Document the boundary** — families keep the CPU reference above 1024 keys,
   recorded with the reason (the generator emits a decode-shaped kernel; the engine's
   prefill is block-shaped), and the generated kernel stays where it is genuinely at
   parity: decode and the Zaya path.
3. **Make the generator block-shaped** — M=8 becomes a row block (tokens) rather than
   the eight heads of one token. That is a kernel redesign, and every verification in
   this session applies to the current shape, not to that one.

I would take **1** for the objective as written (families "match or beat FLM"), with
**2** written down as the honest current state until it is measured — and I would not
start **3** without the user's say, because it invalidates the verification set that
makes everything else here trustworthy.

### The generated attention path runs in the engine for the first time (and is slow)

Implemented option 1 (opt-in, `NPU_ATTN_GEN=1`) and ran it for real:

```
Bf16Mm: generated attention .../attn_gen_2048_nh20_hd128.xclbin   (nh20 hd128 nkv4)
Prefill 2048 [bf16]      native boot = 53898   (FLM reference = 7753)   wall = 1087.92 s
```

**What works.** The path loads the generator's `xclbin` + `insts` through `AttnCtx`,
packs K/V from the engine's own `bKv` using the decoded region layout, quantizes q per
row, and completes without crashing — the first time a kernel from
`n1_core_attn.py` has been driven through the engine rather than the bench. Two
failures were hit and fixed on the way: `AttnCtx` gates on `nq % cols` (nh20 needs
`cols=4`, not the default 8), and it sizes its BOs from `MAX_SEQ`, which it reads from
`NPU_ATTN_MAX_SEQ` — with the default 512 and `seq=2048` it walked off its buffers and
segfaulted. The helper now sets that from the bucket.

**What does not work — the boot token is wrong (53898 vs 7753).** Candidates, in the
order I would test them:

1. **`act` may be raw Q, not RoPE'd Q.** The header's own signature says
   "`act`: attn_rows x qout bf16 [token][head][dim] (Q GEMM output, **raw**)", and the
   captured kernels may apply RoPE internally while `AttnCtx` expects q already
   processed — `zaya_decode.cpp` applies RoPE *before* calling `attn_ctx.run`. If so,
   the generated path is feeding un-rotated q and every head's attention is wrong in a
   way that still looks plausible.
2. The K/V region mapping (decoded from the fill, but the fill's `v_add` and the
   kernel's expectation are two different documents).
3. Which query rows the block covers, and whether `seq` should be the *block's* key
   count rather than `attn_tokens`.

**And it is not usable at this speed.** 1088 s for a 2048-key prefill, against ~2 s
for FLM and ~40 s for the CPU reference it is meant to beat — because
`AttnCtx::run` re-packs all K/V on **every one of the 2048 per-token launches**
(2 G+ conversions). Hoisting that packing out of the row loop is a prerequisite for
the path to be worth having; with it the host cost drops to one pack per block and the
launches dominate (~4-10 s).

So option 1 is real but incomplete: the plumbing exists, the kernel runs, and the two
things between here and a family number above 1024 keys are the q preprocessing
question and the K/V packing hoist — the first being a correctness question I can
answer by reading `zaya_decode.cpp`'s call site, the second a mechanical refactor of
`AttnCtx::run`.

Note: opt-in means nothing else is affected — the six working models still take their
capture path, and the moved-aside shaped ELF stays aside until this produces a right
answer.

### `seq` is the causal bound — the first generated-path run fed every row its own future

Two facts from the engine's own source, both checked before changing anything:

1. **`act`/`bKv` are already RoPE'd** — the fill site says it in words:
   *"attn.xclbin expects PRE-RoPE'd Q and K + raw V — the host applies q_norm/k_norm +
   RoPE, the kernel does NOT"*. `zaya_decode.cpp` passes `cca_prep` output to
   `attn_ctx.run` for the same reason. So the leading candidate in the previous entry
   — "the generated path feeds un-rotated q" — **is refuted**, and the RoPE question
   is closed.
2. **`seq` is the mask.** `AttnCtx`'s contract masks `t >= seq`, so `seq` is not "how
   many keys exist", it is "how far this query may look". The engine sets
   `set_attn_tokens(keys)` = total keys and `set_attn_rows(npt)` = this block's rows,
   so the block occupies positions `[keys - npt, keys)` and **row r must get
   `seq_r = keys - npt + r + 1`**.

The first run passed `attn_tokens` to every row, i.e. gave each row the whole block
including its own future — a wrong answer that looks plausible, which is the failure
mode this file keeps producing. Now:

```cpp
const int key0 = attn_tokens - rows;
for (int r = 0; r < rows; r++) {
    int seq_r = std::clamp(key0 + r + 1, 1, attn_tokens);
    gen_ctx->run(qf.data(), kf.data(), vf.data(), seq_r, ao.data());
    ...
}
```

Verification is running (2048 keys, Nanbeige, `NPU_ATTN_GEN=1`): boot token against
FLM's 7753. Cost is still ~18 minutes per attempt because `AttnCtx::run` re-packs all
K/V on every per-token launch, so the hoist is not optional for iteration speed either
— with it, a rerun is ~30 s and the path becomes the ~4-10 s fallback it was meant to
be instead of something slower than the CPU reference it replaces.

### Option 1 is measured non-viable for prefill (and the token is still wrong)

Ran the per-token generated path twice at 2048 keys, Nanbeige:

| run | K/V cache | wall | boot | FLM reference |
|---|---|---:|---|---|
| first | none | 1087.92 s | 53898 | 7753 |
| second | **yes** (repack only when ko/vo change or seq grows) | **823.86 s** | 152368 | 7753 |
| CPU reference (no generated kernel) | — | ~44 s (from 10808 ms at 1000 keys, `RESULTS-family-attention-shape-2026-09-14.md`) | **7753** ✅ | 7753 |

Three separate reasons it cannot serve prefill:

1. **It is slower than the fallback it was meant to replace.** ~824 s against ~44 s for
   the CPU reference, i.e. **19x slower** — and the CPU reference gets the *right*
   token. The K/V cache halved nothing important because the cost is the launch count:
   32 layers x 8 blocks x 256 rows ~ **65k launches** at ~2 ms each, which is 2-3
   minutes of pure launch time before any host work. This is the per-token kernel shape
   meeting a block-shaped prefill, and no amount of caching fixes a launch per row.
2. **The answer is wrong** (53898, then 152368 for the two different causal
   arrangements — neither is 7753), so the correctness question is not closed either;
   the per-row causal bound was a real fix in principle and did not make it right.
3. **It corrupts the heap** — `free(): invalid size`, the engine's SIGABRT handler
   caught it and re-raised for a core dump. That is a new failure in this path, not a
   pre-existing one (the same run without `NPU_ATTN_GEN` completes correctly), and it
   has to be understood before the code could be trusted even if the other two points
   went away.

**Conclusion: option 1 (per-token loop in the family prefill) is not viable, and I am
not going to keep iterating on it.** A path that is 19x slower than the CPU reference
and wrong cannot be the answer to "match or beat FLM", regardless of further fixes. It
remains committed behind `NPU_ATTN_GEN=1`, off by default, and the six working models
are untouched by it.

What that leaves, unchanged from the three options but now with the first measured out:

* **2. Document the boundary** — families keep the CPU reference above 1024 keys, with
  this measurement as the stated reason. The generated kernel stays where it is at
  parity: decode, and the Zaya path.
* **3. Make the generator block-shaped** — M=8 becomes a row block of tokens rather
  than one token's heads, so a prefill is a handful of launches instead of 65k. That is
  a kernel redesign, and it invalidates every verification in this session.

My recommendation is now **2**, with **3** as the only route to actual family parity
above 1024 keys if that is wanted — and it is a decision for the user, because 3
discards the verification set that makes the rest of this lane trustworthy.
