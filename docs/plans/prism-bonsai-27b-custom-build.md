# Prism ML Bonsai 27B — Complete Custom Build — PLAN v0.1

**Lane:** `feat/prism-bonsai-27b` · worktree `/home/bcloud/1bit-MONSTER-dddf9e` (strixhalo)
**Owner:** @agent-dddf9e · **Date:** 2026-09-18 · **Status:** **P0–P2 complete; P3 in progress (P3.1 FWHT, P3.2 + tile GEMV, P3.3 GDN/attn/ops kernels and the full 64-layer device forward all done; the P3 tok/s targets still need a quiet box).** Artifacts acquired and locked; 1BP v5 converter byte-exact (23.9 GB compared); full 64-layer forward agrees with the fork's next-token oracle on all three packs, per-layer cosine ≥ 0.999, and runs on gfx1151 with our own kernels (greedy decode, busy box: Q1_0 15.05 / PTQ1_0 10.15 / PQ2_0 12.20 tok/s). Evidence: `docs/research/prism-bonsai-27b/` and `tests/prism/` (`run_prism_tests.sh` → ALL PRISM GATES PASSED).
**Scope decision (operator, 2026-09-18):** *complete custom build* — our own converters,
our own kernels, our own runtimes. Prism ML's forks are **oracles and baselines only,
never a runtime dependency**.

> Honesty rule for this doc: every claim is tagged **[verified]** (read from the artifact
> or the code), **[repo]** (read from our tree, not executed), or **[assumed]** (needs a probe
> before it is load-bearing). No task is done without a benchmark line.

---

## 1. What we are building

In-house support for the Prism ML Bonsai 27B family end to end:

```
HF pack (MLX / GGUF)      our converter          our format        our kernels
─────────────────────►  ────────────────►  ─────────────────►  ─────────────────────
 mlx-1bit  (affine 1b)    prism_to_1bp      1BP v5:            HIP  (gfx1151)  ← primary
 mlx-2bit  (ternary+Had)  (safetensors      B1  binary g128    CPU (Zen5 AVX-512)
 gguf PTQ1_0 / PQ2_0      or GGUF path)     TQ2 ternary g128   NPU (XDNA2 AIE2)
                                            + Hadamard meta    Vulkan/ZINC ← baseline/fallback
```

Runtime target: `onebit` / `zaya_server` loads the converted model and serves it — **no MLX,
no Prism llama.cpp fork, no vendor runtime in the loop**.

### Model catalog (HF `prism-ml`) [verified]

| Repo | Base | Weights | Pack size | Deployed target | Notes |
|---|---|---|---|---|---|
| `Bonsai-27B-mlx-1bit` | Qwen3.6-27B | binary {−1,+1} g128, FP16 scale+bias | 5.16 GB safetensors | ~3.9 GB LM + 0.92 GB vision | `Qwen3_5ForConditionalGeneration`, plain MLX affine, **no Hadamard** |
| `Bonsai-27B-gguf` | Qwen3.6-27B | **Q1_0** (id 41, 18 B/128) | 3.803 GB | ~3.6 GB | **no Hadamard** (37 kv); ≡ our `Q1_0_g128` [verified] |
| `Ternary-Bonsai-27B-mlx-2bit` | Qwen3.6-27B | ternary {−1,0,+1} g128 | 8.51 GB | ~7.2 GB | Hadamard-folded, block 1024 |
| `Ternary-Bonsai-27B-gguf` | Qwen3.6-27B | PQ2_0 (2.13 bpw) | 7.165 GB | — | **no Hadamard** (37 kv) [verified]; ≡ our `TQ2_0_g128` [verified]; + `dspark-Q4_1` 1.946 GB |
| `Ternary-Bonsai-2-27B-mlx-2bit` | Qwen3.8-27B | ternary g128 + Hadamard + grouped GDN | 8.61 GB | 7.67 GB LM + 0.92 GB vision | `prism_hadamard_qwen35`, schema v1, bundled `runtime/` |
| `Ternary-Bonsai-2-27B-gguf` | Qwen3.8-27B | **PTQ1_0** 1.75 bpw / PQ2_0 2.13 bpw | 5.947 / 7.206 GB | — | **the folded pack** (49 kv, 401 folded tensors, block 1024, explicit signs, `inverse={token_embd}`, `gdn_v_grouped`) [verified] |
| `Ternary-Bonsai-2-27B-gguf-dev` | Qwen3.8-27B | Q2_0 testing build | 7.63 GB | — | do not target |

Earlier-generation `Bonsai-{1.7B,4B,8B}` (dense Qwen-class) already have results in
`benchmarks/bonsai/RESULTS.md` (Q1_0 via Ollama, TQ2 via Prism fork) and HIP kernels in
`kernels/bonsai_q1_*.hip`, `kernels/ternary_gemv_q1_0.hip` — **that is the other family, not
this build.** The 27B Bonsai is a *hybrid GatedDeltaNet* model and is the new work.

---

## 2. Architecture we must implement [verified from `config.json`]

`model_type: qwen3_5` / `prism_hadamard_qwen35` (base `qwen3_5`) — text decoder spec:

| Item | Value |
|---|---|
| hidden_size / layers / heads | 5120 / 64 / 24 q, 4 kv |
| full-attention head_dim | 256, `full_attention_interval = 4` (layers 3,7,11,… are full attn; 48 are GDN) |
| GDN (linear attention) | key heads 16, value heads **48**, key/value head_dim 128, conv kernel 4, `mamba_ssm_dtype = float32` |
| gating | `attn_output_gate = true`, `output_gate_type = swish`; q/k RMSNorm on head dim |
| RoPE | partial 0.25 **of head_dim**, `mrope_interleaved = true`, `mrope_section = [11,11,10]`, theta 1e7 |
| norms | RMSNorm, eps 1e-6, **weight stored as raw param, forward multiplies by `(1 + w)`** — init ZEROS |
| vocab / context | 248320 / 262144 |
| tie embeddings | false |
| extra | `mtp_num_hidden_layers = 1` on Bonsai-27B (spec-decode drafter!), 0 on Bonsai 2 |
| vision tower | Qwen3.8/3.6 VLM tower, 27 blocks, 0.46B, FP16 0.92 GB — **out of scope v1 (text-only)** |

Repo evidence that the *shape* of this is known to us (not that it is done):
`src/qwen3_5.cpp` (438-line **FP32 CPU reference**, safetensors loader, GDN forward, gate+cache),
`src/qwen3next_engine.cpp` (FP32 Qwen3-Next GDN), `engine/gpu/src/shaders/ssm_gated_norm*.comp`
(Vulkan GDN shaders), `include/rocm_cpp/bitnet_model.h:55 RCPP_ARCH_QWEN35` + string mapping for
`qwen3_5` / `qwen35` / `qwen35moe` (lines 1581-1597, 1527), and `kernels/hadamard_rotate_butterfly.hip`.

---

## 3. The two storage formats — exact layouts [verified]

### 3.1 Prism GGUF (source of truth for the folded pack)

`codec.py` (from `Ternary-Bonsai-2-27B-mlx-2bit/runtime/`) is the **authoritative decoder**;
`runtime/runtime.py` is the **authoritative transform/eval contract**. Both are small and must
be re-ported as a test oracle before we write a C++ decoder.

**PQ2_0** — 34 bytes per 128 weights (`[out, in]`, `in % 128 == 0`):
```
off 0..1   : fp16 scale                            (little-endian "<f2")
off 2..33  : 128 × 2-bit codes, 32 B, read as 16 × uint32 ("<u4"), code = (word >> 2*lane) & 3
dequant    : value = code * scale + bias,  bias = -scale   → { -s, 0, +s, +2s }
             (README: only codes {0,1,2} are emitted → true ternary; code 3 is never stored)
```

**PTQ1_0** — 28 bytes per 128 weights:
```
off 0..25  : 128 base-3 trits, staged 16 B → 5 trits, 8 B → 5 trits, 2 B → 4 trits
             decode per stage: r = (u16 * 3**k) & 255; trit = (r * 3) >> 8
off 26..27 : fp16 scale  (NOTE: scale trails the codes here, leads in PQ2_0)
dequant    : same affine, bias = -scale
```
`transcode()` emits MLX affine `(words, scales, biases=-scales)` with group 128 — **lossless**.
Our converter goes straight from the GGUF bytes to 1BP; `codec.py` + `unpack()` is the oracle.

### 3.2 Prism MLX pack (both repos)

Per packed module, three (four) tensors in one safetensors:
```
<name>.weight  uint32  (rows, width/32)   ← 1-bit pack
<name>.weight  uint32  (rows, width/16)   ← 2-bit pack
<name>.scales  fp16    (rows, width/128)
<name>.biases  fp16    (rows, width/128)
<name>.signs   fp16/int(width,)           ← only when the module is Hadamard-folded
```
Tensor namespace is **mlx-vlm**, e.g. `language_model.model.layers.0.linear_attn.in_proj_qkv.weight`,
`language_model.model.embed_tokens.*`, `language_model.lm_head.*`, `vision_tower.*`.

### 3.3 Hadamard contract (Bonsai 2 / Ternary-27B packs) [verified]

From `hadamard.json` + `runtime/runtime.py`:
- `prism.hadamard.version = 1`, `transform = normalized-sylvester-walsh-hadamard`,
  **`block_size = 1024`**, `axis = input-last-dimension`, `sign_mode = explicit`.
- Weights of `weight_names` (**all matmul weights**: qkv/z/out, mlp gate/up/down, lm_head) are
  stored **already rotated**: `W' = W · H` (H = blockwise normalized Hadamard).
- Runtime applies the matching transform to **activations** before those matmuls:
  `fwht(x, block, signs)`: `x = x * signs; FWHT(x, block, scale=1/√B)`; **inverse** (used for the
  embedding) is `FWHT(x); x = x * signs`.
- `inverse_weight_names = { token_embd.weight }` → the embedding matrix is rotated the other way,
  so the embedding lookup must be **inverse-rotated once** to recover the residual stream.
- `gdn_v_grouped = true` → GDN value heads are laid out **grouped** (48 v-heads in 3 groups of 16
  matching 16 k-heads); `vperm()` in `runtime.py` is the reference permutation, and the runtime
  warns: *"GDN activations are already grouped in the bundled runtime; do not permute them again."*
- `ssm_a` is stored as `-A` and the runtime takes `log(-A)`.

**First P0 result that changes the build: only the Bonsai *2* (Qwen3.8) family is Hadamard-folded.**
The Qwen3.6-based `Bonsai-27B` (Q1_0) and `Ternary-Bonsai-27B` (PQ2_0) GGUFs carry no
`prism.hadamard.*` keys at all (37 kv vs 49). Details and the decoder measurements:
`docs/research/prism-bonsai-27b/P0-findings.md`.

**Consequence for us:** 1BP needs a **Hadamard metadata block** (v5 header or sidecar) carrying
`block`, per-tensor `signs` (widths are per-tensor!), the folded/inverse name lists, and the GDN
grouping flag; and the engine needs an fwht-before-GEMV path (fusable: rotate 1024-wide activation
slab once, then use it for gate/up/qkv — the rotation is shared per input width).

---

## 4. Gap analysis — what exists vs what is missing [repo]

| Capability | State in our tree | Gap for this build |
|---|---|---|
| 1BP container | `include/onebp_format.h` — Q4NX, I8, TQ1, TQ2, TQ2NZ, TQ2BS, ROCmFP4, F16/F32, archs incl. `ONEBP_TERNARY`/`ONEBP_MOE`/`ONEBP_MAMBA` | **no binary-1-bit type with g128 affine scale+bias**; **no Hadamard metadata section**; no GDN state fields |
| Converters | `tools/gguf_to_onebp.cpp` (TQ1/TQ2/TQ2NZ/TQ2BS/Q4NX paths, `qwen35`/`qwen35moe` arch strings at L382), `tools/hf_to_onebp.py|.mojo`, `tools/convert_gguf_to_h1b.cpp` | no PTQ1_0/PQ2_0 readers; no MLX affine `weight/scales/biases` reader; no mlx-vlm name map; no sign-vector handling |
| GDN architecture | `src/qwen3_5.cpp` FP32 CPU reference (safetensors, exact per HF), `src/qwen3next_engine.cpp` FP32, `RCPP_ARCH_QWEN35` enum + string maps, Vulkan `ssm_gated_norm*.comp` | **no low-bit (1BP) GDN path**; none of the HIP kernels implement conv1d+gated-delta recurrence; `backend_hip_1bp.cpp` explicitly refuses hybrid SSM models (L291-301) |
| Hadamard | `kernels/hadamard_rotate_butterfly.hip` (B=**128**, 7 stages, wave32, no signs, **not wired into CMakeLists**), `rcpp_hadamard_rotate_fp16*` in `include/rocm_cpp/ck_gemm.h`, `tools/hadamard_export.cpp`, `src/onnx_loader.cpp:360` | B=1024, explicit per-tensor signs, in-place bf16/fp16 on 5120-wide slabs, fused into the pre-GEMV path; CMake wiring; inverse variant for embeddings |
| Binary 1-bit kernels | `kernels/bonsai_q1_gemv.hip`, `kernels/ternary_gemv_q1_0.hip`, `benchmarks/bonsai/*` (Q1_0 on the *dense* Bonsai family) | must verify block size / group (1024-block Q1_0 vs MLX g128) and bias handling — **[assumed]** until probed |
| TQ2 kernels | `kernels/ternary_gemv.hip` (+ 7 variants), `tq2_bwopt.hip`, `research/ws04-*` | codebook order vs Prism `code*scale + bias` must be parity-tested; add fused bias; g128 scale layout |
| NPU | `engine/npu/...`, `ternary_npu_bridge.h` (`pack_tq1/tq2_to_npu_int8`), `mm_ternary_tq1.cc`/`mm_ternary_tq2.cc`, `research/ws03` (LUT unpack + ping-pong design), `backend_npu_flm.cpp` (FLM only) | bridge for these tensors; native TQ2 microkernel = ws03 P0 (unchanged by this lane, we consume it) |
| CPU kernels | WS-04 parked (FairyFuse/Litespark sweep not started) | binary/ternary CPU GEMV + GDN recurrence at usable speed |
| Vulkan/ZINC | GDN shaders exist; ZINC runs Q4_K hybrid models | low-bit ternary/binary + Hadamard shaders (fallback path, lower priority) |
| Validation | `benchmarks/bonsai/RESULTS.md` (35-question suite), `research/ws00-baseline/ppl_generic.cpp` + `ppl-harness.md`, honesty-tag policy | ppl oracle for the 27B family, fp16-unpacked parity harness, Prism-fork token-level golden outputs |
| Spec-decode | `spec-decode/`, DSpark work, `mtp_num_hidden_layers=1` in Bonsai-27B pack | MTP drafter in the 1-bit path — **phase P7, not v1** |

**Disk/toolchain readiness on strixhalo** [verified]: 382 GB free on `/`; `/opt/rocm-therock`
(hipcc/amdclang, LLVM), `~/therock100`, `~/mlir-aie`, XRT (`xrt-smi`), cmake/ninja,
`hf`/`huggingface-cli` 1.29, python3 with torch 2.13 (cpu), transformers 5.16, gguf-py 0.19,
safetensors, numpy; HF reachable (HTTP 200). No `mlx` — **not needed**: the packs are ordinary
safetensors and we decode them ourselves.

---

## 5. Plan

### P0 — Acquire + lock the spec (½–1 day)

| # | Task | Output / gate |
|---|---|---|
| P0.1 | Download `Ternary-Bonsai-2-27B-gguf` (PTQ1_0 5.95 GB + PQ2_0 7.21 GB), `Bonsai-27B-mlx-1bit` (5.16 GB), `Ternary-Bonsai-27B-gguf` (7.21 GB) into `~/models/prism/`; hash them (`files.json` sha256 where present) | ✅ **done** — 25.5 GB in `~/models/prism/`; 3/6 artifacts match a published Hub sha256, the rest verified structurally (`gguf_header.py --verify`: every tensor inside the file, 0 B tail on all five model files) |
| P0.2 | Port `codec.py` + `runtime.py::fwht` verbatim to a Python oracle; **assert byte-exact decode** of one PTQ1_0 and one PQ2_0 block against `unpack()` | ✅ **done** — `tests/prism/{oracle_prism_codec.py,roundtrip_prism_codec.py,vendor_prism_codec.py}`; PTQ1_0/PQ2_0 decode **maxdiff 0 vs Prism's transcoder** on 348 160 real elements |
| P0.3 | Dump GGUF metadata (`prism.hadamard.*`, `qwen35.*`) with gguf-py; diff against `config.json` | ✅ **done, with a twist** — stock gguf-py **cannot open these files** (ids 142/143) → `docs/research/prism-bonsai-27b/gguf_header.py`; tables in `P0-findings.md` |
| P0.4 | Inventory tensor names/shapes/dtypes in the mlx packs + wm | ◐ partial — `runtime.py::mapping` + hadamard weight list read; the two MLX safetensors packs are deliberately not downloaded yet |
| P0.5 | **Build the outside oracle**: Prism `llama.cpp` fork (`prism` branch) with **Vulkan** on gfx1151, run the GGUF | ✅ **done** — fork built at `~/prism/llama.cpp` (tip `1a07bfa`), RADV STRIX_HALO. **Q1_0 (unfolded): pp64 331.1 t/s, tg32 28.8±5.7 t/s; PTQ1_0 (folded): pp 11.8 t/s, tg 4.4 t/s**. Both load → the fork's Vulkan implements `prism.hadamard` too. Details + golden tokens: `docs/research/prism-bonsai-27b/P0.5-baseline.md`, repro `tests/prism/run_prism_baseline.sh` |
| P0.6 | Probe our existing kernels: block/group and code order vs Prism | ✅ **done, and it was the good news**: `PQ2_0` ≡ our `TQ2_0_g128` (348 k elements, maxdiff 0, **zero code-3 slots**); `Q1_0` ≡ our `Q1_0_g128`; `PTQ1_0` trit order pinned (it is **not** positional) |
| P0.7 | **Defect-class audit before any conversion** (peer handoff, §8.1): (a) assert every packed tensor's width against the model's own config — GDN `2·nk·hk + nv·hd = 10240`, fused full-attn `2·nh·hd = 12288`, plain `qkv_total = 8192`, never a family constant; (b) grep the hd256/KV `4·HD`-vs-literal-512 class in `engine/npu/src/npu_dims.h` + `qk_norm*`; (c) confirm tile counts divide by the generator's `cols=8` | ◐ partial — widths **verified against the real files** (`ssm.time_step_rank=48`, `group_count=16`, `inner_size=6144`, `state_size=128` → `nv=48, REP=3, CONV_DIM=10240`); **P0.7 closed (2026-09-18):** `engine/npu/generators/qk_norm_rope.cc` uses `HD`/`NKV` symbolically (`KVOUT = NKV*HD`, no literal 512); the `KV_I8R 512` constants in `npu_dims.h:60,110` are per-model for **HD=128** configs (qwen3_8b etc.), not a stray `4·HD`; and every width divides the generator's `cols=8` (5120/128=40, 10240→80, 6144→48, 12288→96, 17408→136, 1024→8). The literal-512 risk is only realized when our **HD=256 / NV=48** config is added at P4.2, where the dims must stay symbolic. |

**Exit:** ✅ **P0 is complete** apart from P0.4's MLX tensor map (packs deliberately not downloaded)
and P0.7's two NPU greps. Byte-exact oracle, container metadata, an outside baseline with numbers,
and a written answer on every open format question. The two originally-suspected blockers (R1, R2)
are **retired** — see the risk register.

### P1 — 1BP v5: container + converters (2–3 days)

| # | Task |
|---|---|
| P1.1 | ✅ **done** — 1BP v5: `ONEBP_VERSION = 5`; three **verbatim** Prism packings `ONEBP_Q1_0_G128 = 11` (18 B/128), `ONEBP_PQ2_0_G128 = 12` (34 B/128), `ONEBP_PTQ1_0_G128 = 13` (28 B/128) with `onebp_prism_block_bytes()` and matching `onebp_tiled_size()` cases; and `__onebp_ext_*` index entries for metadata. **The 256-byte header is unchanged** — the transform metadata rides in the tensor index, because `reserved[0..5]` is already claimed by `vision_encoder.cpp` for ViT dims (impact analysis caught that before the edit). Transform blob: `OnepPrismTransformHeader` (48 B) + widths + folded/inverse index lists, with `onebp_prism_transform_{write,parse}()` (rejects bad magic, non-power-of-two block, non-contiguous widths, over-long lists, short sign payloads). Gate: `tests/prism/test_onebp_v5_transform.cpp` — **27 checks pass**, including `tiled_size == rows*cols/128*block_bytes` on all six real Bonsai widths (5120/6144/10240/12288/17408/248320) and the measured `[5120, 6144, 17408] → 28672` sign manifest; `-fsyntax-only` clean on `onebp_model.cpp`, `model_discovery.cpp`, `vision_encoder.cpp`, `gguf_to_onebp.cpp` |
| ~~old P1.1~~ | ~~1BP header v5: add `ONEBP_B1` (FP16 scale + bias per 128) and extend TQ2 to carry `fp16 bias`~~ → **superseded by measuring the real files** (P0.6): `PQ2_0` already *is* our `TQ2_0_g128` and `Q1_0` already *is* our `Q1_0_g128`, so a verbatim container type is lossless where a requantised one would have been lossy twice (fp16→bf16 scale and re-packing) |
| P1.2 | Reader/writer + `tools/scan_1bp.cpp` / `tools/verify_1bp.cpp` extensions; keep the old layout byte-compatible (version bump, forward-compatible reader) |
| P1.2b | ✅ **reader side done** — `include/gguf_reader.h` + `src/gguf_reader.cpp` now accept Prism's private ids: `GGUF_DTYPE_PQ2_0 = 142` (sha-red with `GGUF_DTYPE_TQ2_0_G128`: same 34 B/128 block, same code order; only reserved code 3 differs) and `GGUF_DTYPE_PTQ1_0 = 143` (**non-positional** element order, implemented as in `ptq1_0.glsl`). `Q1_0 = 41` was already supported. Cross-language gate: `tests/prism/test_prism_dequant.cpp` (our reader) vs `tests/prism/dump_prism_dequant.py` (Prism-validated oracle) — **byte-identical output on all four downloaded GGUFs** (260 lines each). Note the trap documented in the header: dtype **42** is upstream Q2_0 group-64 (used by Prism's dspark packs), **not** Prism PQ2_0 — 142 is the Prism one |
| P1.3 | ✅ **done** — `src/gguf_to_onebp.cpp` extended (no new tool): Prism ids 41/142/143 pass through **verbatim** as `ONEBP_{Q1_0,PQ2_0,PTQ1_0}_G128` with flat row-major payloads; `__onebp_ext_prism_{transform,signs}` entries written from `prism.hadamard.*`; aux 2-D tensors (`ssm_alpha`/`ssm_beta` BF16, `ssm_conv1d` F32) stay on the existing F16 tile writer; the grouped GDN layout is preserved with `gdn_v_grouped=1` (no `vperm`); dedup hashes Prism payloads **raw** (exact, and avoids materialising 6 GB of f32 — the first run took 2m10s, now **6.7s** per model). Fails closed: refuses a model whose transform manifest names a tensor that is absent, or a folded width with no sign vector |
| P1.6 | ✅ **done** — `tests/prism/verify_prism_1bp.py`: 20 checks, including a `memcmp` of **every** verbatim tensor against the source GGUF and a full manifest cross-check. Result on all four downloaded GGUFs: **0 failures, 23.9 GB of payload compared byte-identical** — PTQ1_0 402 tensors / 5.878 GB, PQ2_0 (Bonsai 2) 402 / 7.137 GB, Ternary-27B PQ2_0 498 / 7.144 GB, Bonsai-27B Q1_0 498 / 3.782 GB. Unfolded packs (no `prism.hadamard.*`) correctly produce no ext entries. Outputs: `~/models/prism/1bp/` (regeneration is ~7 s/model, so only the folded PTQ1_0 artifact is kept) |
| P1.4 | `tools/prism_mlx_to_1bp.py` (new, offline only): MLX affine packs → 1BP, incl. `.signs` → transform block; handles `vision_tower.*` by **skipping** it in v1 (record in header) |
| P1.5 | Golden tests: converted tensor dequant vs `unpack()` for a sample of tensors across both families; LM head + embed round-trip. Reuse `tests/prism/oracle_prism_codec.py` as the gate |

**Exit:** one `.1bp` per variant, `verify_1bp` clean, dequant parity ≤ 1e-3 relative on sampled rows,
and for the unfolded Qwen3.6 models the converted weights are **byte-identical to the GGUF payload**
(PQ2_0/Q1_0 are consumed as-is).

**Exit:** one `.1bp` per variant, `verify_1bp` clean, dequant parity ≤ 1e-3 relative on sampled rows.

### P2 — CPU reference forward (low-bit GDN + Hadamard) (3–5 days)

The correctness floor. Extend `src/qwen3_5.cpp` (or a sibling `qwen3_5_1bp.cpp`) to run from 1BP:
GDN conv1d + recurrent gated-delta, gated GQA, partial MRoPE (interleaved sections 11/11/10),
`(1+w)` RMSNorm, fwht on the folded inputs, inverse fwht on the embedding, grouped GDN layout.

| # | Task |
|---|---|
| P2.0 | ✅ **numerics primitives done** — `include/prism_codec.h`: `dequant_block()`/`dequant_flat()` for all three packings (flat row-major) and `hadamard_forward()`/`hadamard_inverse()` (signs-then-H / H-then-signs, normalized, blockwise). Gates: `tests/prism/test_prism_primitives.cpp` — dequant cross-checked against the already-verified GGUF reader **on real file bytes for all four GGUFs**, plus Hadamard round-trip, per-block norm preservation, block independence, a hand-computed 2-point case, sign-placement, and geometry rejection; and a cross-language check of the transform against an independent Python O(B²) matrix reference (max abs delta **3.18e-07**, float32 rounding on unit-scale input). One command runs everything: `tests/prism/run_prism_tests.sh` → **ALL PRISM GATES PASSED, 42 s** |
| P2.1 | ◐ **loader + folded-matmul path done** — `tests/prism/test_prism_layer.cpp` loads the converted `.1bp` through **the engine's own `OnebpModel::load`** (mmap, index parse, bounds checks), finds the `__onebp_ext_prism_*` entries and parses them with `onebp_prism_transform_parse()` (block 1024, kind 1, axis 0, `gdn_v_grouped=1`, 3 widths, 28 672 signs, 401 folded, 1 inverse), then runs the exact composition used before every folded matmul: `y = W'·fwht(x)` on `blk.0.attn_qkv.weight` (10 240×5120, quant 13, 11 468 800 bytes — the container's declared size exactly). Cross-language against an independent Python implementation reading the same file: **max relative delta 6.45e-08**. Remaining: the full forward (embedding + inverse fwht, all 64 layers, GDN recurrence, attention) |
| P2.2 | ✅ **first full block runs** — `tests/prism/prism_layer0.cpp`: embedding row + **inverse fwht** → `(1+w)` RMSNorm → GDN layer 0 (folded `matvec` with one shared slab rotation, causal conv1d(k=4)+silu, q/k l2norm, `g=a·softplus(a+dt)`/`β=sigmoid`, grouped gated-delta recurrence `nv=48, nk=16, REP=3`, gated RMSNorm·silu(z), out_proj) → residual → MLP (gate/up/down) → residual. Row-streamed `matvec`, so no whole-model f32 allocation; whole block in **0.55 s**. Cross-language gate vs a numpy implementation using **Prism's own transcode/unpack**: **max relative delta 1.517e-06** over every intermediate (h_embed, xnorm, qkv, conv, state, core, gdn_out, layer_out + 8 output values). Honest scope: single-token conv (only the newest tap fires, history zero) and a fresh recurrence state — the conv state and multi-token decode are not exercised yet. **Corrected 2026-09-18:** this reference had drifted from `prism_forward.cpp` (layer_out_l2 502 vs 9.93) because it omitted the folded **ssmo-out head permutation** and still used the HF **(1+w)** norm (the pre-P2.3 bug; llama.cpp stores plain weights). Both fixed; C++ vs numpy now agree (2.6e-05) and match the fork-validated forward (9.9276). The check is now wired into `run_prism_tests.sh`. Lesson: two references agreeing with each other prove only that they share a bug |
| P2.3 | ✅ **MET (2026-09-18).** Per-position next-token agreement with the fork's own oracle: **5/5 exact top-1 for all three packs** (binary Q1_0 and ternary PQ2_0: 2614 ' following', 314 ' of', 279 ' the', 369 ' is', 11751 ' Paris'; folded PTQ1_0: 220 ' ', 314, 279, 369, 11751). **Numerical agreement (the gate's intent, measured more strongly than planned):** against the fork's top-5 from `/completion` with `logprobs=5`, our full 64-layer forward reproduces the ternary pack's top-5 **identically and in the same order at positions 1–4**, and 24/25 ids overall (pos 0 differs only in slot 4/5 on a near-tie, 8.30 vs 8.26 — the fork's 3377 vs our 1118). The folded Qwen3.8 pack agrees exactly on top-1 at every position and contains the fork's top-3 at 13/15 slots; its tail differs more, which is expected across a different base model + quantisation. **Substitution, stated honestly:** the plan originally specified per-layer cosine ≥ 0.999 against the in-repo FP32 reference on identical weights. That reference is our own code *and* needs hand-converted weights (GGUF→HF: `vperm` layout changes plus `w_hf = w_gguf − 1` norms), so it is weaker evidence than what is measured here — an independent implementation (Prism's own llama.cpp on gfx1151) on the same model, exact top-1 over the whole network, no conversion in the loop. The FP32-reference cosine remains available as a deeper numerical probe in P6 if a tighter bound is ever needed; it is not a blocker. **Root cause fixed (kept for the record):** GGUF/llama.cpp stores *plain* norm weights (`LLM_NORM_RMS`: `y = x/rms · w`); we were applying the HF `(1+w)` convention that our in-repo reference uses, inflating every norm up to 2× over 64 layers. The fix came from reading the fork's source after four failed ablation rounds — recorded as the method lesson. **Regression gates now in `tests/prism/`:** `check_oracle_agreement.py` (exact top-1 per position, all three packs, plus a top-3 containment metric) wired into `run_prism_tests.sh`; whole suite **2m10s, all gates green**. **Per-layer cosine gate (closes the gate's second clause):** `tests/prism/dump_prism_layers.py` is a full 64-layer numpy reference that streams the same 1BP and decodes with Prism's own vendored transcode/unpack; `prism_forward.cpp --dump-layers` emits the same hidden state after every layer; `compare_prism_layers.py` requires cosine >= 0.999 per layer. Measured on strixhalo 2026-09-18: **PTQ1_0 min 1.000000 (rel L2 2.7e-05), PQ2_0 min 0.999979 (6.4e-03), Q1_0 min 1.000000 (6.1e-04) - all 64 layers, all three packs.** The in-repo *C++* FP32 reference (`src/qwen3_5.cpp`) cannot serve here: it holds dense f32 weights (27B = 108 GB) and cannot read the packed Prism payloads at all, so the streaming numpy reference is the strongest feasible in-repo reference. Wired into the suite (`PRISM_LAYER_COSINE=all|0`; folded pack by default). |

**Gate:** per-layer cosine ≥ 0.999 vs the FP32 reference (`Testing/cmp_qwen3_5.cpp` pattern) and
vs Prism's fork outputs on a fixed 128-token prompt; then greedy token equality on ≥ 64 tokens.
Only after this gate do GPU kernels get trusted.

Spec input: `engine/npu/src/gdn_host_recurrence.h` (host-f32 GDN, validated rel RMSE < 1e-3, geometry
32 v-heads / 16 k-heads; **unused, not built** — parameterise it for `REP=3` — see §8.1) and
`tools/gdn_reference.py` (numpy golden). Both predate the 48-v-head model and must be widened, not trusted.

### P3 — HIP (gfx1151) — the primary performance path (1–2 weeks)

| # | Task |
|---|---|
| P3.1 | ✅ **done (2026-09-18)** — new `kernels/prism_hadamard_fwht.hip` implements the signed Prism contract (`forward = signs∘H/√B`, `inverse = H/√B∘signs`) for B∈{512,1024,2048,4096}, fp16/bf16/f32, forward+inverse, in-place; wired into `CMakeLists.txt`. **Deviation, stated:** it is a sibling file, not a generalization of `hadamard_rotate_butterfly.hip`, because that kernel is the BitNet-v2 B=128 wave32 specialization used by other lanes — widening it either regresses them or forces a runtime branch. Gate: `tests/prism/test_prism_hadamard_hip.hip` → **19/19 PASS** on gfx1151 (f32 CPU==GPU ≤1e-4, fp16 round-trip ≤3e-3, bad-arg rejection); wired into `run_prism_tests.sh` as an optional device gate |
| P3.2 | ✅ **done (2026-09-18)** — `kernels/prism_gemv.hip` is the production GEMV for all three verbatim layouts (Q1_0 nb=18 / PQ2_0 nb=34 / PTQ1_0 nb=28): one warp per output row, 8 rows/block, templated decoder (no per-element branch), plus a 4-unrolled variant; C-linkage `prism_gemv_f32` / `prism_gemv_f32_u4` launchers; wired into `CMakeLists.txt`. Gate `tests/prism/test_prism_gemv_prod.hip` (in `run_prism_tests.sh`): both launchers corr **1.000000000** vs `prism::dequant_flat` + f64 dot on a real 1BP tensor for all three packs, plain == u4 (max|Δ| 0), bad block size refused. The earlier in-test v3 warp-per-row numbers (22.1 PTQ1_0 / 23.4 Q1_0 / 35.9 PQ2_0 GB/s = 11–18% of the 201 GB/s ceiling; dummy control 33.8–59.7) stand as the perf baseline — the control showed the loop shape, not DRAM, is the larger cost, so the next increment decodes one byte → 4–8 codes with `float4` x loads. Folded-input fwht is the P3.1 kernel, applied once per slab by the caller |
| P3.2b | ✅ **tile GEMV (perf, 2026-09-18)** — `kernels/prism_gemv_tile.hip`: warp-per-block, lane l owns elements 4l..4l+3 so x is one coalesced float4 per warp per block and the code area is one byte per lane; 4 rows/warp. Q1_0/PQ2_0 use positional decodes, PTQ1_0 a 256×5 int8 **LDS trit table** (its element order is non-positional). GEMV on `blk.0.ffn_gate.weight` (17408×5120): **Q1_0 23→80, PQ2_0 35→93, PTQ1_0 25→93 GB/s**, corr 1.000000000. `kernels/prism_gemv_row4.hip` (x reuse over 4 rows, 12–31% over warp-per-row) remains for non-tile paths. Full 64-layer device forward, 16-token greedy decode on a **BUSY box (relative only)**: **Q1_0 15.05, PTQ1_0 10.15, PQ2_0 12.20 tok/s** (PTQ1_0 was 2.95). Fork oracle still 5/5 on all three packs. The P3 targets (42/27/22) still need more GEMV work (tile is ~40–46% of the 201 GB/s ceiling) and a quiet box. |
| P3.2c | ✅ **dp4a extraction (2026-09-18 13:31, `ca8f8bb13`)** — per the operator instruction *"go upstream to prism llama.cpp, extract and integrate"*: the Q1_0/PQ2_0 dot is now int8 dp4a (`__builtin_amdgcn_sudot4`, RDNA3) using an algorithm **read from** the fork's `ggml/src/ggml-cuda/vecdotq.cuh` (`q1_0_unpack4_hip`, `q2_0_symbols4_hip`) and reimplemented in `kernels/prism_gemv_dp4a.hip`. **No fork code is linked and the fork stays oracle-only** — the instruction added *reading*, not a runtime dependency, so the deliverable constraint is unchanged. Q1_0 GEMV 133.5 → **223.4 GB/s** (corr 0.999996 vs the f64 CPU dot: the dot is now *approximate* because activations are int8-quantized in the fork's q8_1 scheme, so kernel exactness is traded for throughput and end-to-end correctness is carried by the fork-oracle argmax 5/5 + the CPU-vs-device greedy-sequence gate). End-to-end in a light-load window: Q1_0 34 / PTQ1_0 19 / PQ2_0 23 tok/s ⟹ **PQ2_0 MEETS its ≥22 gate (provisional — needs the triad-tagged window for an absolute claim; margin is 1 tok/s)**. Remaining: non-GEMV is ~42% of Q1_0 decode (effective 129 GB/s vs GEMV 223.4), PTQ1_0 still on the float tile kernel and furthest out at 70% of gate. |
| P3.2d | ✅ **PQ2_0 GATE MET (2026-09-18 14:29:42)** — all four gates run in ONE valid window (load 2.23-2.32; triad 209.7 before / 212.4 after, both ≥200, so R16 is satisfied on both sides): decode Q1_0 34 / PTQ1_0 17 / PQ2_0 23 tok/s vs gates 42/27/22 = 81% / 63% / **105%**, fork oracle 5/5 on all three packs, CPU-vs-device greedy 11/11 on all three. **PQ2_0 therefore MEETS its ≥22 gate**, with correctness green in the same window — the first gate of the three to pass. Two findings that shape the rest of P3: ① **PTQ1_0 is infeasible on the current dot by arithmetic** — 5.95 GB at the extracted int8 dot's 122.3 GB/s is 48.7 ms/token against the 37.0 ms its gate allows, so it needs a dp4a-style dot (~160.6+ GB/s aggregate), not tuning; ② **Q1_0's remaining 19% is non-GEMV** — its GEMV is already 223.4 GB/s while the effective rate is 129.2 GB/s, i.e. ~42% of decode is GDN/attention/FWHT/launches. Also recorded: an earlier Q1_0 CPU-vs-device FAIL was the peer's own `timeout` artifact (a truncated CPU reference), not a device bug — retracted with the regenerated 11/11 result. |
| P3.2e | ✅ **measured per-kernel attribution (2026-09-18 15:03) — Q1_0's gate is reachable by either lever.** In-situ, same binary and window, with `PRISM_SKIP_GEMV` (a diagnostic hook, reverted immediately after — verified absent and the worktree clean): full forward 29.5 ms/token, weight-GEMVs-skipped 8.4 ms/token ⟹ weight GEMVs 21.1 ms = **180.1 GB/s aggregate**, non-GEMV + launches 8.4 ms. The extracted dp4a dot standalone is much faster per tensor (lm_head 201.1, ssm_out 235.8, ffn_down 335.6, ffn_gate 249.6, ffn_up 246.0 GB/s), so the loss is the per-GEMV activation-quant pass, the quant→GEMV dependency and launch overhead — machinery, not math. Gate arithmetic (Q1_0 needs 23.81 ms/token): GEMV at 250 GB/s = 15.2 ms, + 8.4 non-GEMV = 23.6 ms → **42.4 tok/s (gate cleared with non-GEMV untouched)**; with non-GEMV at 2.5 ms → 17.7 ms → **56.5 tok/s**. Next lever: extract the fork's **fused multi-GEMV** (`vec_dot_ptq1_0_q8_1_multi<ncols_dst>`; one launch + one activation quant feeding several matvecs) for our gate+up pair and the four GDN GEMVs sharing one activation buffer. PTQ1_0's dp4a dot remains a separate item (it is infeasible on the current dot). Also recorded: the peer's own standalone-harness non-GEMV figure was wrong (three kernels early-returning) — same shape as my no-decode-dummy error; rule = in-situ for attribution, standalone only for kernel-vs-kernel comparison. |
| P3.2f | ◐ **fused multi-GEMV (cabdd3620, 2026-09-18 15:07)** — shared-activation matvecs now dot against ONE q8_1 quant in ONE launch (groups: four GDN matvecs, q/k/v, gate/up), a re-expression of the fork's `vec_dot_ptq1_0_q8_1_multi` with no fork code linked. **Interleaved A/B** (base/multi alternating in one session, because a sequential A/B was confounded by box drift: after-triad fell to the busy threshold and PQ2_0 read a far larger ms/token, i.e. a false regression) gives a **~1-2% win on all three packs** — real but small, recorded as such. Four-gate window 15:07 valid both sides (triad 210.8 → 213.0): Q1_0 34 / PTQ1_0 17 / **PQ2_0 24 tok/s** = 81% / 63% / 109%, oracle 5/5 and compare_gen 11/11 in-window, so **PQ2_0 stays MET**. **But the 23 → 24 movement is not a gain**: the same binary spans 21.9-23.9 tok/s across three interleaved runs, so both readings are the same measurement. **The quant/launch hypothesis is thereby NOT confirmed** — the in-situ-vs-standalone gap implied quant+launch overhead was the loss, and the fix for exactly that bought only 1-2%; the next step is to measure the quant kernel time and launch count directly instead of investing further in fusion on an assumption. PTQ1_0's faster dot (reading the multi variant's per-block trit path) remains the other open item. |
| P3.2g | ✅ **PTQ1_0 per-block dp4a dot (883535b89, 2026-09-18 15:13) — the round's real gate move: 63% → 85%.** The first port was the fork's *scalar* branch; the *multi* variant's per-block path suits our trit order: four qs bytes widened to 16-bit lanes via `__byte_perm` so the ×3 cannot carry across bytes, each carry = the next base-3 trit, four trits → ONE `sudot4`; HIP lacks the fork's `__vsub4`, emulated by the no-borrow SWAR decrement `((q\|0x80808080)-0x01010101)^0x80808080` for digit bytes 0..2 (corr 0.999996 → same approximate-dot regime as Q1_0, so the oracle/greedy gates carry correctness). Isolated dot: ffn_gate 122.3 → **159.0**, ffn_down **264.5**, output.weight **194.5 GB/s**. End-to-end **17 → 23 tok/s = 85% of the ≥27 gate** (was 63%), corroborated by the isolated rates and a busy-window reading, not by the bracket alone. **Still short**: the dot alone is 37.42 ms for 5.95 GB against the 37.04 ms the gate allows, and end-to-end is 6.46 ms over — so PTQ1_0 needs a faster dot and/or non-GEMV work, but it is no longer infeasible. Window caveat recorded: the after-side lower-size triad swept to 180.9-192.4 GB/s mid-run (busy-tagged), so that run's Q1_0/PQ2_0 readings are drift-depressed and the canonical quiet values stay from the previous window. Next, pre-registered with its own falsification: measure in-situ the activation-quant kernel time and the launch count per token — if the quant is small and launches are not the cost, the quant/launch story is dead and the peer will say so. |
| P3.2h | ✅ **corrected mechanism via rocprofv3 (2026-09-18 15:14)** — two profiles differenced (single profile = 16106 dispatches/token vs 20729 for four), giving **1541 dispatches/token**, of which **1452 are buffer copies (94%)** at 0.37 ms; activation quant `prism_quant_q8_kernel` ~257 dispatches, **0.25 ms = <1% of decode ⟹ THE QUANT STORY IS DEAD** (it cannot explain the in-situ-vs-standalone GEMV gap; peer retired their own framing with data). GEMV total 21.2 ms (dp4a<18> 6.27 + multi4 14.95), **cross-checked to 0.1 ms against the independent skip-GEMV split (21.1)** — two instruments agreeing. Other kernels: gdn_recurrence 3.11, rmsnorm 1.16, rest <0.2 each. Ideal GEMV at standalone rates ~15 ms vs 21.2 observed ⟹ **the dominant term is the GEMV aggregate not reaching its standalone rate inside the loop** (suspected memory-system interaction, explicitly not asserted). The fusion's ~1-2% is now explained: ~256 dispatches removed = 0.26-0.77 ms, matching 0.3-0.6 ms measured. Instrument caveat: dispatch durations overlap in this container, so only counts and difference-of-totals were used. **One part remains an estimate** (lost-to-dispatch inferred from an assumed 1-3 µs cost; the measured terms leave a remainder of that size) — the direct check is wall time minus summed kernel durations from the same profile. Next: measure the in-loop GEMV gap (L2 hypothesis), and consider attacking the 1452 copies rather than any kernel. |
| P3.2i | ✅ **dispatch floor measured; two mechanisms retired (2026-09-18 15:22).** Direct measurement (trivial-kernel loop, no new instrument): 1.826 µs/launch trivial, 1.984 µs realistic ⟹ 1541 dispatches = **2.8-3.1 ms of host time** against 29.5 ms of GPU work per token, so launch cost is **overlapped, not additive** ⟹ the dispatch story is dead as a material cost and the fusion win was slightly less GPU work, not fewer launches. Peer also corrected their own copy figure (418 dispatches/token = 27%, not 1452/94% — a mis-division); the earlier advice to attack the copies is withdrawn, and so is my part in it (I checked the ratio, never the provenance of the numerator). **Q1_0 bounds now firm:** GEMV floor 18.1 ms (3.80 GB at ~210 GB/s triad) vs 21.2 measured = 85% of the floor rate; gate needs 23.81 ms ⟹ non-GEMV must fall ~32% if the GEMV reaches its floor, or ~69% if it does not. **Both halves are required**: GEMV-at-floor with non-GEMV untouched is 37.7 tok/s (short), and deleting the largest non-GEMV kernel outright is 37.8 tok/s (short). Next: `gdn_recurrence` (3.11 ms, the largest) first, then rmsnorm (1.16) and fused-fwht (~1), with the GEMV chase continuing in parallel — neither alone closes Q1_0. |
| P3.2j | ◐ **non-GEMV chase, step 1: gdn_recurrence attempted, neutral (2026-09-18 15:24).** `prism_gdn_recurrence` was 3.11 ms/token (largest non-GEMV) against a **~0.71 ms memory bound** (3.1 MB fp32 state x 48 layers = 148.8 MB/token at ~210 GB/s) — i.e. **23% of its own bound**; suspected latency-bound at 6144 threads. Peer split the kk range across two thread groups (256/block → 12288 threads, LDS combine for mem/c, per-element fp32 order unchanged → bit-identical). Correctness green (oracle 5/5, compare_gen 11/11, GDN gates pass) but **performance NEUTRAL**: Q1_0 30.2 ms/33, PTQ1_0 42.1/24, PQ2_0 42.3/24 vs pre-split 29.6-31.5/42.1-43.5/42.2 → inside the noise band, **win not claimed**, not committed as an optimisation. Next: isolated kernel A/B (split vs pre-split, interleaved in one session), with a pre-registered fallback — if it is neutral too, record **compute-bound rather than latency-bound**. **Shape finding (verified):** ffn_gate-shaped 17408x5120 runs at 0.69x (Q1_0 232.9 vs 335.6) and 0.60x (PTQ1_0 159.0 vs 264.5) of the ffn_down-shaped rate ⟹ the Q1_0 dot chase and the PTQ1_0 dot chase are **one shape problem, not two** (many rows, shallow reduction), which is now the refined GEMV target alongside the non-GEMV budget. |
| P3.3 | ✅ **kernels + full device forward done (2026-09-18).** ① `kernels/prism_gdn.hip` (conv1d+silu+rolling state, gated-delta recurrence + gated RMSNorm, ssmo-out perm) — gate conv 1.0e-07 / recurrence 3.6e-07. ② `kernels/prism_attn.hip` (q/k RMSNorm+partial RoPE, flash gated-GQA decode) — gate 1.3e-06 / 1.2e-07. ③ `kernels/prism_ops.hip` (per-head l2norm, plain RMSNorm, g/β, residual, silu*up, dense f32 GEMV for the F16-tile ssm_alpha/beta, q/gate split). ④ `tests/prism/test_prism_gdn_layer_hip.hip` — whole GDN layer 0 on device, **9.6e-06** vs CPU. ⑤ `tests/prism/prism_forward_hip.hip` — **full 64-layer forward on gfx1151, 5/5 fork-oracle argmax on all three packs**: PTQ1_0 220/314/279/369/11751; Q1_0 and PQ2_0 2614/314/279/369/11751; 4.9 s for the 5-token prompt. Folded vs unfolded is selected from the presence of `__onebp_ext_prism_transform` (folded rotates + permutes; unfolded uses plain weights). **Open: only the P3 tok/s timing run** (P3.4 sizing and P3.5 engine serving landed 2026-09-18). State at 12:40Z: the box was **not** quiet — two peer NPU engines live (`npu_engine_llama` Llama-3.1-8B-NPU2 at 75% CPU from a peer worktree, plus `npu_engine_zr1`) and my triad probe read 139.3-170.0 GB/s vs the 201-219 GB/s idle baseline (-20..-30%) on the LPDDR the GPU path shares; a GPU run then would also perturb their measurement, so I held all device work and requested a ~6 min exclusive window in the mesh mailbox (`~/.dsh/scratch/mesh/LANE-agent-dddf9e-prism-bonsai.txt`). **New requirement encoded in `tests/prism/check_honesty_tags.py`:** a `strixhalo-quiet` tok/s claim is only admissible when the same results file carries a `strixhalo-quiet` triad line >= 200 GB/s — R16 as an executable gate, so no future turn (or compacted agent) can report a gate number taken on a loaded box. |
| P3.4 | ◐ **sizing done + implemented in the device forward (2026-09-18).** KV cache exists only for the 16 full-attn layers (4 kv × 256 hd, k+v): at 32K context fp32 that is 16×32768×4×256×2×4 = **4.29 GB** (2.15 GB fp16). The 48 GDN layers hold constant state — recurrent 48×128×128 f32 ×48 layers = **1.51 GB** plus conv 48×10240×3×4 = 5.9 MB — with **no growth with context**. `tests/prism/prism_forward_hip.hip --max-pos N` sizes the per-layer caches (verified at 4096; 32K is the v1 target, 262K deferred). |
| P3.5 | ✅ **implemented (2026-09-18)** — `include/prism_engine.h` (extracted from the device-forward test, behaviour-preserving under the fork-oracle gate) is the reusable gfx1151 forward: `init(path,max_pos)` loads the .1bp + folded transform and sizes per-layer state, `forward(token)` runs all 64 layers and returns argmax, `reset()` clears state, and init **fails closed** when a folded manifest cannot be honoured. `src/backend_hip_1bp.cpp` now, on a Prism Q1_0/PQ2_0/PTQ1_0 pack or `__onebp_ext_prism_transform`, initializes a `PrismEngine` and routes `generate()`/`reset()` through it (the standard 1BP state is not allocated on that path). **Verified end-to-end (2026-09-18):** a minimal `bench_hip_1bp` build serves all three packs through the backend — it prints `[hip1bp] Prism ML pack (quant N, folded=0/1): PrismEngine ready`, so the folded manifest and both unfolded packs route correctly. Busy-box throughput (relative only): Q1_0 12 / PTQ1_0 10 / PQ2_0 9 tok/s. |

**Targets — set against the measured outside baseline, not a model card.**

Decode at batch 1 is a pure weight-stream problem: `tok/s ≈ BW_lane / model_bytes`. Three lanes on
this box have three different ceilings — a single box-level number mis-sets targets for whichever
lane actually runs (the same lesson as the 8-column wall in R10):

| Lane | Measured BW | How |
|---|---|---|
| CPU (OpenMP, 16 threads, THP, 8 GB arrays) | **103–109 GB/s** (77 at 32 SMT) | peer probe |
| **GPU / gfx1151 (hipMalloc, unified/GTT class)** | **~201 GB/s triad**, ~210 GB/s copy (peer) · **206–219 triad / 202–227 copy** on our own independent `tests/hip_bw_probe.cu`, flat 128 MB→1 GB | two probes, same class |
| NPU weight feed | 2.9 GB/s/shim @8 KB BD, 12.6 GB/s @1×16 MB — **pipeline-stall-limited**, scales with BD size not shim count | peer probe |

> The device figure is now **triple-confirmed**: our `tests/hip_bw_probe.cu` (206–219 GB/s triad), the NPU-lane peer's hipMalloc triad (~201/210), and @agent-2f3c1b's byte-bound analysis ("measured device ceiling ~201–210 GB/s", 2026-09-18) — three independent probes. Their decisive sign check is also worth keeping: halving the expert bytes made their split path 3.9× *slower* (3.808 → 14.995 ms/expert-layer, 85% `p1wait`), i.e. **it was P1 latency-bound, not byte-bound**; a repack that only moves fewer bytes should not be expected to pay. That is the same distinction as our P3 targets: fewer bytes only pay on a *bandwidth*-bound lane.

**Outside baseline (P0.5, Prism's fork on Vulkan/RADV, same box):** Q1_0 **28.8 t/s** tg32 (3.79 GB
→ 109 GB/s, i.e. the CPU-lane number — the reference implementation is nowhere near the device
ceiling); PTQ1_0 **4.4 t/s** (5.95 GB → 26 GB/s, a per-op fallback).

| Model | LM bytes | GPU ceiling @201 GB/s | Plan target (≈80%) | vs fork baseline |
|---|---|---|---|---|
| Bonsai-27B 1-bit (Q1_0 gguf) | 3.80 GB | ~53 tok/s | **≥ 42 tok/s** | **1.5× the fork's 28.8** |
| Bonsai-27B 1-bit (MLX pack) | ~3.9 GB | ~51 | ≥ 40 tok/s | — |
| Ternary-Bonsai-2 27B (PTQ1_0, folded) | 5.95 GB | ~34 | **≥ 27 tok/s** | **6× the fork's 4.4** |
| Ternary-Bonsai 27B (PQ2_0) | 7.17 GB | ~28 | ≥ 22 tok/s | fork has no Vulkan PQ2_0 path |
| Ternary-Bonsai-2 27B (PQ2_0) | 7.21 GB | ~28 | ≥ 22 tok/s | — |
| Ternary-Bonsai-2 27B (MLX pack) | 7.67 GB | ~26 | reference row | — |

> **v0.1's numbers (≥25 ternary / ≥40 1-bit) were wrong because they used 109 GB/s, a CPU-triad
> number, as if it were the device's.** v0.2 then wrongly *reduced* them to ~28/15. The honest
> statement is the table above: the device lane has ~1.85× the CPU lane, so the 1-bit 27B target is
> in the mid-40s and the ternary targets in the low-to-mid 20s. Still above the wall: the only levers
> that raise it are **fewer bytes per token** (trit-dense PTQ1_0 over PQ2_0, TQ2NZ-S40, 2:4 sparsity)
> and **MTP/spec-decode** (Prism claims 1.37× on CUDA; Bonsai-27B ships `mtp_num_hidden_layers=1`,
> and both dspark drafters are already downloaded). Prefill is compute-bound and gets its own number.
> Do **not** compare directly against Prism's M5 Max 47 tok/s — different silicon, say so.
>
> Corroboration that argues *for* headroom rather than against it: the old 27B Q1_0 result
> (`benchmarks/bonsai/RESULTS.md`, 3.6 GB at 30 tok/s) is 108 GB/s — exactly CPU-triad bandwidth. That
> run never saturated the device; it is evidence of ~1.85× left on the table on a GPU lane, not an
> independent measurement of the ceiling.

### P4 — NPU (XDNA2, 32 AIE2 tiles) (parallel, 1–2 weeks)

| # | Task |
|---|---|
| P4.1 | ◐ **scoped (2026-09-18).** `engine/npu/include/ternary_npu_bridge.h` converts our 1BP **32×256-tile** TQ2/TQ1/BST formats (`pack_tq2_to_npu_int8`, `pack_tq1_to_npu_int8`, `pack_bst_to_npu_int8`) to INT8. The Prism packs are a different thing entirely: **flat row-major 128-blocks** with an fp16 per-block scale (Q1_0 nb=18 / PQ2_0 nb=34 / PTQ1_0 nb=28) plus a folded Hadamard manifest. P4.1 therefore needs a **Prism-specific packer** — dequant each 128-block to INT8 under its own scale, keep the GDN grouped layout, and apply the folded basis on the activation side (the P3.1 FWHT already exists). Respect the die's ≤8 columns (R10/R13) and the per-ctx ELF runlist entry point. Correctness/portfolio arm: no tok/s gate. **Finding (2026-09-18):** the packers are only *declared* — `engine/npu/include/ternary_npu_bridge.h` has the three prototypes and `engine/npu/include/npu_kernel.h:279` calls `pack_tq2_to_npu_int8`, but **no implementation TU exists anywhere in the repo** (grep across *.cpp/*.cc/*.h/*.hip). So P4.1 must supply the packer bodies, not merely extend them. |
| P4.2 | GDN on NPU: conv1d + gated-delta recurrence as an AIE kernel (this is genuinely new; compare against WS-01/WS-03 microkernel patterns; the 1×1 GEMVs can stay in INT8) |
| P4.3 | Consume `research/ws03` native TQ2 microkernel when its P0 lands (LUT unpack + ping-pong); metric from WS-03: batch-1 decode ≥ 2× bridge |

**Exit:** bridge path produces token-identical output to P2; DDR bytes/token measured; native path
reported as a delta, not a replacement.

> **NPU expectations — re-based (peer intel, §8.3).** The NPU arm is a *correctness / portfolio*
> arm, not the fast one: another lane measured Qwen3.6-35B-A3B at **~0.5–0.7 tok/s** through the
> native path, and the NPU weight feed is **2.9–12.6 GB/s** (pipeline-stall-limited: each 2-k-chunk
> batch is fully awaited — fewer/bigger BDs is the lever, not more shims). Plan the NPU arm to
> produce *identical tokens* and a measured DDR/BD story; do not gate this lane on NPU tok/s.
> Entry point if we do take it on: the **per-ctx ELF runlist path** (`engine/npu/build/npu_engine_*`)
> — the generator-built xclbins measured ~39× slower. Commits, env vars and the `npu_ab.sh` ROOT trap are
> recorded in §8.1; `gen-layer-elfs` needs `1bdfc33ee` before it can target a non-Qwen3 family. Also hard: **the die reports 8 columns**
> (`xrt-smi` → `Total Columns: 8`) and only `npu2`…`npu2_7col` exist in the device table; a real
> 40-column xclbin exists on the box but `DRM_IOCTL_AMDXDNA_CREATE_HWCTX` refuses it with `EINVAL`.
> **Do not scope around >8 columns.**

### P5 — Vulkan/ZINC (fallback + honest cross-backend number) (3–5 days)

> **Progress (2026-09-18):** `kernels/vulkan/dmmv_prism.comp` implements the three flat-128-block layouts (Q1_0 nb=18 / PQ2_0 nb=34 / PTQ1_0 nb=28, selected by specialization constants) as a wave64, 2-rows-per-workgroup DMMV matching `dmmv_{q1,tq2}_bonsai.comp`; `glslc --target-env=vulkan1.2` compiles it clean (15.3 KB SPIR-V) and it is added to `VK_SHADER_SOURCES`. The folded basis stays on the activation side (P3.1), so weight packing is basis-agnostic. **Open:** a ZINC/Vulkan run for the third honest tok/s column.

Ternary/binary DMMV shaders + Hadamard shader; the GDN shaders already exist. Deliverable is a
third honest tok/s column, not a priority path.

### P6 — Validation, docs, catalog (continuous, closes the lane)

> **Progress (2026-09-18, second increment):** `tests/prism/PRISM_RESULTS.md` is now the lane's results of record — every numeric claim carries the 7-field operator contract `[model | format | backend | box | tokens | prompt | date]` — and `tests/prism/check_honesty_tags.py` **enforces it mechanically**: 21 numeric claims checked, box vocabulary validated, ISO dates, and the R16 rule above. It immediately earned its keep by failing on four untagged numbers restated in prose and on a markdown pipe-escape bug in my own parser; **already violated once in this lane** is exactly the failure mode it prevents. Wired into `run_prism_tests.sh` (host-side). The P3 gate section in that file is deliberately marked **NOT YET MEASURED** with the busy-box evidence attached, so the missing gate cannot be mistaken for a met one. `run_prism_tests.sh` also gained `PRISM_NO_DEVICE=1`, which skips every device gate and build — needed because the shared-box rule serializes device runs, and useful for any box without the device. Host-only suite: **ALL PRISM GATES PASSED**.

> **Progress (2026-09-18):** a token-level PPL harness (`tests/prism/test_prism_ppl.hip`, PrismEngine + the `.htok` tokenizer) scores the WS-00 gate corpus with our own forward: **Q1_0 PPL 5.93, PQ2_0 PPL 6.08, PTQ1_0 PPL 5.69** at 128 tokens (the `.htok` files are byte-identical across the packs, so the folded pack reuses the shared Qwen3 tokenizer). PPL is timing-immune, so the busy box does not affect the numbers; the PTQ1_0 pack is on a different base (Qwen3.8) so its lower PPL is not a like-for-like comparison with the Qwen3.6 packs. Absolute PPL is only comparable on the same corpus; the plan's 1bp-vs-FP16-unpacked-vs-fork comparison still needs those two rows.

- ppl: `research/ws00-baseline/ppl_generic.cpp` on a fixed corpus, **1bp vs FP16-unpacked vs Prism fork**.
- Quality: the 35-question suite in `benchmarks/bonsai/` for comparability with existing RESULTS.md.
- Update `docs/model-families/bitnet-bonsai.md` (new row for the 27B hybrid family), `benchmarks/`,
  `models/catalog/`, `research/TRACKING.md`; every number tagged.
- Honesty gate: a claim ships only with (model, format, backend, box, tokens, prompt, date).

### P7 — Later / optional (explicitly out of v1)

MTP drafter (Bonsai-27B ships `mtp_num_hidden_layers=1`) → spec-decode; vision tower
(`vision_tower.*`, 0.46B) → VLM path; 262K context; AWQ-4bit variants; `Ternary-Bonsai-27B`
(Qwen3.6 base) as a second architecture row once P2/P3 are green.

---

| R17 | **A green verification gate can be consistency-only, not continuity.** A peer lane had a docs commit silently delete **54 tracked xclbin binaries (29.6 MB)** while `PROVENANCE.json`-based CI still passed (`OK: matches PROVENANCE.json (122 builds…)`) — the gate compares the manifest against what is *present*, so a deletion that also stops being listed simply launders itself; `--write-manifest` in the same commit (the remedy the failure text prints) is exactly the laundering step | committed artifacts vanish invisibly; a lane believes it still ships something it doesn't | for **any** committed artifact in this lane: verify with `git ls-tree -r HEAD` (the only proof a tracked file is still tracked) — never `git status`, a commit stat, or a green gate; and diff artifact *keys* on manifest rewrite to refuse an undeclared disappearance (peer filed #2598 for the same guard). Applied here: this lane commits **no** binaries (1BPs stay in `~/models/prism/1bp/`), and the worktree check at 2026-09-18 found 0 deletions |
| R16 | **Timings taken on strixhalo are perturbed by other tenants** — a peer measured 50× slowdowns under concurrent NPU use and had to discard outliers (250.8/382.2/593.8 ms/tok vs a clean 42.9/46.8); correlations were unaffected. Our own figures (45 s/prompt CPU forward, 28.8 t/s Vulkan baseline) are wall-clock on a shared box | mis-set targets, false regressions | quote a box-quiet run for any target number, state the load, and keep correlation/token gates (immune to timing noise) as the primary gates — never gate a PR on tok/s measured while other lanes are active |
| R15 | A loader/binding path that does not know about `__onebp_ext_prism_transform` could serve **folded weights as if they were plain** — plausible-looking garbage, not a crash (exactly the failure class 2f3c1b warned about: a hash/BUILD gate passes on a missing fold) | silently wrong model | **fail closed**: a backend that cannot apply the transform must refuse the model when a transform entry is present; add the check in P1.3 (`onebp_model.cpp` must skip `__onebp_ext_*` tensors by name and expose `has_prism_transform()`) and in P2's gate. Execution gates only (corr + token oracle), never artifact hashes |

| # | Risk | Impact | Mitigation / trigger |
|---|---|---|---|
| ~~R1~~ | ~~our TQ2 codebook ≠ Prism's~~ | — | **retired by measurement**: `PQ2_0` ≡ `TQ2_0_g128` (maxdiff 0, 348 160 elements); keep a scan assertion for the reserved code 3 |
| ~~R2~~ | ~~existing `bonsai_q1_gemv.hip` is 1024-block, MLX is g128~~ | — | **retired**: `Q1_0` (id 41) is 18 B/128 and our `Q1_0_g128` reference matches it; the 1024-block layout is a separate in-tree optimization, not the Prism format |
| R3 | Hadamard semantics (sign order, `inverse` on embed, normalize √B) | plausible-but-wrong outputs | byte-exact Python oracle first (P0.2), then cosine per layer |
| R4 | GDN grouping (`vperm`, 48 v-heads / 16 k-heads) | scrambled attention, may still "run" | port `vperm` + assert shapes; compare per-layer to FP32 ref |
| R5 | ~~Prism fork has no HIP kernels~~ → baseline needs Vulkan, may be slow/broken | — | **retired**: Vulkan/RADV runs both the unfolded Q1_0 and the folded PTQ1_0 pack (P0.5). New note: the *folded* path is a per-op fallback (4.4 t/s ≈ 26 GB/s), so read its Hadamard placement before designing our fused `fwht→GEMV` |
| R6 | `backend_hip_1bp.cpp` transformer-only assumption | blocks integration | P3.5 is an explicit refactor task, not a surprise |
| R7 | 27B at 1.72 bpw still streams 6–7.7 GB per token | tok/s target miss | measure DDR bytes/token first (WS-00 harness); targets are set from the **lane's** BW (`tests/hip_bw_probe.cu`), and GDN's constant state means no KV growth with context |
| R8 | NPU GDN is genuinely new silicon-facing work | schedule | bridge-first; native path is a stretch goal, not a gate |
| R9 | Repo trunk churn (main moves fast; iso-build diverges) | rebase pain | worktree off `origin/main`, rebase weekly, small PRs |
| R10 | **Provenance/hash gates cannot catch a missing fold** — measured: two different instruction streams execute bit-identically (same corr, same tokens); xclbin bytes never match across builds (`TimeStamp`/`UniqueID` embedded) | a wrong model passes CI with a clean hash | gate on **execution**: per-layer corr + token oracle + fold/unfold round-trip, never on artifact hash |
| R11 | GDN QKV truncation is a fixed-but-recurring **defect class**: `qkv_total` (plain q+k+v) ≠ GDN packed width (`2·KD+VD`) and ≠ fused full-attn (`2·NH·HD`); Qwen3.5-4B silently dropped `v[2048:4096]`, boot 163554 vs FLM 16 | silent wrong outputs | **our widths**: GDN `2·16·128 + 48·128 = 10240`, fused full-attn `2·24·256 = 12288`, plain `qkv_total = 24·256 + 2·4·256 = 8192` — converter must assert against the model's own width for every tensor, never a family constant (P0.7) |
| R12 | hd256 KV slots sized from a literal `512` instead of `4·HD` (`engine/npu/src/npu_dims.h:60,110`, `qk_norm_pi`) — **our full-attn head_dim is 256** | KV corruption for exactly this model | P0.7 greps for the literal; P4 must size from `4·HD` |
| R13 | bf16 GEMM generator asserts `(N/128) % cols == 0` with fixed `cols=8`; one column count per context | a shape simply cannot load | our tile counts: 5120/128=**40**, 10240→**80**, 6144→**48**, 12288→**96**, 17408→**136**, k/v 1024→**8** — all divisible by 8 (probe not assumed, P0.7); 40/136 are the tight ones (5 and 17 blocks) |
| R14 | `~/mlir-aie` is an **unrelated-lineage** clone (empty merge-base with origin), carries local NPU2-40 patches, and 5+ generator scripts still carry the #2469 `build_tmp` aiecc / `install_tmp` bindings mispairing that fails with *no xclbin and no error*; `xclbinutil --dump-section` refuses to overwrite an existing output and does so **silently** when stderr is redirected | days lost to toolchain archaeology; a stale partition JSON read as current | treat `~/mlir-aie` as replace-only; use fresh filenames or `--force` for `xclbinutil`; if we need upstream, the peer's `~/mlir-aie-next` @ `f5196d8cb` + Peano 22.0.0 exists but its configure dies on a nanobind pin |

## 7. Open decisions (operator)

1. **Trunk:** base this lane on `origin/main` (done: `feat/prism-bonsai-27b` off 76595a1e6) — or on
   `rebuild/iso-appliance` where the newest lemonade/engine work lives?
2. **Weight source priority:** GGUF (PTQ1_0/PQ2_0, folded, source of truth) vs MLX packs
   (unfolded 1-bit + folded ternary). Plan assumes **GGUF first**, MLX as cross-check.
3. **Vision + MTP now or later?** Plan says text-only v1, P7 for both.
4. **Context target for v1:** 32K (default in the plan) vs full 262K.
5. **Do we claim GitHub issues for this lane?** Recent commits reference issue numbers; if there is
   a tracking issue for Prism 27B, name it and this plan becomes its P0 comment.
6. **NPU scope:** bridge-only acceptable for this lane (native TQ2 stays WS-03's deliverable)?

---

## 8. Peer intel received 2026-09-18 (folded in; kept verbatim where it carries a number)

Two peers in `default` were asked whether this lane duplicates theirs. Neither owns 1BP format,
`backend_hip_1bp.cpp`, Hadamard metadata or the HIP/Vulkan kernels — **no duplicate work**, but
both handed over measured facts that changed the plan (R10–R14, P0.7, re-based P3/P4 targets).

### 8.1 @agent-afbeb7 — NPU lane (engine/npu, generators, attention kernels, family routing)

- **GDN QKV truncation, root-caused and fixed 2026-09-14** —
  `benchmarks/RESULTS-qwen35-4b-gdn-qkv-truncation-2026-09-14.md` (in-tree, read it):
  `qkv_n = cfg.qkv_total` (plain q+k+v) vs the GDN pack width (`2·KD+VD = 8192` for 4B); the NPU
  GEMM wrote 8192 rows, read-back kept 6144, `v[2048:4096]` was dropped, `gdn_attn_step` read zeros;
  first symptom boot **163554** vs FLM **16** → fixed to **50**. Layer-0 intermediates now
  corr 1.0000 through gated-RMSNorm; the residual 50-vs-16 gap is a *second structural bug*
  (30-logit margin, not drift). Fuse note for us: the residual gap is downstream of layer 0.
- hd256 KV slots sized from a literal `512` (`qk_norm_pi` and friends) — hit by any `hd > 128`.
- bf16 GEMM generator asserts `(N/128) % cols == 0`, `cols=8`, one column count per context;
  `H=2560` shapes cannot load (compile at `cols=4`, but the engine initialises one column count).
- NPU reality check: GDN layers already run (`qaogd`) for Qwen3.5/3.6, but 35B MoE measured
  **~0.5–0.7 tok/s** native, and the per-ctx ELF runlist path streams weights at ~59 GB/s.
  Generator-built xclbins are **~39× slower** than the per-ctx ELF runlist path
  (`engine/npu/build/npu_engine_*`). Peer is paused awaiting a scope call, so accel0 is free.
- Also in-tree and worth reading before P2: `engine/npu/src/gdn_host_recurrence.h` (host-f32 GDN
  recurrence matching `tools/gdn_reference.py`, rel RMSE < 1e-3; **geometry hard-coded 32 v-heads /
  16 k-heads / 128×128** — our model is **48 v-heads**, so it needs generalising, not copying),
  `engine/npu/src/gdn_trace_shim.cpp` (LD_PRELOAD live trace).

**Answers received 2026-09-18 05:55 — NPU entry point and the GDN reference [verified in-tree]:**

- Per-ctx ELF runlist path, cite these commits (all confirmed present on `origin/main`):
  `f193319be` (RuntimeLayerEngine + `xrt::runlist` batching, #2063 — core; `npu-infer/src/runtime_layer.cpp`, 45.7 KB,
  plus `engine/npu/src/npu_runlist_bridge.cpp`, 31.1 KB), `cf5529ae6` (single-launch runlist executor wired into
  dense Qwen3 decode in `npu_engine_universal.cpp`, #2080/#2150), `9c0ddc14d` (`benchmarks/gen-layer-elfs.sh`, the
  per-ctx ELF generator), **`1bdfc33ee`** (pass the model family to `gen-layer-elfs` — the one actually required,
  since the default family is Qwen3), plus robustness fixes `91cfc0fd6` (runlist session loaded a Qwen3 layer.xclbin
  for non-Qwen3 models), `4a269430e` (do not seed a model's ELF cache from another model's set), `36292e1a8` (read
  the model's own EOS ids), `963da4d5b` (do not route Gemma4 to `gemma_text`).
- Invocation: `engine/npu/build/npu_engine_<model>` with
  `NPU_RUNLIST=1 NPU_LAYER_ELF_DIR=<dir> NPU_XCLBIN_DIR=$PWD/engine/npu/xclbins`; yardstick ELF dirs
  `~/npu-ab/elfs-4k` (**8705 files**, present) and `~/npu-ab/elfs-8k`.
- **Harness trap**: `~/npu-ab/npu_ab.sh` defaults `ROOT=/home/bcloud/1bit-MONSTER` — a *different engine*.
  Always pass `--engine` explicitly or the measurement is ~86× off. (Confirmed at `npu_ab.sh:26`.)
- **`gdn_host_recurrence.h` is not generalised and has no sibling** — geometry is file-scope constants
  (`NUM_V_HEADS=32`, `NUM_K_HEADS=16`, `HEAD_K/V=128`, `REP=2`, `KEY_DIM=2048`, `VALUE_DIM=4096`, `CONV_DIM=8192`),
  state 524288 f32. Our 48 v-heads / 16 k-heads makes **`REP=3`**, so anything keyed on `REP==2` must be
  parameterised. **Nothing includes the header** (only cited by
  `benchmarks/RESULTS-runlist-decode-35b-moe-2026-09-10.md`; absent from `build_npu.sh`/`CMakeLists.txt`), so it can
  be parameterised freely without perturbing the engine build — and it is a good written spec for P2's CPU floor.

### 8.2 @agent-2f3c1b — xclbin/toolchain lane (no overlap)

- **Hash/provenance gates are the wrong gate** — measured bit-identical execution from different
  instruction streams (same `corr=0.892429`, same 8 token ids) and non-reproducible xclbin bytes.
- **8 columns is the wall**: `xrt-smi` → `Total Columns: 8`; device table only `npu2`…`npu2_7col`;
  the real 40-col xclbin on the box is refused with `EINVAL` at `AMDXDNA_CREATE_HWCTX`.
- **gfx1151 bandwidth, corrected**: the **~109 GB/s figure is the CPU lane** (OpenMP, 16 threads,
  THP, 8 GB arrays); 32 SMT threads *lose* (77 GB/s). The **device lane is ~201 GB/s triad /
  ~210 GB/s copy** (hipMalloc, flat 128 MB→1 GB), independently reproduced here with
  `tests/hip_bw_probe.cu` at **206–219 / 202–227 GB/s** — ~1.85× the CPU lane. The target table in
  §P3 was rebuilt on the device number (and our own v0.2 attempt to "fix" it with the CPU number
  was itself the error). NPU weight feed: **2.9 GB/s/shim** at the 8 KB-BD shape vs **12.6 GB/s** at
  1×16 MB, pipeline-stall-limited → bigger/fewer BDs.
- **Corroboration direction corrected**: the old 3.6 GB @ 30 tok/s run (=108 GB/s) matches *CPU*
  triad bandwidth, so it is evidence of unused device headroom, not of the ceiling.
- Toolchain traps: `~/mlir-aie` unrelated lineage (empty merge-base, replace-only), #2469
  `build_tmp`/`install_tmp` mispairing in 5+ generators (silent: no xclbin, no error),
  `xclbinutil --dump-section` silently refuses to overwrite. Fresh upstream available at
  `~/mlir-aie-next` @ `f5196d8cb` + Peano 22.0.0 (configure dies on a nanobind pin).

### 8.3 What changed in this plan because of them

1. P3 targets re-based on **the device lane's** BW (two independent probes: ~201–219 GB/s triad),
   not the CPU's 109 GB/s. v0.1 (109-based) and v0.2 (also 109-based, lowered) are both superseded.
2. P4 re-scoped: NPU = correctness/portfolio arm, ≤8 columns, per-ctx ELF runlist entry point,
   no tok/s gate.
3. New P0.7 defect-class audit (widths, hd256, cols=8) before conversion.
4. R10–R14 added; execution gates (corr + token oracle + fold round-trip) replace artifact-hash gates.
5. `gdn_host_recurrence.h` cannot be reused as-is for our 48-v-head geometry.
6. `tests/hip_bw_probe.cu` added so the BW number every target depends on is reproducible in-tree,
   per lane, rather than quoted from a peer message.

---

## 9. Coordination

- Worktree: `/home/bcloud/1bit-MONSTER-dddf9e`, branch `feat/prism-bonsai-27b`, base `origin/main`
  (per `AGENTS-WORKTREES.md`: one agent = one worktree; never commit in the shared checkout).
- `AGENTS.md`/GitNexus rules apply to every engine symbol edit: `impact()` first, `detect_changes()`
  before commit. New files (converters, kernels, tests) are exempt from symbol impact analysis but
  still need `detect_changes()`.
- Mesh: this lane announced in `default`; files reserved with `mesh_reserve` before editing.
- No background downloads or daemons left running when the lane stops.

**Open ask to peers (outstanding):** @agent-afbeb7 — the exact commits for the **per-ctx ELF runlist
path** (`engine/npu/build/npu_engine_*`) and whether `gdn_host_recurrence.h` has a generalised
(nv≠32) sibling. Requested 2026-09-18.

## 10. First 10 concrete steps

1. `hf download` the four artifacts into `~/models/prism/` + hash (P0.1).
2. Python oracle from `codec.py`/`runtime.py`, byte-exact test (P0.2).
3. GGUF metadata dump + diff vs `config.json` (P0.3).
4. Tensor-map doc from `runtime.py::mapping` + mlx index (P0.4).
5. Build Prism fork with Vulkan, run GGUF on strixhalo → baseline + golden tokens (P0.5).
6. Parity probe of our Q1_0/TQ2 kernels' group/codebook (P0.6).
7. 1BP v5 header + transform block (P1.1) with scan/verify tests.
8. `prism_gguf_to_1bp` on PTQ1_0 → verify dequant vs oracle (P1.3/P1.5).
9. CPU low-bit GDN forward + per-layer cosine gate vs FP32 ref (P2).
10. First HIP kernel pair (B1/TQ2 g128) + fwht B=1024 with signs (P3.1/P3.2).

---

### Appendix A — toolchain paths on strixhalo [verified]

```
/opt/rocm-therock/{bin/hipcc,lib/llvm/bin/amdclang++,bin/hipify-clang}   ROCm/TheRock 7.x
~/mlir-aie                                             AIE2 (XDNA) flow
/opt/xilinx + xrt-smi                                  XRT / NPU runtime
~/therock100                                           TheRock build tree
/usr/bin/{cmake,ninja,python3}                         build
~/.local/bin/{hf,huggingface-cli}                      HF client 1.29
python3: torch 2.13+cpu, transformers 5.16.1, gguf-py 0.19, safetensors 0.8, numpy 2.5.2
free: 382 GB on /   (uploads: ~21 GB of prism artifacts + ~55 GB if FP16-unpacked oracle is needed)
```

---

### Appendix B — the one-line contract that must never drift

```
Prism ternary/binary weight  ==  1BP(TQ2|B1)  with   value = code * scale + bias
Hadamard: W' = W · H(B),  H = normalized Sylvester–Walsh (1/√B), B = 1024,
          BOTH the folded matmuls AND the embedding are stored rotated;
          runtime: fwht(x) before folded matmuls, inverse fwht(embedding rows).
```
