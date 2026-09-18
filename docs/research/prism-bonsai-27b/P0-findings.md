# P0 findings — Prism ML Bonsai 27B (strixhalo, 2026-09-18)

Lane `feat/prism-bonsai-27b`, worktree `/home/bcloud/1bit-MONSTER-dddf9e`.
Companion to `docs/plans/prism-bonsai-27b-custom-build.md`. Every number below was
measured on strixhalo today, not inherited from a model card.

Tools written for this phase (both committed in this worktree):

| File | Purpose |
|---|---|
| `docs/research/prism-bonsai-27b/gguf_header.py` | dependency-free GGUF header/metadata reader + `--verify` structural integrity. **Needed because stock gguf-py 0.19 cannot open these files**: private type ids 142/143 raise `ValueError: np.uint32(143) is not a valid GGMLQuantizationType` |
| `tests/prism/oracle_prism_codec.py` | independent decoders for Q1_0 / PQ2_0 / PTQ1_0, cross-checked against Prism's own `runtime/codec.py` on real tensors, plus the Hadamard manifest checks |
| `tests/prism/roundtrip_prism_codec.py` | synthetic round-trip (Q1_0, PQ2_0) — no model file required |
| `tests/prism/vendor_prism_codec.py` | vendored copy of Prism's `runtime/codec.py` (Apache-2.0, unmodified) so the oracle does not depend on a downloaded pack |

## 1. Artifacts acquired (P0.1) — 25.5 GB in `~/models/prism/`

| File | Bytes | HF-published sha256 | Local sha256 | `--verify` |
|---|---|---|---|---|
| `ternary2-gguf/…PTQ1_0.gguf` | 5 946 648 928 | `53107f53…33ee3` | **match** | PASS |
| `ternary2-gguf/…PQ2_0.gguf` | 7 206 168 928 | n/a (API omits `lfs.oid`) | `3907dc16…62ec1` | PASS |
| `ternary-gguf/…PQ2_0.gguf` | 7 165 121 600 | n/a | `e4781999…b5f6` | PASS |
| `onebit-gguf/Bonsai-27B-Q1_0.gguf` | 3 803 452 480 | `17ef842e…9aa0` | **match** | PASS |
| `onebit-gguf/…dspark-Q4_1.gguf` | 1 787 468 768 | `25e73f9f…fb1b` | **match** | PASS |
| `ternary-gguf/…dspark-Q4_1.gguf` | 1 946 393 568 | n/a | `c4810091…7da9` | PASS |

`--verify` = GGUF parses, every tensor's `offset + size` fits inside the file, no unknown
type ids. For all five model files the highest tensor end equals the file size exactly
(0 B tail). Where the Hub publishes an `lfs.oid`, the local sha256 matches it.

Not downloaded (deliberate): F16 GGUFs (53.8 GB each — only if an fp16 oracle is needed),
vision mmproj packs (0.63/0.93 GB, text-only v1), the two MLX safetensors packs
(5.16 + 8.61 GB, cross-check only), `Ternary-Bonsai-27B-Q2_0/Q2_g64`.

## 2. Container + metadata facts (P0.3)

All files: **gguf v3, alignment 32, 851 tensors** for the full models (79 for a dspark
drafter).

| File | kv | `prism.hadamard.*` | Folded | Weight types |
|---|---|---|---|---|
| Ternary-Bonsai-**2**-27B (Qwen3.8) PTQ1_0 | 49 | **present** | **yes** (401 tensors) | PTQ1_0 ×402, BF16 ×96, F32 ×353 |
| Ternary-Bonsai-**2**-27B (Qwen3.8) PQ2_0 | 49 | **present** | **yes** | PQ2_0 ×402, BF16 ×96, F32 ×353 |
| Ternary-Bonsai-27B (Qwen3.6) PQ2_0 | 37 | **absent** | **no** | PQ2_0 ×402 (+ BF16/F32) |
| Bonsai-27B (Qwen3.6) Q1_0 | 37 | **absent** | **no** | Q1_0 ×498, F32 ×353 |
| dspark-Q4_1 (drafter) | 26 | absent | no | Q4_1, Q2_0 (type 42), F32 |

**This is the single most useful planning fact of P0: only the Bonsai *2* (Qwen3.8) family
is Hadamard-folded.** The Qwen3.6-based 1-bit and ternary 27B GGUFs are ordinary
unfolded Q1_0 / PQ2_0 — no rotation path needed for them at all.

Architecture metadata (`qwen35.*`, identical for both 27B bases except the folded flag):

```
embedding_length 5120  block_count 64  feed_forward_length 17408
attention.head_count 24  head_count_kv 4  key_length 256  value_length 256
full_attention_interval 4        rope.dimension_count 64  sections [11,11,10,0]  freq_base 1e7
ssm.conv_kernel 4  ssm.group_count 16  ssm.inner_size 6144  ssm.state_size 128  ssm.time_step_rank 48
attention.layer_norm_rms_epsilon 1e-6      context_length 262144
```

Derived GDN geometry: `nv = time_step_rank = 48`, `nk = group_count = 16`,
`hd = inner_size / nv = 128`, `hk = state_size = 128`, `CONV_DIM = 2·nk·hk + nv·hd = 10240`,
`REP = nv/nk = 3`. Confirms the peer handoff: `engine/npu/src/gdn_host_recurrence.h`
(hard-coded 32 v-heads, `REP = 2`) cannot be reused as-is.

Hadamard manifest (Bonsai 2 only), all checks pass on both files:

```
version 1   transform normalized-sylvester-walsh-hadamard   axis input-last-dimension
block_size 1024   sign_mode explicit
sign_widths [5120, 6144, 17408]   sign_values 28672 (+/-1)   = sum(sign_widths)
weight_names 401 folded tensors   inverse_weight_names ["token_embd.weight"]
gdn_v_grouped true
```

Every folded tensor exists in the file, and **every folded matmul's input width is one of
the three widths that has a sign vector** — so the transform is keyed by input width, not
per tensor: `5120` (hidden — in_proj_qkv/in_proj_z/mlp gate/up), `6144` (=nv·hd — GDN
out_proj), `17408` (mlp down_proj), plus `lm_head` (5120).

## 3. Codec results (P0.2 + P0.6) — all PASS

Sampled: 12 tensors per file, first/last 2 rows each = 348 160 elements for PTQ1_0 and
PQ2_0, 299 008 for Q1_0.

| Format | Bytes/128 | Scale position | Value map | vs Prism `codec.py` |
|---|---|---|---|---|
| **PTQ1_0** (143) | 28 = qs[24]+qh[2]+fp16 d | **last** (26:28) | trit ∈ {0,1,2} → {-d, 0, +d} | **lossless, maxdiff 0** |
| **PQ2_0** (142) | 34 = fp16 d + qs[32] | first (0:2) | `code*s − s` → {-d,0,+d,+2d} | **lossless, maxdiff 0** |
| **Q1_0** (41) | 18 = fp16 d + qs[16] | first (0:2) | bit ? +d : −d | n/a (no codec.py path) |

Consequences, and they change the plan:

1. **`PQ2_0` is byte-identical to our existing `TQ2_0_g128`** (`tests/q1_tq2_vk_ref.h`):
   same 34-byte block, d-first, 2-bit codes LSB-first 16 per uint32, and our `code − 1`
   map reproduces Prism's `code*s − s` exactly for codes 0..2. **344 000 sampled elements
   show zero code-3 slots**, i.e. the 4th code is genuinely unused as the model card says
   (ours maps 3 → 0, Prism → +2d; a scan assertion must keep it that way).
   → the ternary 27B path needs **no new weight decoder**, only GDN.
2. **`Q1_0` is identical to our `Q1_0_g128`** (18-byte block, bit *l* of byte *il* → element
   `8·il+l`, `+d` when set). Bonsai-27B 1-bit needs **no new weight format either**.
3. **`PTQ1_0` is the only genuinely new packing** for us: base-3 trits, 128-group, fp16
   scale *trailing*, and an **element order that is explicitly not positional** (16-byte
   chunk: byte *j* carries elements *t*·16+*j*; then 8-byte chunk carrying 80+*t*·8+(*j*−16);
   then 2 bytes at 4 trits each carrying 120+*t*·2+*h*). Decoding it via our TQ1-style
   sequential packing would silently misalign every weight — the oracle now pins the order.
4. **Stock gguf-py cannot read these files** (type ids 142/143), so the converter must not
   depend on it; `gguf_header.py` (and later `tools/gguf_reader.cpp`) is the reader.

## 4. Tooling gotchas hit (recorded so they are not rediscovered)

- `hf download <repo> --include "*.json" …` did **not** fetch root-level JSON for the MLX
  pack; passing explicit filenames worked. Always verify the file list afterwards.
- In a non-interactive `ssh 'cmd'`, `cd A && B &` backgrounds the *whole* `A && B` list, so
  later commands run in `$HOME` and "lose" files/logs that are actually fine. Use absolute
  paths.
- `hf`/`huggingface-cli` live in `~/.local/bin`, which is not on the non-login PATH.
- The Hub API omits `lfs.oid` for some files (Xet-backed), so hash verification is only
  possible where it is published; `--verify` covers the rest structurally.

## 5. What P0 changed in the plan

| Plan item | Before | After P0 |
|---|---|---|
| P0.2 oracle | todo | **done** (byte-exact vs Prism's transcoder) |
| P0.3 metadata | todo | **done** (tables above; folded/unfolded per family) |
| P0.6 codebook parity | the biggest unknown (R1/R2) | **done**: PQ2_0 ≡ TQ2_0_g128, Q1_0 ≡ Q1_0_g128, code 3 unused |
| P1 container work | new B1 type + TQ2 bias for Prism parity | **smaller**: PQ2_0/Q1_0 need no new types; PTQ1_0 decode + Hadamard metadata are the real work |
| P3 HIP | new B1 + TQ2 g128 kernels | **existing** `b1`-class (`kernels/bonsai_q1_gemv.hip`) and `ternary_gemv.hip` cover the two unfolded models; new kernels only for PTQ1_0 and the fwht path |
| Risk R1/R2 | codebook/block-size mismatch | **retired** |
| New | — | Hadamard path is needed only by Bonsai 2; GDN geometry nv=48/REP=3 fixed; TQ1-style sequential decode is a trap for PTQ1_0 |

## 6. Still open in P0

- P0.4: MLX-pack tensor map (mlx-vlm namespace → our loader names) — the packs are not
  downloaded yet; `runtime.py::mapping` is already read and is the source.
- P0.5: outside baseline. Prism's fork has **no HIP kernels** (`mmq-instance-ptq1_0.cu`,
  `-pq2_0.cu`, `-q1_0.cu` are CUDA templates; Vulkan has `ptq1_0.glsl`/`dequant_q1_0.comp`),
  so the baseline must be built with `-DGGML_VULKAN=ON` on strixhalo. Not started.
- P0.7: defect-class audit (widths/hd256/cols=8) — widths verified by arithmetic against
  this file's config; the `npu_dims.h` literal-512 and generator `cols=8` greps are todo.
