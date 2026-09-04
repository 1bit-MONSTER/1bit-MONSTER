# W2/W3 — Zaya/type42/1bit-layer port manifest (onto AMD hrx-graph-develop-v2 @ 6319038)

**Date:** 2026-09-03 · **Source (A):** `~/hrx-ws/hrx-v2-src` (= bong fork `hrx-v2` @ 8df3330)
**Target (C):** `~/hrx-ws/amd-hrx-graph` @ 6319038 (round-28 W1 base, loomc@c9855b4)
**Branch to create:** `round-28-zaya-port` in C.

## Drift probe (measured)
A's `src/models/zaya.cpp` syntax-checked against C's headers: **27 errors, 100% additive**
(missing `llama_model.zaya_input_hs_scale/bias`, `llama_layer.cca_val_proj1/2`, conv/state
fields). No structural/API-rewrite errors → port = enumerated additions + registrations.

## Port set (file-level, in dependency order)

### T1 — ggml type42 (Q4NX) registration (prereq for load)
| File (C) | Change | Source (A) |
|---|---|---|
| `ggml/include/ggml.h` | type enum entry + traits decl | port type42 traits (33 refs in A ggml.c) |
| `ggml/src/ggml.c` | `ggml_type_name/size/blck/row_size`, traits table (blck 8192, size 5120), `ggml_is_quantized` | A ggml.c q4nx hunks |
| `ggml/src/ggml-quants.c/.h` | `block_q4nx` (5120B tile), `dequantize_row_q4nx`, `quantize_row_q4nx_ref`, vec_dot if CPU path needed | A quants (validated earlier, bit-exact vs C ref) |

### T2 — llama layer: zaya model + arch registration
| File (C) | Change |
|---|---|
| `src/models/zaya.cpp` | ADD (copy A, ~fixes for member renames) |
| `src/llama-arch.h` | `LLM_ARCH_ZAYA` enum + zaya tensor names (`zaya_router_mlp2/4`, `zaya_router_eda`, `cca_val_proj1/2`, `ssm_convNd`, `cca_conv_grp`, … from A llama-arch.cpp:16 refs) |
| `src/llama-arch.cpp` | arch→hparams mapping (zaya case: n_embd_s = 2*n_qk + n_embd, conv_kernel 2, expert top-1, 17-slot router) |
| `src/llama-model.h/.cpp` | `llama_model`: `zaya_input_hs_scale/bias`; `llama_layer`: conv/cca/router fields; zaya load case (tensor fetch by name, type42 accepted on HRX buffer) |
| `src/llama-graph.cpp` (if shared) | nothing — zaya.cpp is self-contained graph (drift probe OK) |

### T3 — backend: type42 on HRX + 1bit decode kernels (THE large chunk)
| File (C) | Change | Reality |
|---|---|---|
| `ggml/src/ggml-hrx/*` | type42 buffer/copy/mul_mat acceptance | A's ggml-hrx2 glue → C's ggml-hrx (dir renamed/refactored by AMD — port A's q4nx handling hunks; expect the deepest conflicts) |
| `ggml/src/ggml-hrx/kernel-corpus/...` | ADD 1bit decode kernels (r16wb4c/r16wb2/r16x8t q4nx fused decode family + zaya kernels) + routes | A's `ggml-hrx2/catalog/routes` (50 files) + kernels (80 files). C's corpus = compile-time .loom build → port A's .loom sources + route JSON, re-gen corpus. Largest unknown: kernel/route source compatibility with C's loom/merge-mode build |

### T4 — verify gates (repeat Round-28 C-series)
C2': zaya-q4nx.gguf LOAD + single-seq decode clean (no NaN) on C.
C4': zaya multi-seq npl 1-8 no SEGV (recurrent-state copy — verify against C's recurrent framework, which may differ from A's).
Perf: ≥ stale single-seq (tg ≥16.8 t/s @ no DISABLE flags).
Then llama-server continuous batching for Zaya.

## Risk / effort
- T1+T2 = additive/mechanical, 1-3 build-fix cycles.
- T3 = 60-80% of the effort; route/corpus source drift + backend glue conflicts; silent-wrongness risk on decode kernels → gate every step with the dprobe numeric check (NaN=0 + argmax sanity).
- Loader must accept type42 only on HRX buffers (CPU-only load stays impossible by design).
- Keep C's W1 base + stale A both untouched until T4 green (parallel trees).

## State
W1 done (C0/C2/C4 non-1bit green). T1-T4 = W3 execution; recommended as dedicated round cycles with per-T numeric gates, or a fresh agent session starting at T1 with this manifest.
