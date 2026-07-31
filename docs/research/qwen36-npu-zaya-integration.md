# Qwen3.6-35B-A3B on NPU — zaya_server integration notes

_2026-07-31. Strix Halo (Ryzen AI MAX+ 395), FLM v0.9.46, official Q4_K_S weights
from `FastFlowLM/Qwen3.6-35B-A3B-NPU2` (Hugging Face)._

## Status

**Works end-to-end:** `zaya_server --model Qwen3.6-35B-A3B-Q4_K_M.gguf` routes to the
NPU via the FLM backend and returns clean generations ("Silver orb ascends, / Casting
light upon the night, / Silent watcher high.").

## Architecture (final)

The NPU backend (`tests/backends/backend_npu.cpp`) spawns **`flm run <tag>` per
request with FILE stdio** (no pipes):

```
zaya_server → NpuFlmBackend
  → write prompt + "\n/bye\n" to /tmp/flm_in_<pid>.txt
  → fork+exec: flm run qwen3.6-moe:35b-a3b  (stdin/out/err = regular files)
  → poll /tmp/flm_out_<pid>.txt until transcript ends ">>> "
  → parse text after "Model RAW Output:", strip "/bye" echo + ANSI codes
  → kill child
```

Why files instead of pipes or HTTP — three FLM v0.9.46 bugs discovered:

1. **fork+exec children with PIPE stdio hang on the NPU prefill kernel.**
   Reproduced with a minimal C reproducer: child loads the model, prints
   "[FLM] Prefill chunk 1/1", then blocks forever in `drm_syncobj_array_wait`.
   Bash-spawned pipes work; dup2'd pipes hang; FILE stdio works ("2+2 equals 4"
   verified). Root cause: amdxdna/XRT interaction with pipe fds in forked
   children — not yet understood; workaround is file stdio.
2. **`flm serve` mode degenerates into repeated-token loops ("plplpl").**
   Same model/prompt via CLI answers correctly; via the HTTP server the model
   emits "plplpl"/"ypeype" at ~63 t/s (5× NPU speed — bogus). Reproduced
   standalone; not fixed by `--preemption 0/1`. Likely the "Multi-Backend KV
   Cache" path in serve mode. Workaround: per-request CLI spawn.
3. **the-rock HIP libs on LD_LIBRARY_PATH corrupt the FLM NPU runtime.**
   With `/opt/rocm-therock/...` first in the env, the CLI also degenerates.
   The backend sanitizes the child env to keep only FLM lib dirs.

## Other fixes in this session

- `flm_tag_for_model` (src/backend_npu_flm.cpp) + tests/backends/backend_npu.cpp:
  Qwen3.6-35B-A3B (GGUF arch `qwen35moe`, 256 experts) now maps to
  `qwen3.6-moe:35b-a3b` instead of the dense fallback `qwen3:4b`.
- `tests/zaya_server.cpp` GGUF detection reads `expert_count` → `num_experts`.
- Token shift scheme made collision-free: printable ASCII → +100 (132-226),
  control/raw bytes → +300 (300-555). The old +200 scheme collided
  ('e'-'~' → 201-226 overlapped control chars 1-26).
- `npu_flm_set_prompt_text()` extern (NOT a virtual): adding a virtual to
  `InferenceBackend` produced garbage vtable slots in the hipcc-compiled
  adapter TUs (clang RTTI mismatch) → startup segfaults.
- NPU FLM spawn waits for the HTTP port / transcript with the instance
  timeout (300 s) — the 23 GB model load takes 60-90 s cold.
- System HIP (`/usr/lib/.../libamdhip64.so.7.1.52801`, Ollama bundle) is
  broken after a ROCm environment update; zaya_server must run with
  `/opt/rocm-therock/.../_rocm_sdk_devel/lib` first on LD_LIBRARY_PATH.

## Performance

- Per-request model load ~11 s (warm page cache) + prefill + decode.
- Measured decode on this stack: ~12 tok/s (see
  `benchmarks/RESULTS-qwen3.6-35b-a3b-npu-flm-2026-07-30.md` for the full
  `flm bench` sweep: 11.66 @1k → 8.82 @32k tok/s).
- The 11 s/request spawn overhead is the price of FLM's CLI limitations; a
  persistent-session protocol would need the pipe hang or serve degeneration
  fixed upstream (both reproducible with stock FLM v0.9.46).

## Still open

- Native `npu_engine_universal` MoE path (no expert routing in the engine yet).
- The FLM fork-pipe NPU hang and serve-mode degeneration (upstream bugs;
  minimal reproducers in this session's evidence).

## Q4NX byte-format state (for the native engine)

Discovered via ground-truth correlation (GGUF dequant vs Q4NX bytes) + the
`get_quantization_byte_size(m, dtype)` oracle (dlopen'd from
libqwen3_6_moe_npu.so):

- **BF16 tensors (e.g. moe_router) use a stride-8 row interleave**: the
  logical matrix W[i][j] (i=in 0..2047, j=expert 0..255) is stored as
  `flat[(i%8)*65536 + j*256 + i/8]` — rows split into 8 blocks by
  `i mod 8`. Cracked and VALIDATED: router corr=1.000000 vs the GGUF F32
  router, top-8 expert selection identical (152 131 101 115 42 218 127 173).
  Tool: `tools/moe_router_test.cpp`. (Earlier "transposed" note was a
  flatten-ordering illusion — the actual layout is the stride-8 interleave.)
- **Expert FFN tensors (5120-byte rows, dtype 2/4, INT4)**: structure =
  [512 B][512 B][4096 B packed] = [scales][mins][row-major nibbles], and the
  1BP converter (`tools/gguf_to_onebp.cpp`, Q4NX branch) writes exactly this
  layout: `value = q*scale + min`, scale=(max-min)/15, `qd[(rr*256+c)/2]` low
  nibble = even col. The dequant value DISTRIBUTION matches the GGUF ground
  truth (rms 0.0108 vs 0.0069 for the TQ2-quantized GGUF twin), but the value
  POSITIONS still do not correlate with the GGUF under ~40 tested orderings
  (row/col/group-major, stride-8 sub-blocks, expert permutations, scale/min
  swaps, formula variants). Suspected cause: experts are interleaved in
  8-expert NPU-dispatch blocks (FLM_SECRETS `parallel_size 16`,
  `num_groups_per_row_parallel 2`). Next lead: capture the dequant_mm.xclbin
  input/output on the NPU (gen_dequant_mm sequence) as the ground-truth
  oracle, or brute-force expert-block interleavings (256! too big — but the
  block structure constrains it).
- **Attention projections (8704-byte rows) = Q8_0** (dtype 1, 1.0625 B/val =
  34 B per 32 = INT8 + BF16 scale). Scales confirmed at bytes [0:512] (values
  ~1.4e-4..6e-4). The INT8 value→position permutation is NOT yet cracked
  (15+ layout hypotheses tested, all corr≈0; the dequant implementation is
  binary-only in `dequant.lib`). Next lead: reverse the dequant.lib q80
  routine, or use the dequant_mm.xclbin on the NPU as the dequant oracle.
- The dtype oracle method (`/tmp/dtype_probe.c`): probe
  `get_quantization_byte_size(8192, dtype)` — dtype 1 → 8704 B, dtype 2/4 →
  5120 B, dtype 0 → 4608 B (0.5625 = INT4+FP16?), dtype 3/5 → 9216 B.

## Native engine MoE roadmap (npu_engine_universal)

1. ModelConfig: add MoE fields (N_EXPERTS, TOP_K, shared expert, ssm dims).
2. CPU router: moe_router.weight is plain BF16 [2048, 256] — readable now.
3. NPU expert FFN: the 5120-INT4 expert tensors dequantize with the existing
   verified dequantizer; dispatch per-expert G/U/D via dequant_mm.xclbin
   (FLM's gen_dequant_mm sequence, MIT).
4. Attention: 30/40 layers are GatedDeltaNet (linear attention — CPU state
   update, port from llama.cpp ggml gated_ln_net); 10/40 gated full attention
   (attn.xclbin or GPU). This is the "NPU FFN ∥ GPU attention" pattern the
   repo already pipelines (PR #1231).
5. Q8_0 attention projections: needed for on-NPU attention; blocked on the
   value permutation above.

## Decisive next step: NPU-as-oracle harness (spec)

The layout lives in the AIE kernel inside `dequant_mm.xclbin` (the
"torch2aie chunk format" per `dequant_q4nx.cpp`'s own header). Every host-side
permutation search has failed (~60 variants). The definitive oracle: run the
kernel on the NPU and read back the dequantized output.

Harness spec (all pieces are MIT and in-tree):

1. **Kernel signature**: read `dequant_mm.xclbin`'s embedded metadata (xclbin
   is a zip; kernel name + args are in the XML/JSON metadata) — the engine's
   `npu_engine_universal.cpp` HybridFlmCtx shows the pattern for `mm.xclbin`
   (args: bA, bW, bC). `dequant_mm` likely takes (data, scales, out) or a
   single weight BO + dequantized out BO.
2. **Instruction sequence**: dlopen `libqwen3_6_moe_npu.so` and call
   `qwen3_6_moe_npu_sequence::gen_dequant_mm(npu_sequence*, M, K, N, off, m,
   flm_dtype_t)` with dtype=1 (Q8_0) and dtype=2 (INT4) — `npu_sequence` is in
   `third_party/FastFlowLM/src/include/npu_utils/npu_instr_utils.hpp`.
   `get_quantization_byte_size` (already probed) confirms dtypes.
3. **Submit**: XRT (`xrt::device(0)`, register the xclbin, one BO for the
   raw 8704-B rows + one for the output) — same pattern as HybridFlmCtx.
4. **Compare**: the read-back dequantized values vs the GGUF ground truth →
   yields the exact value↔position mapping in one shot.
5. Fallback if the sequence call is finicky: `_move_weights_q80` (same .so)
   generates the weight-load path directly.

Expected result: the layout mapping (likely a per-8/16-element DMA chunk
interleave for the AIE array), unblocking BOTH the Q8_0 attention projections
and the INT4 expert tensors, and thereby the native MoE engine's weight
loader.

### Harness status (2026-07-31, tools/dequant_oracle.cpp)

BUILT AND RUNNING — kernel executes, output verified:

- `Dequant::generate_dequant_q80_packed_in_q4nx_seq(seq, D_in, D_out, w_off,
  mode)` (libdequant.so, MIT) generates the instruction stream; D_in must be
  a multiple of the kernel's `desired_k_dequant` (D_in=2048 ✓).
- The dequant.xclbin kernel (MLIR_AIE, 5 BOs + instr) executes with
  opcode=3; the instruction stream's DDR_PATCH commands (0x81, word8=arg_idx,
  word10=offset) reveal the BO usage: **arg0 = output region (16 patches,
  512 KB stride, 8 MB total), arg1 = input region (17 patches, 320 KB
  stride)**. The input must be in bo1 (leaving it empty = zeros out).
- With a zeroed bo0: chunks 0-7 are fully written (f32, ~131k values each),
  chunks 8-15 partial. The output values are real dequantized weights
  (e.g. 4.48e-3, 2.8e-4, 3.09e-3) but written in an interleaved/strided
  pattern over the BO (every 3rd f32 slot holds a sane value in the current
  config) — the exact write pattern needs the WRITE_DMA BD stride decode
  from the instruction stream (npu_dma_block_cmd fields in
  npu_cmd_write_dma.hpp), which is the remaining step.
- The dtype oracle (get_quantization_byte_size) remains the format key:
  dtype 1 = Q8_0 (8704-B rows), dtype 2/4 = INT4 (5120-B rows).
