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

- **All Q4NX tensors are stored transposed** vs llama.cpp GGUF (verified:
  router BF16 corr=1.0000 on transpose).
- **Expert FFN tensors (5120-byte rows, dtype 2/4, INT4)**: the existing
  `dequant_q4nx.cpp` dequantizes them correctly — gate/up/down_exps verified
  plausible, 0 NaNs. Layout: [512 B BF16 scales 8g×32r][512 B zeros][4096 B
  packed INT4].
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
