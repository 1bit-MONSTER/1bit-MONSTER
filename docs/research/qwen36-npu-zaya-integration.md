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
