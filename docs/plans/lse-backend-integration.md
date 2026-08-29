# LSE (Lemon Seed Engine) backend integration — plan

**Status:** M0–M2 IMPLEMENTED (2026-08-29) · **Owner:** 1bit-MONSTER · **Date:** 2026-08-29
**Upstream:** LSE v0.2.1+ (`lse-workspace/LSE`), `lse-server` OpenAI-API binary
**Gate:** written license from Geramy (email sent 2026-08-29); development can
proceed (non-commercial use is licensed), merge/release is gated.

## 1. Goal

Add `backend_lse` to the `BackendManager` router: a **text-level, subprocess
backend** that runs `lse-server` on localhost and serves the **MLX-format
lane** (Qwen3.5/3.6/3.8 family + lemonseed) on AMD GPU — the only backend in
our tree that can read MLX group-affine checkpoints. Short-context decode is
the win (measured ~28–30 tok/s vs 11.7 NPU on Qwen3.6-35B-A3B MLX 4-bit).

## 2. Architecture decision

**Subprocess + HTTP (not in-process, not REPL pipes):**

- In-process linking is rejected: LSE is C++26 + P2996 reflection behind
  `-freflection`, g++-16-only; would collapse our multi-compiler build and
  import an unstable `std::meta` API. (Quarantined risk.)
- REPL pipes (FLM pattern) are rejected for the *wire*: `lse` CLI is one-shot;
  a per-turn spawn would reload a 19 GB model every turn.
- **Chosen:** fork/exec `lse-server -m <dir> --host 127.0.0.1 --port <ephemeral>
  --api-key <random>` and talk OpenAI wire over localhost HTTP
  (`/v1/completions`, streaming SSE). One persistent daemon per backend,
  mirroring `NpuFlmBackend`'s fork+pipe+execl+retry lifecycle but with an HTTP
  protocol. Pure C++17 — zero LSE headers, zero C++26 coupling.

## 3. Integration points (files/symbols)

| File | Change |
|---|---|
| `include/common.h` | `BackendType::LSE_GPU = 16`; `backend_name()` entry; **add `ModelFormat::MLX = 7`** (routing gate) |
| `src/backend_factory.cpp` | `create_backend()` switch: `case BackendType::LSE_GPU:` → `new LseBackend(cfg)` (mirror `NPU_XRT` case) |
| `src/model_router.cpp` | new route in `select_backend_route()` if-else chain (recipe at lines 57–63) |
| `src/model_discovery.cpp` | detect MLX format: `config.json` has `quantization.mode == "affine"` + `language_model.*` weight prefix (or `model_type` `qwen3_5*`/`lemonseed`) → `cfg.format = MLX` |
| `src/backend_lse.cpp` + `include/backend_lse.h` | **new** — the backend (below) |
| `Testing/router_selfcheck.cpp` | + route checks: MLX qwen3.5-family → LSE first; non-MLX untouched |
| `Testing/lse_backend_selfcheck.cpp` | **new** — spawn/health/generate against a tiny MLX checkpoint |
| `bench/record.sh` | benchmark entry (recipe step 4) |

**Required GitNexus discipline (AGENTS.md):** run `impact` on
`create_backend`, `select_backend_route`, `backend_name` before editing;
`detect_changes` before any commit. Index is 3 commits behind → `analyze`
first.

## 4. `backend_lse` design (mirrors `NpuFlmBackend`)

```
class LseBackend : public Backend {
  pid_t pid_; int sock_;            // daemon + HTTP socket
  std::string port_, api_key_, model_dir_, server_bin_;
  // init(cfg, weights_dir):
  //   server_bin_ = getenv("LSE_SERVER_BIN") or "lse-server"
  //   model_dir_  = cfg.format==MLX ? cfg path : weights_dir
  //   pick ephemeral port (bind :0 → read back), random api-key (openssl-style rand)
  //   fork(); child: dup2 pipes to /dev/null, execl(server_bin_, "lse-server",
  //          "-m", model_dir_, "--host","127.0.0.1","--port",port_,"--api-key",key_)
  //   parent: poll GET /health until OK, retry/backoff (env LSE_SPAWN_RETRIES/
  //           LSE_RETRY_DELAY_S, default 10×5s like NPU_FLM_*)
  //   NOTE: BackendManager init cap is 120 s — cold JIT of a NEW model shape
  //   costs 50–125 s (measured). Pre-warm the LSE disk cache (~/.cache/lse)
  //   before first serve, or raise the cap for this backend via env.
  // generate_text(prompt, max_tokens): POST /v1/completions (stream=false),
  //   parse JSON, return text. Error strings (non-2xx / JSON "error") → "".
  // continue_text(delta): NOT supported by lse-server (stateless OpenAI API,
  //   no KV-reuse endpoint) → return ""; caller falls back to full prompt
  //   (documented re-prefill cost per turn).
  // forward/lm_head/generate: return false/-1 with stderr note (text-level
  //   only, exactly like NpuFlmBackend).
  // destroy(): SIGTERM daemon, reap, close socket.
};
```

Env knobs: `LSE_SERVER_BIN`, `LSE_SPAWN_RETRIES`, `LSE_RETRY_DELAY_S`,
`LSE_INIT_TIMEOUT_S` (raise the 120 s cap if needed), `LSE_PORT` (fixed-port
override for debugging).

## 5. Router semantics

- **MLX format** (new `ModelFormat::MLX`): route `lse` first, `cpu_generic`
  fallback. LSE is the *only* reader of MLX today — this is additive
  capability, no existing route displaced.
- **All other formats/archs**: untouched (GGUF → zinc/ggml_vulkan, Q4NX →
  npu_flm/hip_1bp, 1BP → hip_1bp, safetensors dense → hip/cpu, Zaya/BitNet/
  Zamba2 stay as-is).
- Long-context decode (>~1.5–2k) intentionally stays on NPU/FLM: measured LSE
  decode decays to 4.7 tok/s @3.4k while NPU holds ~11.8.

## 6. Sequencing

| M | Scope | Exit criteria |
|---|---|---|
| **M0** | Enum + `backend_name` + factory case + `LseBackend` skeleton (spawn, `/health`, destroy) | `lse_backend_selfcheck` green: spawns real `lse-server`, health OK, clean kill — **DONE** (fixed: `/health` is auth-gated; `http_get` now sends `Authorization: Bearer`) |
| **M1** | `generate_text` via `/v1/completions` on lemonseed-1.5b (small, already local) | e2e text out; error paths return `""` — **DONE** (`http_post` + nlohmann parse + `timings.predicted_per_second` → `benchmark()`; selfcheck 11/11 incl. live generation) |
| **M2** | MLX detection + router entry + `router_selfcheck` additions | qwen3.5-family MLX → LSE first; all existing checks still green — **DONE** (`read_mlx_metadata` in model_discovery (affine / qwen3_5* / lemonseed structural), `ModelFormat::MLX` route first in `select_backend_route`, LSE registered in `BackendManager::discover()` + `create_instance_rt`, MLX dirs exempt from the S_ISREG guard; router_selfcheck 19/19) |
| **M3** | Qwen3.6-35B-A3B MLX end-to-end + `bench/record.sh` row + eval doc update | same-model A/B (LSE GPU vs NPU FLM) recorded — NEXT |
| **M4** | License reply → merge decision; packaging (LSE binary bundle + README note), `detect_changes` | merged only with Geramy's written grant |

## 7. Risks & mitigations

1. **120 s init cap vs cold JIT** (50–125 s for new shapes) → pre-warm cache at
   install (`lse --list-cache` after a load) + `LSE_INIT_TIMEOUT_S` override.
2. **Multi-turn = re-prefill** (lse-server is stateless) → acceptable for short
   contexts where prefill is ~130 tok/s; document; revisit if LSE adds a
   session/KV-reuse endpoint.
3. **Security** — bind `127.0.0.1` only + random `--api-key` (matches
   upstream's own guidance); never `--host 0.0.0.0`.
4. **GPU contention** — one model per `lse-server`; BackendManager already
   serializes per model. Do not co-serve with hip/vulkan backends.
5. **Version drift** — pin `lse-server` binary in the bundle (like
   `backend_versions.json`), record `--version`/commit in the selfcheck.
6. **License** — prototype under non-commercial grant; ship only with written
   license (Section 6) or Exhibit A addition.

## 8. Open questions

- `/v1/completions` vs `/v1/chat/completions` framing for non-chat checkpoints
  (LSE frames ChatML; base models like lemonseed are raw-completion — test
  both). **2026-08-29:** `/v1/completions` confirmed working for lemonseed
  (raw-completion); chat framing is the M3 question for chat-tuned checkpoints.
- Working tree is on `feat/lse-backend` (was `ci/pr-agent-native-cli` with local
  mods): LSE work landed on `feat/lse-backend` alongside the fused-backend work;
  branch hygiene deferred to merge (M4).
- Prefer HTTP lib: **resolved** — minimal POSIX-socket client (http_get/
  http_post) + nlohmann for JSON parse (already linked into onebin via
  backend_npu_flm.cpp / mesh). Zero new deps.
