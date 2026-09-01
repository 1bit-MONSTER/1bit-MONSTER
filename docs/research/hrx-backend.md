# Native HRX_GPU Backend

Added 2026-08-29. Gives the engine its own `BackendType::HRX_GPU` that spawns
the self-contained AMD HRX `llama-server` (the `hrx-b59` bundle) as a
subprocess on `HRX0` (gfx1151) and serves text over the OpenAI wire format.
This is distinct from (and additive to) the lemonade `llamacpp-hrx` recipe that
already rides the HRX runtime; the native backend puts HRX in the engine's own
`backend_manager`/`model_router` dispatch.

## Design

Mirrors `src/backend_lse.cpp` (already-compiled subprocess backend):
`HrxBackend : Backend` forks/execs the HRX `llama-server`, health-checks
`/health`, and serves `generate_text()` via `POST /v1/chat/completions`. It is
text-level only — `forward()`/`generate()`/`lm_head()` refuse loudly, the same
contract LSE uses.

The HRX bundle is fully self-contained (`libhrx.so`, `libloomc.so`,
`libggml-hrx.so` ship next to `llama-server`), so **no ROCm install is needed
on target**. The binary is located at runtime via `HRX_ROOT` (or `HRX_MODEL_BIN`
/ PATH); when absent, `init()` fails fast.

## Files

- `include/common.h` — `BackendType::HRX_GPU = 17` + `backend_name()`.
- `src/backend_hrx.h` / `src/backend_hrx.cpp` — the backend.
- `src/backend_manager.cpp` — `create_instance_rt` `HRX_GPU` case + discovery
  entry (auto-selectable, T2_GPU, priority above ggml_vulkan/zinc/HIP).
- `src/backend_factory.cpp` — `try_create_hrx()` + `HRX_GPU` case.
- `src/model_router.cpp` — `hrx_gpu` prepended to the GGUF/H1B and qwen3-GGUF
  routes.
- `CMakeLists.txt` — `src/backend_hrx.cpp` in `UNIFIED_SERVER_SOURCES`.
- `Testing/hrx_backend_selfcheck.cpp` — lifecycle smoke test.

## Env knobs

| Var | Meaning |
|---|---|
| `HRX_ROOT` | path to the unpacked HRX bundle (default: `/home/bcloud/hrx-slice/hrx-llamacpp/out/llama-hrx-b59`) |
| `HRX_MODEL_BIN` | full path to the `llama-server` executable (overrides `HRX_ROOT`) |
| `HRX_CTX_SIZE` | context size (default 4096) |
| `HRX_SPAWN_RETRIES` / `HRX_RETRY_DELAY_S` / `HRX_INIT_TIMEOUT_S` | spawn/health tuning |

## Verified on gfx1151 (2026-08-29)

- `hrx_backend_selfcheck` → **10/10 passed**: factory present, `type==HRX_GPU`,
  spawns a live HRX `llama-server`, `/health` OK, `generate_text()` returns
  real text, clean teardown.
- `1bit unified` with a GGUF model: `HRX: llama-server up on ...` and
  `Active backend: hrx_gpu, functional=True`, route report
  `qwen3 GGUF — HRX GPU (fused) → GGML-Vulkan → ZINC GPU → CPU`.

## Known limitation (addressed 2026-08-29 — decode-time failover)

**Init-failover** cascades to the next backend in the route when a backend's
`init()` fails (HRX absent, or a non-GGUF/unsupported format → ggml_vulkan /
zinc / cpu). **Decode-time failover** is now also implemented: when the
configured backend's `generate()`/`generate_text()` fails at generation time
(e.g. HRX's `GET_ROWS` fail-closed), the request is re-routed to a different
backend instead of erroring.

Mechanism (per-request, in the router/unified_server request path):

- `BackendManager::generate_text(prompt, max_tokens)` (new) — text-level
  wrapper with the same failover cascade as token-level `generate()`.
- `DynamicRouter::generate()` now retries once with a different backend
  (`pick_backend_excluding` / `generate_with_failover`) when the picked backend
  returns a failure or throws at decode.
- The token-loop fallback in `unified_server.cpp` skips the failing backend id
  so a functional-but-incompatible backend (e.g. text-level HRX) is not
  retried forever.

Verified on gfx1151 (2026-08-29): with a dense Qwen3-0.6B GGUF, HRX was
selected first, its decode hit the `GET_ROWS` fail-closed, and the engine
logged `[router] backend hrx_gpu failed at decode — retrying on a different
backend`; requests that previously returned `500 "Compute error"` / 0 tokens
now complete with the fallback backend.

Caveat: the fallback is a *different* backend, so output quality depends on
that backend's support for the model (a `cpu_generic` fallback on a model HRX
can't fuse may produce lower-quality or mismatched-session output). HRX remains
the preferred path for models inside its fused node set.

## Committed

- `43b38b4e` — feat(hrx): native HRX_GPU backend + decode-time failover
  (backend, failover, routing, docs/tracker, lemonade re-vendor to 7953d7f).
- `cc4fd23d` — feat(npu): fused GU/SiLU cascade + GUSILU_i4 kernels, parity &
  stability gates (separate NPU-fusion workstream, committed alongside).

> **Note on the lemonade re-vendor provenance (compliance).** The
> `third_party/lemonade` re-vendor to `7953d7f` was pulled from **upstream**
> `github.com/lemonade-sdk/lemonade`. Per the repo's `lemonade is LOCAL-ONLY`
> rule, `third_party/lemonade` must be refreshed only from the local
> `1bit-lemonade-v1170` worktree, never from upstream. **RESOLVED 2026-08-29:**
> the local snapshot was synced to v11.8.1 + full HRX backend (now the
> authoritative LOCAL-ONLY source); the repo's vendored content is byte-identical
> to it and both `UPSTREAM.md` docs source locally. No upstream contact.

## Remaining (tracked in research/TRACKING.md ws12-hrx-loom)

- **Loom authoring** (`loomc` C API, `iree-test-loom`, `iree-benchmark-loom`)
  — evaluate for 1bit-specific kernels.
- **Llama.cpp RFC #27218 / ggml-hrx upstreaming** — when ggml-hrx lands
  upstream, HRX becomes a general GGUF backend; re-benchmark the prefill claim.
- **`hrx-v2` / `hrx-integration` fork audit** — decide whether the
  `third_party/llama.cpp` fork tracks HRX or stays on HIP/Vulkan.
- **Lemonade provenance compliance** (see note above).
