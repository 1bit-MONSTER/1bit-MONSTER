# PLAN: Gut 1bit's agent stack, replace with AMD Gaia

**Date:** 2026-08-01 · **Status:** SUPERSEDED 2026-08-14 — this doc's own doc-status label was stale: D1–D3 were already actioned in code (EMBED_GAIA_CPP wired into CMakeLists.txt, third_party/gaia vendored and linked into `onebin`, replacing tools/onebit.cpp/onebitd.cpp) without this file ever being updated to say so. On the 1bit → 1bit MONSTER migration, that decision was reversed: **both** Gaia and the old `onebit` CLI were removed outright (neither is in the new repo). D3 ("keep repo name 1bit-systems") is moot — the repo is now `1bit-MONSTER`. JARVIS (`tools/jarvis/`) was never gutted and remains the one agent/voice pipeline. Kept here for history only; do not action anything below.

## Why this works (the key insight)

Gaia (amd/gaia v0.22.0) is AMD's **official agent framework** — but it is *not*
an inference engine. It needs a backend: Lemonade Server v11.5.x at
`localhost:13305` (OpenAI-compatible). 1bit **already vendors Lemonade v11.5.1**
(`third_party/lemonade`) and runs it in-process via `unified_server --lemonade`
with all 14 backends. The versions are compatible by construction.

So: **gut the hand-rolled agent layer, keep the inference engine, and let Gaia
drive the engine through the Lemonade pipe that already exists.** 1bit's crown
jewels (NPU reverse-engineering, 9 backends, 1BP format) stay. Gaia replaces
onebit/jarvis/unified-server-as-agent — everything 1bit built that AMD now
ships officially.

## What gets gutted (agent layer — ~7k lines)

| Path | What it is | Replaced by |
|------|-----------|-------------|
| `tools/onebit.cpp`, `tools/onebitd.cpp` | terminal coding agent + daemon | Gaia CLI / agents (`gaia` command, `Agent` base class) |
| `tools/jarvis/` (26 files) + `tools/jarvis_server.cpp` | voice agent server: auth, billing, beacon, rag, planner, persona, TTS | Gaia `talk/`, `audio/`, `rag/`, `governance/`, `daemon/` |
| `tools/unified_server.cpp` agent modes, `tools/unified_router.cpp` | Lemonade embed + routing as agent | Gaia `llm/lemonade_manager.py` + `llm/providers/` (keep only the engine-facing server entry, see below) |
| `tools/image_server.cpp`, `vision_server.cpp`, `whisper_demo.cpp`, `zaya1_vl_demo.cpp` | vision/audio servers | Gaia `vlm/`, `talk/` (Whisper + Kokoro) |
| `src/agent_watchdog.cpp` | self-healing watchdog | Gaia `daemon/custody/` |
| `prompts/`, `personas/`, `skills/` | agent prompts/personas | Gaia `skills/`, own agents' `_get_system_prompt()` |
| `workers/` (auth-url), `adapters/` | jarvis support | delete (adapters = jarvis voice-clone LoRAs; move to `models/` if ever reused) |

Keep: `src/` engine, `kernels/`, `engine/`, `npu-infer/`, `ck-prefill/`,
`spec-decode/`, `include/`, `models/`, `zaya_audio/`, `third_party/`, all
benchmark/convert tools (`bench_*`, `gguf_to_onebp`, `hf_to_onebp`, ...),
`docs/`, `site/` (agent pages rewritten), CI.

## What gets added

- **`third_party/gaia`** — submodule, tag `v0.22.0` (github.com/amd/gaia).
  Python package (full system: RAG, voice, VLM, MCP, hub, TUI, daemon) + `cpp/`
  C++17 port for the zero-Python ethos. Recommendation: **Python Gaia primary**
  (it IS the official system); C++ port is available later if the single-binary
  claim must extend to agents. Decision needed (D1).
- **`agents/`** — 1bit's own agents (coding agent, NPU monitor) as Gaia
  `Agent` subclasses, so they ride the official platform instead of a fork.

## Integration path (phases)

### Phase 0 — Decisions (need user sign-off)
- D1: Python Gaia, C++ Gaia port, or both → recommendation: both vendored,
  Python primary.
- D2: Engine kept as backend (recommended) vs. full-repo replacement where
  1bit becomes a Gaia fork and the engine is dropped. Dropping the engine means
  losing the NPU stack — Gaia has no engine of its own, so nothing replaces it.
- D3: Keep repo name `1bit-systems` (recommended) — Gaia is a dependency, not
  an identity.

### Phase 1 — Repo surgery
1. Add `third_party/gaia` submodule (v0.22.0).
2. Delete gut list above. `unified_server.cpp` shrinks to a thin
   `server_main.cpp`: lemonade embed + engine backends only, no agent modes.
   Rename concept: "inference server", not "agent server".
3. `agents/` with 2 starter agents: `coding_agent` (port of onebit's loop onto
   Gaia `Agent`), `npu_health_agent` (wraps `npu-infer` checks as a tool).
4. CMake: build engine as before; add `third_party/gaia/cpp` (only if D1 = C++).
   Python side: `requirements-gaia.txt` / pip `amd-gaia==0.22.0` + `uv.lock`-style
   pin in `packaging/`.

### Phase 2 — Runtime wiring
1. Start inference: `./build/zaya_server --lemonade serve` (existing path) →
   serves `localhost:13305`, all engine backends registered.
2. Start Gaia: `gaia chat` / `gaia agent run coding_agent` — its
   `lemonade_client` finds the server, engine does the inference.
3. NPU path: Lemonade's `flm` backend is the FastFlowLM stack 1bit rebuilt —
   verify which of 1bit's backends need exposing as Lemonade backends vs.
   routing through llamacpp/flm. The engine's own NPU backend can be registered
   as a custom Lemonade backend (that's the one real integration task; the
   embed already supports the backend registry).

### Phase 3 — Verification
1. `cmake -B build && cmake --build build` green.
2. Smoke: `zaya_server --lemonade serve` + `gaia chat` answering from a 1BP
   model on NPU/GPU.
3. `gaia eval` run on the ported coding agent (Gaia has eval infra — use it
   instead of 1bit's ad-hoc checks).
4. `detect_changes()` / GitNexus impact check on the surviving engine symbols —
   the gut list must not touch `src/` backend code.

### Phase 4 — Docs & release
1. README: agent story becomes "Gaia-powered"; inference story unchanged.
2. `site/` agent pages → point to amd-gaia.ai docs; keep 1bit.systems for the
   engine.
3. CHANGELOG entry; tag `v2026.08.02` with "agent layer = Gaia v0.22.0".

## Risks
- **Python runtime returns to the repo** (Gaia is Python). Engine stays
  zero-Python; agents are Python. If that's unacceptable → D1 = C++ port only
  (thinner: no RAG/voice/hub yet).
- **Gaia moves fast** (v0.22.0 as of today) — pin the submodule, upgrade on a
  schedule, don't fork.
- **Backend registration** in embedded Lemonade is the only hard engineering
  task; everything else is deletion + wiring.
