---
name: impact-analysis
description: Use before editing any function, class, kernel, or public symbol in this repo — to find what would break, who calls the symbol, what its blast radius is, and whether the change is HIGH/CRITICAL risk. Also use when asked "is it safe to change X", "what depends on X", or "what breaks if I change X".
whenToUse: Before every non-trivial code edit, and whenever asked to assess the blast radius of a change.
---

# Impact Analysis

AGENTS.md mandates impact analysis before editing any symbol. This skill makes that runnable even when the GitNexus MCP tools are unavailable.

## Sources of code intelligence

1. **GitNexus MCP** (when available): run `impact({target, direction: "upstream"})`, then `context({name})` for callers/callees. Report the blast radius (direct callers, affected processes, risk level) to the user **before** editing.
2. **GitNexus CLI fallback**: `node .gitnexus/run.cjs analyze` to build/refresh the index; then query via the CLI. If the runner is absent, note it — do not silently skip.
3. **Manual fallback** (always available): for the target symbol run, in order:
   - `grep` for the symbol across `src/`, `kernels/`, `engine/`, `tests/`, `tools/` (headers included — `.h`, `.hip`, `.cpp`).
   - Read the declaration site and the 2-3 call sites that exercise it in the main execution path.
   - Trace the launch/dispatch chain for kernels: which wrapper calls the kernel, which stream/context, what buffer lifetimes it touches.
   - Check `CMakeLists.txt` / `tests/` for harnesses that link the symbol (e.g. `zaya_chain_check`).
   - Note any `extern "C"` or shared-library surface (REST server, FLM bridge) that consumes it.

## Risk classification

| Risk | Definition |
|------|-----------|
| LOW | Local function, no callers outside its translation unit, no wire/format impact |
| MEDIUM | Called from one engine path; tests cover the path |
| HIGH | Multiple callers, cross-backend (NPU/GPU/CPU) dispatch, or affects `.1bp`/GGUF wire format |
| CRITICAL | Public server API, model-file format, memory layout shared with kernels, or merge-queue mainline behavior |

## Before editing

1. State the blast radius to the user (callers, affected paths, risk level).
2. If HIGH or CRITICAL: **warn the user explicitly** and get acknowledgment before proceeding.
3. Never rename symbols with find-and-replace — use `rename` (GitNexus) or manual call-graph walk.

## Before committing

Run `detect_changes()` (GitNexus) or `git diff --stat` + a manual scan of the touched symbols to verify only expected symbols and execution flows changed. For regression review, compare against `main` (`detect_changes({scope: "compare", base_ref: "main"})`).
