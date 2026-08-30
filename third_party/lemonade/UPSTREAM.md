# Vendored: lemonade (LOCAL-ONLY source)

> **LOCAL-ONLY.** This is our own snapshotted lemonade tree. Refresh the
> `third_party/lemonade` snapshots in our projects (1bit-MONSTER, 1bit-MONSTER-pi,
> 1bit-MONSTER-zaya, 1bit-fused-verify, …) **from this local directory**, never
> from `github.com/lemonade-sdk/lemonade` (see `RULES.md`). Do NOT `git fetch` /
> `pull` / `clone`, push PRs, open issues, or run CI against upstream.

This snapshot is synced to the repo's `third_party/lemonade` and carries the
`llamacpp-hrx` backend (`src/cpp/server/backends/hrx/`), its registry entries /
pins in `resources/server_models.json` + `backend_versions.json`, and the
`tools/gen_hrx_model_entries.py` generator.

Local patch: the `CMakeLists.txt` carries the one **embeddability** patch below
(not upstream). It must be re-applied whenever the vendored tree is refreshed,
so `lemonade-server-core` links as a subdirectory of the 1bit-MONSTER engine.

```sh
# Sync FROM this local snapshot INTO a project:
rsync -a --exclude=.git --exclude=UPSTREAM.md third_party/lemonade/ <proj>/third_party/lemonade/
```

## Local patch: embeddability

`CMakeLists.txt` carries one local patch (see the "Embedding" comment near
`lemonade-server-core`):

1. `CMAKE_SOURCE_DIR` → `CMAKE_CURRENT_SOURCE_DIR` everywhere — no-op when
   built standalone, fixes packaging paths when built as a subdirectory of
   the 1bit-monster repo via `add_subdirectory`.
2. Treat the parent's FetchContent-provided `nlohmann_json` and `httplib`
   targets as "system" deps (`USE_SYSTEM_JSON` / `USE_SYSTEM_HTTPLIB` set ON
   when `TARGET nlohmann_json` / `TARGET httplib` exist) so the vendored tree
   does not FetchContent a second copy and collide on target names. The
   `lemonade-httplib` interface target short-circuits to link the parent's
   `httplib` target directly when it exists.
3. PUBLIC include dirs on `lemonade-server-core` so parent targets
   (`unified_server`, `unified_router`) linking the OBJECT library see
   `lemon/` headers + generated headers.
4. `add_test()` police guarded by `BUILD_TESTING` so it does not leak into
   the parent scope when embedded via `add_subdirectory()`.
5. `add_dependencies(lemonade-server-core copy_resources)` so the resource
   copy fires even though `lemond` (whose POST_BUILD would trigger it) is
   never built in the embed.

Drop the patch when upstream adopts any of these changes (do not block on
upstream — this is local-only).
