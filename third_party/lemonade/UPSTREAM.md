# Vendored: lemonade (LOCAL-ONLY source)

> **LOCAL-ONLY.** Refresh this vendored tree from the **local** lemonade source
> (`/home/bcloud/1bit-lemonade-v1170/third_party/lemonade`), never from
> `github.com/lemonade-sdk/lemonade` (see that worktree's `RULES.md`). Do NOT
> `git fetch` / `pull` / `clone`, push PRs, open issues, or run CI against
> upstream.

This snapshot carries the `llamacpp-hrx` backend (`src/cpp/server/backends/hrx/`),
its registry entries / pins in `resources/server_models.json` +
`backend_versions.json`, and the `tools/gen_hrx_model_entries.py` generator.
It is kept in sync with the local source (`1bit-lemonade-v1170/third_party/lemonade`,
v11.8.1).

```sh
# Re-vendor FROM the local source:
rsync -a --exclude=.git --exclude=UPSTREAM.md \
  /home/bcloud/1bit-lemonade-v1170/third_party/lemonade/ third_party/lemonade/
# re-apply the embeddability patch below
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
   `lemon/` headers + generated headers (upstream uses a subdirectory-local
   `include_directories()` that does not propagate to consumers).
4. `add_test()` police guarded by `BUILD_TESTING` so it does not leak into
   the parent scope when embedded via `add_subdirectory()`.
5. `add_dependencies(lemonade-server-core copy_resources)` so the resource
   copy fires even though `lemond` (whose POST_BUILD would trigger it) is
   never built in the embed.

Drop the patch when upstream adopts any of these changes.
