# Vendored: lemonade (LOCAL-ONLY source)

> **LOCAL-ONLY.** Refresh this vendored tree from the **local** lemonade source
> (`/home/bcloud/1bit-lemonade-v1170/third_party/lemonade`), never from
> `github.com/lemonade-sdk/lemonade` (see that worktree's `RULES.md`). Do NOT
> `git fetch` / `pull` / `clone`, push PRs, open issues, or run CI against
> upstream.

This snapshot is at **lemonade v11.9.0**.

## What is upstream vs local

As of v11.9.0, **upstream now carries the `llamacpp-hrx` backend itself**
(`src/cpp/server/backends/hrx/hrx_server.cpp` + `lemon/backends/hrx/`), so that
part of our HRX work is no longer a local patch — the HRX backend code is
byte-identical to upstream.

The **local-only** deltas carried on top of v11.9.0 are:

1. **`hrx-b66` pin** (newer than upstream's `hrx-b59`): in
   `src/cpp/resources/backend_versions.json` and `test/cpp/test_hrx_contract.cpp`.
2. **HRX model-registry annotations**: `src/cpp/resources/server_models.json`
   carries the `*-HRX` entries (`hrx_serve` / `hrx_token_embd` / `hrx_embd_w`),
   `tools/gen_hrx_model_entries.py` and `tools/annotate_hrx_embedding_quants.py`
   are local-only (upstream does not read or generate these).
3. **Embeddability patch** in `CMakeLists.txt` (see below).

> Note: the `stream_stall_timeout` config key that our v11.8.x snapshot carried
> was **dropped** in this re-vendor — v11.9.0 handles the streaming-stall bound
> via `global_timeout` (upstream #3386), and local review confirmed the extra
> config key is not needed.

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
