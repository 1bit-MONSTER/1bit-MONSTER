# Vendored: lemonade-sdk/lemonade (embedded server core)

Vendored from https://github.com/lemonade-sdk/lemonade at commit
`2b6a7d77c71e551736f6cc8473dc46f479cd156b` (tag `v11.7.0`).

Vendored (instead of a submodule) because the embedded server core needs a
patch that only exists locally, and CI can't fetch unpublished submodule
SHAs. Re-vendor on upstream sync:

```sh
git clone https://github.com/lemonade-sdk/lemonade /tmp/lemonade
cd /tmp/lemonade
git checkout 2b6a7d77c71e551736f6cc8473dc46f479cd156b  # v11.7.0
# re-apply the embeddability patch below
rsync -a --exclude=.git /tmp/lemonade/ third_party/lemonade/
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

Drop the patch when upstream adopts any of these changes.
