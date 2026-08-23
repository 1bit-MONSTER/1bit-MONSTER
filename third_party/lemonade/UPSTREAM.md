# Vendored: lemonade-sdk/lemonade (embedded server core)

Vendored from https://github.com/lemonade-sdk/lemonade at commit
`fc4f2439a9225355a63c38cefd5fe16c23525cf7` (fix(installer): handle slow
tarball listings (#2830)).

Vendored (instead of a submodule) because the embedded server core needs a
patch that only exists locally, and CI can't fetch unpublished submodule
SHAs. Re-vendor on upstream sync:

```sh
git clone https://github.com/lemonade-sdk/lemonade /tmp/lemonade
cd /tmp/lemonade
git checkout fc4f2439a9225355a63c38cefd5fe16c23525cf7
# re-apply the embeddability patch below
rsync -a --exclude=.git /tmp/lemonade/ third_party/lemonade/
```

## Local patch: embeddability

`CMakeLists.txt` carries one local patch (see the "Embedding" comment near
`lemonade-server-core`):

1. `CMAKE_SOURCE_DIR` → `CMAKE_CURRENT_SOURCE_DIR` everywhere — no-op when
   built standalone, fixes packaging paths when built as a subdirectory of
   the 1bit-monster repo via `add_subdirectory`.
2. PUBLIC include dirs on `lemonade-server-core` so parent targets
   (`unified_server`, `unified_router`) linking the OBJECT library see
   `lemon/` headers + generated headers (upstream uses a subdirectory-local
   `include_directories()` that does not propagate to consumers).

Drop the patch when upstream adopts either change.
