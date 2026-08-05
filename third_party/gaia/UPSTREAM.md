# Gaia C++ framework — vendored copy

Vendored from **amd/gaia v0.22.0** (github.com/amd/gaia), the `cpp/` tree only
(the Python package lives upstream; we use the C++17 port to keep 1bit
zero-Python). License: MIT, see `LICENSE.md`.

Contents: `gaia_core` library (agent loop, tool registry, lemonade/mcp
clients, session, REPL, console, file/git tools, security) + `gaia-bash`
coding agent binary. Same agent loop as the official Python framework — agents
written against `gaia::Agent` ride the official AMD platform.

## Sync

```bash
git clone --depth 1 --branch v0.22.0 https://github.com/amd/gaia /tmp/gaia-upstream
rsync -a --delete --exclude='.git' /tmp/gaia-upstream/cpp/ third_party/gaia/
cp /tmp/gaia-upstream/LICENSE.md third_party/gaia/LICENSE.md
# bump the tag in this file + CHANGELOG
```

Upstream tags: `git ls-remote --tags https://github.com/amd/gaia | tail`

## Build notes

Vendored patches (re-apply after each sync, listed in `CMakeLists.txt`):
1. **httplib**: skip FetchContent when a `httplib::httplib` target already
   exists — the 1bit top-level build fetches cpp-httplib itself, and a second
   FetchContent collides on the target name.
2. **nlohmann_json**: when `find_package(nlohmann_json)` resolves to the
   top-level build's FetchContent copy (non-exportable), use include-dir-only
   instead of a PUBLIC link so `install(EXPORT)` stays valid.
3. **onebin dispatch**: `agents/bash/main.cpp`'s `int main()` is guarded by
   `#ifdef ONE_BIN_DISPATCH` to compile as `gaia_bash_main()` when its
   sources are linked into the 1bit `onebin` ELF instead of the standalone
   `gaia-bash` binary (see `tools/onebin.cpp` and the `EMBED_GAIA_CPP` block
   in the top-level `CMakeLists.txt`).

Included from the top-level `CMakeLists.txt`; options default OFF when built
as a subproject (tests/examples). TUI (FTXUI) is disabled — engine has its own
TUI; enable with `-DGAIA_BUILD_TUI=ON` if the Gaia console is wanted.
`gaia-bash` expects a Lemonade-compatible server at `http://localhost:13305`
(1bit: `zaya_server --lemonade serve`).
