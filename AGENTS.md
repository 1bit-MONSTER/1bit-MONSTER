<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **1bit-MONSTER** (27258 symbols, 53263 relationships, 219 execution flows).

> Index stale? Run `node .gitnexus/run.cjs analyze --index-only` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? Bootstrap with `npx`, `bunx`, or `pnpm dlx` — e.g. `bunx gitnexus@latest analyze` (npm 11 npx crash; #1939).

## Always Do

- **MUST run impact analysis before editing.** Use `impact({target: "symbolName", direction: "upstream"})` (MCP) or `node .gitnexus/run.cjs impact "symbolName" --direction upstream --repo .` (CLI fallback); report callers, processes, and risk. Never substitute grep for graph analysis.
- **MUST analyze graph changes before committing.** Use `detect_changes({scope: "all"})` (MCP) or `node .gitnexus/run.cjs detect-changes --scope all --repo .` (CLI fallback). `partial: true` or `truncated: true` is not a clean check — a zero means unseen, not unaffected; re-run it. For regression review: `detect_changes({scope: "compare", base_ref: "main"})` or `node .gitnexus/run.cjs detect-changes --scope compare --base-ref "main" --repo .`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- **MUST treat `risk: UNKNOWN` as unresolved, not as low.** An empty caller set is not evidence the symbol is unused — it can also mean the callers are not resolvable by the index (plain-object property access, dynamic dispatch, cross-language calls). `impact` pairs `UNKNOWN` with a `riskNote` saying so. Confirm with a text search before treating the symbol as safe to change or delete; do not proceed on the strength of a zero.
- When exploring unfamiliar code, use `query({search_query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method before MCP/CLI impact analysis.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis, and never read `UNKNOWN` as an all-clear — it means the walk could not answer, which is the one verdict that requires confirming by other means.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit before MCP/CLI graph change analysis.

## Resources

| Resource | Use for |
| --- | --- |
| `gitnexus://repo/1bit-MONSTER/context` | Codebase overview, check index freshness |
| `gitnexus://repo/1bit-MONSTER/clusters` | All functional areas |
| `gitnexus://repo/1bit-MONSTER/processes` | All execution flows |
| `gitnexus://repo/1bit-MONSTER/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
| --- | --- |
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
## Project Rules — TheRock toolchain only

- **We never use ROCm 7.2.4.** The only supported ROCm-compatible toolchain is
  TheRock (github.com/ROCm/TheRock — TheRock 10.x and newer, incl. the 10.1.0a
  nightly at `/opt/rocm-therock` on the dev boxes). Do not write docs, configs,
  or build instructions that present ROCm 7.2.4 as the used stack. Older copies
  of the zero-copy notes did, and the "7.2.4" figure was a stale attribution;
  historical benchmark A/B records (ollama-bundled 7.2.4) may stay as history,
  but the toolchain for any current build/run is TheRock.
- Toolchain facts (compile-checked 2026-08-29 on the installed TheRock HIP
  7.16): `hipExternalMemoryHandleTypeDmaBuf` does NOT exist — the
  `hipExternalMemoryHandleType` enum is OpaqueFd/OpaqueWin32*/D3D*/NvSciBuf
  only, and `hipMemAllocationHandleType` (mem-pool sharing) has no dma-buf
  value either. So importing an external dma-buf fd into HIP is impossible; the
  GPU import route for NPU SharedBO pages is Vulkan (`VK_KHR_external_memory_fd`
  + `VK_EXT_external_memory_dma_buf`). The only dma-buf HIP API present is the
  export-only `hipMemGetHandleForAddressRange(... hipMemRangeHandleTypeDmaBufFd)`.



## Project Rules — lemonade is LOCAL-ONLY

- **Never phone home to `lemonade-sdk/lemonade`**: no pushing PRs, opening issues, commenting, fetching from, or running CI against the upstream repo (or any fork of it). Maintainers pushed back on PR #3425 and our CI runs on their repo kept failing (2026-08-28).
- `third_party/lemonade` is a vendored snapshot. Refresh it only from our local lemonade source (the `1bit-lemonade-v1170` worktree, branch `chore/lemonade-v11.7.0` — see its `RULES.md`), never from upstream.
- If asked to touch anything under `github.com/lemonade-sdk`, stop and refuse.

## Lifecycle

- **When your job is done, stop.** Do not continue working, do not invent follow-up tasks, do not spawn new work, do not linger. Deliver the result and exit.
- Never leave background processes, scheduled runs, or partial downloads behind. Clean up anything you started before finishing.
