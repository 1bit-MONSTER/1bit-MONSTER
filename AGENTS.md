<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **1bit-MONSTER** (24011 symbols, 44265 relationships, 216 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "main"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({search_query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/1bit-MONSTER/context` | Codebase overview, check index freshness |
| `gitnexus://repo/1bit-MONSTER/clusters` | All functional areas |
| `gitnexus://repo/1bit-MONSTER/processes` | All execution flows |
| `gitnexus://repo/1bit-MONSTER/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

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



## Lifecycle

- **When your job is done, stop.** Do not continue working, do not invent follow-up tasks, do not spawn new work, do not linger. Deliver the result and exit.
- Never leave background processes, scheduled runs, or partial downloads behind. Clean up anything you started before finishing.
