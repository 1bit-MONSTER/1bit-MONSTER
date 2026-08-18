---
name: test-driven-development
description: Use when implementing or fixing behavior — write the failing test first, watch it fail for the right reason, then make it pass (red-green-refactor). Especially for engine kernels, loaders, and server handlers in this repo.
whenToUse: Any implementation or bug fix with observable behavior, unless the change is a trivial comment/format-only edit.
---

# Test-Driven Development (Red-Green-Refactor)

The repo's test surface: `tests/` with ctest registration in `CMakeLists.txt`, HIP kernel harnesses, and standalone check binaries (e.g. `zaya_chain_check`, `zaya_gpu_decode`, `bench_*`). CI runs the suite per PR; some tests carry the `gpu_required` label and SKIP_RETURN_CODE 77.

## The cycle

1. **RED — write the failing test first.**
   - Name the behavior precisely: input state → expected output.
   - For kernels: prefer a CPU reference implementation in the test (golden values), not just "no crash".
   - For loaders: use a fixture file or a tiny generated `.1bp`/GGUF; for the real 44.6 GB model, guard with a skip when `models/` lacks the file (SKIP_RETURN_CODE 77 pattern).
   - Register in CMake (`add_test`) so CI and `ctest` pick it up; label `gpu_required` when it needs the HIP device.
2. **Run it and confirm it fails for the right reason** — assertion failure, not a compile error, not a hang. If it hangs or crashes, the test needs fixing before the code does.
3. **GREEN — minimal change to pass.** No speculative refactors in this step.
4. **REFACTOR — only after green.** Collapse duplication, tighten names, keep the suite green.

## Testing anti-patterns to avoid

- Golden-testing the implementation against itself (mirror implementations that share the same bug).
- Asserting on timing/performance in the same test as correctness (split into separate benchmarks).
- Skipping the test because the model file is big — build small synthetic fixtures instead.
- Over-mocking: in this codebase, prefer real in-process harnesses (engine is `#include`-linkable, no main()) over stubbing HIP.

## Kernel-specific guidance

- Compute expected values in host code with doubles, compare against device output with a tolerance appropriate to the accumulation order (bf16 kernels: compare to a bf16 CPU reference, not an f32 one).
- Exercise the A/B masks where the engine supports them (e.g. `zaya_set_gemv_mode`) — legacy vs new path must agree.
