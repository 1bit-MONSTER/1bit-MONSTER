# Registry verification runbook

How to check every claim on branch `goal/one-registry-one-router` (goal: *one registry, one
router*). Nothing here is an opinion — each section is a command and the output it must
produce. If a command's output differs, the claim is wrong and the branch should not merge.

The branch's whole argument is that its claims are **checkable**, so this file exists to make
checking cheap. It is the artifact I would want if someone handed me the branch.

---

## 0. What the branch adds, in one paragraph

An artifact-first **model registry** (`ModelRegistry`) whose unit is an *artifact* — an id, a
container, a dtype-space, 1..N shard files, capabilities, a tokenizer and a lineage — rather
than a file or a flat directory scan. On top of it: a **capability table** with declared
constraints and provenance, a **resolver** (`id → artifact → capability → route`), a
**bridge** translating capabilities onto the engine's backend vocabulary, and a **merge
primitive** that lets the registry reorder a route while never removing a lane — with its two
properties **asserted** by a checker rather than claimed. Plus the diagnostics and checks listed
in §6–§9, and the full list of tools this branch adds is
`git diff --name-only origin/main...HEAD -- tools/`. Nothing consumes the resolver yet: the
caller flip is one line and is the operator's decision.

---

## 1. Build it (no engine link needed for most of this)

The registry module is deliberately engine-independent, so the oracle builds with a stock
compiler:

```sh
g++ -std=c++20 -O2 -Iinclude -DREGISTRY_SCAN_STANDALONE \
    tools/registry_scan.cpp src/model_registry.cpp -o registry_scan
```

Expected: builds clean with no HIP/XRT. `-Wall -Wextra` clean too.

For the engine-side tools (bridge, route map) and the real binary, on a box with the TheRock
SDK (recipe measured by @agent-44437c on strixhalo; configure ≈ 66 s, onebin ≈ 72 s):

```sh
cmake -S . -B b -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/rocm-therock/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/opt/rocm-therock/bin/amdclang++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm-therock/bin/amdclang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151
cmake --build b --target onebin -j32            # artifact is b/1bit (OUTPUT_NAME)
cmake --build b --target registry_scan -j32     # standalone CLI
cmake --build b --target registry_route_map -j32
```

Expected: `b/1bit`, `b/registry_scan`, `b/registry_route_map`.

---

## 2. The registry describes the store truthfully

```sh
./registry_scan ~/models                       # or any store
./registry_scan --resolve <artifact> ~/models
./registry_scan --json ~/models | python3 -c 'import json,sys; print(json.load(sys.stdin)["report"])'
```

Expected: every artifact carries `container`, `dtype_space`, `capabilities`, `quantization`,
`tokenizer`, and — where readable — `tok_embd_dtype`, `declared_experts`, `declared_hidden`,
`declared_layers`, `tensor_count`, `rope_theta`, `native.*`. Rows that cannot be trusted carry
a named marker rather than a silent value: `[type42!]`, `[name-says-q4nx-no-type42]`,
`[native-name-mismatch]`, `[display-name-suspect]`, `[arch-suspect]`,
`[experts-underdeclared!]`, `[expert-fields-absent-v1]`, `[geometry-cannot-hold-file!]`.

**The claim to check:** every marker corresponds to a documented finding whose evidence is in
the OKF bundle's `model-artifact-inventory.md` (F1–F13, D5a–g). If a marker fires on a file
whose finding says it should not, the finding or the code is wrong.

---

## 3. Capability constraints carry provenance and enforcement scope

```sh
./registry_scan --capability HRX-GGUF --at-context 2048 ~/models   # NONZERO artifacts
./registry_scan --capability HRX-GGUF --at-context 4096 ~/models   # 0 artifacts
./registry_scan --engine-limit HRX-GGUF=4096:b66-fork \
                --route <a gguf> --at-context 4096 --prefer HRX-GGUF ~/models
```

Expected: at 2048 the lane is offered; at 4096 it is refused **with the reason and the bundle
identity in the text**. Deliberately no artifact COUNT here: a number in a doc is a measured
claim that drifts with the store (it was 20 when written, 19 an hour later), which is the same
fault as the reason string that said "4 files" after the fifth was measured. Assert the shape,
not the count. the `--engine-limit` override makes the same query succeed and the
bundle id appears in the output. The footnote under the table must state `[measured, bundle
hrx-b66]`, `enforced by: engine server + shim`, and `NOT enforced in: --lemonade`.

**The claim to check:** a limit is a *report* with a *scope*, not a gate — the tool says so in
its own output, and `report.gate_enforces` is `false` in the JSON.

---

## 4. The resolver refuses with distinguishable reasons

```sh
./registry_scan --route <id> ~/models                              # YES -> <capability>
./registry_scan --route <id> --prefer NPU-1BP ~/models             # NO  -> not available
./registry_scan --route <id> --at-context 4096 --prefer HRX-GGUF ~/models
./registry_scan --route no-such-artifact.gguf ~/models             # exit 1
```

Expected: **three** distinct rejection kinds that must not read the same — *no backend
advertises it*, *hardware absent*, *constraint violated* — plus refused-by-design
(`HRX2-GGUF-Q4NX`) and `BLOCKED: KNOWN ABORT` for the measured-SIGABRT class.

**The claim to check:** the reason determines the fix, and the categories are separate
because they have different fixes. If two of them produce identical text, the distinction has
been lost.

---

## 5. The capability → backend map, and why the type is not enough

```sh
./registry_route_map --table
```

Expected: the 9-row table, `HRX2-GGUF-Q4NX` REFUSED, a **PROBE CONTRACT** block showing
`availability_for` producing all four outcomes (ABSENT / UNKNOWN / REGISTERED_DRY / PRESENT),
a **TYPE-UNIQUENESS** block marking the four collapsed types, and an **EVIDENCE STATUS** block
that states per row how it is known (`BOX-VERIFIED` for two rows, `CODE-READING` for one, the
rest `ROUTER-STRING`).

**The claim to check:** no row is presented as stronger than its evidence, and
`BackendType` is never treated as a hardware category — `npu_xrt` vs `npu_flm` share a type
with **opposite** availability, and Vulkan-capable ids span **three** types.

---

## 6. The flip decision table (what a caller would actually change)

```sh
./route_compare <file>...          # inside onebin: 1bit route-compare <file>...
```

Expected per file: `router`, `merged` (what a flip would do), and `registry`, with
`[HEAD DIFFERS - a correction]` or `[HEAD SAME, list extended]`.

### 6.1 The invariant assertion (engine-side, onebin)

```sh
1bit registry-merge-invariants <dir>
```

**INV-1** never remove a lane: every router id survives in the merged list.
**INV-2** move the head only for a **stated** exclusion: the router's head id must appear in
`refused` / `unavailable_here` / `blocked` / `skipped_by_context` / `conditional`. *"The registry
named a different lane" is not a reason* — which is the general form of the rule I had to learn
by breaking it (an unconditional override demoted `cpu_qwen3_5` on the 35B-A3B).

Expected on the store measured: `checked=18  INV-1 violations=0  INV-2 violations=0`. This is a
**gate**: it exits non-zero on violation, unlike §6's table.

**How it was validated** — and this is the part to preserve: compiled against the pre-defer
merge (`git show 7a6778728^:src/model_registry_route.cpp`) it fails naming the file and the
fault (`INV-2 VIOLATED  qwen3-6-35b-a3b-q8-0.gguf: head moved cpu_qwen3_5 -> ggml_vulkan with NO
stated exclusion`), and `INV-1` was never violated by that bug — so INV-2 is precisely the
invariant that catches it rather than a blanket check that would have passed anyway.

**The claim to check, and the reason this is a table rather than a leap:** the merge moves the
router's head **only where the registry has excluded that lane for a stated reason**. On the
store measured, that is exactly two files (both measured FAIL on HRX), three deferrals, and
every file gains a lane with none removed. If a head moves on a file where the registry merely
*named* a different lane, the defer rule has regressed.

---

## 7. The step-2 evidence: `registry-diff` (engine-side, onebin)

```sh
1bit registry-diff <dir>
```

Prints the legacy `discover_models()` view beside the registry's and the delta between them:
`same-file` / **id-divergent** / **legacy-invisible**, plus every legacy id that names more than
one file. Read-only; wired into no caller. It exists because step 2 ("extend
`src/model_discovery.cpp` into the registry of record") is a behaviour change and therefore the
operator's call — the evidence for it should be numbers, not an impression.

Measured on the live store: **`same-file=18  id-divergent=18  legacy-invisible=13`** — every file
the flat scan finds carries a different canonical id, 13 artifacts are invisible to a
non-recursive scan, and three sets of distinct files share one legacy id (the `-m` silent-pick
hazard). Caveat: run it over a directory that EXCLUDES a file with an unknown dtype until the
`fix/gguf-unknown-dtype-guard` branch lands, because the unfixed scan SIGFPEs on one (F14).

**The claim to check:** the three numbers, and the fact that a native `.q4nx`'s legacy id is its
**containing directory** (proved by running it in two differently-named directories — same files,
id follows the directory).

## 8. The checks (two in CI, informational)

```sh
sh tools/dispatch_key_check.sh src/backend_manager.cpp     # exit 0; 1 = the set CHANGED
python3 tools/registry_flag_audit.py --static-only         # A + C, source-only
python3 tools/registry_fixture.py /tmp/fixture
python3 tools/registry_flag_audit.py --binary ./registry_scan --fixture /tmp/fixture   # A-D
```

Plus the one enforcement that lives outside `tools/` at runtime — the commit-msg tripwire for
a shell-interpolated commit message (a backticked span the shell EXECUTED, deleting the text
silently, leaving zero backticks and no marker):

```sh
hk=$(git rev-parse --git-path hooks/commit-msg) && ln -sf "$PWD/tools/commit-msg-hook.sh" "$hk"
```

**In a worktree that path resolves to the COMMON hooks directory**, so installing from any
worktree applies a commit-blocking hook to every worktree of the repo — a repo-wide decision, not
a local install. The cause-level fix needs no hook at all: pass messages with `-F <file>`.

Expected: exit 0 on a clean tree; exit 1 with a named flag, type or message pattern when
something drifts.

**The claim to check — and the discipline that matters more than the checks:** every one of
these was validated against an input *known to be bad* before being trusted, because a check
that has never failed is not a check. `registry_flag_audit.py` had **eight** defects, all
found that way and none by reading it. If you change a check, run it against a deliberately
broken input first.

---

## 9. What is deliberately NOT done

- **The caller flip.** One line in `backend_manager`, verified primitive, **left to the
  operator** — it changes what runs where, and every hardware assertion on this branch had to
  be corrected by someone measuring a box.
- **`cpu_qwen3_5` (class-specific CPU engine).** A real vocabulary gap, and the lane owner's
  judgement is to leave it documented rather than encode it: *"leaving it as a documented gap
  and not flipping that lane is the right call."* Closing it because it is closable would
  repeat the fault this branch spent its last commits correcting.
- **A dense native `.q4nx`.** The MoE branch does not catch it and no Q4NX chain exists behind
  it; the store has no such artifact, so the honest entry is **"branch unexercised"**, not a
  guessed chain.
- **`--lemonade`.** That mode returns before registering the engine's routes, so the registry
  is invisible there. Corrected in the ADR; not implemented.
- **The three checks are informational in CI.** Promote the dispatch-key check only once the
  recorded four-type set stops moving — a required check would have the first person it blocks
  delete it rather than deliberately extend the list.
