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
./b/1bit registry-merge-invariants <dir>    # or the standalone `registry-diff`-style build
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
./b/1bit registry-diff <dir>          # or the standalone build below, no engine link
```

Prints the legacy `discover_models()` view beside the registry's and the delta between them:
`same-file` / **id-divergent** / **legacy-invisible**, plus every legacy id that names more than
one file. Read-only; wired into no caller. It exists because step 2 ("extend
`src/model_discovery.cpp` into the registry of record") is a behaviour change and therefore the
operator's call — the evidence for it should be numbers, not an impression.

**RUNNABLE WITHOUT THE ENGINE LINK** — the header used to say "engine-side, onebin", which reads as
*not checkable until the link lands*. It is checkable now, in one line, with plain `clang++`:

```sh
clang++ -std=c++23 -O2 -Iinclude -Isrc -DREGISTRY_DIFF_STANDALONE \
  tools/registry_diff.cpp src/model_registry.cpp src/model_discovery.cpp \
  src/safetensors_reader.cpp src/q4nx_reader.cpp src/gguf_reader.cpp -o registry-diff
```

**FOUR POPULATIONS, all measured** (this is the bad-input run §8's rule requires, and the reason the
tool had to become runnable):

| population | same-file | id-divergent | legacy-invisible |
|---|---|---|---|
| the store (F14 file excluded) | 18 | 18 | 13 |
| `/home/bcloud/bench-models` | 4 | 4 | 0 |
| `/home/bcloud/models` | 0 | 0 | 2 |
| the synthetic fixture (before) | 0 | 0 | 5 |

The counters **move with the input** — the tool reads the directory rather than emitting a constant.
That is what the control establishes, and it is all it establishes: the fixture's own numbers are
**instrument evidence, not a finding about artifacts.**

**`same-file == id-divergent` in all four, and that equality IS the claim** (@agent-ec855d, who
nearly reported it as a redundancy and it is the opposite). My gloss — *"every file the flat scan
finds carries a different canonical id"* — **is** the equality of the two counters, so `18/18` is the
evidence rather than two views of one fact. Two counters that always agree usually deserve suspicion;
here the agreement is the measurement.

**AND THE COUNTER HAD NO CASE WHERE IT COULD DISAGREE — now closed.** No population had ever made the
counters differ, so the branch that reports a difference was dead code: had `id-divergent` regressed
to a constant `0`, no fixture or store we have would have noticed. `tools/registry_fixture.py` now
emits **two complete GGUFs** — one whose `general.name` equals the registry's id for it (the
basename), and one where they differ:

```
fixture now : same-file=2  id-divergent=1  legacy-invisible=5     (both branches live)
```

Both are needed. With only the matching case the fixture still reads `id-divergent=0`, and a
regression to a constant 0 stays invisible; with both, the expected `id-divergent` is **non-zero**,
so the counter has to actually count for the fixture to agree.

**Verified by mutating the counter** (§8's negative-control rule applied to the *counter*, not just
the tool): pinning `id_divergent` to a constant `0` yields `same-file=2 id-divergent=0` against the
fixture's expected `1` — **the fixture disagrees, so the regression is caught.**

**Why the fixture's other artifacts are invisible to the legacy scan, now known rather than
assumed:** they are metadata-only, and the strict engine reader refuses them (*"GGUF truncated:
'blk.2.weight' needs 53760 bytes at offset 2336 but file is 2318 bytes"*) while the registry's
metadata reader accepts them. So `legacy-invisible` there is **reader strictness, not artifact
class** — the complete GGUFs above are seen by both. That is exactly why a control's own output must
not be read as a second finding.

Measured on the live store: **`same-file=18  id-divergent=18  legacy-invisible=13`**
Measured on the live store: **`same-file=18  id-divergent=18  legacy-invisible=13`** — every file
the flat scan finds carries a different canonical id, 13 artifacts are invisible to a
non-recursive scan, and three sets of distinct files share one legacy id (the `-m` silent-pick
hazard). Caveat: run it over a directory that EXCLUDES a file with an unknown dtype until the
`fix/gguf-unknown-dtype-guard` branch lands, because the unfixed scan SIGFPEs on one (F14).

**The claim to check:** the three numbers, and the fact that a native `.q4nx`'s legacy id is its
**containing directory** (proved by running it in two differently-named directories — same files,
id follows the directory).

## 8. The checks (two configured in CI, informational)

**WHEN CI ACTUALLY RUNS THEM — verified, not assumed.** `ci.yml` triggers on
`push: branches: [main]`, `pull_request: branches: [main]` and `merge_group:`. A
**feature-branch push runs nothing**: `api.github.com/.../actions/runs?branch=<feature>` returned
**0 runs**, so on this branch the two steps below have never executed in CI, and neither have the
rest of `ci.yml`'s jobs. They begin to run when a PR is opened against `main`.

So the honest description of the enforcement today is: **procedural (this file) with CI as the
destination once a PR exists.** The steps are correct and verified by running them by hand — the
composite exits 0 — but nothing runs them automatically on a feature branch. **"Configured" must
not be read as "enforced", and it must not be read as "enforced somewhere, probably".** Whether to
add a branch pattern to the `push` trigger is a repo-wide runner-capacity decision, not a local one.

**WHY THERE IS NO CHEAPER ENFORCEMENT POINT — this was hunted (@agent-ec855d), so it is not
re-hunted.** The only any-branch `push` workflow is `bench.yml`, and it is path-filtered to
`engine/npu/{src,kernel,xclbins}/**` and to itself, on a SELF-HOSTED runner with a 60-minute
timeout — a source check does not belong in an NPU benchmark job on the NPU box. Six workflows
(`census-watch`, `end-to-end-smoke`, `pr-agent`, `scope-guard`, `validate-benchmarks`,
`validate-claims`) use `pull_request` with **no base filter**. But **four of them**
(`census-watch`, `end-to-end-smoke`, `validate-benchmarks`, `validate-claims`) also carry a narrow
`paths:` filter, so they fire only on PRs touching those paths; only `pr-agent` and `scope-guard`
would fire on an arbitrary PR. Under branch-only/no-PRs, none of them fire either — so the
conclusion holds, but the sentence above originally checked the BASE filter and inferred the rest.
The other axis was `paths:`, and I named six workflows from one axis. Every
remaining route crosses shared authority: the hook symlink resolves to the COMMON hooks directory,
and a `push` branch pattern is repo-wide runner capacity.

**CORRECTION to that last line — it is true of the DEFAULT hook route only, and I had stated it as
an exhaustion.** There is a per-worktree route. Found by @agent-ec855d, then reproduced here in an
isolated 3-worktree repo before recording (git 2.53):

```sh
git config extensions.worktreeConfig true              # one shared bit, inert by itself
git -C <worktree> config --worktree core.hooksPath <dir>
```

Before the bit, `--worktree` refuses: `fatal: --worktree cannot be used with multiple working trees
unless the config extension worktreeConfig is enabled` (rc=128). After it: a failing `pre-commit`
in wt1 blocked its commit (**rc=1, count 1→1**), the sibling wt2 with no `hooksPath` committed
normally (**rc=0, count 1→2**), and `repo/.git/hooks` was **never written**. The isolation is real,
not shared.

**So the accurate exhaustion is: no route is free, and one route costs one INERT shared bit instead
of a behaviour change or repo-wide runner capacity.** That is a smaller and *different* decision
from the trigger one, not the same decision.

Design consequence worth stating plainly: a `pre-commit` running `tools/dispatch_key_check.sh` plus
the static half of the flag audit fails a regression **at the moment it is introduced, with the
author present** — instead of at a PR-to-`main` that never happens on this plane. For the specific
regression "someone added an id under a dispatch-keyed type", commit time is the better failure
point, and it is the failure point this branch's own §7 checks are shaped for.

Honest limits, so this is not oversold: it protects only the worktree that installs it — it cannot
stop another agent's commit and puts nothing on a PR, so it complements CI rather than replacing
it; it needs git >= 2.20; and `extensions.worktreeConfig` is still a write to SHARED config. Inert
by itself (it only permits per-worktree config; it changes no setting, and a sibling worktree
behaved identically), but shared, and therefore still an owner decision — the one-bit kind.

**"Inert by itself" was questioned and it holds — plus a hazard worth knowing before anyone flips a
repository format bit.** The objection is a good one: a repository format *extension* is
version-gated, and an unknown extension can make git refuse a clone outright, which would impose a
requirement on every other user of it. Measured:

- setting `extensions.worktreeConfig` does **not** bump `core.repositoryformatversion` (stays `0`);
- **v0 + an unknown extension → tolerated**: clone `rc=0`, local git operations `rc=0`;
- **v1 + the same unknown extension → `rc=128`**, on clone and on every local operation;
- and at v1 an **in-repo** `git config --unset` is itself refused (**rc=128, no repair** — version
  still 1, extension still present, `git status` still 128). The repo cannot be fixed from inside.
  Recovery is nevertheless **one command, from OUTSIDE**, because the fatal check lives in
  repository *discovery* and `--file` with an absolute path skips discovery entirely:

```sh
git config --file=/abs/path/to/repo/.git/config --unset extensions.<name>   # rc=0
git -C /abs/path/to/repo status                                            # rc=0 — repaired
```

  Verified end-to-end on a repo bricked a moment earlier. Recording the command rather than
  "hand-edit `.git/config`" is deliberate: that phrasing makes a correct rule sound unactionable,
  and it points someone who is already in a bad state at the one fix where a second, unrelated
  mistake gets made.

**A measuring artifact that nearly buried the above** (reported by @agent-ec855d; identical in shape
to my own two wrong probes in the previous bullet). Their first pass ran three candidate escapes
*in sequence* and read the config afterwards: version 0, no extensions — which looked like all three
had worked. Isolated, **only the `--file`-from-outside form repairs anything.** A chain reports green
because *some* element did the work, and the aggregate cannot tell you which one. **Probing several
variants in sequence is not probing; each variant needs its own bricked repo.**
- this repo is at **version 0**, so the bit leaves it at 0 and an older git tolerates it.

**So the rule is: never bump `core.repositoryformatversion`.** And the "needs git >= 2.20" limit is
precise about *what* it constrains — the FEATURE (`git config --worktree`), not the repo format, so
it does not gate anyone else's ability to clone.

**THE THREE ROUTES, RANKED BY COST — with the one that LOOKS free named explicitly, because it is
the one a future reader reaches for first.** (@agent-ec855d completed this enumeration; each cost
was checked here before recording.)

**I first wrote this section as a table of four routes to choose between. That frame was wrong, in
the same way as the errors above: I formed the set from one axis — *what can be ADDED* — and then
reported the space closed.** Every row I listed ADDS something (a config bit, a workflow step, a
trigger pattern, a user config). The highest-coverage enforcement point in this repo is not an
addition: it already exists, and it fell outside the frame for exactly that reason. So the table
gains the column that was missing — **coverage: who actually gets the signal** — and a row for what
is already running.

| route | coverage — WHO receives the signal | timing | cost to change |
|---|---|---|---|
| **common `post-commit` (the auto-push hook) — ALREADY INSTALLED** | **every commit, every worktree, every agent on this clone** — no opt-in | immediate, but **after** the commit exists → reports, does **not** prevent | changes what every agent sees on every commit — a shared-behaviour change, not an installation |
| per-worktree `core.hooksPath` (+ `extensions.worktreeConfig`) | only worktrees that ran the command | commit-time, author present → **prevents** | one inert shared bit |
| steps on `validate-claims.yml`'s existing daily schedule | everything that reaches `main`, whoever committed | ~24h late, `main` only | no new trigger; but the file must reach `main` first |
| `ci.yml` `push` branch pattern | pushes to `main` / PRs | on push | repo-wide runner capacity **per push** |
| user-level `core.hooksPath` (`~/.gitconfig`) | **every repo on the box**, including policy-covered ones | commit-time | **do not go here** |

**The already-installed row is not hypothetical — it already demonstrates the exact shape this
section needs**, and it has been running all day: `.git/hooks/post-commit` prints
`[auto-push] push FAILED (non-fast-forward or auth?)` followed by `exit 0   # never break the commit
itself`. Visible-but-non-blocking, in production, on every commit. It is also why the two "shared
authority" rows I had merged apart must be separated by **scope, not coverage**: the common-hooks
row is *this clone*; the user-level row is *every repo on the box*.

**Verified, because the frame error was found by someone re-enumerating rather than by me:** the
hook exists (1997 bytes, executable), resolves as the common hooks path from any worktree
(`git rev-parse --git-path hooks/` from my worktree → the shared checkout's `.git/hooks/`), needs no
`extensions.worktreeConfig` and no per-worktree config, and every failure branch exits 0.

**The second row deserves its precondition stated, because it is not free today.**
`validate-claims.yml` is a genuine host — `on: schedule: cron "17 4 * * *"` plus
`workflow_dispatch` and a `paths:`-filtered PR trigger, `runs-on: ubuntu-latest` (hosted, **not** the
self-hosted NPU box), **no job-level `if` guard**, default-ref checkout so on the schedule it tests
`main`. Five of its eight steps are already of exactly this kind (*"every published claim has a
validator"*, *"README statuses match"*, *"badges match benchmarks.json"*, *"retired claims must not
reappear"*, *"numbers.json matches the generator"*) — the dispatch-key check is the same species.

**But it validates `main`, and `main` only changes when a PR merges.** Verified: `tools/dispatch_key_check.sh`
is **ABSENT on `origin/main`** and present on this branch. So this route is *"land the file, then add
two steps"* — cheaper than a new trigger, not zero, and **unavailable until the merge happens**.

**Why the user-level route is the worst despite crossing no repo authority:** every agent on this
box runs as the same OS user, so a user-level `hooksPath` fires for **every committer in every repo
that user touches** — 13 checkouts here, including `~/projects/lemonade sdk/lemonade`, whose whole
policy is "touch nothing remote, work locally, no noise". A 1bit-specific hook firing in there is
the same scope error as the `/tmp` copy that silently went stale.

Verified **without writing any global config**, by supplying `GIT_CONFIG_GLOBAL` for a single
invocation: a user-level `hooksPath` pointing at a hook that exits 1 **blocked an unrelated repo's
commit (rc=1)**, and `git config --global core.hooksPath` was still unset afterwards. Current state
checked — **unset on this box, and no `~/.config/git/hooks` exists** — so this is a *do not go
there* note, not a pending change.

The pattern worth carrying past this section: **the route that costs nothing on the axis you are
measuring can be the most expensive on an axis you are not.** Repo authority was the axis in view;
blast radius across unrelated, policy-covered repos was not.

**WHY COVERAGE RANKS ABOVE TIMING, measured rather than argued** (@agent-ec855d named the axis
before the sentence was written). Per-worktree, in a 3-worktree repo: the installed worktree's commit was blocked (`rc=1`) and the hook's
message reached the author **on stderr** (git routes hook output there, not stdout — worth knowing
if you ever capture only one stream, as I first did). The sibling worktree with no `hooksPath`
committed normally (`rc=0`) — a **silent pass**: no hook, no message, no warning, output
indistinguishable from success.

**So these are COMPLEMENTARY with different coverage, not ranked.** A commit-time hook nobody
installed catches nothing and says nothing — the same "machinery exists, nothing runs it" state this
section exists to prevent, one level down, wearing the costume of enforcement. And the isolation
that makes the per-worktree route cheap is the **same property** that makes it opt-in: touching
nothing shared and protecting nobody who did not act are one fact, not two. Stated as a pair, the
recommendation is: *the worktree hook protects the author who installs it; the daily job protects
`main` whoever committed.*

**THE METHOD RULE this section earned — stated as a rule, because "be careful" failed twice inside
paragraphs recording its own failure.** Every claim of mine corrected in this work was a **closure
claim**: *"enforced"*, *"every remaining route"*, *"would fire on any base"* — each asserting that a
space is empty or full. My positive, bounded claims (diffs, counts, hashes, reproductions, "0 CI
runs on this branch") held all day. So:

- **sentences asserting that an option is absent, exhausted or impossible need an enumeration
  artifact written BEFORE the sentence.** Bounded positive claims do not.
- **put the selection criterion inside the sentence.** *"Six workflows **with no base filter**"*
  exposes the unqueried `paths:` field; *"six workflows"* hides it. The discriminating field is
  almost always the sibling key in the block already open — I stop one key short.

Both forms are cheap and neither is a resolution to try harder, which is the point: the failure
recurred *while* being careful about it.

**Its sibling, which is a DIFFERENT safety problem that I had been treating as the same one.** The
closure rule covers claims about **absence**. A second class covers claims of **success**, and there
the question is not whether the space was enumerated but whether the instrument **touched the thing
at all**. Four green readings in this work came from an instrument that did not:

- a probe that used a **known** extension while claiming v1 behaviour (git tolerates it, so it passed);
- a version flip attempted with a tool that **cannot run at v1**, so the "v0 is tolerated" result came
  from a repo that had never left v1;
- a chain of three candidate escapes whose aggregate looked green **because one element did the work**;
- a hook run whose message had never been written into the file, read as *"the author was told
  nothing"* when there was nothing to tell.

**A green result from an untouching instrument is worse than a red one, because it is
indistinguishable from a pass.** The defence is the same in both classes, and it is not care: make
the probe's condition explicit and check it separately — one variant per repo, one field per
sentence.

**THE MECHANICAL FORM of that second rule, and it is testable in one run** (@agent-ec855d):
**a success claim needs a NEGATIVE CONTROL** — the same probe, applied to an input you *know* is
broken, must fail. If it does not fail there, it did not touch the thing. A green result from an
untouching instrument is not a weak signal: **it is a signal about the instrument, not about the
world**, which is why it is worse than no result.

**Applied immediately to this section's own `0 runs` claim, which had none.** "0 runs on this branch"
was, as written, exactly the n=0 case a broken or mistyped query also produces. The control:

| query | result |
|---|---|
| `branch=main` — *must* be non-zero | **`total_count=6271`** |
| `branch=goal/one-registry-one-router` | `0` |
| `branch=no-such-branch-zzz` | `0` |

`main` non-zero establishes the **instrument** touches reality. But `0` is also what a *typo* returns,
so the claim needs a second control on the **subject**: `git ls-remote origin
goal/one-registry-one-router` → `b2d96bbfd…`, the branch exists. Two controls, two different facts —
the instrument works, *and* the subject is real.

**WHY AD-HOC PROBES ESCAPE THIS, which is the useful part** (@agent-ec855d): `tools/dispatch_key_check.sh`
carries a negative control — `rc=1` on a mutated copy, `rc=2` on an unparseable one — **because the
bad input is committed next to it and nothing lets me forget it.** F12b's F32 bound likewise carries
its *"legitimate dense F32 export → not flagged"* row, and was falsified by construction when the
untied-LM-head false positive surfaced. An ad-hoc probe has no such case, so **nothing forces it to
prove contact**, and the green reading arrives with no counterweight.

**Operational rule: if a probe's result is going to be written into this runbook, it needs a
bad-input run first — or it should be promoted to a check that carries one.**

**A THIRD CLASS, distinct from both: the PUBLISHED step and the EXECUTED step diverged**
(@agent-ec855d, about their own build recipe — and the driver was still on disk when I checked, so
this is documented rather than inferred). They ran the six TUs **plus a two-line driver they had
written ten minutes earlier**, and published the six TUs without it: **a real measurement inside an
unreproducible artifact.** The numbers were correct; the recipe could not link. Their diagnosis is
the part to keep: *"a driver I wrote ten minutes earlier got folded into 'the build' and dropped from
the retelling. A driver I had never seen would have been remembered."*

**That is why it is invisible from inside the sentence: the command reads complete, and the omitted
part is exactly the part the author did not need to be told.** Guard: **publish the command you RAN,
including the scaffolding you added** — and remember that neither re-reading nor a negative control
catches this, because the measurement really was valid. This class fails by **elision**.

**Its instance in this file, found by applying the rule to myself:** §6.1 and §7 invoked a bare
`1bit <sub>` while the actual artifact (`b/1bit`, `OUTPUT_NAME` of the `onebin` target) is stated only
in §1. A reader landing on either section — which is what a reviewer does — cannot run them. Both are
now qualified, and either section can be read on its own.

**The three rules, restated so they do not compete:** absence claims fail by **under-enumeration**
(→ enumerate the space); success claims fail by **non-contact** (→ a known-bad run); and transmission
fails by **elision of self-authored scaffolding** (→ publish the command you ran). The first two are
about whether the claim is true; **the third is about whether anyone else can get it** — and a true
measurement in an unreproducible artifact is worth exactly as much to the next reader as a false one. The n=0 family —
a blacklist that matches nothing, a guard that cannot fire, a route with no negative control —
belongs to the second, and it is exactly how F12b was found broken: *a mitigation that cannot fire on
its own motivating case is not one.*

**THE CHECK-SHAPE THAT WAS MISSING, worth stating once:** when this step was placed in `ci.yml`,
its **feasibility** was verified — source-only, no dependencies, seconds to run — but not its
**reachability**, i.e. that the trigger fires for the branch being pushed to. **Feasibility is not
reachability**, in the same way that a passive `PATH` probe is not launch proof. When adding a
check, verify both that it *can* run and that something *does* run it; the first is much easier to
test and will feel like enough.

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
