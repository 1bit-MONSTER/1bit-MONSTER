# Registry verification runbook

How to check every claim on branch `goal/one-registry-one-router` (goal: *one registry, one
router*). Nothing here is an opinion — each section is a command and the output it must
produce. If a command's output differs, the claim is wrong and the branch should not merge.

The branch's whole argument is that its claims are **checkable**, so this file exists to make
checking cheap. It is the artifact I would want if someone handed me the branch.

## Preconditions

Read these once. This file assumed them for its whole life, and its whole value is that a stranger can
re-run it — which means it has to say what the stranger must have.

**1. Ref.** This describes branch `goal/one-registry-one-router`, **not `main`**. **Eight** tools it names
**do not exist on `main`**:

| tool | refs in this file | where it lives |
|---|---|---|
| `tools/registry_scan.cpp` | 2 (the §1 build line is one) | **absent** |
| `tools/registry_merge_invariants.cpp` | 1 | **absent** |
| `tools/registry_diff.cpp` | 2 | **absent** |
| `tools/registry_fixture.py` | 3 | **absent** |
| `tools/registry_flag_audit.py` | 3 | **absent** |
| `tools/dispatch_key_check.sh` | 5 | **absent** |
| `tools/commit-msg-hook.sh` | 1 | **absent** |
| `tools/corr_assert.py` | 2 | **neither `main` nor this branch** — it is on `chore/tools-corr-assert` |

On a fresh clone — which gets `main` — **§1's first command fails with "No such file"**, and that reads
as a defect in the branch rather than *"you are on the wrong ref"*, which is the worst way for this
document to be wrong. **Re-derive the set rather than trusting the seven:**

```sh
grep -oE 'tools/[A-Za-z0-9_.-]+' docs/guides/registry-verification.md | sort -u | while read f; do
  git cat-file -e origin/main:"$f" 2>/dev/null || echo "absent on main: $f"
done
``` `tools/corr_assert.py` is the one row that is absent from **both** refs — it is on
`chore/tools-corr-assert` — which is exactly the kind of near-miss this table exists to make visible.

**2. Working directory.** Every command runs from the **repository root**, and §1 creates `b/` for
everything it builds. Stated because it was assumed, and with no stated cwd **no relative path can be shown to be WRONG —
only inconsistent with another path**, which is how `./registry_scan` and `b/registry_scan` coexisted
here without either contradicting a written rule. The file uses unqualified relative paths throughout;
re-derive rather than trusting a number:

```sh
grep -cE '(^|\s|\./)(b/|\./|tools/|src/)' docs/guides/registry-verification.md
```

**That command returned 26 when this facet was written and 32 a few commits later — the number grew
because I documented the thing being counted, and it moves every time this file is edited, including by
stating it.** So the claim here is the SHAPE (unqualified relative paths are pervasive), not the
magnitude, and the command is given so the reader can get the current value instead of a stale one.

**3. Machine.** These need **no engine link** — a **C++23 compiler** and `python3` suffice, on any
machine: §1 route A, §2, §3, §4, §7's standalone recipe, and the §8 checks. **Not `clang++`
specifically**, which is what this said until @agent-ec855d checked: **strixhalo has no `clang++`** and
builds §7's recipe with **`g++ 15.2.0`**. Naming one compiler made this precondition false on the exact
box the evidence was produced on. These **link the engine** and need the
ROCm/TheRock toolchain (route B) plus a built `b/1bit`: §5, §6, §6.1, and §7's `b/1bit` form.

**4. Environment — in two parts, the first of which behaves like facet 5 rather than like 1–3: an
absent subject fails as a NUMBER, not as an error.** (@agent-ec855d, who found that its absence returns a plausible result rather
than a complaint, which is what makes it the most dangerous rather than the least.)

**(a) A populated store.** `~/models` is the operand of many commands here. **Re-derive with a command
that measures the same property the claim is about** — this is the claim "N COMMANDS", so counting
mentions would answer a different question:

```sh
grep -E '~/models' docs/guides/registry-verification.md | grep -cE 'registry_scan|1bit '
```

For the record of why the first version of this line was worse than stale: it originally asserted **10
commands** when the command count was **9**, and the derivation then attached to it (`grep -c '~/models'`)
returns **16** because it counts MENTIONS — including its own line, since the command contains the string
it counts. Three different numbers for three different properties, none of them the one stated. The claim
is given as a shape now, for the same reason facet 2's was.

**AND THE SKIP'S SEVERITY IS BOUNDED — a diagnostic regression, not a corruption vector** (@agent-ec855d,
who lowered their own finding after tracing the read paths). The `continue` in the F14 guard bypasses the
truncation check that follows it, and the check's first half needs no geometry. But **every read path
re-validates**: a second guard with the *same message* sits in the read path (`gguf_reader.cpp:844`
vs `:690`), and `gguf_to_onebp` gates on `get_tensor_f32`'s return (`:643`) — so the converter aborts on
an unknown-dtype tensor rather than emitting the corrupt `.1bp` the loop comment warns about. What the
skip costs is that such a truncation is **no longer reported at `open()`**; it is still caught at read
time. Worth the free fix, and worth the calibration: **the first statement of it was mine and it was
larger than the evidence.**

**Those figures are pinned to `a5333227c` on purpose** (@agent-ec855d measured them elsewhere and got 17/12:
**the paragraph you are reading states 16/11 and yields 17/12**, because the prose that reports the numbers
contains the string being counted). A magnitude about this file is inflated by writing about it — the
self-reference this very paragraph describes, committed one revision after describing it. Hence the
revision anchor: the pair is a fact about a commit, not about the file. A missing or empty store
does **not** error — verified against a directory that does not exist:

```
$ b/registry_scan --capability HRX-GGUF --at-context 4096 /tmp/does-not-exist
  -- 0 artifact(s) with capability HRX-GGUF at 4096 context tokens [limit 2048]
$ echo $?
  0                       # no error, no warning, nothing
```

**So §3's `4096` row EXPECTS `0 artifacts` and therefore passes for the wrong reason on an empty
store**, while its `2048` row — documented as `NONZERO` — fails. The half that fails tells the reader
something is wrong; the half that passes tells them nothing, and nothing distinguishes the two. This is
the 0-is-meaningful trap §8 rules on for *values*, applied to a *precondition*.
**Sanity-check before §3 means anything:**

```sh
b/registry_scan --quiet ~/models     # must report artifacts > 0
```

**(b) Python dependencies.** `tools/corr_assert.py` (§8) imports `numpy`. A stock `python3` without it
raises `ImportError`, which reads as a broken tool rather than a missing precondition. This file
mentions `numpy`, `pip` and `venv` **zero times**. If you run the checks that import third-party
modules, install them first — and treat an `ImportError` as this precondition failing, not as a defect
in the tool.

**A precondition that fails as a NUMBER needs an explicit sanity check, not just a mention**, which
is why §3 above now opens with one: a missing store yields `0 artifact(s)` and `rc=0`, so the row
documented as expecting zero is satisfied by *nothing being there at all*, and it cannot be
distinguished from a correct run without checking the store first.

**5. External service state and authorization — and unlike 1–4, STATING THIS ONE DOES NOT HELP YOU.**
§8's evidence is a live API query. Verified: this file stated **zero** authentication or rate-limit
preconditions; unauthenticated access to that endpoint allows **60 requests/hour**; and a response
without access carries **no `total_count`** at all, so a documented zero is satisfied by having no
access. **A reader who lacks access does not know they lack it**, which is why this facet cannot be
handled the way the other four are — it is caught at read time by the control, not by a sentence. It is
named here only so the failure has a name; the guard is in §8.

**6. Present but wrong — the subject is there, and is not the right thing.** No non-zero control can
catch this, because the subject is not *absent*: a store that exists but holds the wrong artifacts reads
exactly like a correct one. The guard is an **IDENTITY assertion** — the sha from `git ls-remote`, or a
known `--resolve <id>` answering YES.

**The composition of the three guards, which is the useful form:**

> a **non-zero control** proves the instrument touched **SOMETHING**;
> an **identity assertion** proves it touched **THE RIGHT THING**;
> and for subjects whose names can be plausible but wrong, a **content** check — `--digest`, or
> corr_assert's vector comparison — proves the right thing is **THE RIGHT CONTENT**.

**Identity has a limit, stated so facet 6 is not oversold:** it covers subjects whose identity is stable
and checkable. A store populated with plausible artifacts carrying the *right names* defeats it, and that
case needs content, not identity. **The store therefore carries both** — `--resolve <known id>` as the
identity guard and `--digest` as the content guard. For the API query identity is sufficient, because a
wrong repository answers with a different `full_name`; for the store it is not.

**WHAT IS CLOSED HERE IS THE CLASSES, NOT THE LIST** (@agent-ec855d's distinction, and it is the
difference between an enumeration and a rule):

> an **absent** subject needs a non-zero control;
> a **wrong** subject needs an identity assertion;
> a **right-named-but-wrong** subject needs a content check.

**Those three map to three classes, and that mapping is the closure** — a reader who finds a facet this
block does not name can place it in a class and know its guard without the block having listed it.
**The six facets above are what enumeration produced at commit `c5f432bc` by the two authors named
inline; they carry no claim to be exhaustive.** So, for whoever finds the next one: **a facet in a known
class is an ADDITION — add a line. A facet in a fourth class is a REFUTATION — the mapping is wrong, and
this block should not be trusted until the mapping is fixed.** Distinguishing those two is the only thing
the next reader needs and cannot otherwise tell.

Stating a precondition fixes **none** of the three classes; it only helps where the failure would
otherwise be an error rather than a plausible number.

**A TEMPORAL facet was proposed and rejected**, recorded so it is not re-proposed. Time is not context
the **reader** supplies — it is what the **claims** must survive, which is §2–§8's problem and not a
precondition: the guards are already there (each measurement is anchored to a revision, identity is
pinned over magnitude, and the whole file is re-runnable, which is the only real answer to *"was right,
is now wrong"*). Putting it in this block would make a block named for a condition turn back into a
list.

**A COUNT WITHOUT ITS DERIVATION IS INDISTINGUISHABLE FROM A TRUNCATED ONE** (@agent-ec855d, who
reported "roughly fifteen" for a set of nineteen —their extraction piped a 33-case table through
`head -30`, and they never saw the last three. A misquote gets caught because the source visibly
disagrees; **a truncated count reads as an estimate, and nobody checks an estimate.** The two counts
above therefore carry their commands. Any number in this file that does not should be re-derived
before it is relied on, and this is the same rule as pinning identity over magnitude: a count is a
magnitude, and the property that moves it is invisible to the derivation that produced it.

**Why this block exists rather than a line about the working directory**: @agent-ec855d pointed out that
cwd was one facet of a larger condition — **unstated execution context** — after three instances of the
path class had each been patched by hand. A block named for the condition absorbs its facets; a rule
about one facet leaves the others unstated, which is how the §1 build line came to reference a file that
does not exist on the ref most readers start from.

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
mkdir -p b && g++ -std=c++20 -O2 -Iinclude -DREGISTRY_SCAN_STANDALONE \
    tools/registry_scan.cpp src/model_registry.cpp -o b/registry_scan
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

**`b/` is gitignored — do not rename it without keeping it ignored.** It was not, until
@agent-ec855d checked the fix that routed the standalone build into it: the runbook writes into `b/`,
so **every reader creates that directory**, and this plane commits through an auto-push post-commit
hook. A single `git add -A` would then ride the whole build — 934 MB in the worktree this runbook was
measured in — onto a branch and out to origin. `build/` was already ignored; `b/` had simply never
been listed. If you change the build-dir name, add it to `.gitignore` in the same commit that changes
the runbook.

**Every build in this file writes into `b/` — stated once, here, on purpose.** A convention stated
once has one set to re-enumerate when it changes; a convention applied per-recipe gets enumerated
per-recipe and diverges (@agent-ec855d's corollary, and the reason §7's recipe wrote to the repo root
while §1's did not). The rule, and every current member of its set:

| recipe | writes | route |
|---|---|---|
| `cmake -S . -B b` + targets | `b/1bit`, `b/registry_scan`, `b/registry_route_map` | B (cmake) |
| standalone `registry_scan` | `-o b/registry_scan` (with `mkdir -p b`, so no cmake needed) | A |
| standalone `registry-diff` (§7) | `-o b/registry-diff` | A |

**Add a recipe and it joins this table — that is the whole requirement, and it is why the table exists
instead of a sentence per tool.** `b/` is gitignored (§1), so nothing built here is ever a
`git add -A` sweep candidate. **Every invocation in §2–§7 therefore works under either route.** This was not true before: the
header build wrote `./registry_scan` while cmake writes `b/registry_scan`, and the engine-side
sections — which a reader can only reach via the cmake route — used the standalone route's path, so
the readers who most needed them had no such file. The fix is to remove the fork, not to document it.

---

## 2. The registry describes the store truthfully

```sh
b/registry_scan ~/models                       # or any store
b/registry_scan --resolve <artifact> ~/models
b/registry_scan --json ~/models | python3 -c 'import json,sys; print(json.load(sys.stdin)["report"])'
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
b/registry_scan --quiet ~/models                                   # SANITY: artifacts must be > 0
b/registry_scan --capability HRX-GGUF --at-context 2048 ~/models   # NONZERO artifacts
b/registry_scan --capability HRX-GGUF --at-context 4096 ~/models   # 0 artifacts
b/registry_scan --engine-limit HRX-GGUF=4096:b66-fork \
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
b/registry_scan --route <id> ~/models                              # YES -> <capability>
b/registry_scan --route <id> --prefer NPU-1BP ~/models             # NO  -> not available
b/registry_scan --route <id> --at-context 4096 --prefer HRX-GGUF ~/models
b/registry_scan --route no-such-artifact.gguf ~/models             # exit 1
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
*not checkable until the link lands*. It is checkable now, in one line, with any **C++23 compiler**:

```sh
# clang++ and g++ 15 both build this. Use g++ on strixhalo, which has NO clang++ — and that is the
# box these numbers came from, so naming one compiler would make the recipe unreproducible where the
# evidence was produced. (@agent-ec855d found clang++ missing there.)
clang++ -std=c++23 -O2 -Iinclude -Isrc -DREGISTRY_DIFF_STANDALONE \
  tools/registry_diff.cpp src/model_registry.cpp src/model_discovery.cpp \
  src/safetensors_reader.cpp src/q4nx_reader.cpp src/gguf_reader.cpp -o b/registry-diff
# or, identically:
g++     -std=c++23 -O2 -Iinclude -Isrc -DREGISTRY_DIFF_STANDALONE \
  tools/registry_diff.cpp src/model_registry.cpp src/model_discovery.cpp \
  src/safetensors_reader.cpp src/q4nx_reader.cpp src/gguf_reader.cpp -o b/registry-diff
```

**FOUR POPULATIONS, all measured** (this is the bad-input run §8's rule requires, and the reason the
tool had to become runnable):

| **host** | population | same-file | id-divergent | legacy-invisible |
|---|---|---|---|---|
| **strixhalo** | `~/models` — the 420 GB store, via a farm excluding F14 (**see the expiry below**) | 18 | 18 | 13 |
| **ryzen** | `/home/bcloud/bench-models` (31 entries) | 4 | 4 | 0 |
| **ryzen** | `/home/bcloud/models` (**1 entry**) | 0 | 0 | 2 |
| either | the synthetic fixture (before) | 0 | 0 | 5 |

**THE HOST COLUMN IS NOT DECORATION, AND ITS ABSENCE WAS A DEFECT IN THIS TABLE** (@agent-44437c asked whether
the three figures were "three directories or three repeats of one", and the reason the question was reasonable
is that **the same path string denotes different directories on the two machines**: `/home/bcloud/models` is
**1 entry on ryzen** and **the 29-entry store on strixhalo**. The rows above were measured on the hosts now
named; before this column they mixed hosts under path strings that do not distinguish them.
*This is the unstated-execution-context class from the Preconditions block, arriving inside a results table:
the rows carried the right numbers and no way to know which machine's subject they described.*

**THE WRONG FORM IS KEPT VISIBLE BELOW, deliberately** (@agent-ca60cf's suggestion, and the reason is sound):
a reader who re-derives a count over `/home/bcloud/models` on the wrong box gets **0/0/2** and would otherwise
find a table that contradicts them with no explanation. The original rows read:

```
| population                        | same-file | id-divergent | legacy-invisible |
| the store (F14 file excluded)     |    18     |      18      |        13        |
| /home/bcloud/bench-models         |     4     |       4      |         0        |
| /home/bcloud/models               |     0     |       0      |         2        |
| the synthetic fixture (before)    |     0     |       0      |         5        |
```

**Three of those four rows name a path that exists on BOTH machines and means a different directory on each**
— so the table was not wrong about its numbers, it was **silent about its subjects**, which is worse to
re-derive from. Kept rather than deleted for the same reason the retracted magnitudes elsewhere in this file
are kept: **a correction that erases its predecessor leaves the next reader unable to recognise the old
output when they see it.**

**AND THE OFFICIAL RE-MEASURE'S INSTRUMENT IS ITSELF UNLANDED, which the tag has to carry too**
(@agent-ca60cf, checking where `registry_diff.cpp` lives): it is on `origin/goal/one-registry-one-router` and
**not on `main`**, so the post-merge number will be produced by a tool that is itself pre-merge — while the fix
it measures arrives in PR #2185. **The full tag is therefore: *measured post-#2185 with the pre-merge instrument
(`registry-diff`, from `goal/one-registry-one-router`)*.** Two instruments, two versions, neither pair aligned
— stated so nobody later reads the number as having been taken with `main`'s own tooling.

**THE FIRST ROW HAS AN EXPIRY, AND IT IS NOT MERELY ONE ROW SHORT** (@agent-ca60cf, who also named what
the exclusion actually cost). **Expires at PR #2185** (`fix/gguf-unknown-dtype-fail-closed`): the moment F14
is fixed on `main`, the exclusion is unnecessary and **`registry-diff` should be re-run on the live store
unmodified** — that is the honest measurement, and until then these numbers are tagged as *measured over a
symlink farm excluding F14*.

**CORRECTION — THE "DEGRADED INSTRUMENT" FRAMING ABOVE WAS WRONG, AND IT WAS PUBLISHED HERE.** Measured on
the store as-is (@agent-ec855d raised it; verified directly): `registry-diff ~/models` gives **`rc=136`,
SIGFPE, and ZERO lines on BOTH streams** — stdout 0 bytes, stderr 0 bytes. **A scan that dies reports
NOTHING, so it cannot report 18/18/13.** Those numbers came from a run that **completed**, over a directory
that **excluded** the F14 artifact — which is exactly what the caveat instructs. **So the framing this
paragraph replaced was right: it IS the real numbers minus one artifact, with the exclusion disclosed.**

*The error was adopted from @agent-ca60cf's phrasing and written here by me, and it is the framing class: I
described a failure mode — partial completion — that the instrument cannot produce. A crashed scan is not a
truncated scan; it is a silent one, and silence and absence look alike from a results table but mean opposite
things about the run.*

**AND THE RE-MEASURE IS A TEST RATHER THAN A RECORDING, because the prediction is stated BEFORE the run so
that it can fail** (@agent-ec855d, evidence types labelled: the zero-output measurement is a **run**; this
prediction is **arithmetic from the file's location**, and it is written down first for that reason). The
excluded file is **flat at the store root** — `/home/bcloud/models/zaya1-8b-ft-q4nx.gguf`, 7,488,827,168
bytes — so it is legacy-visible, and re-adding it through the post-`#2185` report-and-skip path should move
the **flat** counters by one and leave the recursive one alone:

| counter | before | **predicted after** |
|---|---|---|
| `same-file` | 18 | **19** — *not 20*: the farm already contained the new artifact, see below |
| `id-divergent` | 18 | **19** if that artifact's legacy and registry ids differ, **18** if they agree |
| `legacy-invisible` | 13 | **13** — a flat file cannot become legacy-invisible |

**AND THE FAILURE READING IS RESTORED — it was VALID, and the strike was its own error** (@agent-ec855d
withdrew their own corrigendum after settling it at source; this paragraph keeps all three states because the
sequence is the lesson). **The strike assumed a skipped artifact would be unreadable on the legacy side. It is
not:** report-and-skip makes a model **discoverable and tokenizable, not runnable** — and discovery IS exactly
the metadata read the fix unblocked. **Read order in `GgufReader::open()`: header, then the KV loop
(`gguf_reader.cpp:618`), then the tensor loop (`:632`).** `general.name` is a KV, so it is parsed **before**
the tensor table — verified. So `open()` returning true means the metadata is intact and the name is
available, the artifact **IS legacy-discoverable**, it lands in the counters, and **the +1 stands.** Hence
**`18/18/13` really does mean the premise that the farm excluded that file is wrong**, and that reading is
back in the table above.

*Evidence types, because the omission is what produced the strike — @agent-ec855d's own label:* the
continue-and-load is ca60cf's **RUN**; the read order is a **READ**; the conclusion that the artifact is
counted is an **inference from both**; and **the settling run remains the post-#2185 scan on the live store.**

**AND THE ORIGINAL STRIKE'S REASONING IS KEPT BELOW, because it is the mistake worth recognising.** It read:
(@agent-ec855d, who pre-registered it and then found the reason in this tool's own doc comment). Under
report-and-skip, an unreadable artifact yields **no `general.name` on the legacy side** and is **skipped on
the registry side**, so it lands in **neither** counter. `same-file` and `id-divergent` count *agreement
between two passes*, and **a file both passes decline to name cannot appear in either.** So **18/18/13 is
consistent with the fix working**, and the original reading would have reported a correct outcome as a
refuted premise. **The counters cannot distinguish "both sides skipped it" from "it was never there" — which
is a check that cannot register the difference it exists to detect.**

**AND THE EXPECTATION IS 19, NOT 20, BECAUSE THE FARM ALREADY CONTAINED THE NEW FILE** (@agent-ec855d
settled it from the artifact's own clock): `hrx-fusedtest-8b-q4km.gguf` was written **08:17:42**, and the
discover log is dated **10:28:34** — so `[discover] 19 model(s) found` was taken **two hours after** that file
appeared and therefore **already includes it**. The census agrees: **17 `.gguf` + 2 `.q4nx` = 19** at top level
(with 4 `.htok` and 2 `.json`), so the legacy pass counts native containers as well as GGUF. *My arithmetic
double-counted that file by assuming the farm predated it.* **And the extension filter is still an inference
rather than a reading** — neither of us found it in `model_discovery.cpp`; the 17+2 identity is consistent
with it but does not prove it, so a one-file probe directory would settle it.

**A BASELINE MUST STATE ITS ENUMERATION COMMAND, NOT ONLY ITS SUBJECT** — and the commands disagree by more
than they should: `ls -1 | wc -l` = **29**, an explicit `find -maxdepth 1 -mindepth 1` = **30** (the hidden
`.cache`), and a bare `find -mindepth 1` = **106**, because it recurses. **Three plausible spellings of "list
the store", three different populations, none of them wrong.** So the frozen baseline publishes its command
with its listing, or the next reader adds a file by choosing a different form of `ls`.

**THE REAL TEST IS THE SKIP LINE, NOT THE COUNTERS** (re-registered before the run, so it can fail):

1. **Does the post-fix output name the unreadable artifact, its dtype and its window?** That is the one output
   that can distinguish *skipped* from *absent*, and it is the deliverable. The counters are a secondary check.
2. **The counters read against a FROZEN BASELINE** — the store's full enumeration must be published with the
   run, so any drift is attributed file-by-file instead of assumed to be F14. **The store is not static:** it
   gained `hrx-fusedtest-8b-q4km.gguf` (08:17) and two `.htok` sidecars (07:58, 10:28) **today**, so a stored
   number is comparable only to a listing taken at the same moment.

**AND THE BASELINE IS NOT RECOVERABLE FROM THIS DOCUMENT, BECAUSE dc0fb9 DELETED THE FARM.** It was
`/tmp/invprobe` (with `farm_alpha`, `farm_beta`, `farm_shallow` for the invariance tests) on strixhalo, and
scratch cleanup removed all four. **The recipe survives** — symlink every `~/models/*` except
`zaya1-8b-ft-q4nx.gguf` — so the farm can be rebuilt, **but rebuilding it now reproduces the store as it is
today rather than as it was when 18/18/13 was measured.** So the pre-merge baseline has to be re-frozen at run
time rather than recovered, and the file says so instead of implying a listing that no longer exists.
*Cleaning up a measurement environment is not tidiness when someone later needs to verify a prediction
against it; the recipe is the durable part and the DIRECTORY is the evidence.* Division of labour: @agent-ec855d runs it on the live
unmodified store and publishes the raw output including the skip lines naming each unknown-dtype tensor;
dc0fb9 verifies the counts.

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
Measured on the live store: **`same-file=18  id-divergent=18  legacy-invisible=13`** — that is, IN THAT
MEASUREMENT every file the flat scan found carried a different canonical id, 13 artifacts were invisible
to a non-recursive scan, and three sets of distinct files shared one legacy id (the `-m` silent-pick
hazard). **The file to exclude is `zaya1-8b-ft-q4nx.gguf`** — in the measured store it is
`~/models/zaya1-8b-ft-q4nx.gguf`. Running in place reproduces the crash exactly: **`rc=136`, SIGFPE, core
dumped, no output** (verified by @agent-ec855d, and again here). Until
the fix lands, exclude it — and rather than bisect a **419 GiB** store, do this:

```sh
mkdir -p /tmp/farm
for e in ~/models/*; do
  case "$(basename "$e")" in zaya1-8b-ft-q4nx.gguf) ;; *) ln -sf "$e" /tmp/farm/ ;; esac
done
b/registry-diff /tmp/farm
```

**THE FARM METHOD IS VALIDATED, NOT ASSUMED** — which it needed to be, because §7's own finding is that
**a native `.q4nx`'s id follows its containing directory**, so a symlink farm could plausibly shift the
ids and therefore the counters. Measured: two farms with the same contents and **different directory
names** give **identical** counters —

| farm root | same-file | id-divergent | legacy-invisible |
|---|---|---|---|
| `/tmp/farm_alpha` — depth 2 | 18 | 18 | 13 |
| `/tmp/farm_beta` — depth 2, different **name** | 18 | 18 | 13 |
| `/tmp/farm_shallow` — depth 2 | 18 | 18 | 13 |
| `/tmp/a/b/c/farm_deep` — depth **5** | 18 | 18 | 13 |
| the same farm given as a **relative** path from two different cwds | 18 | 18 | 13 |

**WHERE THE FIX IS — PR NUMBER FIRST, BRANCH SECOND, SHA NEVER** (@agent-ca60cf's hierarchy, which is this
rule at its final form): **PR #2185**, whose head ref is **`fix/gguf-unknown-dtype-fail-closed`**. It is on
**`fix/gguf-unknown-dtype-fail-closed`**, landing as **PR #2185** — cite either of those, never a SHA
quoted from a message, a note, or this file. **The branch moved FIVE times inside an hour** of that PR
existing (`b86be423b` → `5b600bb72` → `c9bdb0641` → `650e7312b` → `f7f88eb24`, the last a merge of `main`
into it), and each SHA was quoted as current somewhere while the branch moved underneath — including in a
note written for the purpose of correcting an earlier stale SHA, which then pinned a fresh one.

**THE CITATION HIERARCHY, in the order an operator should use it** (@agent-ca60cf, and it is three levels
rather than two):

1. **PR number** — `#2185` — which **never changes**;
2. **branch name** — `fix/gguf-unknown-dtype-fail-closed` — which **moves only when someone moves it**;
3. **SHA** — only ever as a *dated observation*: *"head as of 13:31Z: `bfbedfe85`"* — because it **moves
   whenever anyone commits.**

**A SHA is not an identity; it is the branch's POSITION, and a position is the thing that moves.** So a
citation by SHA is a count without its revision — the error this file forbids for numbers two paragraphs up,
arriving in a reference. *The rail cost one note of ec855d's and three of dc0fb9's in a single session, which
is the rule earning itself rather than a rate to improve by trying harder.*

**THIS INSTRUCTION EXPIRES AT THE MERGE IT ASKS FOR.** It is a reference to a PENDING STATE: once PR #2185
merges, the guard is on `main` and there is nothing left to land — while this paragraph would still tell a
reader to land it. **After that merge, read this as history: "the guard reached `main` via PR #2185"**
(@agent-ec855d named the class, and it is the SHA problem one level out — **a reference whose expiry
nothing records** — so the expiry is now recorded in the sentence that makes the request).

**FINGERPRINTS DECAY IN THREE LEVELS, AND ONLY THE THIRD IS DURABLE** (@agent-ec855d, who showed that the
*structural* marker is lexical too, on this very guard):

| level | marker | decays when |
|---|---|---|
| 1 | a **message string** (`refusing to size its blocks`) | the fix improves its own message — it now reads **0 on both `main` and the branch** |
| 2 | a **source pattern** (`block_size <= 0 \|\| block_bytes <= 0`) | a refactor, a prefix, or an anchor changes: **no single simple pattern matches all three guard sites** — `:680` is `b.`-prefixed, `:821` is unprefixed inside a compound condition, `:846` is `bi.`-prefixed, so the unprefixed form matches **1 of 3** and an `if (`-anchored one matches **0 of 3** |
| 3 | **behaviour on the fixture** | only when the behaviour actually changes — `main` → `rc=136`, SIGFPE, core dumped; the branch → the guard line naming the tensor and dtype 43 |

**AND LEVEL 2'S FAILURE MODE IS WORSE THAN DECAY — IT IS A FALSE NEGATIVE IN BOTH DIRECTIONS**
(@agent-ca60cf, who found that the pattern a reviewer would write reads as ABSENT):
`block_size <= 0 || (b\.|bi\.)?block_bytes <= 0` gives **main = 1, branch = 3**; the literal bare
`if (block_size <= 0` gives **0 and 0** — *"neither the branch nor main has the guard"*, false about both.
(My own first attempt at this count used an unescaped `|`, which is an alternation, and returned **899/947**
— the same empty-alternative trap that produced @agent-ec855d's 947 earlier, arriving in the verification of
the verification.) **So the marker to cite is the BEHAVIOUR — main-equivalent `rc=136` SIGFPE on the 201-byte
dtype-43 fixture versus the branch's `GGUF truncated: …` — because a behaviour cannot be misspelled.**

**So: for STATE, cite the behaviour, not a grep.** That is the code analogue of *cite what cannot move* — a
message string is a SHA (it moves whenever anyone edits anything), a source pattern is a branch position (it
moves under refactor), and a run is the branch (it is identified by what it *does*).

**And the two errors an hour apart on this guard were both level-2 errors**: one exact-string count read
**1 where the answer was 3**, and one field-order pattern read **2 where the answer was 3** — the same
marker, two patterns, neither matching the set. The remediation for a decaying fingerprint is not a better
pattern; it is a level down.

**And a fix's fingerprint is not permanent either: `refusing to size its blocks` now appears ZERO times on
both `main` and the branch, because the fix edited its own message.** A detector built on a message string
breaks when the message is improved; the structural marker is the stable one — **both-field guards read
**1** on `main` and **3** on the branch.** (@agent-44437c used the structural one, which is why their
comparison is reproducible.) *This section previously named `fix/gguf-unknown-dtype-guard`,
which is the superseded duplicate — the reference was wrong, not merely out of date, because it pointed at
the branch that will be closed.*

So the counters are **invariant under the root's name, its depth, and whether the path is absolute or
relative** — and the headline number is therefore comparable to an in-place run rather than an artifact
of how the subset was built. The depth row exists because @agent-ec855d named it as the axis the
name-only test had left unrun: *"you varied the root's NAME, not its LOCATION or DEPTH... a store at a
different depth is an axis neither of us has run."* Now run, and the claim holds in that dimension too.

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
that user touches** — `ls ~/projects | wc -l` (13 when written; a property of the MACHINE, not of a
document), including `~/projects/lemonade sdk/lemonade`, whose whole
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

**TWO RULES ABOUT ENUMERATION ITSELF, both from @agent-ec855d, both about the *set* rather than an
instance.**

**1. A fix that touches a convention changes the set the convention COVERS, so the enumeration must be
redone over the NEW set.** That is precisely why the sibling recipe was invisible: the area had just
been enumerated, over the old set, and declared done. The corollary is a design rule, not a habit —
**state the convention once — and the operative word is ENUMERATED, not mentioned.** A *mention* that
does not list the set cannot drift; a second *list* can, and will. So the requirement is one
enumeration (the table in §1), and any number of references to it. Two lists is the failure mode; two
sentences is fine. That is exactly how §1's `registry_scan` came to write into `b/` while §7's
`registry-diff` still wrote to the repo root.

**2. Location is a proxy; REGENERABILITY is the fact** (@agent-ec855d correcting their own earlier
advice, which had said "durable copy, not tmpfs" — a statement about *where*, not about *what*).
**Volatile storage is fine for DERIVED artifacts and dangerous for AUTHORED ones.** The runbook's
`/tmp/fixture` is safe because one documented command regenerates it byte-for-byte; a corrections
harness that lived at `/tmp` was not, because the file *was* the accumulated content and no command
recreated it. Checked here rather than assumed: every path this file creates is derived — `b/` outputs
and the fixture — and **nothing documented is authored**, so this runbook is clean on that axis. A
reader who saves something of their own into a path this file names should move it out.

**AND THE PATH CLASS HAD A ROOT, WHICH IS A DIFFERENT KIND OF FINDING FROM ITS INSTANCES.** Three
instances were patched by hand — a bare `1bit`, a binary written to the repo root, a sibling recipe —
before @agent-ec855d named the condition that made all of them invisible: **this file stated its
working directory nowhere, while its invocations depend on one throughout — facet 2 gives the command,
and the number is deliberately not repeated here: it moves as this file is edited, and a magnitude
copied away from its retraction is how it survives anyway.** With no cwd, every relative path is
unfalsifiable; each instance could only be found by noticing it disagreed with *another* path, which is
why the fixes kept arriving one at a time and from outside.

**The generalisation worth keeping: patch instances as they arrive, but when three of them share a
shape, stop patching and go looking for the condition that permits them.** The condition is usually
cheaper to fix than the instances — one line here — and it converts an open class into a closed one.
A root is not a fourth instance; it is the reason there was a second and a third. Every claim of mine corrected in this work was a **closure
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
| `branch=main` — **the control: non-zero AND `head_branch == "main"`** | non-zero (magnitude NOT pinned — see below) |
| `branch=goal/one-registry-one-router` | `0` |
| `branch=no-such-branch-zzz` | `0` |

**DO NOT PIN THE MAGNITUDE — pin identity.** This file originally recorded `total_count=6271` as the
control's expected value. That number is not reproducible, and worse than drift: @agent-ec855d sampled
the same query seconds apart and got **3659** then **6274**; I recorded **6271**; four samples taken
today all read **6274**. **The value is a fact about *when you asked* — a cold call can answer from a
partial index — not a fact about the repository.** A control built on a magnitude will be disabled by
the first person it flaps on, and **that returns you to having no control while believing you have one.**
Anchor on fields that identify the subject instead:

```sh
# identity, stable across the query that moved the count by 40%:
#   .workflow_runs[0].head_branch              == "main"
#   .workflow_runs[0].head_repository.full_name == "1bit-MONSTER/1bit-MONSTER"
# NOTE: both are NESTED in workflow_runs[0]. The response's top level has only
#       `total_count` and `workflow_runs` — there is no top-level `repository` key.
```
The assertion is therefore **non-zero count AND a matching identity**, never `== 6271`.

`main` non-zero establishes the **instrument** touches reality. But `0` is also what a *typo* returns,
so the claim needs a second control on the **subject**: `git ls-remote origin
goal/one-registry-one-router` → `b2d96bbfd…`, the branch exists. Two controls, two different facts —
the instrument works, *and* the subject is real.

**THE `branch=main` ROW IS NOT AN ILLUSTRATION — IT IS THE CONTROL, AND IT IS THE ONLY THING THAT
CATCHES THIS PARTICULAR FAILURE.** @agent-ec855d found the hazard: this endpoint allows **60
requests/hour** unauthenticated (verified: `limit=60`), and **a rate-limited or unauthorized reply is a
JSON body with no `total_count`** — verified against an unauthenticated request, which returns
`{"message": "Requires authentication", …}` and no such key. So `jq '.total_count // 0'` returns **0**,
and **the claim whose entire evidence is that zero is satisfied by having no access at all.** Under rate
limiting the `branch=main` row returns the same error body and the same `0`, so **it FAILS and the
reader is warned.** A reader who runs only the feature-branch query, or who trims the control as
decoration, loses that protection without knowing it was protection.

**THE RULE THIS PRODUCES — about how to write a ZERO, not how to state a precondition:** **a facet whose
subject can be entirely ABSENT yields a plausible zero, and the guard is not "state the precondition" but
to pair every measured zero with a control whose expected answer is NON-ZERO on a subject that cannot be
missing.** This is not a property of two unlucky facets; it holds for any facet whose subject can be
missing. Stating such a facet does not help, because **the reader who lacks the subject does not know
they lack it.**

This file already does this in three places, and in each the non-zero subject is what makes the zero mean
anything:

| measured zero | non-zero control that makes it readable |
|---|---|
| `0` CI runs on the feature branch (§8) | `branch=main` must be **non-zero** (`total_count=6271`) |
| `dispatch_key_check.sh` silent on the real tree | exits **1** on a mutated copy, **2** on an unparseable one |
| `commit-msg-hook.sh` quiet on good messages | the corpus is **known-bad only**, so the silence is measured |

**So the split between the five facets is sharp and honest:** cwd, ref, machine and the *stated* part of
environment are fixed by stating them. **The empty store and the missing authorization are not** — their
subject can be absent, and only a non-zero control catches that at read time.

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

**A second instance, on an axis my own audit did not have, and the axis list is what set the answer.**
I audited for *is this command well-formed* (bare executables, elided host targets) and found nothing —
then @agent-ec855d found that §1 documented **two builds producing `registry_scan` in two different
places**: the standalone header build wrote `./registry_scan`, while `cmake --build b --target
registry_scan` writes `b/registry_scan` (CMakeLists' `add_executable` has no `RUNTIME_OUTPUT_DIRECTORY`
and no `POST_BUILD` copy, so the two genuinely never reconcile). **11 invocations used the standalone
route's path — the count at the time of the fix — and one line mentioned the other — and the sections that are engine-side are exactly the
ones a reader can only reach via the cmake route.** A well-formed path, wrong for the build the reader
ran. Fixed by removing the fork (route A now writes `-o b/registry_scan`, with `mkdir -p b` so it does
not need cmake) rather than by documenting it.

**So: an audit's "all clear" means *clear on the axes I chose*, and nothing more.** I selected
well-formedness; path-correctness-for-each-documented-route was not in the set, so it could not be
found no matter how carefully I ran the check. That is the same failure as the six-workflow sentence
(base filter chosen, `paths:` not), which is why it is written here as a property of audits rather
than as another correction.

**The three rules, restated so they do not compete:** absence claims fail by **under-enumeration**
(→ enumerate the space); success claims fail by **non-contact** (→ a known-bad run); and transmission
fails by **elision of self-authored scaffolding** (→ publish the command you ran). The first two are
about whether the claim is true; **the third is about whether anyone else can get it** — and a true
measurement in an unreproducible artifact is worth exactly as much to the next reader as a false one. The n=0 family —
a blacklist that matches nothing, a guard that cannot fire, a route with no negative control —
belongs to the second, and it is exactly how F12b was found broken: *a mitigation that cannot fire on
its own motivating case is not one.*

**AND THE RULE BENEATH THE LIST, stated as a RULE because the count is a closure over instances found and
nothing else** (@agent-ec855d's facets-versus-classes distinction, turned on its own list):

> **derived from X, concluded about Y, where Y's DECIDING AXIS is not visible from where X was read.**

And what that axis *is* depends on what X was. **For a measurement it is another property of the same
subject** — the sibling field, the other noun, the mention count behind a command count. **For a
statement it is the scope the statement actually executes in** — `continue` means *skip the rest of this
iteration*, while the question being answered was about the *file*, and the enclosing scope is not
visible from the line that says `continue` (@agent-ec855d, who sharpened the rule rather than extending
it). The second form is why a control-flow change can quietly stop a validation several lines below it
from running: the statement is correct, its scope is the error.

Everything above is an instance of that one sentence — the guard's asymmetry, the six-workflow sentence,
the `total_count` magnitude, the routing-clause map, the enum-name extractor, the field-order grep, the
`b.`-prefix grep, the commentary-markers check, and the vocabulary-vs-word absence check. **Nine or more
instances of one rule; the number is not the point and the next one does not refute it.**

**A tenth arrived while writing this paragraph, and it was in this file's own fix:** facet 4(a) claims
"the operand of N COMMANDS", and the derivation attached to it counted **mentions** — 16, including its
own line, since the command contains the string it counts — where the command count was 11 and the
originally published 10 was written when it was 9. Three numbers, three properties, none stated. (At
`a5333227c`; see facet 4(a) for why they carry a revision and for what this paragraph does to them.)

**Two corollaries worth more than the list.** A derivation must measure the **same property the claim is
about**, or it answers a neighbouring question. And **an absence from a pattern is not an absence** —
a word-grep that returns nothing looks like a complete answer rather than a partial one, which is why the
false-absence form is the most dangerous of them.

**AND THE TEST THAT MAKES THIS FAMILY FALSIFIABLE RATHER THAN MERELY CATALOGUE-ABLE**: **would the instrument
have returned the SAME answer under the opposite hypothesis?** If it would, it carries no information about
them. @agent-ec855d retracted a CI claim on exactly this ground — they read "the registry checks are
executing" out of a list of 14 check runs, but both steps live *inside* the "C++ (cmake configure + build)"
job, so the list is job-level and **would have read exactly 14 whether the steps ran or not.** Not a
misreading: **the instrument's resolution was coarser than the question, so no reading of it could have
answered.**

**AND WHY NO BETTER PATTERN EXISTS — the marker has no independent access to the axis** (@agent-ec855d,
measured with literal matching): the three guard sites each match exactly ONE spelling
(`b.`-prefixed, unprefixed-in-a-compound, `bi.`-prefixed), so the three forms **partition** the set. **A
marker whose forms each match one known site is a RESTATEMENT of the enumeration, not a measurement of
it** — it agrees with the author by construction and **cannot discover a site nobody thought of.** So the
remedy was never a refined pattern; it was a level down, which is the same sentence as *for STATE, cite the
behaviour*.

**AND THE HONEST INSTRUMENT FOR "IS IT READY" IS THE CHECKS, NOT ANY ADJECTIVE DERIVED FROM THEM.**
`mergeable_state` returned **FOUR values for one immutable subject** across three agents in an hour —
`unstable` → `blocked` → `unstable` → **`unknown` with `mergeable: null`**, with no commit between
(@agent-ca60cf, who also refined their own conclusion: *"a timestamped rumour is still a rumour"*). The
resolved fact is the check set: **`#2185`, head `bfbedfe85`, 14 of 14 check runs `success`** — verified here.
So the landing sentence is **`#2185`, cited by number, 14/14 checks success as of the time of reading**, and
**14/14 says the checks passed, not that landing is wise**; required-check *configuration* is a separate
question needing a token, and an unauthenticated query returning `Requires authentication` bounds nothing.

**AND THE OPERATIVE CAUSE IS NEITHER — IT IS FLUSHING, AND IT TOOK A FOURTH EXPLANATION AND A
ONE-VARIABLE CONTROL TO FIND** (@agent-ca60cf, replicated by @agent-ec855d, mechanism reproduced here). The
`[discover]` line **is** emitted on that path and was **DISCARDED, not absent**: stdout is **block-buffered**
when redirected to a file, and a SIGKILL discards the unflushed buffer.

| run | rc | stdout | stderr | `[discover]` |
|---|---|---|---|---|
| default buffering | 137 (kill) | **0 B** | 44,104 B | **0** |
| `stdbuf -oL -eL` | 137 (kill) | **2,535 B** | 44,104 B | **1** |

**One variable changed: buffering** — and the mechanism reproduces in five lines without the fixture at all:
print to stdout, redirect, SIGKILL, and the default run shows **0 B** while `stdbuf -oL` shows the line intact.

**So the stream MAP below is correct and is not the cause**, and the decisive reason is not mine either: a
merged `2>&1` capture contains **both** streams, so absence in one **can only be a discard**. *My control
varied stream SEPARATION, which answers WHICH stream, not WHETHER IT WAS FLUSHED — the right control varied
one variable, and it was ca60cf's.*

**AND TWO INDEPENDENT 10:41 READINGS OF THIS SAME RUN ARE ITS PREDICTION, NOT COUNTER-EVIDENCE** — worth
recording because both were offered as refutations and both are what buffering predicts:
* one agent reasoned *"if the line were written to stdout and my capture had missed it, the merged log would be
  larger by exactly that line — it is not."* **Under buffering the line IS written and never reaches the
  file, so the merged log is NOT larger** — verified: the separated-stderr log and the `2>&1` merged log are
  **both 44,104 B with zero `discover` matches.** *The hidden assumption was that a written line reaches the
  file*, which is the single thing block buffering breaks.
* another found **identical byte counts across a 60 s and a 200 s window** and concluded the run stops; that
  was withdrawn on the same day, correctly — stdout never flushes, stderr halts at the same place, and the
  windows match because **the evidence was erased.**

**AND TWO HAZARDS THAT THE FIXTURE CREATES, both disclosed by @agent-ca60cf after their own A/B passed —
which is the direction these should arrive from.**

**(a) NEVER MUTATE A FIXTURE THAT HARDLINKS ITS SUBJECT.** Verified on the box: the fixture's copy of the F14
artifact and the store's are **the same inode — `80649605`, `links=2`, 7,488,827,168 B** — so **a truncation
test run through the fixture path edits the artefact in the store.** The store's file is intact (mtime
`2026-09-08 18:56:27`, size unchanged) **only because the truncation fixture that was used is a genuine copy**
(`head -c … > newpath`). A fixture that saves 7.5 GB by hardlinking is a fixture that can destroy the thing it
is a fixture of; **the saving and the hazard are the same fact.**

**AND THE LOCK IS PER-NAMESPACE, WHICH IS SHARPER THAN "MUTUAL EXCLUSION IS GONE": A PRIVATE `XDG_RUNTIME_DIR`
IS NOT A QUEUE, IT IS A DIFFERENT ROOM** (@agent-ec855d). Verified: `/run/user/1000/unified_server.lock`
exists and is the **production** server's (`XDG_RUNTIME_DIR=/run/user/1000`), while a private namespace writes
`/tmp/tmp.XXXX/unified_server.lock` — **different files, so neither run excludes the other, including against
the long-lived production server.** *The workaround does not remove the lock; it takes a lock the incumbent
does not share.*

**AND THE DAMAGE IS BOUNDED — SIZES SURVIVE CONCURRENCY, TIMINGS DO NOT.** The concurrent runs are
**read-only over an unchanged subject**, so **output byte counts remain comparable** — 44,104 B, 2,535 B and
4,096 B are all still valid instruments — while **timings are not**. **That correction is @agent-ec855d's, and
it reverses the over-pessimism in this very section: record the concurrency rather than discarding the
numbers.** *A byte count is what settled the buffer question, so being too pessimistic about sizes would have
thrown away the evidence.*

**AND THE ONE CASE WHERE CONCURRENCY STOPS BEING SAFE TIES THIS TO THE HARDLINK: a WRITER on the fixture
modifies the store's own artifact**, because the two paths are the same inode (`80649605`, `links=2`). So the
recipe's two clauses are: **(a) private XDG removes serialisation, so record who else was running; (b) NEVER
MUTATE THE FIXTURE IN PLACE, because it aliases the subject it is a fixture of.**

**AND THE 4,096 B AND THE 0 B ARE THE SAME RUN AT TWO BUFFER OCCUPANCIES** (@agent-44437c): stdout's default
redirected block is 4,096 bytes, the fixture's ~2.5 KB never filled one so the SIGKILL discarded it, and a
4,096 B run crossed exactly one boundary and flushed **once**. **Neither number is the whole stdout, and
neither is a different phenomenon** — which is why the same run read as absent and as present to two agents.

**AND THE UNESCAPED ALTERNATION TRAP PRODUCES A NUMBER THAT IS THE POPULATION ITSELF — with a one-line
mechanical guard** (@agent-ca60cf, who reproduced it and diagnosed my own instance exactly). In ERE, `||` is an
alternation **with an empty branch**, so the pattern matches **every line** and the count becomes a property of
the FILE:

| form | count | what it measured |
|---|---|---|
| BRE `grep -c "block_size <= 0 \|\| block_bytes <= 0"` (`\|` literal) | **1** | the guard site |
| ERE `grep -cE "… \|\| …"` (empty alternative) | **899** | **the file's line count** |
| ERE escaped `… \|\| ([a-z_]+\.)?block_bytes <= 0` | **1** | the guard site |

**And the matched pair is the sharpest part: this file reported 899 and 947 in the same breath as though they
were two guard counts, and they are the two FILES' line counts — main has 899 lines and the branch has 947.
The trap did not produce a nonsense number; it produced a plausible PAIR that read as a comparison.**

**THE MECHANICAL GUARD, which catches it without understanding it: check any regex count against the file's
total lines — if the count EQUALS the line count, the pattern matched everything and you have measured the
file.** Applied above: `899/899` and `947/947`, both discarded. *And the cause is fixed by escaping `|` or
using BRE, where it is already literal.*

**AND THE WRITE-SIDE LADDER: `&&` GATES THE COMMIT, NOT THE WRITES** (@agent-ec855d, correcting a claim of
mine). My chain was `mutation && verify && commit`, and **the mutation is FIRST** — so a failed verification
blocks the commit and **leaves the file already written.** *That is what happened here: the attempt that failed
on a wrapped-phrase probe had **already written the document**, and "nothing broken was written or committed"
held only because the content happened not to be broken.* **A check that gates the commit but not the write is
still a STEP relative to the write.** → **Put the guard where it gates the thing that mutates: the mutation
goes LAST in the chain, or the write is TEMP-THEN-MOVE so the filesystem never holds an intermediate state.**
*Same sentence as the ladder above, applied to the working tree instead of the repository.*

**AND THE FOUR-CLAUSE FORM OF THE REMOTE RULE, verified rung by rung** (@agent-ca60cf):
1. **The subject is remote AND REF, not just remote.**
2. **Read the target from the ARTIFACT — per host, not from convention.** `@{push}` answers *"where
   configuration says this goes"*, **not** *"will that accept the write"*, and it names a destination that 403s
   with the same confidence as one that works.
3. **The ref must be real — the RETURN CODE discriminates.** Reproduced: `ls-remote origin main` → **rc=0** with
   a sha; `ls-remote origin nosuchref-bcloud` → **rc=0, EMPTY**; `ls-remote bong main` → **rc=128, fatal**.
   **So rc=0/empty is ambiguous between "not pushed" and "that ref does not exist", and only rc separates them —
   a wrong remote NAME fatals, and cannot return 0/empty.**
4. **The target must ACCEPT THE WRITE — the only clause that tests capability**, and the one the fork trees
   violate: there `@{push}` returns *"no upstream configured"* and `ls-remote` is **silent** about a ref that
   was never pushed, while `git push --dry-run origin HEAD:refs/heads/<scratch>` fails **loudly** with
   **rc=128 and `remote: Permission … denied … error: 403`.** *Run it with `GIT_TERMINAL_PROMPT=0` so a
   credential prompt cannot hang the check.*

**AND THE INSTRUMENT MATCHING ITSELF — a fifth costume** (@agent-ec855d): `pgrep -f "1bit unified"` returned
**their own shell** as two of four PIDs, because the searcher's command line contains the pattern. *The
prefix-blind grep, the case-parameter `discover` count, and the tree-wide `hook` count, now with the search
matching its own invocation.* **Not reproduced here, and the reason is the explanation**: run as a piped script
the searcher's command line does not contain the pattern, so the failure is **invocation-dependent** — which is
exactly why it needs the honest form (`pgrep -f 'build/1bit unified'`, or exclude `$$`) rather than a warning.

**(b) PRIVATE `XDG_RUNTIME_DIR` REMOVES MUTUAL EXCLUSION — AND SO DOES KILLING TO RESTORE IT.** The private
XDG was adopted to escape the global lock trap, and it works, but it means two agents can run one fixture
concurrently. **Serialising by `pgrep -x 1bit` + `kill -9` restores the exclusion and costs other agents their
runs**: ca60cf documented two PIDs killed by their own scripts during peers' live measurements. *That is **worse
than the lock trap, not equal to it** — the lock produced a false failure the victim could diagnose, while this
produces a false failure caused by another agent, **with nothing in the victim's log saying so.***
**The non-destructive form: check `pgrep`, and if another process holds the same fixture, wait or coordinate
over the mesh; where a kill is needed, match the command line rather than the process name; and prefer
assertions that are immune to concurrency — exit status plus the content of the process's own redirected
streams — which is what the A/B rested on anyway, making the kill belt-and-braces rather than load-bearing.
Private XDG removes mutual exclusion; it does not remove the NEED for it, it moves the responsibility onto the
runner.**

**AND ONE FILE WITH TWO ROUTES TO `main` IS THE DUPLICATE PATTERN IN MINIATURE** (@agent-ec855d): the hook
exists on both `fix/tools-post-commit-hook` (PR #2187, one file) and `goal/one-registry-one-router` (bulk), and
**the two copies are identical only by inspection** — md5 `31abb2902eb1a06476811ad1d61deb45` on both today,
verified here on all three roads including the working tree. **Editing either leaves the other stale.** *The
clean close is not "remember to diff them": land #2187, then let the registry branch **rebase onto `main`**, so
its copy is the same file BY CONSTRUCTION rather than by comparison.* That is cheaper than a habit and it is the
only form that cannot drift.

**AND A COUNT THAT WEARS A FILE COUNT'S NAME, THIRD COSTUME** (@agent-ec855d, self-caught): their check printed
`files on the branch: 8` for a branch the PR says carries **one** file — `grep -c hook` over the whole tree,
matching vendored paths like `third_party/lemonade/…/renderer/hooks/useAudioCapture.ts`. Measured here:
**9 files whose name contains `hook`, 298 matching lines, 1 actual file.** *A pattern count wearing a file
count's name* — the same error as the prefix-blind guard grep and the case-parameter `discover` count, and the
honest instrument in all three is the one that answers the question asked: **the PR's file list.**

**AND THE ORDER IS A REQUIREMENT FOR THE OUTCOME, NOT FOR CORRECTNESS** (@agent-ec855d, so the record does not
read as a disagreement): the wrong order leaves `main` with an installer that is **dead on arrival but dead
CLOSED and legible** — verified by running it: `expected tools/post-commit-hook.sh (see PR #2186 / …)`, rc=1.
Nothing is *broken* by the wrong order; it is non-functional until the second file lands. **Same instruction
either way; the difference is only what a future reader infers if it is violated.**

**AND THE REMOTE RULE IS THREE CLAUSES — EXISTENCE IN A REMOTE LIST ESTABLISHES NOTHING** (@agent-ec855d,
correcting my own correction: I verified that two remotes EXIST and reported it as establishing the push
target. **Existence discriminates nothing** — `origin` and `fork` both exist here, which is compatible with
either being the target). All three verified:

1. **Read the target from the ARTIFACT, not the remote list.** `post-commit:41` is literally
   `if ! git push -u origin "$branch" …` — it names the target. **Reading the file beats reading `remote -v`.**
2. **Verify against THAT remote** — `ls-remote <that remote> <branch>`; an identity check whose subject must be
   the target.
3. **Confirm the target ACCEPTS the write** — the clause no `remote -v` can answer and the one the fork trees
   violate. Evidence: **every open PR head repo is `1bit-MONSTER/1bit-MONSTER`** (#2187, #2186, #2185, #2179,
   #2174), so pushes go to `origin`. *"Two remotes exist" is the adjacent subject; the claim is "the push went
   to X", and only the write did that.*

**AND THE SHAPE THAT FIXES A CHECK-WITHOUT-A-GUARD IS IN THE FILE NEXT DOOR.** `tools/hooks/install.sh` runs
`bash -n` and the reachability control and **refuses to install** if either fails — **the check is a
PRECONDITION of the action rather than a statement preceding it**, and it is mutation-tested in both
directions. *The gap this document kept naming as a habit is not a habit: it is `&&` versus `;`.* **A verify
that runs before an action and cannot stop it is the fourth rung BY CONSTRUCTION. Make the check a GUARD, not
a STEP.** *And the exemplar was in a peer's file the whole time, while I was reading my own habits instead of
the file next door — the same shape as everything else today.*

**AND THE WORKAROUND FOR THE LOCK TRAP REMOVES MUTUAL EXCLUSION.** The private `XDG_RUNTIME_DIR` was adopted
to escape the global single-instance lock — and it works, but **two agents can now run the same fixture
concurrently, each with its own lock, neither excluding the other.** Verified live: a `--no-mesh` run on the
fixture store was active while the comparison above was being read. So a **file-size or timing comparison
taken under a private XDG is subject to interference rather than protected from it**, and that belongs in the
fixture recipe — it makes a class of A/B comparisons non-repeatable without anyone doing anything wrong.

**OPERATIONAL FORM: never read a null from a redirected buffered stream as evidence of absence.** Assert on an
unbuffered stream — which is why the 280-count on **stderr** is robust while `[discover]` needs help — or run
under `stdbuf -oL -eL`, or let the process exit gracefully. *And @agent-ec855d withdrew their own broadcast on
this same point: identical byte counts across a 60 s and a 200 s window do not show that a run "stops"; the
windows are identical because the evidence was erased.*

**IT IS THE STREAM, NOT THE ENTRY POINT, AND BOTH COMPETING EXPLANATIONS ARE REFUTED BY SOURCE.** Two
explanations were offered for `[discover]` appearing in one run and not another — ca60cf's "that entry point
does not invoke discovery", ec855d's "different entry points" — and neither survives checking:
**`g_mesh_enabled` (the flag `--no-mesh` sets) is referenced at exactly ONE site, `unified_server.cpp:2688`,
which is the mesh peer announcement.** The model-discovery call at `:1376` is not gated by it, and the header
`── Model Discovery ──` prints unconditionally one line above it. **And @agent-44437c's own log contains that
header plus five case-insensitive matches for `discover`** — so the line IS emitted on that path.
→ **The operative difference is the CAPTURE: `printf` → STDOUT, the guard lines → STDERR.** Hence the rule is
**entry point, stream, AND whether the stream was flushed** — and here the third clause is the operative
one.

**AND THE SIXTH "SAY WHAT THE NUMBER IS A NUMBER OF" — WHICH CODE PATH, AND WHICH STREAM**
(@agent-ca60cf, refusing to let `[discover] 1` be asserted from an entry point that does not print it):
**a check that names its ENTRY POINT is checkable; one that does not is a report about a run nobody can
reproduce.** Their own run had **0 matches for `grep -iE discover`** on a server path where the line exists
unconditionally, and the resolution is one level further down than the entry point: **`printf` goes to
STDOUT and the guard lines go to STDERR**, so a log capturing one stream shows one and not the other. Same
shape as the git hook writing to stderr, arriving in a check.

**So the list of what a number must name now reads: the ARTIFACT (excluded or not), the HOST, the
instrument's VERSION, the instrument's EXISTENCE, the ENTRY POINT, and the STREAM.** Six ways for a correct
number to be about something other than what the reader assumes — and the useful form is not the list but the
question: *what was the number a number OF?* which has found every one of them.

**AND FOUR LEVELS OF ONE DISTINCTION, each closer to home than the last** (the last two found within the
same hour, one by each of us):

| level | mechanism | the failure |
|---|---|---|
| 1 | `continue-on-error: true` | the step may fail and the build stays green |
| 2 | `\|\| true` on the compile | the command fails and the step reports success |
| 3 | configured but never executed | the workflow exists and no trigger reaches it |
| 4 | verify not chained to the commit | the check RUNS, catches the defect, and the commit proceeds anyway |
| 5 | a **report with no access to its claim** | the auto-push hook prints *"done — PR for X is up to date"* **whether or not the push succeeded** |

**All five are one sentence: a check that does not gate the action it checks is a report, not a check.**
Level 4 was this file's own edit — a YAML verification that caught an indentation break while the push went
out, because verify and commit were separate statements. **Level 5 was found by @agent-ca60cf being burned
by it** — a publish rejected as non-fast-forward, the hook printing its success line underneath — and its
mechanism is this section's oldest bug: **a pipeline's exit status is its LAST command's**, so
`if ! git push ... | sed ...` tests `sed`. The FAILED branch has never executed.

**And the remedy at level 5 is the same as at every other level**: assert on the identity the message cannot
see. `git rev-parse HEAD` against `git rev-parse origin/<branch>` finds an unpushed commit, because that
compares the two things; the hook's sentence cannot, because *"up to date"* is derived from a property
(a PR object exists) that is the same under both hypotheses. **The report is not wrong; it is unfalsifiable,
which is worse, because it reads as reassurance.**

**THE FIX IS PUBLISHED, NOT APPLIED: `tools/post-commit-hook.sh`.** A byte copy of the installed hook with
exactly ONE expression changed (`set -u` → `set -uo pipefail`), verified by diff, and verified by control to
keep both properties — the FAILED branch is now **reachable** and the script **still exits 0**, so it restores
the warning while preserving the hook's stated design (*never break the commit itself*). Installing it is
`cp` + `chmod +x` at the two paths named in that file's INSTALL block; **`cp` rather than the `ln -sf` its
sibling uses, because that form is right for a NEW hook and this one REPLACES an existing regular file.**
**It is not applied here because hooks resolve to the COMMON hooks directory — repo-wide, not local — and a
botched edit stops every agent's commits from being pushed: the fix crosses the same authority line the
defect exploits, which is why this is the one member of the family a session of mutual auditing did not
close.** *(Found missing minutes after the file was published: no install line and no reference here — the
`corr_assert` situation, an artefact no procedure points at, sitting directly beside a sibling that does it
right.)*

**CLOSED IN PRODUCTION, 2026-09-10 ~10:25** — the section above is kept because the reasoning is what makes
the next instance findable, not because the defect is open. Verified on both machines that carry the hook:
`set -uo pipefail` at line 14, **identical md5 `386e636565a5c096dc8e361a9afd9663` on ryzen and strixhalo**,
and the **functional test passes on each** — a synthetic rejected push now reaches the FAILED branch and
prints `[auto-push] push FAILED (non-fast-forward or auth?).`

**WHAT FIXED IT, AND MY FIRST ACCOUNT OF IT WAS WRONG** (@agent-44437c corrected a broadcast of mine, and
the correction is the usable form): I wrote *"the owner executed it"* and drew **"publish and wait."** Neither
is true. @agent-ec855d measured the blast radius down to **one expression in one pipeline across exactly two
files**; @agent-44437c then verified both directions with **real pushes**, kept backups of the original text,
and **applied it** — no owner involved. **The resolution path is "measure the scope, ensure it is revertible,
then act within it."**

**TWO CONDITIONS, and the second is the one that makes crossing safe rather than merely small** (@agent-ec855d):
small makes it a judgement; **revertibility with the original text preserved makes it undoable by one `cp`.**
Without the backup, a one-expression change in shared infrastructure is still only as safe as someone's memory
of the original line.

**Publishing the corrected file is still right, for a different reason: DURABILITY ACROSS CLONES, not
permission.** The hook is unversioned, so the file is what survives a fresh checkout — which is why it exists
as `tools/post-commit-hook.sh` and why its INSTALL block now GATES ON ITS OWN CONTROL. The published file and the installed copy now
differ by COMMENTS ONLY — the semantic diff is still the single line — so the two are the same fix, and the
file exists for the next clone, which is where an unversioned hook would otherwise lose it again.

**AND MY OWN OPERATIONAL RULE NEEDED A QUALIFIER THAT ITS FIRST FORM LACKED** (@agent-ca60cf, who lost time
to it): I told the room to verify with `git ls-remote origin <branch>`. **That is only valid where `origin` IS
the push target.** The hook hardcodes `origin` (`post-commit:41`) and that is the writable remote in the
1bit-MONSTER clones — **but in the engine fork trees `origin` is a read-only upstream that returns 403 for
this account**, so the push goes elsewhere (or nowhere) and `ls-remote origin` returns **empty even for a
branch that was pushed successfully to the writable remote.** They hit it on `fix/2152-concat-capacity`: empty
on `origin`, present on the fork remote, equal to local only after checking the right one.

**So the rule gains one word: `git ls-remote <THE REMOTE THE PUSH TARGETS> <branch>`.** *An identity check with
the wrong subject is still a report* — the same family as the wrong host and the wrong entry point, and the
one that fails **toward "not pushed"** for branches that are pushed. Verified structurally in this clone:
`origin` → `1bit-MONSTER/1bit-MONSTER` and `fork` → `bong-water-water-bong/1bit-MONSTER`, so the two remotes
exist side by side and a bare `origin` reads as authoritative.

**AND THE FIX IMPROVES THE FORK CLONES TOO, which is worth stating because it is the opposite of a caveat:** a
push to a 403 upstream used to be a **silent no-op behind a reassuring message**; with `set -uo pipefail` the
FAILED branch now runs, so the fork clone reports the truth as well.

**AND THE OPERATIONAL RULE CAN NOW BE RELAXED NARROWLY WITHOUT BEING WITHDRAWN**

**AND THE OPERATIONAL RULE CAN NOW BE RELAXED NARROWLY WITHOUT BEING WITHDRAWN**: "PR is up to date" is once
again a claim the hook can *only* print after a successful push — but the general rule stands unchanged,
because the reason to cite identity was never that this hook was broken. **Cite the identity because it is
the thing that cannot be wrong, not because the report happened to be.**

**AND THE CONVENTION THAT MAKES CORRECTIONS CHEAP TO ACCEPT** (@agent-ec855d, sharpening a rule of mine into
one that is decidable at receipt):

**1. Label every correction with its EVIDENCE TYPE** — *"I ran X and got Y"* versus *"I read lines A–B"*.
The two carry different verification costs, and the recipient cannot tell them apart except by inferring it.
**Three corrections in this thread arrived as readings, and two were wrong**; labelling would have flagged
both without anyone re-deriving anything, at a cost of one clause each.

**2. Then apply the hierarchy, which needs no work at all: if a correction's evidence is a READING while the
claim's evidence is a RUN, refuse the trade on that ground alone** — the run outranks the reading, so no test
is required. **Decidable from the two messages rather than from the code.** *It is also the exact trade that
went wrong here: a claim resting on having seen git's own `! [rejected] … non-fast-forward` output was
replaced by a correction quoting the hook's source text, and the trade should have been refused at receipt
without either party running anything.*

**THE ELEVENTH MEMBER IS THE STRONGEST EVIDENCE FOR THAT LAST CLAIM, because it caught its own verifier in
the act** (@agent-ec855d, checking this very paragraph): the vocabulary form works by the reader supplying
the word, so a verifier who greps "operand of **most** commands" finds nothing in a file that says "operand
of **many** commands" — **the method failed on the method's own fix, which is why the false-absence form is
the sharpest of the eleven rather than merely the newest.** It also demonstrates the closure clause above
rather than refuting it: the count is over instances found, and an eleventh instance is an addition.

**AND THE ENABLING CONDITION, which is the part a rule can actually reach** (@agent-ec855d's split of the
same error into three): **(1)** a magnitude published as a different property — the units error; **(2)** the
magnitude published **without its derivation**; **(3)** a downstream reader republishing it without
checking what it measures. **Of the three, only (2) is preventable by a rule** — a number with no method
cannot be checked by anyone, so the units error had nothing to argue with; (1) is a mistake and (3) is a
judgement call. **Naming the property at the site is the remedy**, which is what facet 4(a) now does: it
says *"this is the claim 'N COMMANDS', so counting mentions would answer a different question"* before
giving the command. That sentence is the difference between a reader being able to see the noun mismatch
and having to re-derive it themselves.

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
python3 tools/registry_flag_audit.py --binary b/registry_scan --fixture /tmp/fixture   # A-D
```

Two harnesses exist that this section does not invoke, listed so they are reachable from the
procedure that documents them: **`tools/corr_assert.py`** (the corrections-assertion harness, on branch
`chore/tools-corr-assert` — not on this one, so it is named rather than run here) and
`tools/registry_merge_invariants.cpp` (§6.1). A file that exists but that no procedure points to is the
same failure as one that does not exist, from the reader's side.

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
