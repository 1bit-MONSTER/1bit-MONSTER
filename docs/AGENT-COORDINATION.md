# Agent Coordination — 1bit-MONSTER (ryzen ↔ strixhalo)

> This repo is worked by **DeepSeek Harness agents on two machines** (ryzen ↔
> strixhalo); several sessions can be live on one box at the same time.
> Both edit this same codebase; this file is the shared handoff ledger.
> **Read it before starting work. Update it when you change lanes or land
> something. Keep both machines' clones in sync (protocol at the bottom).**

## 2026-09-13 (evening) — strixhalo: census/CI lane, landed and left

Nine PRs landed on this lane today; the midday entry below covers the first two.
This is the rest, written as *state* rather than a changelog, so the next session
does not redo any of it.

**Behaviour that changed — expect it on the next scheduled runs** (`census-watch`
04:30Z, `census-autopr` 04:45Z):

- The alert **comments on an open `census-watch` issue** when the class set moves
  and stays silent when it has not (#2322). If #2178 gains an "Update" comment,
  that is the mechanism, not a person.
- `census-autopr` **retries its push three times**, and re-checks for an existing
  PR before retrying `gh pr create` (#2326) — so a missing daily post now means
  three failures, not one.
- The provenance manifest writer **preserves** `build.toolchain` /
  `generating_script_revision` instead of hard-coding null (#2334). Regenerating
  with `--write-manifest` no longer erases a toolchain a build recorded; set one
  with `--toolchain "…"`. That field is the blocker #2262 stays open for.

**Guidance that changed:** the two texts that steer the coverage lanes now agree
(#2336, #2337) — decide an alias on the checkpoint's *tensors*, not the class name
or the config, and record a non-alias in `Testing/arch-gaps.md` (*Uncovered classes
reviewed later*). The auto-filed alias draft tells its reviewer the same thing.

**Reviewed evidence — do not re-triage.** All five classes today's sweeps surfaced
(`englishbase`, `gdn2`, `haiku`, `vapor`, `fidel`) are in `Testing/arch-gaps.md`
with the evidence that none is an alias; `vapor` is the instructive one (LFM2's
config verbatim, factorized FFN in the checkpoint). `research/TRACKING.md` carries a
2026-09-13 delta closing its `NPU_PREFILL_MAX` question and re-pointing the Census
row at the 09-13 sweep (`4eb5dc43a`; 323,793 / 323,904 = 99.966%).

**Left alone deliberately:** the site's "100% HuggingFace coverage" wording (#2178
rules that the fix is to map the class, not soften the text); the census's local run
state (`hf_new_models_state.json` / `significant_arrivals.json` are *expected* dirty
on a box that runs the watcher — `scripts/census-watch.sh:60` excludes exactly those
two); and the decoder gate's Gemma3-1B skip, which is honest — its `lm_head` is
`[147456, 1280]` against an embed of `[262144, 1152]`, and the ceil-tiled decode
that would "fix" it returns noise.

**Open, not this lane:** ops #2333 (the dead strixhalo→Pi backup) was being fixed on
the Pi while I looked, so I verified its numbers independently and commented instead
of racing it. The siblings `backup-ryzen.sh` and `backup-minisforum.sh` still have
the same shape (no precheck, rc recorded but never read, no status file), and
nothing yet reads the new status file or the syslog line.

## 2026-09-13 — unlanded-work audit: four branches existed only on this box

`git branch` here has **~78 refs that are not on `origin`**. Most are harmless —
they are branches whose work landed via squash-merge, so they *look* ahead of
`main` while their content is already in it. Four are not harmless: work with no
PR and no remote ref, i.e. one `git worktree remove` away from being lost. All
four are now pushed to `origin` (preservation only, deliberately no PRs):

| branch | commits | what it holds |
|---|---|---|
| `fix/2114-cascade-qn-guard` | 26 | the #2114 fold-scale guard — whose *result* is posted on the issue (no per-layer guard can fix it, the healthy per-layer error is already above the recurrence's stability threshold) — **plus** #2078/#2113 cascade launch perf (per-task-group batched fills) and the VECFIX/RE xclbins + build recipes. **None of that perf work or those artifacts are in `main`.** |
| `run-2145`, `fix/2145-ctx-guard-verify` | 4 + 4 | verification runs for the still-open #2145 (state-imported context on the HRX device) |
| `census-sweep-fix` | 1 | census one-liner; largely superseded by the #2315 sweep |

Method, so it can be repeated: `git for-each-ref` for branches ahead of `main`
that have no `origin/<branch>`, then `gh pr list --head <branch> --state all` —
**per branch**. Do not batch that check through a `--limit N` list: it truncates
*silently at exactly N*, and a capped list produces confident false "NO PR"
verdicts. That is how this audit initially mislabelled a fifth branch
(`docs/tracking-20260913-verify`, which already had PR #2330).

**Disposition (checked against `main` the same day).** None of the four is
landable as-is, so this section is a record rather than a to-do:

- `run-2145` + `fix/2145-ctx-guard-verify` — **duplicates of merged work.** The
  ctx guard is in `main` (`src/hrx_inprocess.cpp:380,391`, env
  `HRX_MAX_CTX_TOKENS`) via PR **#2203**, and the #2147 `GGML_HRX_CPU_OPS`
  change is in `main` (`src/backend_hrx.cpp:112-114`). Note `run-2145`'s merge
  commit names `origin/fix/2145-hrx-ctx-limit`, which **does not exist** on
  origin — the landed form is #2203. When auditing, check `origin/<same-name>`
  *and* any branch names referenced in the branch's own commits; work can be
  pushed under a different name.
- `census-sweep-fix` — **superseded** by the 09-13 sweep `4eb5dc43a` (#2315);
  its own sweep is the 09-12 one.
- `fix/2114-cascade-qn-guard` — the guard's *result* is already posted on #2114
  (negative: no per-layer guard can fix it). Its perf commits (`88020e0c8`,
  `5527e3226`) touch **only the generator**
  (`engine/npu/generators/n1_core_fused_gu_silu_d_iron.py`), so they change no
  runtime behaviour until the cascade xclbin is regenerated — and `main`'s copy
  of that generator has since diverged (36+/12-). Reviving the perf is therefore
  gated on the xclbin build flow (#2262) plus a silicon re-verify and a
  provenance update. The VECFIX/RE xclbins on that branch are artifacts of the
  same investigation.

## 2026-09-13 (midday) — strixhalo: census alert plumbing

- **PR #2322** (`fix/census-watch-alert-delta`, commit `3ce96c476`): the census
  alert deduped on *"an issue is open"*, so it dropped every finding that arrived
  while one stayed open. The 2026-09-13 run found `fidel`, `moonfrost` and `vapor`
  beyond the classes #2178 names, and that run's log was the only record of them.
  The step now dedupes the class SET (an HTML marker in the body and in each
  update comment) and comments on the open issue when the set moves — silent when
  it has not. Five stubbed-`gh` cases over the real 09-13 log lines are in the PR.
- **`Census autopr` was red for 2026-09-13** (run 34750206963): it generated the 15
  post files, then died at `git push origin post/significant-20260913` with
  `remote: fatal error in commit_refs`. Server-side/transient, not policy — the only
  active ruleset targets `main`, the same push succeeded on 09-12, and **a re-run of
  the failed job succeeded** (13:39Z), opening the day's draft post as PR #2323 on
  branch `post/significant-20260913`. If it recurs, that push wants a bounded retry:
  the workflow runs once a day, so one transient refusal silently costs a day.
  **PR #2326** (`fix/census-autopr-push-retry`) is that retry — 3 attempts on the
  push, and on `gh pr create` a re-check for an existing PR before each retry,
  since a create error can arrive after the PR is up.
- **#2322 verified in production** (2026-09-13 14:32Z): a `workflow_dispatch` of
  `census-watch` on `main` ran the watcher and posted the update comment against
  #2178 — `newly reported: deepseekv41,englishbase,gdn2,haiku,moonfrost` plus the
  `<!-- census-classes: … -->` marker. The rolling 120-model window had already
  retired `fidel`/`vapor` and surfaced three new classes, which is exactly the
  churn the old skip-the-whole-alert behaviour used to hide.
- **Registry gaps still open** — issue #2178, reviewed from the official configs:
  `deepseekv41` is a real V4.1 multimodal family (engram + candidate-block
  machinery), not an alias; new on 09-13 are `fidel` (`4E-AI/Fidel1.1-1B`, custom
  hyper/cross rank config), `moonfrost` (`whoashish115/Moonfrost-777M`, MLA-style
  compressed KV + 32-expert MoE) and `vapor` (`Neeze/Vapor-2B-V1.2`, which carries
  LFM2's config schema verbatim — all 30 `layer_types` identical — but with vocab
  128000, rope theta 1e7 and a `rank_config` of `SharedSwiGLULinear` layers LFM2
  does not have). None of the three is a one-line mapping, so the census "100%"
  claim stays false until someone implements them — do not "fix" it with aliases.

## 2026-09-12 (late) — strixhalo snapshot

- **`main` is at `f3825fbb6`** — PR #2282 (the #2199 fused-int4 `.data` fix +
  rebuilt xclbin + regenerated provenance manifest) — and level with `origin/main`.
  Re-verified 2026-09-12: `bash engine/npu/tests/check_kernel_bss.sh` →
  `SUMMARY: symbols=ok duplicate=ok bss=0` / `RESULT: PASS` (was
  `FAIL_BSS_ONLY` when #2262 measured it).
- **The NPU work no longer lives in this clone.** It moved to the worktree
  `~/1bit-MONSTER-goal` on branch `goal/runlist-decode-wire` (330 ahead / 119
  behind `origin/main`, pushed, **no PR yet**). Dense-Qwen3 @1k prefill and the
  generated long-context attention ELF live there; read
  `engine/npu/generators/FK3-STATUS-2026-09-12.md` before touching it.
- **Sessions, not a pair.** Several `dsh --profile tui` sessions run on strixhalo
  at once (SEO/site publishing, the NPU thread, housekeeping). Coordination is
  still this file + git — assume a concurrent session holds the NPU.
- **Devices are invisible in the default bash sandbox.** `dsh` runs commands in a
  bwrap sandbox with a synthetic `/dev`, so `/dev/accel/accel0`, `/dev/dri` and
  `/dev/mem` are absent and `xrt-smi` reports "0 devices". NPU commands need the
  full-access sandbox. `/tmp` is a per-command tmpfs — never build or park
  artifacts there; use `~/npu-build/` or the repo.
- **Single-NPU contention is measurable.** Two `npu_engine_qwen3_0_6b` runs
  started a minute apart (22:08 / 22:09, 2026-09-12) turned a 150 ms @1k
  attention into **227 s**. Sync-protocol rule 4 is not theoretical: serialize
  before you record a benchmark number.
- **Untracked duplicates that will collide on merge.** `benchmarks/FLM-PARITY-DATA-SOURCES.md`
  and `benchmarks/prompts/reclaimer.txt` are untracked in the main clone and are
  tracked on `goal/runlist-decode-wire` with **different content** — merging that
  branch into `main` will trip over them. The branch owner should reconcile.
- **Disk is at 87 % (247 G free) and the big consumers are all load-bearing.**
  models 425 G, Xilinx toolchains 142 G, and `~/.cache/moe-cap-rb` **77 G — that
  last one is the 35B MoE capture oracle (byte-verified in Rounds 81–83 of the
  NPU work), not a cache you may prune**. Reclaiming space here is a human
  decision; a cleanup script must not touch the oracle. (A 1.1 GB `flm` crash
  dump and ~7.9 GiB of swap were reclaimed on 2026-09-12; nothing larger is
  safely free.)
- **Scratch left untracked in this clone** (~41 entries): NPU probe sources and
  their compiled binaries under `engine/npu/pool/`, `npu-infer/tools/`,
  `tools/zaya_*.cpp`, plus `npu-infer/captures/txn-elfs-moe35b/` and
  `npu-infer/build-rl/`. Left in place deliberately — they are someone's working
  set, not litter. The dated evidence among them was landed in PR #2306.

## Agents & machines

| Agent | Machine | LAN IP | Workspace | Repo remote | GitHub identity |
|-------|---------|--------|-----------|-------------|-----------------|
| **Co-worker** | ryzen (Ryzen 7 9800X3D) | 192.168.50.100 | `~/projects/1bit-MONSTER` | `fork` → `bong-water-water-bong/1bit-MONSTER` (branch `fix/triage-round`) | bong-water-water-bong |
| **Strixhalo agent (this one)** | strixhalo (Ryzen AI Max+ 395, NPU box) | 192.168.50.110 | `/home/bcloud/1bit-MONSTER` | `origin` → `1bit-MONSTER/1bit-MONSTER` (branch `main`) | bong-water-water-bong (same account) |

- Both agents share the GitHub identity `bong-water-water-bong` — either can push to the
  fork and to upstream `main`.
- DSH Web GUI: ryzen `http://127.0.0.1:3080` (local), strixhalo `http://127.0.0.1:3080`.
  There is **no DSH↔DSH chat API** — coordination happens through this file + git.

## Ownership map (who fixes what)

| Area | Owner | Notes |
|------|-------|-------|
| GitHub issue triage + fixes (#1832, #1834, #1837, #1864, #1865, #1870, #1872, #1874, #1878, #1882, #1908, #1909, #1910, #1911, #1913, #1776 gate, #1831 interim, census #1900/#1906) | **Co-worker** (ryzen) | branch `fix/triage-round` on fork; 14+ issues landed 2026-08-28 |
| Fused GU→SiLU→D cascade kernel work (BUG-001..011, #1775/#1769) — p1/p2 two-launch production | **Strixhalo agent** | committed on `main` (aecfad54): BUG-005 D-cascade fix silicon-verified; single-launch premise REJECTED (BUG-011) |
| NPU HW verification on strixhalo (`/dev/accel0`) | **Strixhalo agent** (only machine with the NPU) | verify co-worker's kernel changes + cascade work |
| Upstream escalations (llvm-aie/peano: #1836, #1844, #1912, #1866, #1835) | **Co-worker** leads; strixhalo agent provides reproducers/evidence from HW | tracked in #1882 |

## Shared file guard

**`engine/npu/generators/mm_kernel_reference.cc` is edited by BOTH agents.** It has been
merged (2026-08-28, merge commits 88de972c + 5c78007b — full sync): the file now contains
the co-worker's complete `fix/triage-round` work (KERNEL_STATIC .data, #1865 arg-based
h2/pC + retired #1842 pins, #1874 I4_SCALAR_C1 production default, #1872 direct-vector
dequant, #1878/#1912 unpack_i4_sx shim) AND the strixhalo agent's `silu_quant_i8_fused_q22`
+ `cascade_reduce_*_i32` single-pass forms. Verified: 0 syntax errors with the cascade
defines (`-DDIM_M=8 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY -DM8_VECTORIZED`).

**#1872 BUILD — now compiles (ported 2026-08-28, strixhalo agent).** The committed
`I4_DIRECT_VECTOR_DEQ` path did NOT compile on the repo toolchain (aie API mismatches:
`aie::to_vector` on plain vectors, `acc[e]` on an accum, and `aie::mul` for 64-wide
int32 yielding a 32-lane accum). `69973241` rewrote the dequant as register-only scalar
int64 math (`B''[e]=sat8(round(q4<<4*ratioQ22>>22))` — still NO Bb memory round-trip),
so it compiles (0 errors) and the int4 fused xclbin builds
(`final_i8_MOE_GUSILU_i4_zaya.xclbin`). **NPU corr gate still open**: the `npu_engine_universal`
decode ran >13 min at ~850% CPU without reaching the per-layer corr/byte-identity gates
(host CPU-bound at the reference; needs investigation or a longer/bounded run).

**Rule for this file:** pull/merge before touching it; never overwrite the other side's
functions; if a conflict appears, preserve BOTH sides and note it in the merge commit.

## Status snapshot (2026-08-28, fully synced — resolution)

- **Sync complete**: `main` contains ALL co-worker work (fork state + the 16 newest
  commits from upstream `fix/triage-round` @ 954dc298) + strixhalo agent's cascade work
  (aecfad54) + upstream main. Merge commits: 88de972c, 5c78007b. Open PR #1917 (MERGEABLE).
- Co-worker: 27 issues triaged; 14+ fixed; HW-verified on strixhalo (#1865 byte-identical,
  #1874 corr 1.0, #1832 live q4nx decode, #1872 arithmetic-verified 512000/512000; NPU gate
  on #1872 pending) + XL features (#1907 baretorch, #1831 HIP) + upstream escalations.
- Strixhalo agent: fused-cascade work committed (aecfad54) + bug reports BUG-001..011;
  BUG-011 decision: single-launch zero-DMA premise REJECTED — p1/p2 two-launch (h2 via
  DDR) is the production path.
- **RESOLUTION (zero-h2-DMA single launch): PROVEN BLOCKED** — see next section. The
  objective's "prove which blocker is fatal" branch is satisfied with controlled silicon
  evidence; production path = p1/p2 two-launch.

## Zero-h2-DMA single-launch: PROVEN BLOCKED (2026-08-28)

The doc's option (a) (2-channel dataflow multiplexing B_d over a GU channel) was
implemented and tested on silicon: split GU A/B into two 2-D single-stream fifos
(ch0 A-tile, ch1 B-tile carrying B_gu then B_d). It **builds** but the shared-B
fifo's D-cascade writeback does NOT fire (C2=0x5A, reproduces cleanly even for a
`--no-gu` D-only probe). Controls: the ORIGINAL (combined-AB GU + dedicated
`of_b_d`) D-only design IS silicon-exact (C2=2048, bad=0); a 3-fifo dedicated-B
variant won't place (2-input-DMA, BUG-007). So the complete zero-h2-DMA fused
single launch remains blocked by the iron ObjectFifo + 2-input-DMA constraint.
Production path stays p1/p2 two-launch (h2 via DDR). Next options: (b) an iron
FIFO primitive that pipelines merged/segmented elements (toolchain-level), or a
way to reuse one channel without the shared-B writeback regression.

## Sync protocol (both agents)

1. **Before starting work:** `git fetch origin` (and the fork), merge/rebase `main`
   (and `fix/triage-round` if you touch kernel files), read this file.
2. **After landing anything:** push immediately; update this file's snapshot table;
   mention the commit SHA.
3. **Kernel file rule:** see above — pull first, preserve both sides, never force-push.
4. **NPU is single-device:** do not run the 1bit engine / xclbin benchmarks on strixhalo
   while the other side is validating there (documented AMD-Vi IO_PAGE_FAULT storms).
5. **Coordination messages:** commit them here (append a dated note) rather than relying
   on chat; the other agent reads this file on its next pull.
