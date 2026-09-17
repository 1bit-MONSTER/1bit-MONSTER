#!/usr/bin/env python3
"""census_autopr.py — auto-file a draft PR when the new-model watcher finds an
uncovered architecture whose name is a known name spelled differently.

The watcher (hf_new_models.py) detects new HF archs the engine registry
doesn't map. Mapping is normally a manual bitnet_model.h edit. When HF's
model_type is the SAME NAME spelled differently (e.g. `kimik3` vs `kimi_k3`),
this module proposes the one-line alias as a DRAFT pull request — turning
"discover + edit" into "review + merge".

A NAME MATCH IS NOT EVIDENCE OF A FAMILY, and this module has already forgotten
that once: it filed `language` -> OBILANGUAGE (#2443) and `picolm` -> PICO
(#2444), both closed unjustified, because both were name-similarity guesses.
So the two fuzzy rules are ALERT-ONLY — they print the candidate for a human
and open no PR. Only an exact normalized-name match (rule 1) files, because a
spelling difference is not a claim about the architecture. Set
CENSUS_AUTOPR_FUZZY=1 to file on a fuzzy candidate as well; it is not
recommended, since the reviewer must establish the family from the checkpoint's
tensors either way — which is the work the draft was meant to save. (Both
docstring examples above are already aliases in bitnet_model.h, and rule 1 is
what reaches them, so no fuzzy rule has a demonstrated true positive.)

Both fuzzy rules over-fired because the table they match against is an
exact-match DISPATCH table, not a family list:
  * rule 3 read every key as a family prefix, and the table legitimately holds
    1-4 char aliases (`h` -> LLAMA, `rw` -> FALCON, `new` -> QWEN3,
    `h3`/`i3` -> LLAMA). Every uncovered class starting with `h` therefore
    "looked like" LLaMA. Measured 2026-09-17: 24 of 38 real non-family arch
    names produced a candidate, and 175 keys were <= 4 chars.
  * rule 2's comment always said "<=2 substitutions/dels" but the code tested
    `difflib.ratio() > 0.8`, which fired on `language` vs `obilanguage`
    (ratio 0.842, edit distance 3).
The rules below implement what those comments already claimed. Genuinely-new
architectures are left as a manual alert — they need a real engine
implementation, not an alias.

A class that has ALREADY been reviewed is never proposed again, whatever the
name looks like: a heading in `Testing/arch-gaps.md` naming the class is the
record of a review that concluded "not an alias", and this module now reads it.
Before that it was write-only, so `language` and `picolm` were filed, closed
unjustified, and re-filed by the next run that saw them.

Usage (from hf_new_models.py after finding uncovered):
    from census_autopr import maybe_file_draft_pr
    maybe_file_draft_pr(uncovered, dry_run=...)

Environment:
  GITHUB_TOKEN / gh CLI — uses `gh` if available, else no-ops with a log line.
  CENSUS_DRY_RUN=1 — print what would be filed, don't touch GitHub.
  CENSUS_SKIP_PR=1 — never open PRs (alert only).
  CENSUS_AUTOPR_FUZZY=1 — also open PRs for name-similarity (fuzzy) candidates,
      which the default now only alerts on. Not recommended; see above.
"""
import json, os, re, subprocess, sys
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(ROOT, "include", "rocm_cpp", "bitnet_model.h")
SELFCHECK = os.path.join(ROOT, "Testing", "census_tail_sweep.py")
# Where a reviewed class is recorded when it turns out NOT to be an alias.
ARCH_GAPS = os.path.join(ROOT, "Testing", "arch-gaps.md")

# Token names the engine defines (for the proposed line's RHS). Pulled from the
# header once, cached.
_TOKEN_RE = re.compile(r"RCPP_ARCH_[A-Z0-9_]+")

_KNOWN = None  # {arch_string: token_name}


def _known_mappings():
    """Parse include/rocm_cpp/bitnet_model.h for string -> RCPP_ARCH_* pairs.

    FIRST match wins, mirroring rcpp_arch_from_string's linear `if` chain. A
    plain assignment took the LAST, and the two disagree: `qwen3_5moe` is
    mapped at two places in that chain (-> QWEN35 early, -> QWEN3 late), so the
    engine resolves QWEN35 while this table reported QWEN3. Querying the header
    this way means the tool can never be more wrong than the engine.
    """
    global _KNOWN
    if _KNOWN is not None:
        return _KNOWN
    known = {}
    try:
        with open(ENGINE) as f:
            for line in f:
                m = re.search(r'strcmp\(s,\s*"([a-z0-9_]+)"\)\s*==\s*0\)\s*return\s+(RCPP_ARCH_[A-Z0-9_]+)', line)
                if m:
                    known.setdefault(m.group(1), m.group(2))
    except OSError as e:
        print(f"[autopr] cannot read {ENGINE}: {e}", file=sys.stderr)
    _KNOWN = known
    return known


def _norm(s):
    """Normalize an arch string for comparison: drop non-alnum, lowercase."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


# The mapping table is an exact-match dispatch table and legitimately holds very
# short aliases, so using it as a family list needs floors or the shortest alias
# becomes a wildcard (`h` -> every h-initial class "is" LLaMA). See docstring.
_MIN_FAMILY_LEN = 4   # below this a key is an alias, not a family name
_MIN_FUZZY_LEN = 8    # an <=2-edit match is a rename only on strings this long


def _distance(a, b):
    """Levenshtein distance — the metric rule 2 always claimed to use."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _guess_ex(uncovered_arch):
    """Return (candidate_arch_string, token_name, kind) or None.

    kind is "exact" — the same name spelled differently, safe to draft an alias
    for — or "fuzzy", a name-similarity guess that is ALERT ONLY (docstring).
    """
    known = _known_mappings()
    n = _norm(uncovered_arch)
    if not n:
        return None

    # 1. Exact match on a normalized known string (e.g. kimik3 vs kimi_k3).
    for s, tok in known.items():
        if _norm(s) == n:
            return s, tok, "exact"

    # 2. <=2 edits from a known name, both long enough for that to mean
    #    "spelling", not "coincidence". The old test was `ratio() > 0.8`, which
    #    called `language`/`obilanguage` (3 edits) a match.
    best, best_d = None, 3
    if len(n) >= _MIN_FUZZY_LEN:
        for s, tok in known.items():
            ns = _norm(s)
            if len(ns) < _MIN_FUZZY_LEN or s == uncovered_arch:
                continue
            d = _distance(n, ns)
            if d <= 2 and d < best_d:
                best, best_d = (s, tok), d
    if best:
        return best[0], best[1], "fuzzy"

    # 3. Known-family substring prefix (e.g. "qwen35moe" starts with "qwen35").
    #    The floor is what stops the 1-4 char aliases acting as wildcards.
    fams = sorted(known.keys(), key=len, reverse=True)
    for s in fams:
        ns = _norm(s)
        if len(ns) >= _MIN_FAMILY_LEN and n.startswith(ns) and len(n) > len(ns):
            return s, known[s], "fuzzy"

    return None


def _proposed_lines(arch, target):
    """Proposed bitnet_model.h line(s) for arch -> target token."""
    return [f'    if (strcmp(s, "{arch}") == 0) return {target};  // census-auto']


def _selfcheck_has(arch):
    try:
        with open(SELFCHECK) as f:
            return arch in f.read()
    except OSError:
        return False


def _documented_in_arch_gaps(arch):
    """True if `arch` is named in a Testing/arch-gaps.md heading.

    That file is where a class goes once it has been reviewed and found NOT to be
    an alias ("Uncovered classes reviewed later — same standard, and still not
    aliases"). Nothing read it, so a reviewed class could be proposed again every
    time it reappeared in the watcher's window: `language` and `picolm` were
    filed, closed unjustified, and re-filed by the next run that saw them. A
    review is a decision, and this is what makes it one the tool can see.

    Headings carry one or two backticked names (`### `blockmtp` / `looped_block_mtp``),
    so every token on a `##`-or-deeper heading line is matched, normalized.
    """
    n = _norm(arch)
    if not n:
        return False
    try:
        with open(ARCH_GAPS) as f:
            for line in f:
                if not line.startswith("##"):
                    continue
                if any(_norm(tok) == n for tok in re.findall(r"`([^`]+)`", line)):
                    return True
    except OSError:
        return False
    return False


def _header_has(arch):
    try:
        with open(ENGINE) as f:
            return f'"{arch}"' in f.read()
    except OSError:
        return False


def _apply_alias(arch, target):
    """Insert the proposed mapping into the engine header. True if it changed.

    2026-09-12: the alias was proposed but NEVER APPLIED, so `git add` staged
    nothing and `git commit` died with "nothing to commit" (on stdout, which is
    why the logged failure had no detail). No draft PR was ever filed, and the
    run left the checkout switched to an empty branch — census/auto-map-maba
    was byte-identical to main and `maba` was absent from the header. Applying
    the edit is what makes the rest of this function work.
    """
    if _header_has(arch):
        return False
    with open(ENGINE) as f:
        src = f.read()
    head, sep, tail = src.rpartition("\n    return RCPP_ARCH_UNKNOWN;")
    if not sep:
        raise RuntimeError(f"mapping anchor not found in {ENGINE}")
    src = head + "\n" + "\n".join(_proposed_lines(arch, target)) + sep + tail
    with open(ENGINE, "w") as f:
        f.write(src)
    return True


def _git_out(cmd):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def _open_draft_pr(arch, target, models):
    """Open a draft PR proposing the alias. Returns PR url or None."""
    branch = f"census/auto-map-{arch}"
    body = (
        f"## Auto-filed by the census watcher\n\n"
        f"The daily new-model watcher found an uncovered HF architecture "
        f"(`{arch}`) that looks like a known family. This draft proposes "
        f"the one-line alias so it stops surfacing as a coverage gap.\n\n"
        f"**Proposed mapping:** `{arch}` → `{target}` "
        f"(candidate, needs human review)\n\n"
        f"**Models triggering it:** {', '.join(models[:5])}\n\n"
        f"**Steps for the reviewer:**\n"
        # "check the model config / paper" was the old step 1, and the config is
        # exactly the surface that lies: `vapor` carries LFM2's schema verbatim
        # — all 30 `layer_types` identical, every `block_*`/`conv_*` key equal —
        # and its checkpoint stores a factorized FFN (`A2`, `A_shared`, `B1`-`B3`,
        # `side_*`) that no reader here understands. A reviewer following the old
        # step would have confirmed it and merged a mapping for a model the
        # engine cannot load. The tensors are what settle this.
        f"1. Confirm `{arch}` really is the `{target}` family — on the "
        f"checkpoint's tensors, not on the class name or the config. A config "
        f"can match a family exactly and still be a different model (see "
        f"`vapor` in `Testing/arch-gaps.md`). `model.safetensors.index.json` "
        f"answers it in one request, or the safetensors header by range "
        f"request when the model is a single shard.\n"
        f"2. If yes, mark ready + merge. If it's a *new* architecture, close "
        f"this and record the evidence in `Testing/arch-gaps.md` under "
        f"*Uncovered classes reviewed later* instead — a class that is neither "
        f"a rename nor a shape match is not a mapping problem.\n\n"
        f"Auto-generated by `Testing/census_autopr.py` — the mapping is a "
        f"*guess*, not verified."
    )
    # Remember where this checkout was. An autopr must not leave it somewhere
    # else: the census tree has to stay on main, and a lane's tree must keep
    # its branch. (`finally` below is what guarantees it.)
    orig_branch = _git_out(["git", "symbolic-ref", "-q", "--short", "HEAD"])
    orig_head = _git_out(["git", "rev-parse", "HEAD"])
    try:
        if not _apply_alias(arch, target):
            print(f"[autopr] {arch}: already mapped in bitnet_model.h — "
                  f"nothing to propose", file=sys.stderr)
            return None
        cmds = [
            # -C: create the branch, or rebuild it from the current HEAD when a
            # previous run left one behind (that case used to fall through to a
            # commit with nothing staged and fail).
            ["git", "switch", "-C", branch],
            ["git", "add", os.path.relpath(ENGINE, ROOT)],
            ["git", "commit", "-m", f"fix(census): auto-propose {arch} -> {target}"],
            ["git", "push", "-u", "origin", branch],
            ["gh", "pr", "create", "--base", "main", "--head", branch,
             "--draft", "--title", f"census: map {arch} -> {target} (auto-draft)",
             "--body", body],
        ]
        r = None
        for cmd in cmds:
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            if r.returncode != 0 and cmd[1] == "push":
                # A rebuilt bot branch is usually behind its remote; retry once
                # with a lease — census/auto-map-* belongs to this tool alone.
                r = subprocess.run(["git", "push", "--force-with-lease", "-u",
                                    "origin", branch],
                                   cwd=ROOT, capture_output=True, text=True)
            if r.returncode != 0:
                text = (r.stderr or r.stdout).strip()
                if cmd[1] == "pr" and "already exists" in text:
                    # an open draft PR for this branch IS the desired end state
                    return _git_out(["gh", "pr", "view", branch, "--json",
                                     "url", "--jq", ".url"]) or None
                print(f"[autopr] cmd failed: {' '.join(cmd)}\n{text[:300]}",
                      file=sys.stderr)
                return None
        # return the PR url from the last command — only accept a real github.com https URL
        out = (r.stdout if r else "").strip()
        if not out:
            return None
        try:
            parsed = urlparse(out)
        except ValueError:
            return None
        return out if parsed.scheme == "https" and parsed.netloc == "github.com" else None
    finally:
        # Hand the checkout back exactly as we found it.
        if orig_branch:
            subprocess.run(["git", "switch", "--quiet", orig_branch], cwd=ROOT,
                           capture_output=True)
        elif orig_head:
            subprocess.run(["git", "switch", "--quiet", "--detach", orig_head],
                           cwd=ROOT, capture_output=True)


def maybe_file_draft_pr(uncovered, models=None, dry_run=None):
    """For each uncovered class, open a draft PR for a candidate alias.

    Files only for an "exact" candidate — the same name spelled differently.
    A fuzzy (name-similarity) candidate is reported and NOT filed, unless
    CENSUS_AUTOPR_FUZZY=1. Returns list of (arch, target, pr_url_or_None).
    """
    if dry_run is None:
        dry_run = os.getenv("CENSUS_DRY_RUN") == "1"
    if os.getenv("CENSUS_SKIP_PR") == "1":
        print("[autopr] CENSUS_SKIP_PR=1 — alert only, no PR", file=sys.stderr)
        return []

    models = models or {}
    filed = []
    for arch in sorted(uncovered):
        guess = _guess_ex(arch)
        if not guess:
            print(f"[autopr] {arch}: no plausible known-family candidate — manual", file=sys.stderr)
            continue
        target_arch, token, kind = guess
        if _selfcheck_has(arch):
            print(f"[autopr] {arch}: already in selfcheck — manual", file=sys.stderr)
            continue
        if _documented_in_arch_gaps(arch):
            # Reviewed already, and the verdict was "not an alias". Re-filing it
            # would re-open a settled question, so this outranks the candidate
            # kind: even an exact-looking match has been looked at.
            print(f"[autopr] {arch}: reviewed in Testing/arch-gaps.md — not an "
                  f"alias, no PR (candidate would have been {token})",
                  file=sys.stderr)
            continue
        if kind == "fuzzy" and os.getenv("CENSUS_AUTOPR_FUZZY") != "1":
            # A name is not a family. Both draft PRs this module has ever filed
            # came from a fuzzy match and were closed unjustified (#2443, #2444),
            # so the printed candidate is the whole useful output here.
            print(f"[autopr] {arch}: name-similar to {target_arch} — ALERT ONLY, "
                  f"no PR (candidate {token}; settle it on the checkpoint's "
                  f"tensors, not on the name)", file=sys.stderr)
            filed.append((arch, token, None))
            continue
        print(f"[autopr] {arch} -> {token} (candidate from {target_arch}, {kind})")
        if dry_run:
            filed.append((arch, token, "DRY-RUN"))
            continue
        if not _open_draft_pr(arch, token, models.get(arch, [])):
            filed.append((arch, token, None))
        else:
            filed.append((arch, token, "PR-OPENED"))
    return filed


def _self_test():
    """Guard the over-fires this module already shipped. 0 = ok, 1 = broken."""
    bad = []
    known = _known_mappings()

    # A missing/short table would make every check below pass by having nothing
    # to look at — the "could not determine" that reads as "no move".
    if len(known) < 100:
        bad.append(f"parsed only {len(known)} aliases from {ENGINE} — cannot "
                   f"judge anything, and a missing header must not pass")
        print("census_autopr --self-test: FAIL")
        for b in bad:
            print("  - " + b)
        return 1

    # The engine's chain is first-match-wins; the table must agree with it.
    first = {}
    with open(ENGINE) as f:
        for line in f:
            m = re.search(r'strcmp\(s,\s*"([a-z0-9_]+)"\)\s*==\s*0\)\s*return\s+(RCPP_ARCH_[A-Z0-9_]+)', line)
            if m:
                first.setdefault(m.group(1), m.group(2))
    disagree = sorted(a for a, t in first.items() if known.get(a) != t)
    if disagree:
        bad.append(f"table disagrees with the engine (first match wins) on "
                   f"{len(disagree)} alias(es), e.g. {disagree[:3]}")

    # Rule 1 must still recognize a real rename — this is the case that files.
    if _guess_ex("kimik3") != ("kimi_k3", "RCPP_ARCH_KIMI_K3", "exact"):
        bad.append(f'kimik3 no longer an exact match: {_guess_ex("kimik3")!r}')

    # The two draft PRs that were closed unjustified must not be fileable again.
    for arch in ("picolm", "language"):
        got = _guess_ex(arch)
        if got and got[2] == "exact":
            bad.append(f"{arch} resolves exactly to {got[1]} — would file a PR")

    # A class that has been REVIEWED must not be re-proposed, whatever its name
    # scores. `haiku` and `picolm` are both documented in arch-gaps.md now.
    if not _documented_in_arch_gaps("haiku"):
        bad.append("haiku is documented in Testing/arch-gaps.md but reads as "
                   "undocumented — the review record is not being consulted")
    if _documented_in_arch_gaps("zzzznotaclass"):
        bad.append("a name that appears in no heading reads as documented")
    # `picolm` is the discriminating case: documented AND still a candidate, so
    # the pair below is satisfiable only if the review record is what suppresses
    # it. Asserting the suppression alone would pass for the wrong reason — as a
    # first version of this check did, using `haiku`, which no longer produces a
    # candidate at all.
    if _guess_ex("picolm") is None:
        bad.append("picolm no longer produces a candidate — the arch-gaps "
                   "suppression assertion below would pass vacuously")
    if not _documented_in_arch_gaps("picolm"):
        bad.append("picolm is recorded in arch-gaps.md but reads as undocumented")
    _prev_skip = os.environ.pop("CENSUS_SKIP_PR", None)
    try:
        proposed = maybe_file_draft_pr(["picolm", "haiku"], dry_run=True)
    finally:
        if _prev_skip is not None:
            os.environ["CENSUS_SKIP_PR"] = _prev_skip
    if proposed:
        bad.append(f"class(es) documented as reviewed were still proposed: {proposed}")

    # No alias shorter than this may act as a family-prefix wildcard (`h` made
    # every h-initial class "LLaMA"). The bound is a LITERAL on purpose: a first
    # version of this check skipped keys using _MIN_FAMILY_LEN, so lowering that
    # constant to re-introduce the bug also switched the guard off — it passed
    # while broken. A guard must not be parameterised by what it guards.
    _NEVER_A_FAMILY_BELOW = 4
    for s in sorted(known):
        ns = _norm(s)
        if len(ns) >= _NEVER_A_FAMILY_BELOW:
            continue
        probe = ns + "xq"
        got = _guess_ex(probe)
        if got and _norm(got[0]) == ns:
            bad.append(f'short alias "{s}" acts as a wildcard: '
                       f'{probe} -> {got[0]} ({got[2]})')

    if bad:
        print("census_autopr --self-test: FAIL")
        for b in bad:
            print("  - " + b)
        return 1
    print(f"census_autopr --self-test: PASS — {len(known)} aliases, "
          f"exact-only filing, no short-alias wildcard")
    return 0


if __name__ == "__main__":
    # CLI: census_autopr.py <uncovered_arch> [model_ids...] | --self-test
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        sys.exit(_self_test())
    if len(sys.argv) < 2:
        print("usage: census_autopr.py <uncovered_arch> [model_ids...]\n"
              "       census_autopr.py --self-test", file=sys.stderr)
        sys.exit(2)
    maybe_file_draft_pr([sys.argv[1]], {sys.argv[1]: sys.argv[2:]})
