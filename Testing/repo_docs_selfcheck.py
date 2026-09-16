#!/usr/bin/env python3
"""Docs-and-repo consistency, precision first: does what we SAY match what IS?

Every check here exists because a real, user-visible mismatch got through, and
each one is written against a false positive that a naive version produced while
this was being built — those notes are in the code, because a checker that cries
wolf gets ignored and then catches nothing.

Checks:
  1. relative markdown links that point at nothing
  2. repo paths named in a doc that resolve nowhere — resolved against the repo
     root, the enclosing section's own base, and the doc's directory
  3. commands from bash-fenced / `$ ` lines whose target does not exist
     (CLI subcommands, run.sh, make targets, referenced scripts)
  4. CI workflows that invoke a script which is not in the tree, and trigger
     `paths:` entries naming files that do not exist (a trigger for a file that
     is not there can never fire — a gate that silently never runs)
  5. `-D` variables in OUR configure commands that CMakeLists.txt does not
     declare (and that are not standard CMake variables) — a documented knob
     that does nothing;

  6. flags passed to OUR tools (1bit, run.sh, scripts/, packaging/, tools/,
     Testing/) that appear nowhere in the source — a documented knob nothing
     parses. Flags of other people's tools on the same line are ignored.

Two scopes, because they are different defects. The GATED surface is what a user
follows — the front page, the guides and wiki, the packaging and site READMEs —
and a finding there fails the run. Everything else (plans, goals, issue records,
internal design docs) is printed as advisory: a stale path in a three-week-old
plan is not the same defect as one on the front page, and gating on it would keep
the check red until the whole archive is tidy. `--all` gates both.

Deliberately NOT checked:
  * historical records (CHANGELOG, AUDIT_ISSUES, docs/archive, docs/journey.md,
    docs/research/, research/) — they describe a past state, so they are correct
    as history;
  * packaging/live/build/rootfs — a copied Ubuntu root filesystem, not our content;
  * version drift — scripts/sync-version.sh --check owns that, and the Version
    consistency CI job runs it;
  * documented environment variables: also measured and dropped. A variable name in
    backticks is indistinguishable from a BackendType enum value (docs/CODEBASE.md
    lists NPU_XRT, ZINC_GPU, Q4NX_FUSION), a CMake option (docs/guides/building.md
    lists ZAYA_ENABLE_GPU_DECODE beside CMAKE_HIP_ARCHITECTURES) or another tool's
    variable (OPENAI_API_BASE). Filtering by our own prefixes still left 26
    "orphans" and every one triaged to one of those three. Flags ARE checked —
    the same scan over command lines produced 0 findings and no false positives;
  * "tracked while .gitignore says generated": it cannot be made precise. Without
    --no-index git skips tracked files so it never fires; with it, 239
    deliberately committed files match a broad ignore pattern and drown the
    signal. The one real instance (snap/prime/, a snap build artifact) was fixed
    directly instead.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXCLUDE_DIRS = (
    ".git", "build", "build100", "third_party", "node_modules", ".gitnexus",
    "site/downloads", "packaging/live/build/rootfs", "packaging/build",
    "docs/archive", "docs/superpowers", "docs/research", "research", "spec-decode",
    "mobile", "integrations", "community", "benchmarks", "experimental",
    "npu-infer", "fastflowlm_analysis", "hackathon", "docs-site", ".superpowers",
)
# Historical records: they describe a past state and are correct as history.
# docs/AGENT-COORDINATION.md is here on purpose too — it is the cross-machine
# handoff ledger and names files that live in other clones/worktrees by design
# (that is the point of it), so a path that misses this tree is expected.
EXCLUDE_FILES = {"CHANGELOG.md", "AUDIT_ISSUES.md", "TODO_TRACKING.md",
                 "docs/journey.md", "docs/AGENT-COORDINATION.md"}

# A path mentioned in a doc may be relative to the repo root, to the section it
# sits under, or to the document itself. CODEBASE.md is written section-relative
# ("### 7.1 NPU … (`engine/npu/`)" then "`src/zaya_decode.cpp`"), so a checker
# that only tries the repo root reports a dozen paths that are all correct.
HEADING_BASE = re.compile(r"^#{1,6}\s+.*?\(`([^`]+)`")
PATH_TOKEN = re.compile(r"`([^`\n]+)`")
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
PATHISH = re.compile(r"^(?:\./)?(?:[\w.-]+/)+(?:[\w.-]+\.[A-Za-z0-9]{1,6}|[\w.-]+/)$")

# Docs legitimately cite files in OTHER trees — docs/aiesim-debugging.md names
# `include/aie/Runtime/TxnEncoding.h` and `cmake/modulesXilinx` while explaining
# that they are patches to the `~/mlir-aie` checkout. A path on a line (or under a
# heading) that names a foreign tree is not a claim about this repo.
EXTERNAL_TREES = ("mlir-aie", "torch2aie", "llama.cpp", "xdna-driver", "peano",
                  "iron/", "xrt", "vitis", "ggml", "fastflowlm", "lemonade",
                  "hrx-system", "torch2aie", "aiecompiler", "chesscc")

# A doc may also name a file precisely BECAUSE it is gone ("src/codec_decoder.cpp
# went with the voice-cloning stack"). That is history, not a broken pointer.
REMOVAL_WORDS = ("went with", "is gone", "are gone", "was removed", "were removed",
                 "retired", "deleted", "no longer", "removed in", "self-reverted",
                 "renamed", "superseded", "is now")

# A file the doc plans to write is not a file the doc claims exists.
PLANNED_WORDS = ("next steps", "roadmap", "planned", "to be written", "will be added",
                 "not yet")
PLANNED_LINE = re.compile(r"^\s*(?:[-*]\s*)?(?:\d+\.\s*)?(?:Write|Create|Add|Port)\s+`")

# `feat/jarvis-v2-rewrite`, `fix/…`, `goal/…` on a line mean the line is talking
# about a branch, the same way a commit hash does.
BRANCHISH = re.compile(r"\b(?:feat|fix|chore|docs|test|refactor|goal|rebuild|run|census|npu)/[\w.-]+")


# The gated surface is what a user follows: the front page, the guides and wiki,
# the packaging and site READMEs. Everything else (plans, goals, issue records,
# internal design docs) is still audited and printed, but as advisory — a stale
# path in a plan from three weeks ago is not the same defect as one on the front
# page, and gating on it would keep the check red until the whole archive is tidy.
GATE_PREFIXES = ("docs/guides/", "docs/wiki/", "docs/model-families/")
GATE_FILES = {"README.md", "CONTRIBUTING.md", "SECURITY.md", "CODE_OF_CONDUCT.md",
              "packaging/README.md", "packaging/iso/README.md", "site/README.md"}


def gated(relpath: str, mode: str) -> bool:
    if mode == "all":
        return True
    # A CI workflow is not a doc that ages: a trigger naming a file that is not
    # there, or a step invoking a script that does not exist, is a gate that
    # silently never runs. Those always gate, wherever they sit.
    if relpath.startswith(".github/"):
        return True
    if relpath in GATE_FILES or relpath.startswith(GATE_PREFIXES):
        return True
    if relpath.endswith(".md") and relpath.count("/") == 0:
        return True                      # root docs
    if relpath.startswith("docs/") and relpath.count("/") == 1:
        return True                      # docs/*.md
    return False


def included(relpath: str) -> bool:
    if relpath in EXCLUDE_FILES:
        return False
    # dated triage records describe a past state, like the other history files
    if re.match(r"docs/TRIAGE-ISSUES-.*\.md$", relpath):
        return False
    return not any(relpath == d or relpath.startswith(d + "/") for d in EXCLUDE_DIRS)


def markdown_files():
    for dp, dns, fns in os.walk(ROOT):
        rel = str(Path(dp).relative_to(ROOT))
        if rel != "." and not included(rel):
            dns[:] = []
            continue
        for fn in fns:
            if fn.endswith(".md"):
                p = Path(dp) / fn
                if included(str(p.relative_to(ROOT))):
                    yield p


def git(*args: str, input: str | None = None) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          input=input).stdout


def resolves(doc: Path, section_base: str | None, token: str) -> bool:
    token = token.rstrip("/")
    candidates = [ROOT / token]
    if section_base:
        candidates.append((ROOT / section_base.strip("./")) / token)
    candidates.append(doc.parent / token)
    # engine/npu/tests/README-COMPILER-AB.md says `tests/bench_gemm_analytical.cpp`
    # for engine/npu/tests/bench_gemm_analytical.cpp — the doc names its component
    # tree from the component root, one level up from itself.
    candidates.append(doc.parent.parent / token)
    return any(c.exists() for c in candidates)


def command_lines(text: str):
    """Yield (first_lineno, logical_command, fence_mentions_external_cd).

    Backslash continuations are joined first: a cmake invocation and the -D flags
    it sets are one command spread over several lines, and a rule that looks at
    the `cmake` line alone sees none of them.
    """
    fence = None
    external_cd = False
    foreign_tree = False
    pending = ""
    start = 0
    for i, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if s.startswith("```"):
            lang = s[3:].strip().lower()
            if fence is None:
                fence = lang
                external_cd = False
                foreign_tree = False
            else:
                fence = None
            pending = ""
            continue
        if re.search(r"cd\s+(~|/|\$HOME|\$\{HOME|\.\./|third_party/|path/to)", s):
            external_cd = True  # another project's tree, e.g. ~/torch2aie, third_party/llama.cpp
        # Narrower, for the -D rule: a `cd ~`/`cd <our clone>` is still OUR tree —
        # the main build example does exactly that, and treating it as foreign is
        # how the first version of this rule managed to miss the very line it was
        # written for.
        if re.search(r"cd\s+[^\n]*(third_party/|torch2aie|path/to|\.\./)", s):
            foreign_tree = True
        in_code = fence in ("bash", "sh", "shell", "console", "")
        if not in_code and fence is not None:
            continue
        if not in_code and not s.startswith("$ "):
            continue
        body = s[2:] if (not in_code and s.startswith("$ ")) else s
        if not pending:
            start = i
        pending = f"{pending} {body}".strip() if pending else body
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue  # continuation: keep collecting
        yield start, pending, (external_cd, foreign_tree)
        pending = ""


# `-D` variables that are not ours to declare: standard CMake variables, which
# are legitimate on a configure line without an option() entry.
STANDARD_CMAKE_VARS = {
    "CMAKE_BUILD_TYPE", "CMAKE_HIP_ARCHITECTURES", "CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER",
    "CMAKE_HIP_COMPILER", "CMAKE_INSTALL_PREFIX", "CMAKE_PREFIX_PATH", "CMAKE_CUDA_ARCHITECTURES",
    "CMAKE_EXPORT_COMPILE_COMMANDS", "CMAKE_TOOLCHAIN_FILE", "CMAKE_VERBOSE_MAKEFILE",
    "BUILD_SHARED_LIBS", "BUILD_TESTING",
}


def declared_cmake_options() -> set[str]:
    names: set[str] = set()
    for pat in ("CMakeLists.txt", "cmake/*.cmake", "packaging/**/*.cmake"):
        for f in ROOT.glob(pat):
            names |= set(re.findall(r"^\s*option\s*\(\s*([A-Za-z0-9_]+)",
                                    f.read_text(encoding="utf-8", errors="replace"), re.M))
    return names


def source_corpus() -> str:
    """Every line of our own source, for "does this flag exist anywhere" checks."""
    exts = (".cpp", ".h", ".hpp", ".cc", ".sh", ".py", ".yml", ".yaml")
    roots = ("src", "tools", "engine", "tests", "scripts", "packaging", "Testing", "cmake")
    out: list[str] = []
    for r in roots:
        for f in (ROOT / r).rglob("*"):
            if f.suffix in exts and f.is_file():
                rel = str(f.relative_to(ROOT))
                if included(rel):
                    out.append(f.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(out)


def make_targets() -> set[str]:
    targets: set[str] = set()
    for mf in ROOT.rglob("Makefile*"):
        rel = str(mf.relative_to(ROOT))
        if not included(rel):
            continue
        for line in mf.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"^([A-Za-z][\w.-]*)\s*:(?!=)", line)
            if m:
                targets.add(m.group(1))
    return targets


def main() -> int:
    mode = "all" if "--all" in sys.argv else "gate"
    findings: list[tuple[str, int, str]] = []
    version = (ROOT / "VERSION").read_text().strip()

    onebin = (ROOT / "tools/onebin.cpp").read_text(encoding="utf-8", errors="replace")
    cli = set(re.findall(r'cmd\s*==\s*"([\w.-]+)"', onebin))
    cli |= set(re.findall(r'prog\s*==\s*"([\w.-]+)"', onebin))
    targets = make_targets()
    tracked = set(git("ls-files").split())
    corpus = source_corpus()
    cmake_options = declared_cmake_options()
    FLAG = re.compile(r"--[a-z][a-z0-9][a-z0-9-]+")
    OUR_TOOL = re.compile(r"^(?:1bit|run\.sh|scripts/[\w.-]+|packaging/[\w.-]+|Testing/[\w.-]+|tools/[\w.-]+)")

    # ── 1 + 2 + 3: walk the docs once ──
    for doc in markdown_files():
        rel = str(doc.relative_to(ROOT))
        text = doc.read_text(encoding="utf-8", errors="replace")
        section_base = None
        section_text = ""
        lines = text.splitlines()
        for lineno, line in enumerate(lines, 1):
            if line.startswith("#"):
                section_text = line
            hb = HEADING_BASE.match(line)
            if hb:
                section_base = hb.group(1)
            # A path named next to a commit hash or the word "branch" is a claim
            # about another branch, not about this tree — packaging/iso/README.md
            # documents a launcher that existed in one commit of another branch
            # and was self-reverted (`96578e2c` … "(on that branch)").
            context = " ".join(lines[max(0, lineno - 12):lineno + 1])
            low_ctx = (context + " " + section_text).lower()
            branch_context = bool(re.search(r"\b[\da-f]{7,40}\b", context)) or "branch" in low_ctx
            foreign = any(t in low_ctx for t in EXTERNAL_TREES)
            removed = any(w in low_ctx for w in REMOVAL_WORDS)
            planned = any(w in low_ctx for w in PLANNED_WORDS) or bool(PLANNED_LINE.match(line))
            branch_context = branch_context or bool(BRANCHISH.search(line))

            for m in LINK.finditer(line):
                t = m.group(1).split("#")[0]
                if not t or t.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                if t.startswith("/") or ".." in t.split("/"):
                    continue  # absolute or out-of-tree: a different question
                if branch_context:
                    continue
                if not (doc.parent / t).exists():
                    findings.append((rel, lineno, f"link points at nothing: {t}"))

            for m in PATH_TOKEN.finditer(line):
                token = m.group(1).strip()
                if not PATHISH.match(token) or token.startswith(("http", "/", "~")):
                    continue
                if "..." in token or "<" in token:
                    continue  # an abbreviated or templated path
                if not token.startswith(("src/", "docs/", "scripts/", "tools/", "packaging/",
                                         "Testing/", "engine/", "include/", "site/", "cmake/",
                                         "kernels/", "snap/", "tests/", "spec-decode/")):
                    continue
                if branch_context or foreign or removed or planned:
                    continue
                if not resolves(doc, section_base, token):
                    findings.append((rel, lineno, f"path does not resolve: {token}"))

        for lineno, cmd, tree_flags in command_lines(text):
            external_cd, foreign_tree = tree_flags
            # our own configure lines, before the tool match below (which does not
            # cover `cmake`, so this rule used to sit behind an unreachable branch)
            if not foreign_tree and re.match(r"^cmake\b", cmd):
                for var in re.findall(r"-D([A-Za-z][A-Za-z0-9_]*)", cmd):
                    if var not in cmake_options and var not in STANDARD_CMAKE_VARS:
                        findings.append((rel, lineno,
                                         f"-D{var}: CMakeLists.txt does not declare this option"))
            m = re.match(r"^(?:\$\s*)?(?:\./)?(?:build/)?(1bit|run\.sh|make|scripts/[\w.-]+\.sh|packaging/[\w.-]+\.sh)\b\s*(.*)$", cmd)
            if not m:
                continue
            tool, rest = m.group(1), m.group(2).strip()
            args = rest.split()
            # `make -C ~/elsewhere …` / `cd ~/torch2aie && make full-build`:
            # another repository's Makefile — its targets are not ours to know.
            if external_cd:
                continue
            if tool == "make" and "-C" in args:
                idx = args.index("-C")
                if idx + 1 < len(args) and (args[idx + 1].startswith(("~", "/", "$"))):
                    continue
            if tool == "1bit" and args and not args[0].startswith("-") and args[0] not in cli | {"zaya", "unified"}:
                findings.append((rel, lineno, f"`1bit {args[0]}` is not a subcommand"))
            elif tool == "run.sh" and args and args[0] not in cli | {"chat"}:
                findings.append((rel, lineno, f"`./run.sh {args[0]}` is not a subcommand"))
            elif tool == "make" and args:
                for a in args:
                    if a.startswith("-") or "=" in a or "/" in a:
                        continue
                    if a not in targets and a not in {"install", "clean", "test", "all"}:
                        findings.append((rel, lineno, f"`make {a}` — no such target"))
                    break
            elif tool.endswith(".sh") and not (ROOT / tool).exists():
                findings.append((rel, lineno, f"script does not exist: {tool}"))
            if OUR_TOOL.match(tool):
                # Everything after a `#` belongs to a comment, not to the command:
                # docs/mobile/RUNBOOK.md mentions `systemd-run --unit=…` in a
                # trailing comment on a line that starts with our binary, and
                # --unit is systemd-run's flag, not ours.
                for flag in FLAG.findall(rest.split("#", 1)[0]):
                    if flag not in corpus:
                        findings.append((rel, lineno, f"{tool} {flag} — no such flag in the source"))

    # ── 4: CI invokes a script that is not in the tree ──
    # Note on a rule that is deliberately absent: "tracked while .gitignore says
    # it is generated" cannot be made precise. Without --no-index git skips
    # tracked files, so the rule never fires at all; with it, 239 deliberately
    # committed files match a broad ignore pattern (*.bin, benchmark sources) and
    # the signal drowns. The one real instance found (snap/prime/, a snap build
    # artifact committed despite .gitignore) was fixed directly instead.
    #
    # A path in a `paths:` trigger list is a subscription, not an invocation, and
    # a call guarded by `[ -f … ]` is a deliberate retirement — validate-claims.yml
    # documents exactly that for its retired drivers. Neither is a finding.
    for wf in (ROOT / ".github/workflows").glob("*.yml"):
        lines = wf.read_text(encoding="utf-8", errors="replace").splitlines()
        in_paths = False
        for i, line in enumerate(lines, 1):
            if re.match(r"^\s*paths(-ignore)?:", line):
                in_paths = True
                continue
            if in_paths:
                entry = re.match(r"^\s*-\s*[\'\"]?([^\'\"\s]+)", line)
                if entry:
                    pat = entry.group(1)
                    # a glob cannot be judged by existence
                    if not any(c in pat for c in "*?["):
                        if not (ROOT / pat).exists():
                            findings.append((str(wf.relative_to(ROOT)), i,
                                             f"trigger path names nothing: {pat}"))
                    continue
                if re.match(r"^\s{0,4}\w+:", line):
                    in_paths = False
            if line.lstrip().startswith("#"):
                continue  # a comment may describe a retired script without invoking it
            for m in re.finditer(r"\b((?:scripts|tools|packaging)/[\w./-]+\.(?:sh|py))\b", line):
                ref = m.group(1)
                if in_paths or (ROOT / ref).exists():
                    continue
                window = "\n".join(lines[max(0, i - 8):i + 1])
                if re.search(r"\[\s*-[fd]\s", window):
                    continue
                findings.append((str(wf.relative_to(ROOT)), i, f"invokes missing {ref}"))

    gating = [f for f in findings if gated(f[0], mode)]
    advisory = [f for f in findings if not gated(f[0], mode)]

    for doc, line, msg in advisory:
        print(f"  advisory  {doc}:{line}: {msg}")
    if advisory:
        print(f"  ({len(advisory)} advisory finding(s) outside the gated surface — "
              f"history, plans and internal design docs; run with --all to gate them too)\n")

    if gating:
        print(f"repo consistency FAILED — {len(gating)} finding(s) in the user-facing docs:")
        for doc, line, msg in gating[:40]:
            print(f"  - {doc}:{line}: {msg}")
        if len(gating) > 40:
            print(f"  … and {len(gating) - 40} more")
        return 1
    print(f"repo consistency ok — {len(advisory)} advisory finding(s), 0 in the gated surface")
    return 0


if __name__ == "__main__":
    sys.exit(main())
