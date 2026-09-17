#!/usr/bin/env python3
"""check_xclbin_provenance.py — static provenance + consistency check for engine/npu/xclbins.

INTENT
------
The NPU xclbin set is tracked in git precisely because the MLIR/chess toolchain cannot
reliably rebuild it (see engine/npu/build_npu_ternary.sh's header: "the aiecc MLIR
toolchain has a pre-existing version mismatch that prevents clean builds"). That makes the
*committed* artifacts the source of truth, and it means two classes of defect can only be
caught by inspecting the committed set rather than by building it:

  1. Hygiene regressions that a clean checkout reproduces deterministically. An in-tree
     alias must stay inside engine/npu/xclbins/ and name a tracked sibling *.xclbin - a link
     that escapes the directory, or names a path the commit does not carry, is broken in a
     fresh clone no matter which machine made it. Third-party links may also be tracked,
     pointing into a machine-local FastFlowLM install (/opt/fastflowlm/...), and for those
     whether they resolve is HOST state, not a property of the commit: the check asserts
     their TARGETS - nobody may repoint them, materialise them, or drop a different binary
     in their place - and reports the dangling/resolving split as host information instead
     of asserting it. (Asserting the split was an earlier design error: it made the gate red
     on any box with fastflowlm installed while staying green in CI, which is how a check
     gets ignored. A clean checkout does not get those files; a box with FLM does. No such
     link is tracked today - the four that motivated this mechanism were deleted in #2218 -
     but the invariant is what the check is for, so it stays.)
  2. Provenance loss. The artifacts record their producer, format version, UUID and
     timestamp, but no toolchain version, no source revision and no generating script - so
     a green rebuild could never be shown to reproduce what ships. PROVENANCE.json is the
     missing half: a manifest keyed by the embedded XclBinUUID, recording the content hash,
     the aliases, and (to be filled by the build) the toolchain and script revision.

This check runs with stdlib Python only and needs no NPU, so it can gate on stock CI
runners. It asserts hygiene invariants that need no baseline, then compares the observed
tree against the committed manifest; a legitimate change to the artifact set is landed by
regenerating the manifest in the same commit (--write-manifest), which is the review point.
Every field the manifest records about the committed set is compared, not a hand-picked
subset of it: population.tracked_paths_under_dir counted 519 against a tree of 520 without
the gate noticing (issue #2513), which is what a recorded-but-unread field buys you. The
only exclusions are the two host-dependent population keys, named in HOST_DEPENDENT_KEYS.

POPULATION, UNITS, TIMEZONE (stated because a census is meaningless without them)
--------------------------------------------------------------------------------
* Population: tracked paths under engine/npu/xclbins/ (via `git ls-files -s`), not the
  working directory listing - the tracked set is what a clean checkout gets.
* Top-level *.xclbin entries = regular artifacts + symlinks. Directory recursion is
  deliberately excluded from the artifact counts.
* Content hash: sha256 of the artifact bytes. Distinctness key: the embedded XclBinUUID.
* Timezone: census days are UTC, derived from the embedded AXLF TimeStamp. Never mtime -
  102 of the files carry one checkout mtime (2026-08-20 00:49), which yields a different
  census to anyone reaching for `ls -l`.
* Sizes are bytes; percentages are unit-independent (MB decimal vs MiB binary differ).

EXIT CODES
----------
0  consistent (and, if a manifest exists, matching)
1  violation (details on stderr/stdout)
2  usage or environment error (not a git work tree, unreadable artifact, ...)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

XCLBIN_DIR = "engine/npu/xclbins"
MANIFEST_NAME = "PROVENANCE.json"

UUID_RE = re.compile(r'"XclBinUUID":"([0-9a-fA-F]{32})"')
TS_RE = re.compile(r'"TimeStamp":"(\d+)"')
DANGLING_TARGET_PREFIX = "/opt/fastflowlm/"

MODE_SYMLINK = "120000"


class Failure(Exception):
    """A conformance violation (exit 1)."""


class EnvError(Exception):
    """A usage/environment problem (exit 2)."""


def git_tracked(root: Path) -> list[tuple[str, str]]:
    """[(repo-relative path, git mode)] for every tracked path under XCLBIN_DIR."""
    proc = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-s", "--", XCLBIN_DIR],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise EnvError(f"git ls-files failed in {root}: {proc.stderr.strip()}")
    rows = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        meta, _, path = line.partition("\t")
        fields = meta.split()
        if len(fields) < 2 or not path:
            raise EnvError(f"unparseable ls-files line: {line!r}")
        rows.append((path, fields[0]))
    if not rows:
        raise EnvError(f"no tracked paths under {XCLBIN_DIR}/ - wrong root or empty checkout")
    return rows


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError as exc:
        raise EnvError(f"cannot read {path}: {exc}") from exc
    return h.hexdigest()


def read_artifact(path: Path) -> tuple[str, int]:
    """(XclBinUUID, TimeStamp epoch seconds) from the embedded AXLF mirror block."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise EnvError(f"cannot read {path}: {exc}") from exc
    text = raw.decode("latin-1", "replace")
    uuid_m = UUID_RE.search(text)
    ts_m = TS_RE.search(text)
    if not uuid_m or not ts_m:
        raise Failure(
            f"{path.name}: no XclBinUUID/TimeStamp in the AXLF metadata "
            f"(uuid={'yes' if uuid_m else 'NO'}, timestamp={'yes' if ts_m else 'NO'})"
        )
    return uuid_m.group(1).lower(), int(ts_m.group(1))


def observe(root: Path) -> dict:
    """Snapshot the tracked artifact set: population, artifacts, symlinks, census."""
    rows = git_tracked(root)
    top_level = [p for p, _ in rows if os.path.dirname(p) == XCLBIN_DIR]
    top_xclbins = sorted(p for p in top_level if p.endswith(".xclbin"))
    if not top_xclbins:
        raise EnvError(f"no top-level *.xclbin tracked under {XCLBIN_DIR}/")

    modes = dict(rows)
    symlinks: dict[str, str] = {}
    regular: list[str] = []
    for p in top_xclbins:
        path = root / p
        if modes[p] == MODE_SYMLINK:
            if not os.path.islink(path):
                raise Failure(
                    f"{p}: tracked as a symlink (git mode 120000) but the worktree entry "
                    "is not a symlink - an alias was materialised into a real file"
                )
            symlinks[p] = os.readlink(path)
        else:
            if not path.exists():
                raise Failure(f"{p}: tracked artifact is missing from the worktree")
            regular.append(p)
    txt_symlinks = [p for p in top_level if p.endswith(".txt") and modes[p] == MODE_SYMLINK]

    # which symlinks resolve, and which are declared-dangling (target absent by design)
    dangling: dict[str, str] = {}
    resolving: dict[str, str] = {}
    for p, target in symlinks.items():
        resolution = target if os.path.isabs(target) else os.path.join(os.path.dirname(p), target)
        if (root / resolution).exists():
            resolving[p] = target
        else:
            dangling[p] = target

    artifacts: dict[str, dict] = {}
    for p in regular:
        path = root / p
        uuid, ts = read_artifact(path)
        digest = sha256_of(path)
        entry = artifacts.setdefault(
            uuid,
            {"sha256": digest, "timestamp_utc": None, "paths": []},
        )
        if entry["sha256"] != digest:
            raise Failure(
                "duplicate UUID with differing bytes - dedup-by-UUID is unsafe:\n"
                f"  uuid {uuid}\n"
                f"    {entry['sha256']}  (already seen)\n"
                f"    {digest}  ({path.name})"
            )
        entry["paths"].append(os.path.basename(p))
        stamp = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if entry["timestamp_utc"] is None:
            entry["timestamp_utc"] = stamp
        elif stamp < entry["timestamp_utc"]:
            entry["timestamp_utc"] = stamp

    for entry in artifacts.values():
        entry["paths"].sort()
        entry["aliases"] = entry["paths"][1:]

    # census over distinct artifacts, UTC days
    days: dict[str, int] = {}
    for entry in artifacts.values():
        day = (entry["timestamp_utc"] or "")[:10]
        if day:
            days[day] = days.get(day, 0) + 1
    top_days = sorted(days.items(), key=lambda kv: (-kv[1], kv[0]))

    payload_bytes = sum((root / p).stat().st_size for p in regular)
    # redundant = the second and later copies of a duplicated artifact
    redundant_bytes = 0
    for entry in artifacts.values():
        if len(entry["paths"]) > 1:
            size = (root / XCLBIN_DIR / entry["paths"][0]).stat().st_size
            redundant_bytes += size * (len(entry["paths"]) - 1)

    # OBSERVED, never asserted: the model-tagged op set pairs final_i8_<op>_<model>.xclbin with
    # insts_i8_<op>_<model>.txt. ATTN has no insts, and no engine in the tree references ATTN at
    # all (grep for final_i8_ATTN/insts_i8_ATTN over engine/ = 0; the only engines to name ops are
    # npu_engine_fused/gpurender/overlap and the two zero_copy tests for GU+D, and
    # npu_engine_spec for QKV+O) - so pairing was never an invariant of the tree, and asserting
    # it would encode a non-invariant as one. Same error class as asserting a link's resolution
    # state. Recorded as information so an inconsistency is visible without being a verdict.
    xclbin_ops = {}
    for p_ in regular:
        name = os.path.basename(p_)
        if name.startswith("final_i8_") and name.endswith(".xclbin"):
            xclbin_ops[name[len("final_i8_"):-len(".xclbin")]] = name
    insts_ops = {
        os.path.basename(q)[len("insts_i8_"):-len(".txt")]
        for q in top_level
        if os.path.basename(q).startswith("insts_i8_") and q.endswith(".txt")
    }
    unpaired = sorted(op for op in xclbin_ops if op not in insts_ops)

    return {
        "population": {
            "tracked_paths_under_dir": len(rows),
            "tracked_top_level_entries": len(top_level),
            "top_level_xclbin": len(top_xclbins),
            "regular_artifacts": len(regular),
            "xclbin_symlinks": len(symlinks),
            "alias_symlinks_in_tree": len(resolving),
            "dangling_symlinks": len(dangling),
            "txt_symlinks": len(txt_symlinks),
            "distinct_builds_by_uuid": len(artifacts),
        },
        "census": {
            "unit": "one row per distinct XclBinUUID",
            "timezone": "UTC (embedded AXLF TimeStamp, never mtime)",
            "distinct_builds": len(artifacts),
            "days_utc": dict(sorted(days.items())),
            "top_days_utc": [{"day": d, "builds": n} for d, n in top_days[:5]],
            "payload_bytes": payload_bytes,
            "redundant_bytes": redundant_bytes,
            "redundant_percent": round(redundant_bytes * 100.0 / payload_bytes, 2)
            if payload_bytes
            else 0.0,
        },
        "observed": {
            "note": "host/path-variant facts, reported not asserted",
            "unpaired_model_xclbins": unpaired,
            "pairs_with_insts": sorted(op for op in xclbin_ops if op in insts_ops),
        },
        "declared_dangling": dict(sorted(dangling.items())),
        "symlinks": dict(sorted(symlinks.items())),
        "artifacts": dict(sorted(artifacts.items())),
    }


def hygiene(obs: dict) -> list[str]:
    """Invariants that need no baseline. Returns warnings; raises Failure on violations."""
    notes: list[str] = []

    # (b) dangling links must be declared, and must not be "fixed" with a foreign binary
    for path, target in obs["declared_dangling"].items():
        if not target.startswith(DANGLING_TARGET_PREFIX):
            notes.append(
                f"{path}: dangling to {target} - an unexpected dangling target; "
                "if this is intentional, regenerate the manifest"
            )

    # (a) An in-tree alias must stay inside the artifact directory and name a tracked,
    #     regular sibling artifact. This needs no baseline, so it still holds after the
    #     same commit regenerates the manifest - which is the case the manifest diff
    #     cannot see. Absolute targets are the declared third-party case: whether they
    #     resolve is host state (noted above), and which target each one carries is
    #     recorded and asserted by compare().
    #
    #     This replaces two blocks that could not fail. One asked whether a symlink's
    #     basename was also a regular artifact - impossible, since observe() classifies
    #     each top-level path as a symlink XOR a regular file. The other walked every
    #     (UUID group x symlink) pair looking for an alias that left its group, but
    #     artifacts[uuid]["paths"] only ever holds *regular* names, so its guard
    #     `if name not in regular: continue` fired on all 1068 iterations and the
    #     assertion below it was unreachable. The invariant they were reaching for is the
    #     one asserted here; "still points at the same twin" is a manifest fact and is
    #     asserted by compare()'s whole-map symlink comparison.
    tracked_regular = {
        f"{XCLBIN_DIR}/{name}" for entry in obs["artifacts"].values() for name in entry["paths"]
    }
    for path, target in sorted(obs["symlinks"].items()):
        if os.path.isabs(target):
            continue
        if ".." in Path(target).parts:
            raise Failure(
                f"{path}: target {target!r} leaves {XCLBIN_DIR}/ - a tracked alias may only "
                "name a sibling artifact, never a path outside the artifact directory "
                "(a fresh clone gets the link, not the bytes)"
            )
        resolved = f"{XCLBIN_DIR}/{target}"
        if resolved not in tracked_regular:
            raise Failure(
                f"{path}: target {target!r} is not a tracked *.xclbin directly under "
                f"{XCLBIN_DIR}/ (resolves to {resolved}) - a fresh clone would not get it"
            )
    return notes


# The two population keys that describe the HOST rather than the commit: whether a
# third-party link resolves depends on whether FastFlowLM is installed at
# /opt/fastflowlm. Comparing them made the gate red on strixhalo while staying green in
# CI, which is how a check trains people to ignore it; the caller reports them instead.
HOST_DEPENDENT_KEYS = {
    ("population", "alias_symlinks_in_tree"),
    ("population", "dangling_symlinks"),
}
# Sections of the manifest that describe the COMMITTED set. `generated`/`build` record the
# generation event and the arm that produced the artifacts, and are not properties of the
# tree, so they are not compared.
COMPARED_SECTIONS = ("population", "census", "observed")


def _walk_recorded(prefix: tuple, want, got, problems: list[str]) -> None:
    """Report every leaf of a recorded section that differs from the observation."""
    if isinstance(want, dict) and isinstance(got, dict):
        for key in sorted(set(want) | set(got)):
            if prefix + (key,) in HOST_DEPENDENT_KEYS:
                continue
            _walk_recorded(prefix + (key,), want.get(key), got.get(key), problems)
        return
    if want != got:
        problems.append(f"{'.'.join(prefix)}: manifest {want!r} != observed {got!r}")


def compare(obs: dict, manifest: dict) -> list[str]:
    """Diff the observed snapshot against the committed manifest."""
    problems: list[str] = []

    # Everything the manifest records about the committed set is compared, not a
    # hand-picked subset: population.tracked_paths_under_dir said 519 against a tree of 520
    # (one path of 4aa5fa1ca's 13 landed after the manifest was regenerated) and the gate
    # stayed green, because only four population keys were read. A field nothing reads is
    # not a check, it is a note to the future (issue #2513).
    for section in COMPARED_SECTIONS:
        _walk_recorded((section,), manifest.get(section) or {}, obs.get(section) or {}, problems)

    # The invariant is the TARGET, not the resolution state. Every recorded third-party
    # link must still point where it pointed (nobody repoints it, materialises it, or
    # puts a different binary in its place); whether that target exists on this box is
    # host state and is reported by the caller, never asserted here.
    m_dang = manifest.get("declared_dangling", {})
    for path, target in sorted(m_dang.items()):
        got = obs["symlinks"].get(path)
        if got is None:
            problems.append(f"third-party link {path} is gone (recorded target {target})")
        elif got != target:
            problems.append(f"third-party link {path} repointed: {target} -> {got}")
    # A tracked symlink that the manifest does not know at all is caught by the
    # symlinks-map comparison below (it compares the whole target map).

    m_sym = manifest.get("symlinks", {})
    if m_sym != obs["symlinks"]:
        for k in sorted(set(m_sym) | set(obs["symlinks"])):
            if m_sym.get(k) != obs["symlinks"].get(k):
                problems.append(
                    f"symlink {k}: manifest {m_sym.get(k)!r} != observed {obs['symlinks'].get(k)!r}"
                )

    m_art = manifest.get("artifacts", {})
    for uuid in sorted(set(m_art) | set(obs["artifacts"])):
        a, b = m_art.get(uuid), obs["artifacts"].get(uuid)
        if a is None:
            problems.append(f"artifact {uuid}: present now, absent from the manifest ({b['paths']})")
        elif b is None:
            problems.append(f"artifact {uuid}: in the manifest, absent now ({a.get('paths')})")
        elif a.get("sha256") != b.get("sha256"):
            problems.append(
                f"artifact {uuid}: content changed ({a.get('paths')}) "
                f"{str(a.get('sha256'))[:12]} -> {str(b.get('sha256'))[:12]}"
            )
        elif a.get("paths") != b.get("paths"):
            problems.append(
                f"artifact {uuid}: alias set changed {a.get('paths')} -> {b.get('paths')}"
            )
    return problems


def render_census(obs: dict) -> str:
    pop, cen = obs["population"], obs["census"]
    lines = [
        f"  paths under {XCLBIN_DIR}/            {pop['tracked_paths_under_dir']}",
        f"  top-level *.xclbin entries           {pop['top_level_xclbin']}",
        f"    regular artifacts                  {pop['regular_artifacts']}",
        f"    symlinks                           {pop['xclbin_symlinks']}"
        f"  ({pop['alias_symlinks_in_tree']} in-tree aliases, {pop['dangling_symlinks']} dangling)",
        f"  *.txt symlinks                       {pop['txt_symlinks']}",
        f"  distinct builds by XclBinUUID        {pop['distinct_builds_by_uuid']}",
        f"  payload / redundant (bytes)          {cen['payload_bytes']} / {cen['redundant_bytes']}"
        f"  = {cen['redundant_percent']}% redundant",
        f"  census days (UTC)                    {len(cen['days_utc'])}",
    ]
    for row in cen["top_days_utc"][:3]:
        lines.append(f"    {row['day']}  {row['builds']} builds")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None, help="repo root (default: inferred from this file)")
    ap.add_argument("--manifest", default=None, help=f"path to {MANIFEST_NAME}")
    ap.add_argument("--write-manifest", action="store_true",
                    help="regenerate the manifest from the observed tree (land the change in the same commit)")
    ap.add_argument("--toolchain", default=None, metavar="STR",
                    help="with --write-manifest, record the toolchain that produced these xclbins in "
                         "build.toolchain; without it an existing recorded value is PRESERVED, never erased")
    ap.add_argument("--script-revision", default=None, metavar="STR",
                    help="with --write-manifest, record the generating script's revision in "
                         "build.generating_script_revision; without it an existing value is preserved")
    ap.add_argument("--allow-null-toolchain", action="store_true",
                    help="with --write-manifest, permit build.toolchain to be written as null when there "
                         "is no recorded value to preserve; without it the write is REFUSED (issue #2262)")
    ap.add_argument("--allow-missing-manifest", action="store_true",
                    help="do not fail when the manifest is absent (bootstrap only)")
    args = ap.parse_args(argv)

    if not args.write_manifest and (args.toolchain is not None or args.script_revision is not None):
        print("note: --toolchain/--script-revision only apply with --write-manifest; ignoring them",
              file=sys.stderr)

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[3]
    manifest_path = Path(args.manifest) if args.manifest else root / XCLBIN_DIR / MANIFEST_NAME

    # Issue #2262. The artifacts carry no compiler marker, so build.toolchain can only come
    # from the build that produced them - this tool cannot derive it. It used to write null
    # in silence when --toolchain was omitted, which is exactly how the field stayed empty
    # through the one rebuild that was supposed to fill it. Refuse to re-record a blank
    # instead: a manifest write must either name the arm, or say explicitly that it cannot.
    # Checked here, before observe(), so the refusal does not depend on the tree being valid.
    if args.write_manifest and args.toolchain is None and not args.allow_null_toolchain:
        prev_toolchain = None
        if manifest_path.exists():
            try:
                prev_toolchain = ((json.loads(manifest_path.read_text()).get("build") or {})
                                  .get("toolchain"))
            except (OSError, json.JSONDecodeError):
                prev_toolchain = None
        if prev_toolchain is None:
            print(
                "REFUSING to write a manifest with build.toolchain = null (issue #2262).\n"
                "  The artifacts record no compiler arm or version, so this value cannot be\n"
                "  derived from them - only the build that produced them can supply it, and a\n"
                "  rebuild that forgets it is how the field stayed empty. Pass one of:\n"
                "    --toolchain \"<compiler arm + versions>\"\n"
                "        e.g. --toolchain 'aiecc (llvm-aie, LLVM 23.0.0), XRT 2.26f'\n"
                "    --allow-null-toolchain\n"
                "        to record the blank as a decision rather than a default.",
                file=sys.stderr)
            return 1

    try:
        obs = observe(root)
        notes = hygiene(obs)
    except (Failure, EnvError) as exc:
        kind = "VIOLATION" if isinstance(exc, Failure) else "ENVIRONMENT"
        print(f"{kind}: {exc}", file=sys.stderr)
        return 1 if isinstance(exc, Failure) else 2

    print(f"xclbin provenance - {root}")
    print(render_census(obs))
    for note in notes:
        print(f"  note: {note}")
    observed = obs.get("observed", {})
    unpaired = observed.get("unpaired_model_xclbins") or []
    if unpaired:
        print(f"  observed (not asserted): {len(unpaired)} model-tagged xclbin(s) with no matching "
              f"insts_i8_*.txt: {', '.join(unpaired)}")

    if args.write_manifest:
        # build.toolchain and build.generating_script_revision are the two fields this
        # tool cannot observe: the artifacts carry no compiler marker, so the values can
        # only come from the build that produced them. The writer used to hard-code both
        # to null, which silently ERASED a value a build had recorded - and this file's
        # own `intent` tells the next person to regenerate with exactly this command, so
        # following the documented flow destroyed the field. That is the "missing half of
        # the provenance" issue #2262 is open for. Precedence is now: explicit flag, then
        # the existing manifest, then null with the note below.
        previous_build: dict = {}
        previous_artifacts: set[str] = set()
        if manifest_path.exists():
            try:
                previous = json.loads(manifest_path.read_text())
                previous_build = previous.get("build") or {}
                previous_artifacts = set((previous.get("artifacts") or {}).keys())
            except (OSError, json.JSONDecodeError):
                previous_build, previous_artifacts = {}, set()
        toolchain = args.toolchain if args.toolchain is not None else previous_build.get("toolchain")
        script_revision = (args.script_revision if args.script_revision is not None
                           else previous_build.get("generating_script_revision"))
        payload = {
            "schema": 1,
            "intent": (
                "Provenance for the committed NPU xclbin set, keyed by the artifact's embedded "
                "XclBinUUID. Regenerate with engine/npu/tests/check_xclbin_provenance.py "
                "--write-manifest --toolchain \"<compiler arm + versions>\" "
                "--script-revision \"<git rev of the generating script>\" and land it in the same "
                "commit as the artifact change. Those two flags are the point of this file: the "
                "artifacts carry no compiler marker to derive them from, so a rebuild that omits "
                "--toolchain records build.toolchain as null again (issue #2262)."
            ),
            "generated": {
                "by": "pi/coding-agent",
                "at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "check": "engine/npu/tests/check_xclbin_provenance.py",
                "host": socket.gethostname(),
                "ref": subprocess.run(
                    ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                    capture_output=True, text=True).stdout.strip(),
            },
            "notes": [
                "population.dangling_symlinks and population.alias_symlinks_in_tree are "
                "HOST-DEPENDENT: they were observed on generated.host. Installing "
                "fastflowlm at /opt/fastflowlm makes third-party links resolve, which says "
                "nothing about the repo. They are the only two keys compare() excludes - "
                "every other recorded field is compared against the tree, so a counter that "
                "drifts turns the gate red instead of sitting in this file unread "
                "(issue #2513). Symlink TARGETS are asserted (the symlinks and "
                "declared_dangling maps), never this split.",
                "declared_dangling lists the third-party targets that did not resolve on "
                "generated.host. It is a target map, not a resolution requirement.",
            ],
            "build": {
                "toolchain": toolchain,
                "toolchain_note": (
                    "Empty by design, and written only because --allow-null-toolchain was "
                    "passed explicitly (issue #2262). The artifacts record no compiler arm or "
                    "version (no chesscc/peano/llvm/aiecc/clang marker in the AXLF metadata; "
                    "PlatformVBNV is empty), so this tool cannot derive one, and it now REFUSES "
                    "to write this blank by default. The build that regenerates these artifacts "
                    "is the only thing that can fill it in - that is the missing half of the "
                    "provenance."
                    if toolchain is None else
                    "Recorded from the build that produced these artifacts, not read out of "
                    "them: the AXLF metadata carries no chesscc/peano/llvm/aiecc/clang marker "
                    "and PlatformVBNV is empty, so this value is only as good as the build "
                    "that wrote it. Pass --toolchain to replace it."
                ),
                "generating_script_revision": script_revision,
            },
            "population": obs["population"],
            "census": obs["census"],
            "observed": obs["observed"],
            "declared_dangling": obs["declared_dangling"],
            "symlinks": obs["symlinks"],
            "artifacts": obs["artifacts"],
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(f"\nwrote {manifest_path.relative_to(root)} "
              f"({manifest_path.stat().st_size} B, {len(obs['artifacts'])} artifacts)")
        # Say out loud what happened to the two unobservable fields - a silent null is
        # how a recorded toolchain disappeared before, and a silent carry-over would be
        # the mirror-image mistake.
        if toolchain is None:
            print('  build.toolchain: null - no build has recorded one '
                  '(set it with --toolchain "<compiler + version>")')
        elif args.toolchain is not None:
            print(f"  build.toolchain: {toolchain!r} (from --toolchain)")
        else:
            print(f"  build.toolchain: {toolchain!r} preserved from the existing manifest "
                  f"(pass --toolchain to replace it)")
            if previous_artifacts and previous_artifacts != set(payload["artifacts"]):
                print("  WARNING: the artifact set changed and no --toolchain was given, so "
                      "the preserved value may not describe these builds", file=sys.stderr)
        print(f"  build.generating_script_revision: {script_revision!r}"
              + (" (from --script-revision)" if args.script_revision is not None else
                 " preserved from the existing manifest" if script_revision is not None else
                 " - not recorded"))
        return 0

    if not manifest_path.exists():
        if args.allow_missing_manifest:
            print(f"\nno manifest at {manifest_path} (allowed by --allow-missing-manifest)")
            return 0
        print(
            f"\nVIOLATION: no manifest at {manifest_path.relative_to(root)}\n"
            "  The artifact set has no recorded provenance. Bootstrap it with:\n"
            "    engine/npu/tests/check_xclbin_provenance.py --write-manifest \\\n"
            "      --toolchain \"<compiler arm + versions, e.g. 'aiecc (mlir-aie venv), LLVM 23.0.0, XRT 2.26f'>\" \\\n"
            "      --script-revision \"$(git rev-parse --short HEAD)\"\n"
            "  Without --toolchain the manifest records build.toolchain as null, which is the\n"
            "  open half of issue #2262.",
            file=sys.stderr,
        )
        return 1

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ENVIRONMENT: cannot parse {manifest_path}: {exc}", file=sys.stderr)
        return 2

    problems = compare(obs, manifest)

    # Host state, reported and never asserted: whether the recorded third-party links
    # resolve depends on the machine (FastFlowLM installed at /opt/fastflowlm makes them
    # resolve; a plain checkout does not). Both are fine; a change of TARGET is not, and
    # that is what compare() checks.
    third_party = manifest.get("declared_dangling", {})
    if third_party:
        resolving = [p for p in third_party if p not in obs["declared_dangling"]]
        print(f"\n  third-party links recorded: {len(third_party)}  "
              f"resolving here: {len(resolving)}  not resolvable here: "
              f"{len(third_party) - len(resolving)}")
        if resolving:
            print(f"  note: {len(resolving)} of them resolve on this host "
                  f"(a fastflowlm install is present) - host-dependent, not a violation")
        print("  (their TARGETS are asserted; their resolution is not)")

    if problems:
        print(f"\nVIOLATION: the tracked set differs from {manifest_path.name} "
              f"({len(problems)} difference(s)):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print(
            "\n  If the change is intentional, regenerate and land the manifest in the SAME\n"
            "  commit so the provenance moves with the artifacts:\n"
            "    engine/npu/tests/check_xclbin_provenance.py --write-manifest\n"
            "  Do not: copy a foreign xclbin over a dangling link, or materialise an alias\n"
            "  symlink into a real file (it doubles the payload and hides the aliasing).",
            file=sys.stderr,
        )
        return 1

    print(f"\nOK: matches {manifest_path.name} "
          f"({obs['population']['distinct_builds_by_uuid']} distinct builds, "
          f"{obs['population']['dangling_symlinks']} third-party link(s) not resolvable here)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
