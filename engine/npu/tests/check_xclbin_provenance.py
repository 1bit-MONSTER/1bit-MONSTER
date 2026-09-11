#!/usr/bin/env python3
"""check_xclbin_provenance.py — static provenance + consistency check for engine/npu/xclbins.

INTENT
------
The NPU xclbin set is tracked in git precisely because the MLIR/chess toolchain cannot
reliably rebuild it (see engine/npu/build_npu_ternary.sh's header: "the aiecc MLIR
toolchain has a pre-existing version mismatch that prevents clean builds"). That makes the
*committed* artifacts the source of truth, and it means two classes of defect can only be
caught by inspecting the committed set rather than by building it:

  1. Hygiene regressions that a clean checkout reproduces deterministically. Four of the
     tracked *.xclbin symlinks point at /opt/fastflowlm/... which does not exist on any of
     our machines, so a fresh clone is already broken; and one build ships as a real file,
     a byte-identical duplicate, and a symlink, so "deduping" the set either way loses
     something (materialising the aliases doubles the payload; copying a foreign binary
     over a dangling link hides that it was never built here).
  2. Provenance loss. The artifacts record their producer, format version, UUID and
     timestamp, but no toolchain version, no source revision and no generating script - so
     a green rebuild could never be shown to reproduce what ships. PROVENANCE.json is the
     missing half: a manifest keyed by the embedded XclBinUUID, recording the content hash,
     the aliases, and (to be filled by the build) the toolchain and script revision.

This check runs with stdlib Python only and needs no NPU, so it can gate on stock CI
runners. It asserts hygiene invariants that need no baseline, then compares the observed
tree against the committed manifest; a legitimate change to the artifact set is landed by
regenerating the manifest in the same commit (--write-manifest), which is the review point.

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
        "declared_dangling": dict(sorted(dangling.items())),
        "symlinks": dict(sorted(symlinks.items())),
        "artifacts": dict(sorted(artifacts.items())),
    }


def hygiene(obs: dict) -> list[str]:
    """Invariants that need no baseline. Returns warnings; raises Failure on violations."""
    notes: list[str] = []
    pop = obs["population"]

    # (b) dangling links must be declared, and must not be "fixed" with a foreign binary
    for path, target in obs["declared_dangling"].items():
        if not target.startswith(DANGLING_TARGET_PREFIX):
            notes.append(
                f"{path}: dangling to {target} - an unexpected dangling target; "
                "if this is intentional, regenerate the manifest"
            )
    # (a) every non-dangling symlink must resolve (by construction, but assert the split)
    if pop["alias_symlinks_in_tree"] + pop["dangling_symlinks"] != pop["xclbin_symlinks"]:
        raise Failure(
            "symlink accounting does not add up: "
            f"{pop['alias_symlinks_in_tree']} resolving + {pop['dangling_symlinks']} dangling "
            f"!= {pop['xclbin_symlinks']} total"
        )
    # (d) aliases must stay symlinks: they are counted as symlinks, and no alias may also
    #     exist as a regular artifact of the same name
    regular_names = {
        name for entry in obs["artifacts"].values() for name in entry["paths"]
    }
    for path in obs["symlinks"]:
        if os.path.basename(path) in regular_names:
            raise Failure(
                f"{path}: exists BOTH as a symlink and as a regular artifact - "
                "an alias was materialised instead of linked"
            )
    # (c) dedupe correctness: paths sharing a UUID must be byte-identical (checked in observe)
    # (d) aliases must stay symlinks AND keep pointing at their own twin. A materialised
    #     alias shows up as an extra REGULAR path in its UUID group (caught by the manifest
    #     comparison, which also sees the payload/redundancy move); a "repaired" alias
    #     pointing somewhere else is caught here, without needing a baseline.
    for uuid, entry in obs["artifacts"].items():
        regular = set(entry["paths"])
        for path, target in obs["symlinks"].items():
            name = os.path.basename(path)
            if name not in regular:
                continue  # this symlink's own artifact is elsewhere (e.g. dangling targets)
            if os.path.basename(target) not in regular:
                raise Failure(
                    f"{name}: is a link into UUID {uuid}'s group but points at "
                    f"{target!r}, which is not a member of that group"
                )
    return notes


def compare(obs: dict, manifest: dict) -> list[str]:
    """Diff the observed snapshot against the committed manifest."""
    problems: list[str] = []

    m_pop = manifest.get("population", {})
    for key in ("top_level_xclbin", "regular_artifacts", "xclbin_symlinks",
                "alias_symlinks_in_tree", "dangling_symlinks", "distinct_builds_by_uuid"):
        want, got = m_pop.get(key), obs["population"].get(key)
        if want != got:
            problems.append(f"population.{key}: manifest {want} != observed {got}")

    m_dang = manifest.get("declared_dangling", {})
    if m_dang != obs["declared_dangling"]:
        only_m = {k: v for k, v in m_dang.items() if k not in obs["declared_dangling"]}
        only_o = {k: v for k, v in obs["declared_dangling"].items() if k not in m_dang}
        for k, v in only_m.items():
            problems.append(f"declared_dangling: {k} was dangling to {v}, now it RESOLVES")
        for k, v in only_o.items():
            problems.append(f"declared_dangling: {k} is newly dangling to {v}")

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
    ap.add_argument("--allow-missing-manifest", action="store_true",
                    help="do not fail when the manifest is absent (bootstrap only)")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[3]
    manifest_path = Path(args.manifest) if args.manifest else root / XCLBIN_DIR / MANIFEST_NAME

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

    if args.write_manifest:
        payload = {
            "schema": 1,
            "intent": (
                "Provenance for the committed NPU xclbin set, keyed by the artifact's embedded "
                "XclBinUUID. Regenerate with engine/npu/tests/check_xclbin_provenance.py "
                "--write-manifest and land it in the same commit as the artifact change."
            ),
            "generated": {
                "by": "pi/coding-agent",
                "at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "check": "engine/npu/tests/check_xclbin_provenance.py",
                "ref": subprocess.run(
                    ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                    capture_output=True, text=True).stdout.strip(),
            },
            "build": {
                "toolchain": None,
                "toolchain_note": (
                    "Empty by design: the artifacts record no compiler arm or version "
                    "(no chesscc/peano/llvm/aiecc/clang marker in the AXLF metadata; "
                    "PlatformVBNV is empty). A build that regenerates these artifacts must "
                    "fill this in - that is the missing half of the provenance."
                ),
                "generating_script_revision": None,
            },
            "population": obs["population"],
            "census": obs["census"],
            "declared_dangling": obs["declared_dangling"],
            "symlinks": obs["symlinks"],
            "artifacts": obs["artifacts"],
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(f"\nwrote {manifest_path.relative_to(root)} "
              f"({manifest_path.stat().st_size} B, {len(obs['artifacts'])} artifacts)")
        return 0

    if not manifest_path.exists():
        if args.allow_missing_manifest:
            print(f"\nno manifest at {manifest_path} (allowed by --allow-missing-manifest)")
            return 0
        print(
            f"\nVIOLATION: no manifest at {manifest_path.relative_to(root)}\n"
            "  The artifact set has no recorded provenance. Bootstrap it with:\n"
            "    engine/npu/tests/check_xclbin_provenance.py --write-manifest",
            file=sys.stderr,
        )
        return 1

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ENVIRONMENT: cannot parse {manifest_path}: {exc}", file=sys.stderr)
        return 2

    problems = compare(obs, manifest)
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
          f"{obs['population']['dangling_symlinks']} dangling as declared)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
