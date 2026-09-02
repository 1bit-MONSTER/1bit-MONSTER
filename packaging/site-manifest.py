#!/usr/bin/env python3
"""site-manifest.py — write SHA256SUMS + manifest.json for site/downloads.

Consumed by `make package-site` (packaging/Makefile). Given the release
version and the site/downloads directory, it:
  * (re)writes SHA256SUMS for every canonical package present, and
  * writes manifest.json — the machine-readable list the Downloads page
    (site/1bit-downloads.html) reads client-side to render live file sizes
    and checksums without a server.

Only files that exist are listed, so a release that skipped the AppImage
(needs appimagetool) still produces a valid manifest.

Usage:
    python3 packaging/site-manifest.py <VERSION> <site/downloads-dir>
"""
import hashlib
import json
import os
import sys

VERSION = sys.argv[1]
DL_DIR = sys.argv[2]

CANONICAL = [
    f"1bit-monster_{VERSION}_amd64.deb",
    f"1bit-monster-{VERSION}-x86_64.AppImage",
    f"1bit-monster-{VERSION}-linux-amd64.tar.xz",
]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    present = [n for n in CANONICAL if os.path.exists(os.path.join(DL_DIR, n))]
    if not present:
        sys.stderr.write(f"site-manifest: no canonical packages for {VERSION} in {DL_DIR}\n")
        return 1

    with open(os.path.join(DL_DIR, "SHA256SUMS"), "w", encoding="utf-8") as f:
        for name in present:
            f.write(f"{sha256(os.path.join(DL_DIR, name))}  {name}\n")

    manifest = {
        "version": VERSION,
        "files": [
            {
                "name": name,
                "size": os.path.getsize(os.path.join(DL_DIR, name)),
                "sha256": sha256(os.path.join(DL_DIR, name)),
                "url": "downloads/" + name,
            }
            for name in present
        ],
    }
    with open(os.path.join(DL_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"site-manifest: {len(present)} packages -> SHA256SUMS + manifest.json in {DL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
