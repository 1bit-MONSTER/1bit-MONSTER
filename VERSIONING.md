# Versioning

1bit.systems uses **CalVer** — calendar versioning — not SemVer.

```
YYYY.MM.DD           e.g. 2026.07.20
YYYY.MM.DD.N         e.g. 2026.07.20.1   (second+ release on the same day)
```

Git tags are the same string with a `v` prefix: `v2026.07.20`.

## Why CalVer

1bit.systems is a fast-moving product/inference stack, not a library with an API
contract that downstream code pins against. SemVer's major/minor/patch promises
don't map onto "we shipped new engine work today." Dates are unambiguous,
sortable, and tell you at a glance how fresh a build is.

> History note: tags older than the CalVer switch (`v0.1.0`…`v1.0.0` and the
> descriptive tags like `v2026.07.06-fusion`, `-jarvis`, `-all5models`) are
> **early-prototype history**. Don't make new ones. Everything current is
> `vYYYY.MM.DD`.

## Single source of truth

The root [`VERSION`](VERSION) file holds the current version and **nothing else
should hardcode it.** Every packaging manifest is stamped from `VERSION` by
[`scripts/version-sync.sh`](scripts/version-sync.sh):

| File | Field |
|------|-------|
| `package.json` | `"version"` |
| `packaging/deb/DEBIAN/control` | `Version:` |
| `packaging/deb/DEBIAN/postinst` | banner line |
| `packaging/snap/snapcraft.yaml` | `version:` + `source-tag:` |
| `snap/snapcraft.yaml` | `version:` |
| `packaging/aur/PKGBUILD` | `pkgver=` |
| `packaging/README.md` | title, release links, docker tags |

`packaging/Makefile` and `scripts/release.sh` read `VERSION` directly.

`site/releases.json` is a **release-history feed** (many versions), not a single
version to stamp — it is maintained per release, not by `version-sync.sh`.

## Cutting a release

```bash
scripts/release.sh              # version = today's date (auto-.N if it collides)
scripts/release.sh 2026.08.01   # or pick the version explicitly
```

It bumps `VERSION`, rolls the `## Unreleased` CHANGELOG section into a dated
heading, stamps all manifests, then commits (`release: vX`) and tags (`vX`).
Publish with:

```bash
git push origin main && git push origin vYYYY.MM.DD
```

## Keeping manifests honest

Run this any time (and ideally in CI) to fail on drift:

```bash
scripts/version-sync.sh --check
```
