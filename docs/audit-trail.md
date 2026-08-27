# The Audit Trail — 1.5 TB of Evidence

> Every claim in [journey.md](journey.md) is backed by preserved artifacts. This document is the map: where the data lives, what it contains, how it maps to this repo's structure, and how to verify any number in the journey. **This is not for the weak** — it is the unedited record of ~600 hours of engineering, archived nightly since 2026-08-05 and kept append-only forever.

## At a glance

| Archive | Location | Size | Contents |
|---------|----------|------|----------|
| Original snapshot tarballs (2026-08-05) | Raspberry Pi SD card: `/mnt/strixhalo-backup/` | ~272 GB | Full project tree, home src, git repos — the frozen originals |
| Live nightly archive | Raspberry Pi ZFS pool: `/ZFSPool/backups/` | ~2 TB used (1.5 TB project data) | Nightly rsync of every machine that touches the project |
| Model weights & checkpoints | archived working tree: `1bit-systems/models/` | 326 GB | 1BP/GGUF weights for all 47 models, incl. ZAYA1-74B, Qwen3-4B-NPU2 |
| Spec-decode experiments | archived working tree: `1bit-systems/spec-decode/` | 24 GB | Draft/target experiment dumps |
| Reference logits | archived working tree: `*_ref_logits.txt` | ~7 MB | torch/numpy reference outputs used by the validation gates |
| Session transcripts | archived home: `.claude/`, `.codewhale/`, `.reasonix/` | continuous | Agent session logs from every working session |

## The machine

**Raspberry Pi backup server — `192.168.50.216` (hostname `pi`, aarch64, ZFS 2.4.1).**

- **SD card** (469 GB ext4, `/`): the original 2026-08-05 dump lives at `/mnt/strixhalo-backup/*.tar`. This is the "frozen in time" snapshot of the entire project as it stood that day.
- **ZFS pool `ZFSPool`**: 5 × WD 1 TB laptop drives in RAID-Z1 (~4.55 TB raw, ~1.6 TB free), imported manually at boot. Datasets:
  - `ZFSPool/backups/strixhalo` — **1.41 TB** — the project's home machine: `1bit-MONSTER/` (this repo), `1bit-systems/` working tree (366 GB incl. models, spec-decode, reference logits), `1bit/`, `1bit-mlx/`, `bitnet.cpp/`, `fastflowlm_analysis/`, `npu-infer/`, `torch2aie/`, and the agent session transcripts
  - `ZFSPool/backups/minisforum` — **539 GB** — Windows workstation user profile (the 1bit Windows build path, `1bit.cpp/`, `1bit-systems/`)
  - `ZFSPool/backups/ryzen` — **16 GB** — Linux workstation home (1bit-systems clone, agent configs)
  - `ZFSPool/backups/sliger` — placeholder (next machine)

> ⚠️ **Gotcha documented for future archaeologists:** the pool's mountpoint is **`/ZFSPool`** (capital P). `/ZFSpool` (lowercase p) is a decoy directory left on the SD card root — the data is **not** there. If a dataset ever appears empty, run `sudo zfs mount -a` first.

## Original dump manifest (SD card, 2026-08-05)

| Tarball | Size | Contents |
|---------|------|----------|
| `01-credentials-configs.tar` | 3.2 MB | Credentials & configs (access-restricted; do not redistribute) |
| `02-projects.tar` | 231 GB | `projects/1bit-systems/` — the full working tree: engine, kernels, models, npu-infer, fastflowlm_analysis, research, docs, site, tools, tests (10,521 entries) |
| `03-home-src.tar` | 29 GB | Home source trees incl. the `1bit/` HIP kernels and `zaya-llama.cpp` build objects |
| `04-onebit-systems.tar` | 186 MB | The 1bit-systems repo checkout incl. `third_party/` (stable-diffusion.cpp, stb, lemonade) |
| `05-home-git-repos.tar` | 13 GB | Git repos incl. `audit-1bit-systems/` (hackathon track 1/3 specs, demo scripts, submission checklists) |

## Backup cadence (cron on the Pi)

| Job | Time | Source → Destination |
|-----|------|----------------------|
| `backup-strixhalo.sh` | 03:30 daily | `bcloud@192.168.50.110:/home/bcloud/` → `ZFSPool/backups/strixhalo/home/bcloud/` |
| `backup-ryzen.sh` | 03:45 daily | `bcloud@192.168.50.100:/home/bcloud/` → `ZFSPool/backups/ryzen/home/bcloud/` |
| `backup-minisforum.sh` | 04:00 daily | Windows SMB `//192.168.50.61/C$/Users/bcloud` → `ZFSPool/backups/minisforum/home/bcloud/` |

- **Append-only by design** — rsync runs have no `--delete`; the backup server never auto-deletes.
- **Weekly ZFS snapshots** every Monday (`ZFSPool/backups@<dataset>-YYYY-MM-DD`), 90-day retention.
- Logs: `~/backup.log`, `~/backup-ryzen.log`, `~/backup-minisforum.log`.

## How the dump maps to this repo

| Repo path | Role | Raw evidence lives at (on the Pi) |
|-----------|------|-----------------------------------|
| `docs/research/` | RE reports, kernel analysis, format mining, NPU contract guides | archived `1bit-systems/docs/research/` + `fastflowlm_analysis/` + `npu-infer/` |
| `docs/archive/` | Blockers, handoffs, dead ends (INT8-blocked, weight-stream, fused-integration…) | archived `docs/archive/` — every dead end preserved |
| `kernels/`, `engine/`, `npu-infer/` | The code itself — the 4-day RE sprint output | archived working tree, plus the `1bit/` HIP kernels and `zaya-llama.cpp` fork |
| `models/` (git-ignored) | Runnable 1BP/GGUF weights for 47 models | `1bit-systems/models/` (326 GB) — incl. `ZAYA1-74B.1bp`, `Qwen3-4B-NPU2`, `kl-test` (168 GB) |
| `benchmarks/` | Measured numbers, sweep tools | `benchmarks/` + `*_ref_logits.txt` reference outputs + `bloom_ref_logits.txt` (3.9 MB) |
| `hackathon/` | Competition artifacts | archived `hackathon/` + `audit-1bit-systems/hackathon/` (specs, demo scripts, demo-video.mp4) |
| `docs/journey.md` | The timeline | this doc + the raw session transcripts (`.claude/projects/*.jsonl`) that the timeline was written from |
| `docs/plans/` | The roadmap & pivot plans (one-heap, tilefuse, mojo-fold…) | archived `docs/plans/` |

## Verifying a claim from the journey

Every UPDATE in `docs/journey.md` is traceable:

1. **Code claims** → the commit exists in this repo's history (e.g., the UPDATE 34 burn is `cbce9630`; the 2026-08-14 validation gates are `0d1803bd` bloom, `4d355071` step1, `6f4fc4b1` deepseek-mla, `41869540` gpt-oss).
2. **Measurement claims** → `benchmarks/` in the repo, reference logits in the archive, or the archived working tree's build outputs.
3. **Real-time claims** ("documented in real time") → the session transcripts on the Pi under `ZFSPool/backups/strixhalo/home/bcloud/.claude/projects/` and `history.jsonl`.
4. **The 1.5 TB itself** → the tarball hashes are recorded at archive time; the ZFS pool reports no errors (`zpool status`: "No known data errors").

## Access

```bash
# the Pi (key auth as bcloud; sudo password: bcloud)
ssh bcloud@192.168.50.216

# live archive
ls /ZFSPool/backups/strixhalo/home/bcloud/1bit-MONSTER/
sudo zpool status        # pool health
sudo zfs list -t snapshot # weekly snapshots
sudo zfs mount -a        # if a dataset appears unmounted

# original dump
ls -lh /mnt/strixhalo-backup/*.tar
```

**Restore procedure:** `tar -xf /mnt/strixhalo-backup/02-projects.tar -C /target` (231 GB — allow hours over LAN; the ZFS datasets are the fast path for anything newer than 2026-08-05).
