# Agent worktree layout (2026-08-19)

Two agents share this repo on strixhalo. To stop branches and uncommitted
changes colliding in one checkout, each agent works in its OWN git worktree
(shares the same `.git`, separate working directory + build dir):

| Worktree | Agent | Default branch |
|---|---|---|
| `/home/bcloud/1bit-MONSTER` | (primary checkout — ISO/branding lane) | `fix/blog-therock-7.14` |
| `/home/bcloud/1bit-MONSTER-agent` | (harness/deployment lane) | `feat/rocm-therock-7.14-lane-pin` |

Rules:
- Never run `git checkout <branch>` to switch between lanes inside a shared
  worktree — use `git worktree add <path> <branch>` so each lane keeps its
  own working directory.
- `git status` / `git stash` only ever see YOUR worktree's files.
- Build dirs are per-worktree and gitignored (`build/`), so a build in one
  lane never touches another lane's artifacts.
- List: `git worktree list`. Remove: `git worktree remove <path>`.

New lanes: `git worktree add /home/bcloud/1bit-MONSTER-<lane> <branch>`
