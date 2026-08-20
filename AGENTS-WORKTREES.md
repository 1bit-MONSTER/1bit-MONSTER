# Agent worktrees on the shared strixhalo checkout

All agents working on the 1bit-MONSTER repo on `strixhalo` use **one git
worktree per agent**, never the shared checkout itself. This keeps two agents
from colliding on branches, uncommitted changes, or build output in the same
working directory.

## Layout

```
/home/bcloud/1bit-MONSTER          # shared checkout (git home) — read-mostly
/home/bcloud/1bit-MONSTER-<agent>  # one worktree per agent, own branch
```

Create a worktree for an agent or branch with:

```bash
git -C /home/bcloud/1bit-MONSTER fetch origin
git -C /home/bcloud/1bit-MONSTER worktree add -b <branch> \
    /home/bcloud/1bit-MONSTER-<agent> origin/main
```

Example in use today: `/home/bcloud/1bit-MONSTER-agent` on
`feat/rocm-therock-7.14-lane-pin`.

## Rules

1. **One branch per worktree.** A worktree checks out exactly one branch; two
   agents on different branches use two worktrees.
2. **Never commit or stage in the shared checkout.** All commits happen inside
   the agent worktrees, so scratch changes never block another agent's
   checkout or `git status`.
3. **Fork from fresh main.** Fetch before `worktree add` and base the new
   branch on `origin/main`, never on another agent's in-flight branch.
4. **Push and open PRs from the worktree:**
   `git -C /home/bcloud/1bit-MONSTER-<agent> push origin <branch>`.
5. **Don't touch another agent's worktree** — no resets, checkouts, or removals
   without asking.
6. **Clean up when done.** After the branch is merged (or abandoned), remove
   the worktree with `git -C /home/bcloud/1bit-MONSTER worktree remove
   /home/bcloud/1bit-MONSTER-<agent>` so the box doesn't accumulate stale
   checkouts.
