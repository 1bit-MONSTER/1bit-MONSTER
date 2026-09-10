#!/usr/bin/env bash
# install.sh — install the repo's auto-push post-commit hook, verifying BEFORE installing.
#
# Why an installer instead of "copy this file into .git/hooks/": hooks are
# unversioned, so a fix applied by hand dies with the next clone or `git init`.
# This keeps the hook in the repo and installs it — and it targets the COMMON git
# dir, so one install covers every worktree of the clone.
#
# The verification is CHAINED to the install deliberately. A check that does not
# gate the action it checks is a report, not a check — the defect class this hook
# itself was fixed for (its failure branch was unreachable, so a rejected push
# printed "PR is up to date"). The control below is the same shape: it proves the
# FAILED branch is reachable under the hook's own shell options before we install
# anything.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git rev-parse --show-toplevel)"

# The hook has ONE home in this repo: tools/post-commit-hook.sh (the sibling of
# tools/commit-msg-hook.sh). This installer deliberately does NOT carry a copy of
# the hook — a second copy of the hook is the duplicate pattern this repo has
# already paid for. It carries the ENFORCEMENT instead: without a script, the
# control can only be a comment, and a comment cannot refuse a copy.
#
# HOOK_SOURCE overrides the file under test and HOOK_TARGET the destination, both
# so this installer can be mutation-tested without touching real hooks.
# NOTE the mutual dependency: this installer needs tools/post-commit-hook.sh, and
# that file's INSTALL block points back here. They should land in the same cycle.
hook="${HOOK_SOURCE:-$repo_root/tools/post-commit-hook.sh}"
hooks_dir="$(git rev-parse --git-common-dir)/hooks"
target="${HOOK_TARGET:-$hooks_dir/post-commit}"

if [ ! -f "$hook" ]; then
  echo "no hook at $hook" >&2
  echo "  expected tools/post-commit-hook.sh (see PR #2186 / goal/one-registry-one-router)" >&2
  exit 1
fi

# 1. syntax
bash -n "$hook"

# 2. shape control: a failing command inside a pipeline must reach the failure
#    branch when the HOOK'S OWN shell options are in force. Without pipefail the
#    pipeline's status is the LAST command's (sed's, which succeeds) — exactly how
#    the original hook's FAILED branch became dead code.
#
#    ONE RULE, which both failures above reduce to (@agent-dc0fb9's phrasing):
#      a control's options must be the FILE's options, and nothing else's.
#    Failure 1 set them itself; failure 2 inherited them from the caller.
#
#    THE CONTROL MUST TAKE THE OPTIONS FROM THE FILE UNDER TEST, not set them
#    itself. The first version of this installer wrote `( set -uo pipefail; ... )`,
#    which tests THIS SCRIPT'S shell options and therefore passes for ANY file,
#    including a hook with the defect — a gate that cannot see the property it
#    asserts. Found by @agent-dc0fb9 via mutation test (a copy with `set -u`
#    reintroduced installed cleanly); fixed by sourcing the file's own options.
opts="$(grep -E '^set ' "$hook" | head -1)"
if [ -z "$opts" ]; then
  echo "control FAILED: no 'set' line found in $hook — refusing to install" >&2
  exit 1
fi
#    Run it in a FRESH bash that first clears the options, because a subshell
#    would INHERIT this installer's own `set -o pipefail` — which makes the control
#    pass for any file, including one with the defect. (That was the second
#    mutation-test failure: `set -u` in the file under test was invisible because
#    pipefail came in from the parent. The gate has to have no opinion of its own.)
if bash -c "set +e +u +o pipefail; $opts; ! false 2>&1 | sed 's/^/x/' >/dev/null"; then
  :
else
  echo "control FAILED: failure branch unreachable under the hook's own options ($opts)" >&2
  echo "  that hook would report a rejected push as 'PR is up to date' — refusing to install" >&2
  exit 1
fi

# 3. install, keeping a timestamped backup of any non-symlink that is already there
mkdir -p "$hooks_dir"
if [ -e "$target" ] && [ ! -L "$target" ]; then
  cp "$target" "$target.bak.$(date +%s)"
fi
install -m 0755 "$hook" "$target"

echo "installed $target ($(md5sum "$target" | cut -c1-10))"
echo "verify a push landed with: git ls-remote origin \"\$(git branch --show-current)\" vs git rev-parse HEAD"
