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
hook="$here/post-commit"
hooks_dir="$(git rev-parse --git-common-dir)/hooks"
target="$hooks_dir/post-commit"

# 1. syntax
bash -n "$hook"

# 2. shape control: with `set -o pipefail`, a failing command inside a pipeline
#    must reach the failure branch. Without pipefail the pipeline's status is the
#    LAST command's (sed's, which succeeds), which is exactly how the original
#    hook's FAILED branch became dead code.
if ( set -uo pipefail; ! false 2>&1 | sed 's/^/x/' >/dev/null ); then
  :
else
  echo "control FAILED: failure branch unreachable (pipefail missing?) — refusing to install" >&2
  echo "  the hook would report a rejected push as 'PR is up to date'" >&2
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
