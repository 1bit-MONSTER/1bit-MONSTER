#!/bin/sh
# WRITTEN BY @agent-ec855d (2026-09-10), ADOPTED HERE and validated independently: the
# known-bad message (c42659f23) is BLOCKED, a clean message passes. Their own caveat
# travels with it — the fingerprint was measured over 25 commits on this branch
# (broad rule 6/25 = 24% false positives and therefore useless; narrow rule 1/25 = the
# known-bad only), which is a small corpus, so this is evidence the rule is narrow
# enough for this repo's style rather than a guarantee. And their limit, demonstrated
# rather than asserted: a message whose only vanished span is MID-SENTENCE passes.
#
# THE HOOK CATCHES THE CONSEQUENCE, NOT THE CAUSE. The cause-level fix is procedural and
# no hook can enforce it: pass commit messages with -F (a file) so the shell never
# evaluates them. I interpolated `-m` twice this session; -F is the actual fix.
#
# INSTALL (git hooks are not versioned — per-clone, and this repo has several clones):
#     ln -sf ../../tools/commit-msg-hook.sh .git/hooks/commit-msg
# Without that symlink this file is the flag-audit situation: a check nobody runs.
# commit-msg — catch the shell-interpolation artifact that silently EMPTIES a
# backticked span out of a commit message.
#
# WHY THIS EXISTS. `git commit -m "text with `cmd` ..."` lets the shell run cmd and
# substitute its stdout. When cmd prints nothing the surrounding text does not just
# lose formatting, it VANISHES — leaving a gap that reads like ordinary prose.
# Observed on c42659f23:
#     "two commands in one  block, and under the default\n   the first failure ABORTS"
#     "Added\n    so the intent is explicit"
#     "a repo-wide . It would"
# The message contains ZERO backticks, so there is no marker left to notice. It reads
# fine if you read it approvingly, which is the only way its author reads it.
#
# THE FINGERPRINT IS MEASURED, NOT GUESSED. Over the last 25 commits on
# goal/one-registry-one-router:
#   broad  (any double space, or space before period)       -> 6/25 flagged = 24% FP  USELESS
#   narrow (emptied span adjacent to sentence punctuation)  -> 1/25 = the known-bad only, 0 FP
# The broad rule fires on GOOD messages because correct commit bodies legitimately
# contain code spans with double spaces and inline dots ("XX ... BLOCKED",
# "hip_1bp_gpu  predicate=yes available=yes ...").
#
# HONEST LIMIT, so nobody trusts it further than it goes: the narrow rule catches a
# vanished span only where it sits next to sentence punctuation. A message whose only
# vanished span is mid-sentence is MISSED. This is a zero-false-positive tripwire for
# one specific artifact, not a general detector for the class. The durable fix is
# procedural and no hook can enforce it: pass messages with -F (a file) or
# single-quote them so nothing in them is evaluated.
#
# Install (hooks are NOT versioned, so this is a per-clone step):
#     ln -sf ../../tools/commit-msg-hook.sh .git/hooks/commit-msg   # from the repo root
#
# Bypass for a legitimate message that trips it:  git commit --no-verify

msg="$1"
[ -n "$msg" ] && [ -f "$msg" ] || exit 0

pattern='[[:alnum:]]  \.|[[:alnum:]] \. [[:upper:]]'
if grep -qE "$pattern" "$msg"; then
    echo "commit-msg: SUSPECT SHELL INTERPOLATION in this message." >&2
    echo "  Likely cause: a backticked span was executed and substituted with EMPTY" >&2
    echo "  output, deleting the text silently. Line(s):" >&2
    grep -nE "$pattern" "$msg" | sed 's/^/      /' >&2
    echo "  Fix: rewrite the message via -F (a file) or single quotes so nothing is evaluated." >&2
    echo "  Deliberate? git commit --no-verify    (and prefer -F next time)" >&2
    exit 1
fi
exit 0
