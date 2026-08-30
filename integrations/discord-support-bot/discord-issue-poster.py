#!/usr/bin/env python3
"""discord-issue-poster.py — mirror GitHub issues to #issue-tracker as FORUM posts.

Two jobs, one 15-minute cron run:

  1. POST   — new open issues (number > last handled) become tagged forum
              posts in #issue-tracker (a forum channel).
  2. SYNC   — every tracked post is reconciled with the live GitHub issue:
              * closed   → tag `resolved` + archive the post
              * reopened → unarchive + tag `pending`
              * escalation label (priority/critical/...) → tag `escalated`
              * label / body changes re-derive type + DEFCON severity
              Tag ids resolve from the channel at runtime; posts are only
              PATCHed when something actually changed.

State persists in ~/.cache/discord-issue-poster-state.json (last issue
handled + a {issue_number: {thread, state, tags, archived}} map), so a
re-run never double-posts and the sync knows each post's last-known state.

Cron (strixhalo): */15 * * * * (every 15 min; cheap when nothing new)

Token: ~/.secrets/Discord Bot token.txt (same as the other discord bots).
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "/home/bcloud/1bit-MONSTER/integrations/discord-support-bot")
from post_issue import (  # noqa: E402
    TAG_SEVERITY,
    desired_tags,
    forum_tags,
    gh_issue,
    post_issue_post,
    update_post,
)

REPO = "1bit-MONSTER/1bit-MONSTER"
STATE_FILE = os.path.expanduser("~/.cache/discord-issue-poster-state.json")


def load_state() -> dict:
    try:
        return json.load(open(STATE_FILE))
    except Exception:
        return {"last_issue": 0, "posts": {}}


def save_state(state: dict) -> None:
    state["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    json.dump(state, open(STATE_FILE, "w"), indent=2)


def new_open_issues(after: int) -> list[dict]:
    """Open issues with number > after, oldest first."""
    out = subprocess.run(
        ["gh", "issue", "list", "--repo", REPO, "--state", "open",
         "--limit", "100", "--json", "number,title,createdAt"],
        capture_output=True, text=True, check=True).stdout
    issues = [i for i in json.loads(out) if i["number"] > after]
    return sorted(issues, key=lambda i: i["number"])


def sync_post(state: dict, number: int, rec: dict, tags: dict) -> None:
    """Reconcile one tracked post with the live issue; PATCH only on change."""
    try:
        issue = gh_issue(REPO, number)
    except Exception as exc:  # noqa: BLE001 — issue may be gone; leave post alone
        print(f"sync #{number}: skipped ({type(exc).__name__}: {exc})")
        return
    want = [t for t in desired_tags(issue) if t in tags]
    ids = [tags[t] for t in want]
    closed = (issue.get("state") or "").lower() == "closed"
    changed = False

    if want != rec.get("tags"):
        update_post(rec["thread"], applied_tags=ids)
        rec["tags"] = want
        changed = True
        print(f"sync #{number}: tags -> {want}")

    if closed and not rec.get("archived"):
        update_post(rec["thread"], archived=True)
        rec["archived"] = True
        changed = True
        print(f"sync #{number}: archived (closed)")
    elif not closed and rec.get("archived"):
        update_post(rec["thread"], archived=False)
        rec["archived"] = False
        changed = True
        print(f"sync #{number}: unarchived (reopened)")

    if changed:
        rec["state"] = "CLOSED" if closed else "OPEN"
        save_state(state)


def main() -> int:
    state = load_state()
    after = int(state.get("last_issue", 0))
    tags = forum_tags()

    # ── job 1: post new issues (plus retry previously failed numbers) ─────
    issues = new_open_issues(after)
    posts = state.setdefault("posts", {})
    failed = state.setdefault("failed", [])
    # Retry failed numbers first (ascending), then any new ones — a
    # transient failure must never drop an issue: if #N fails but #N+1
    # succeeds, last_issue only advances on success, and #N stays in
    # `failed` until it posts.
    for num in sorted(set(failed) | {i["number"] for i in issues}):
        try:
            # post_issue_post needs the FULL issue dict (url/labels/state/
            # body) — the list payload above only carries number/title/
            # createdAt.
            full = gh_issue(REPO, num)
        except Exception as exc:  # noqa: BLE001
            if num not in failed:
                failed.append(num)
            print(f"fetch #{num} FAILED (will retry): {type(exc).__name__}: {exc}")
            save_state(state)  # persist NOW — a later success must not orphan it
            continue
        if (full.get("state") or "").lower() == "closed":
            # Closed before we could post it (e.g. while sitting in
            # `failed`) — never mirror a closed issue, and stop retrying it.
            if num in failed:
                failed.remove(num)
            print(f"#{num} closed before posting — skipped")
            save_state(state)
            continue
        try:
            tid = post_issue_post(full)
        except Exception as exc:  # noqa: BLE001
            if num not in failed:
                failed.append(num)
            print(f"post #{num} FAILED (will retry): {type(exc).__name__}: {exc}")
            save_state(state)  # persist NOW — a later success must not orphan it
            continue
        if num in failed:
            failed.remove(num)
        posts[str(num)] = {
            "thread": tid,
            "state": "OPEN",
            "tags": [t for t in desired_tags(full) if t in tags],
            "archived": False,
        }
        print(f"posted #{num} '{full['title']}' as forum post {tid}")
        if num > after:
            state["last_issue"] = num
        save_state(state)
        time.sleep(2)  # rate-limit politeness between posts

    # ── job 2: reconcile tracked posts with live issue state ──────────────
    for num, rec in list(posts.items()):
        sync_post(state, int(num), rec, tags)
    save_state(state)

    print(f"done: {len(issues)} new, {len(posts)} tracked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
