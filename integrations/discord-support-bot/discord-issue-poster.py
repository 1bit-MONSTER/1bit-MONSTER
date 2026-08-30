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
import datetime
import fcntl
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, "/home/bcloud/1bit-MONSTER/integrations/discord-support-bot")
from post_issue import (  # noqa: E402
    TAG_SEVERITY,
    desired_tags,
    forum_tags,
    forum_threads,
    gh_issue,
    post_issue_post,
    update_post,
)

REPO = "1bit-MONSTER/1bit-MONSTER"
STATE_FILE = os.path.expanduser("~/.cache/discord-issue-poster-state.json")
# Only auto-post issues created within this many days on a FRESH state
# (missing/corrupt state file). Prevents a duplicate-post flood — with no
# baseline, a fresh host would treat every open issue as new and mirror
# hundreds of historical issues. Set 0 to mirror every open issue.
BOOTSTRAP_SINCE_DAYS = int(os.getenv("BOOTSTRAP_SINCE_DAYS", "1"))


def load_state() -> dict:
    try:
        return json.load(open(STATE_FILE))
    except Exception:
        return {"last_issue": 0, "posts": {}}


def _iso_ts(value: str) -> float | None:
    """Parse a GitHub ISO-8601 timestamp (e.g. 2026-08-30T13:42:00Z) → epoch.

    Returns None when unparseable — callers keep such issues rather than
    silently dropping them.
    """
    try:
        return datetime.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return None


def save_state(state: dict) -> None:
    state["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    # Atomic write (tmp + os.replace): an in-place json.dump truncates the
    # file first, so a crash mid-write corrupts state and load_state would
    # fall back to last_issue=0 — re-posting every open issue as a
    # duplicate. os.replace is atomic on POSIX.
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp, STATE_FILE)


def new_open_issues(after: int) -> list[dict]:
    """Open issues with number > after, oldest first.

    --limit 1000 (gh paginates internally): with >100 open issues, a
    --limit 100 payload would silently drop older issues and the
    last_issue cursor could skip them forever.
    """
    out = subprocess.run(
        ["gh", "issue", "list", "--repo", REPO, "--state", "open",
         "--limit", "1000", "--json", "number,title,createdAt"],
        capture_output=True, text=True, check=True, timeout=30).stdout
    issues = [i for i in json.loads(out) if i["number"] > after]
    return sorted(issues, key=lambda i: i["number"])


def _is_gone(exc: Exception) -> bool:
    """True when the Discord API says the thread no longer exists (404)."""
    return getattr(exc, "code", None) == 404 or "404" in str(exc)


def _drop_dead_post(state: dict, number: int) -> None:
    """A tracked post no longer exists (deleted / config change).

    Drop it from the map, and if the issue is STILL OPEN, re-queue the
    number so it is mirrored again on the next run — otherwise it would
    silently vanish from the triage board (the number is below the
    last_issue cursor, so job 1 would never revisit it).
    """
    state["posts"].pop(str(number), None)
    try:
        issue = gh_issue(REPO, number)
        if (issue.get("state") or "").lower() != "closed":
            failed = state.setdefault("failed", [])
            if number not in failed:
                failed.append(number)
            print(f"sync #{number}: post gone (404), issue open — re-queued for re-post")
        else:
            print(f"sync #{number}: post gone (404), issue closed — dropped")
    except Exception:  # noqa: BLE001 — unknown state; re-queue, closed-skip guard protects
        failed = state.setdefault("failed", [])
        if number not in failed:
            failed.append(number)
        print(f"sync #{number}: post gone (404), issue state unknown — re-queued (closed-skip guard)")
    save_state(state)


def sync_post(state: dict, number: int, rec: dict, tags: dict) -> None:
    """Reconcile one tracked post with the live issue; PATCH only on change.

    Every Discord PATCH is individually guarded: one failure (post deleted
    → 404, or rate-limit 429 mid-batch) must not abort the whole job-2
    loop and disable lifecycle sync for every tracked post.
    """
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
        try:
            update_post(rec["thread"], applied_tags=ids)
        except Exception as exc:  # noqa: BLE001
            if _is_gone(exc):
                _drop_dead_post(state, number)
            else:
                print(f"sync #{number}: tag PATCH failed (will retry): "
                      f"{type(exc).__name__}: {exc}")
            return
        rec["tags"] = want
        changed = True
        print(f"sync #{number}: tags -> {want}")

    if closed and not rec.get("archived"):
        try:
            update_post(rec["thread"], archived=True)
        except Exception as exc:  # noqa: BLE001
            if _is_gone(exc):
                _drop_dead_post(state, number)
            else:
                print(f"sync #{number}: archive failed (will retry): "
                      f"{type(exc).__name__}: {exc}")
            return
        rec["archived"] = True
        changed = True
        print(f"sync #{number}: archived (closed)")
    elif not closed and rec.get("archived"):
        try:
            update_post(rec["thread"], archived=False)
        except Exception as exc:  # noqa: BLE001
            if _is_gone(exc):
                _drop_dead_post(state, number)
            else:
                print(f"sync #{number}: unarchive failed (will retry): "
                      f"{type(exc).__name__}: {exc}")
            return
        rec["archived"] = False
        changed = True
        print(f"sync #{number}: unarchived (reopened)")

    if changed:
        rec["state"] = "CLOSED" if closed else "OPEN"
        save_state(state)
    time.sleep(0.5)  # rate-limit politeness across a batch sync


def main() -> int:
    # Overlapping cron runs would double-post: one lock per run.
    lock_fh = open(STATE_FILE + ".lock", "w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another run is in progress — skipping")
        return 0

    state = load_state()
    after = int(state.get("last_issue", 0))
    tags = forum_tags()

    # Idempotency: map issue number → existing forum post (active + archived).
    # A post named "#N ..." means the issue is already mirrored — e.g. a
    # previous run posted it but crashed before saving state, or the Discord
    # POST succeeded server-side while the client timed out. `listing_ok`
    # guards the deleted-post check: an absent thread only counts as deleted
    # when the listing itself succeeded.
    threads, listing_ok = forum_threads()
    existing = {}
    for t in threads:
        m = re.match(r"^#(\d+)\s", t.get("name") or "")
        if m:
            existing[int(m.group(1))] = t
    thread_ids = {t["id"] for t in threads}

    # ── job 1: post new issues (plus retry previously failed numbers) ─────
    issues = new_open_issues(after)
    if not state.get("posts") and BOOTSTRAP_SINCE_DAYS > 0:
        # Fresh/unknown baseline: only mirror issues created recently, so a
        # lost state file can't flood the forum with every historical issue.
        # (Skipped entirely when BOOTSTRAP_SINCE_DAYS=0 = mirror everything.)
        cutoff = time.time() - BOOTSTRAP_SINCE_DAYS * 86400
        kept = []
        for i in issues:
            ts = _iso_ts(i.get("createdAt", ""))
            if ts is None:
                print(f"#{i['number']}: unparseable createdAt — keeping (guard is best-effort)")
                kept.append(i)
            elif ts >= cutoff:
                kept.append(i)
        issues = kept
    posts = state.setdefault("posts", {})
    failed = state.setdefault("failed", [])
    # Retry failed numbers first (ascending), then any new ones — a
    # transient failure must never drop an issue: if #N fails but #N+1
    # succeeds, last_issue only advances on success, and #N stays in
    # `failed` until it posts.
    for num in sorted(set(failed) | {i["number"] for i in issues}):
        if num in existing:
            # Already mirrored (crash/timeout recovery) — record, don't re-post.
            if num in failed:
                failed.remove(num)
            posts[str(num)] = {"thread": existing[num]["id"], "state": "OPEN",
                               "tags": [],
                               "archived": bool((existing[num].get("thread_metadata") or {}).get("archived"))}
            print(f"#{num} already posted ({existing[num]['id']}) — recorded, not re-posted")
            if num > after:
                state["last_issue"] = num
            save_state(state)
            continue
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
        # Deleted-post detection without a PATCH: an open issue with stable
        # labels/severity never triggers a PATCH, so a hand-deleted post
        # would otherwise go unnoticed forever. Only when the listing
        # succeeded and the thread id is absent is it truly gone.
        if listing_ok and rec["thread"] not in thread_ids:
            print(f"sync #{num}: post {rec['thread']} no longer in forum — dropping/re-queueing")
            _drop_dead_post(state, num)
            continue
        # Re-read the REAL archived state from the forum scan: Discord
        # auto-archives posts after auto_archive_duration (7 days) of
        # inactivity, and rec["archived"] only tracks our own PATCHes — an
        # open issue's auto-archived post would otherwise never be
        # unarchived and would silently vanish from the active view. The
        # flag lives under thread_metadata.archived in list responses.
        t = existing.get(int(num))
        if t and t.get("thread_metadata"):
            rec["archived"] = bool(t["thread_metadata"].get("archived"))
        sync_post(state, int(num), rec, tags)
    save_state(state)

    print(f"done: {len(issues)} new, {len(posts)} tracked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
