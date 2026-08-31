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
    forum_search_issue,
    forum_tags,
    forum_threads,
    gh_issue,
    post_issue_post,
    thread_exists,
    update_post,
)

REPO = "1bit-MONSTER/1bit-MONSTER"
STATE_FILE = os.path.expanduser("~/.cache/discord-issue-poster-state.json")
# Only auto-post issues created within this many days on a FRESH state
# (missing/corrupt state file). Prevents a duplicate-post flood — with no
# baseline, a fresh host would treat every open issue as new and mirror
# hundreds of historical issues. Set 0 to mirror every open issue.
BOOTSTRAP_SINCE_DAYS = int(os.getenv("BOOTSTRAP_SINCE_DAYS", "1"))
# A failed number is only retried after this long: a client-side timeout
# may have actually created the post server-side, and Discord's search
# index (the retry dedupe) is updated asynchronously. 20 min >> index
# delay, so the combined-miss duplicate window closes.
RETRY_DELAY_SECONDS = int(os.getenv("RETRY_DELAY_SECONDS", str(20 * 60)))


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


def fetch_open_issues() -> list[dict]:
    """ONE gh call for every open issue, full fields included.

    gh paginates internally past 100. Both job 1 (posting) and job 2
    (sync) read from this single list, so a run stays fast with hundreds
    of tracked posts — no per-post subprocess.
    """
    out = subprocess.run(
        ["gh", "issue", "list", "--repo", REPO, "--state", "open",
         "--limit", "1000",
         "--json", "number,title,url,state,labels,author,createdAt,body"],
        capture_output=True, text=True, check=True, timeout=30).stdout
    return json.loads(out)


def _is_gone(exc: Exception) -> bool:
    """True when the Discord API says the thread no longer exists (404)."""
    return getattr(exc, "code", None) == 404 or "404" in str(exc)


def _drop_dead_post(state: dict, number: int | str) -> None:
    """A tracked post no longer exists (deleted / config change).

    Drop it from the map, and if the issue is STILL OPEN, re-queue the
    number so it is mirrored again on the next run — otherwise it would
    silently vanish from the triage board (the number is below the
    last_issue cursor, so job 1 would never revisit it).
    """
    number = int(number)  # callers pass posts keys (strings) — keep `failed` int-only
    state["posts"].pop(str(number), None)
    state.setdefault("failed_at", {})
    try:
        issue = gh_issue(REPO, number)
        if (issue.get("state") or "").lower() != "closed":
            failed = state.setdefault("failed", [])
            if number not in failed:
                failed.append(number)
                state["failed_at"][number] = time.time()
            print(f"sync #{number}: post gone (404), issue open — re-queued for re-post")
        else:
            print(f"sync #{number}: post gone (404), issue closed — dropped")
    except Exception:  # noqa: BLE001 — unknown state; re-queue, closed-skip guard protects
        failed = state.setdefault("failed", [])
        if number not in failed:
            failed.append(number)
            state["failed_at"][number] = time.time()
        print(f"sync #{number}: post gone (404), issue state unknown — re-queued (closed-skip guard)")
    save_state(state)


def sync_post(state: dict, number: int, rec: dict, tags: dict, issue: dict) -> None:
    """Reconcile one tracked post with the (already-fetched) live issue.

    Every Discord PATCH is individually guarded: one failure (post deleted
    → 404, or rate-limit 429 mid-batch) must not abort the whole job-2
    loop and disable lifecycle sync for every tracked post.
    """
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
        time.sleep(0.5)  # rate-limit politeness only after an actual PATCH


def main() -> int:
    # Overlapping cron runs would double-post: one lock per run. The lock
    # file lives next to the state file — create the parent dir first
    # (~/.cache may not exist on a fresh host, and this runs BEFORE
    # save_state's makedirs).
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    lock_fh = open(STATE_FILE + ".lock", "w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another run is in progress — skipping")
        return 0

    state = load_state()
    after = int(state.get("last_issue", 0))
    tags = forum_tags()
    # One gh call for the whole run — feeds both job 1 and job 2.
    open_list = fetch_open_issues()
    open_map = {i["number"]: i for i in open_list}

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
    posts = state.setdefault("posts", {})
    # Normalize `failed` to ints: job-1 failures append ints, while
    # _drop_dead_post used to append posts keys (strings) — a mixed list
    # makes sorted(candidates) raise TypeError and aborts the whole run.
    failed = [int(n) for n in state.setdefault("failed", [])]
    state["failed"] = failed
    issues: list[dict] = []
    if not listing_ok:
        # Fail closed: without a forum listing the idempotency map is
        # empty, so posting OR retrying `failed` could duplicate a post
        # that actually exists server-side (e.g. a client-side timeout on
        # the earlier POST). Skip everything; retry next run.
        print("forum listing failed — skipping posting and retries this run (fail closed)")
        candidates: set[int] = set()
    else:
        # New issues above the cursor, from the already-fetched list.
        issues = sorted((i for i in open_list if i["number"] > after),
                        key=lambda i: i["number"])
        # The bootstrap cutoff applies ONLY when the baseline is genuinely
        # unknown (no state file, or last_issue == 0) — NOT when the posts
        # map happens to be empty. A host that already advanced last_issue
        # (e.g. the pre-forum state file with last_issue=1957 and no posts
        # key) has a known baseline: older open issues above last_issue
        # must still be mirrored, or they'd be skipped forever once a newer
        # issue advances the cursor.
        baseline_known = int(state.get("last_issue", 0)) > 0 or bool(state.get("posts"))
        if not baseline_known and BOOTSTRAP_SINCE_DAYS > 0:
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
        # Retry failed numbers only once they've been failing long enough
        # for Discord's search index (and any listing propagation) to have
        # seen a post that a client-side timeout may actually have created
        # server-side — closing the combined-miss duplicate window. New
        # issues are always candidates. (JSON keys are strings — normalize.)
        failed_at = {int(k): v for k, v in state.get("failed_at", {}).items()}
        state["failed_at"] = failed_at
        candidates = ({n for n in failed if time.time() - failed_at.get(n, 0) > RETRY_DELAY_SECONDS}
                      | {i["number"] for i in issues})
    fresh_ids: set[str] = set()  # posts created THIS run — not in the pre-posting snapshot
    # Retry failed numbers first (ascending), then any new ones — a
    # transient failure must never drop an issue: if #N fails but #N+1
    # succeeds, last_issue only advances on success, and #N stays in
    # `failed` until it posts.
    for num in sorted(candidates):
        post = existing.get(num)
        if post is None:
            # /threads/active is unreliable (404 on this API version), so
            # the name-based map may miss active posts — search-scope the
            # idempotency by the issue URL fragment before posting.
            tid = forum_search_issue(num)
            if tid:
                post = {"id": tid, "name": f"#{num} (search)", "thread_metadata": {}}
                print(f"#{num}: found via search ({tid}) — recording, not re-posting")
        if post is not None:
            # Already mirrored (crash/timeout recovery) — record, don't re-post.
            if num in failed:
                failed.remove(num)
                failed_at.pop(num, None)
            posts[str(num)] = {"thread": post["id"], "state": "OPEN",
                               "tags": [],
                               "archived": bool((post.get("thread_metadata") or {}).get("archived"))}
            print(f"#{num} already posted ({post['id']}) — recorded, not re-posted")
            if num > after:
                state["last_issue"] = num
            save_state(state)
            continue
        try:
            # post_issue_post needs the FULL issue dict — served from the
            # single open-list fetch (url/labels/state/body all included).
            full = open_map.get(num)
            if full is None:
                # Not in the open list anymore (closed/deleted while it sat
                # in `failed`) — never mirror it, and stop retrying.
                if num in failed:
                    failed.remove(num)
                    failed_at.pop(num, None)
                print(f"#{num} closed before posting — skipped")
                save_state(state)
                continue
            tid = post_issue_post(full)
        except Exception as exc:  # noqa: BLE001
            if num not in failed:
                failed.append(num)
                failed_at[num] = time.time()
            print(f"post #{num} FAILED (will retry): {type(exc).__name__}: {exc}")
            save_state(state)  # persist NOW — a later success must not orphan it
            continue
        if num in failed:
            failed.remove(num)
            failed_at.pop(num, None)
        posts[str(num)] = {
            "thread": tid,
            "state": "OPEN",
            "tags": [t for t in desired_tags(full) if t in tags],
            "archived": False,
        }
        print(f"posted #{num} '{full['title']}' as forum post {tid}")
        fresh_ids.add(tid)
        if num > after:
            state["last_issue"] = num
        save_state(state)
        time.sleep(2)  # rate-limit politeness between posts

    # ── job 2: reconcile tracked posts with live issue state ──────────────
    # Adopt untracked "#N" posts (e.g. created by a manual post_issue.py
    # run for an issue at/below last_issue): they would otherwise keep
    # frozen tags forever, never closing/archiving with the issue.
    id_to_name = {v: k for k, v in tags.items()}
    for num, t in list(existing.items()):
        if str(num) in posts:
            continue
        posts[str(num)] = {
            "thread": t["id"],
            "state": "OPEN",  # unknown; sync_post corrects from gh
            "tags": [id_to_name[i] for i in (t.get("applied_tags") or [])
                     if i in id_to_name],
            "archived": bool((t.get("thread_metadata") or {}).get("archived")),
        }
        print(f"adopted untracked post #{num} ({t['id']})")

    for num, rec in list(posts.items()):
        # Deleted-post detection without a PATCH: an open issue with stable
        # labels/severity never triggers a PATCH, so a hand-deleted post
        # would otherwise go unnoticed forever. A listing absence is only a
        # hint — the active listing caps at 100 and can truncate, so a
        # targeted 404 check confirms before dropping. Posts created
        # earlier in THIS run (fresh_ids) are exempt (snapshot predates).
        if (listing_ok and rec["thread"] not in thread_ids
                and rec["thread"] not in fresh_ids):
            exists = thread_exists(rec["thread"])
            if exists is False:
                print(f"sync #{num}: post {rec['thread']} 404 — dropping/re-queueing")
                _drop_dead_post(state, num)
                continue
            if exists is True:
                print(f"sync #{num}: post exists but missing from listing (truncated?) — keeping")
            # exists is None (unverifiable) — proceed; the PATCH paths below
            # surface a real 404 on their own.
        # Re-read the REAL archived state from the forum scan: Discord
        # auto-archives posts after auto_archive_duration (7 days) of
        # inactivity, and rec["archived"] only tracks our own PATCHes — an
        # open issue's auto-archived post would otherwise never be
        # unarchived and would silently vanish from the active view. The
        # flag lives under thread_metadata.archived in list responses.
        t = existing.get(int(num))
        if t and t.get("thread_metadata"):
            rec["archived"] = bool(t["thread_metadata"].get("archived"))
        # Serve the issue from the single open-list fetch; only closed /
        # deleted issues (absent from it) need an individual gh call.
        # posts keys are strings — int() for the open_map lookup.
        issue = open_map.get(int(num))
        if issue is None:
            try:
                issue = gh_issue(REPO, num)
            except Exception as exc:  # noqa: BLE001 — gone; leave post as-is
                print(f"sync #{num}: skipped ({type(exc).__name__}: {exc})")
                continue
        sync_post(state, int(num), rec, tags, issue)
    save_state(state)

    print(f"done: {len(issues)} new, {len(posts)} tracked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
