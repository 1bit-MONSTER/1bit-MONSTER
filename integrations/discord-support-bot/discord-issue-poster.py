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
    TAG_STATE_ESCALATED,
    TAG_STATE_PENDING,
    TAG_STATE_RESOLVED,
    TAG_TYPE_FEATURE,
    TAG_TYPE_INQUIRY,
    TAG_TYPE_TROUBLESHOOTING,
    desired_tags,
    forum_search_issue,
    forum_tags,
    forum_threads,
    gh_issue,
    post_issue_post,
    post_tags,
    thread_exists,
    update_post,
)

# The bot manages exactly these tags; ANY other tag on a post (human-added)
# is preserved across syncs. Human triage on the state axis (flipping an
# open post to resolved/escalated) is also respected while the issue stays
# open — only close/reopen transitions are auto-enforced.
MANAGED_TAGS = (set(TAG_SEVERITY.values())
                | {TAG_TYPE_TROUBLESHOOTING, TAG_TYPE_FEATURE, TAG_TYPE_INQUIRY,
                   TAG_STATE_PENDING, TAG_STATE_RESOLVED, TAG_STATE_ESCALATED})
STATE_TAGS = {TAG_STATE_PENDING, TAG_STATE_RESOLVED, TAG_STATE_ESCALATED}

REPO = "1bit-MONSTER/1bit-MONSTER"
STATE_FILE = os.path.expanduser("~/.cache/discord-issue-poster-state.json")
def _env_int(name: str, default: int) -> int:
    """int() with a fallback — an empty/non-numeric env value (e.g. a
    .env line left as 'RETRY_DELAY_SECONDS=') must not crash the import."""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# Only auto-post issues created within this many days on a FRESH state
# (missing/corrupt state file). Prevents a duplicate-post flood — with no
# baseline, a fresh host would treat every open issue as new and mirror
# hundreds of historical issues. Set 0 to mirror every open issue.
BOOTSTRAP_SINCE_DAYS = _env_int("BOOTSTRAP_SINCE_DAYS", 1)
# A failed number is only retried after this long: a client-side timeout
# may have actually created the post server-side, and Discord's search
# index (the retry dedupe) is updated asynchronously. 20 min >> index
# delay, so the combined-miss duplicate window closes.
RETRY_DELAY_SECONDS = _env_int("RETRY_DELAY_SECONDS", 20 * 60)
# A post parked in retry longer than this is not self-healing — surface it
# as a FAILED run (watchdog alert) instead of hiding behind the cooldown.
STUCK_SECONDS = _env_int("STUCK_SECONDS", 2 * 3600)


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
        capture_output=True, text=True, check=True, timeout=60).stdout
    return json.loads(out)


def _is_gone(exc: Exception) -> bool:
    """True when the Discord API says the thread no longer exists (404)."""
    return getattr(exc, "code", None) == 404 or "404" in str(exc)


def _rec_state_tag(rec: dict) -> str | None:
    """The state tag inside rec['tags'] — our last-known write."""
    return next((t for t in (rec.get("tags") or []) if t in STATE_TAGS), None)


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
    state.setdefault("failed_since", {})
    try:
        issue = gh_issue(REPO, number)
        if (issue.get("state") or "").lower() != "closed":
            failed = state.setdefault("failed", [])
            if number not in failed:
                failed.append(number)
                state["failed_since"][number] = time.time()
            state["failed_at"][number] = time.time()
            print(f"sync #{number}: post gone (404), issue open — re-queued for re-post")
        else:
            print(f"sync #{number}: post gone (404), issue closed — dropped")
    except Exception:  # noqa: BLE001 — unknown state; re-queue, closed-skip guard protects
        failed = state.setdefault("failed", [])
        if number not in failed:
            failed.append(number)
            state["failed_since"][number] = time.time()
        state["failed_at"][number] = time.time()
        print(f"sync #{number}: post gone (404), issue state unknown — re-queued (closed-skip guard)")
    save_state(state)


def sync_post(state: dict, number: int, rec: dict, tags: dict,
              id_to_name: dict[str, str], issue: dict) -> bool:
    """Reconcile one tracked post with the (already-fetched) live issue.

    Returns True when the post was reconciled without failure. Every
    Discord PATCH is individually guarded: one failure (post deleted →
    404, or rate-limit 429 mid-batch) must not abort the whole job-2 loop.

    Tag policy — the bot manages ONLY the three derived dimension tags.
    Other (human-added) tags are preserved, and while an issue stays open
    a human's state tag (resolved/escalated) is respected rather than
    overwritten; close/reopen transitions are auto-enforced.
    """
    derived = desired_tags(issue)          # [type, state, severity]
    ttype, derived_state, sev = derived[0], derived[1], derived[2]
    closed = (issue.get("state") or "").lower() == "closed"
    # Reopen detection: open issue + post archived + the record previously
    # said CLOSED = a genuine reopen -> force pending + unarchive. Discord
    # ALSO auto-archives an OPEN issue's post after 7 days of inactivity;
    # that must still unarchive, but must NOT force pending over a human's
    # state tag — so the rec["state"] == "CLOSED" discriminator matters.
    reopened = not closed and bool(rec.get("archived")) and rec.get("state") == "CLOSED"
    changed = False

    # State policy with ownership tracking (rec["state_owner"]: "bot" |
    # "human") so bot-applied state tags can be downgraded again (e.g.
    # escalation removed) while a human's triage choice persists:
    #   * close forces resolved; reopen forces pending (bot)
    #   * an escalation label forces escalated (bot) — and its removal
    #     downgrades back to pending
    #   * otherwise a HUMAN-owned state tag is kept; a bot-owned one is
    #     re-derived (downgrade allowed). If the live tag differs from the
    #     bot's last write, a human intervened — adopt it as human-owned.
    force_state = None
    if closed:
        force_state = TAG_STATE_RESOLVED
    elif reopened:
        force_state = TAG_STATE_PENDING
    elif derived_state == TAG_STATE_ESCALATED:
        force_state = TAG_STATE_ESCALATED

    want = [ttype, force_state or TAG_STATE_PENDING, sev]
    tags_ok = True  # tag reconciliation completed (or nothing to do) — gates the prune

    if want != rec.get("tags"):
        live = post_tags(rec["thread"], id_to_name)
        if live is None:
            # Transient read failure — never treat it as "no tags" (that
            # would PATCH away human tags). Skip only the TAG update; the
            # archive/unarchive step below must still run. The prune is
            # gated on tags_ok so a close run with a failed tag read does
            # not prune a post that still carries stale tags.
            print(f"sync #{number}: live-tag read failed — skipping tag update")
            tags_ok = False
        else:
            live_state = next((t for t in live if t in STATE_TAGS), None)
            owner = rec.get("state_owner", "bot")
            if force_state is None:
                if owner == "human" and live_state:
                    state_tag = live_state              # human triage persists
                elif live_state and live_state != _rec_state_tag(rec):
                    state_tag = live_state              # bot-owned but changed -> human intervened
                    owner = "human"
                else:
                    state_tag = TAG_STATE_PENDING       # bot-owned -> downgrade allowed
                want[1] = state_tag
            preserved = [t for t in live if t not in MANAGED_TAGS]
            final = want + sorted(set(preserved))
            if final != live:
                try:
                    # Guard against tags removed/renamed on the channel:
                    # only PATCH ids that still resolve. Record ONLY what
                    # was actually applied — a missing derived tag stays
                    # out of rec["tags"] so a later run re-attempts it
                    # once the tag is restored to the channel.
                    applied = [t for t in final if t in tags]
                    update_post(rec["thread"], applied_tags=[tags[t] for t in applied])
                except Exception as exc:  # noqa: BLE001
                    if _is_gone(exc):
                        _drop_dead_post(state, number)
                    else:
                        print(f"sync #{number}: tag PATCH failed (will retry): "
                              f"{type(exc).__name__}: {exc}")
                    return False
                rec["tags"] = applied
                if force_state is not None:
                    rec["state_owner"] = "bot"
                else:
                    rec["state_owner"] = owner
                changed = True
                print(f"sync #{number}: tags -> {applied}")
                save_state(state)  # persist NOW — a later archive failure must not lose this

    if closed and not rec.get("archived"):
        try:
            update_post(rec["thread"], archived=True)
        except Exception as exc:  # noqa: BLE001
            if _is_gone(exc):
                _drop_dead_post(state, number)
            else:
                print(f"sync #{number}: archive failed (will retry): "
                      f"{type(exc).__name__}: {exc}")
            return False
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
            return False
        rec["archived"] = False
        changed = True
        print(f"sync #{number}: unarchived (reopened)")

    if changed:
        rec["state"] = "CLOSED" if closed else "OPEN"
        save_state(state)
        time.sleep(0.5)  # rate-limit politeness only after an actual PATCH

    if closed and rec.get("archived") and tags_ok:
        # Fully handled (resolved + archived): prune the record so closed
        # issues don't accumulate and cost a gh issue view subprocess every
        # run forever. Remember the number so a later REOPEN can be told
        # apart from a mere auto-archive when the post is re-adopted.
        # Gated on tags_ok: a skipped/failed tag update must not prune a
        # post that still carries stale tags.
        state["posts"].pop(str(number), None)
        state.setdefault("closed", {})[number] = time.time()
        print(f"sync #{number}: resolved + archived — pruned from tracking")
        save_state(state)
    return True


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
        # One-off contention (a backfill overrunning the 15-min interval)
        # is normal and matches the watchdog's success markers. But if the
        # PREVIOUS line is also contention, the holder looks permanently
        # stuck — print a FAILED line so the watchdog alerts.
        prev = ""
        log_path = os.path.expanduser("~/.local/share/discord-issue-poster.log")
        try:
            with open(log_path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        prev = line
        except OSError:
            pass
        if "another run is in progress" in prev:
            print("FAILED: lock contended across consecutive runs — previous run appears stuck")
        else:
            print("another run is in progress — skipping")
        return 0

    state = load_state()
    after = int(state.get("last_issue", 0))
    tags = forum_tags()
    id_to_name = {v: k for k, v in tags.items()}  # used by recovery + adoption + sync
    # One gh call for the whole run — feeds both job 1 and job 2. A slow /
    # rate-limited GitHub API must not leave a raw traceback in the cron
    # log: fail cleanly (the watchdog's success-marker check then alerts).
    try:
        open_list = fetch_open_issues()
    except Exception as exc:  # noqa: BLE001
        print(f"open-issue fetch FAILED ({type(exc).__name__}: {exc}) — skipping this run")
        return 0
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
    # Normalize `failed` to ints and DROP non-numeric garbage (earlier
    # iterations appended posts-map keys; a stray non-numeric entry must
    # not ValueError the whole run).
    failed = [int(n) for n in state.setdefault("failed", [])
              if str(n).lstrip("-").isdigit()]
    state["failed"] = failed
    issues: list[dict] = []
    # Defined for BOTH branches — the summary reads them even when the
    # listing failed (otherwise the fail-closed path would NameError).
    failed_at: dict[int, float] = {}
    failed_since: dict[int, float] = {}
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
        # unknown (no state file, or last_issue == 0). The decision must
        # be last_issue ALONE — job 2's adoption populates state['posts']
        # from the live forum, so a fresh host that merely adopts existing
        # posts must not then treat the baseline as known and flood the
        # forum with every historical open issue. (A pre-forum state file
        # with last_issue=1957 and no posts key is still known via
        # last_issue.)
        baseline_known = int(state.get("last_issue", 0)) > 0
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
        # issues are candidates EXCEPT numbers still inside their own
        # cooldown (a failed new issue would otherwise retry through the
        # new-issues side before the 20-min window elapses). (JSON keys are
        # strings — normalize.)
        failed_at = {int(k): v for k, v in state.get("failed_at", {}).items()
                     if str(k).lstrip("-").isdigit()}
        state["failed_at"] = failed_at
        failed_since = {int(k): v for k, v in state.get("failed_since", {}).items()
                        if str(k).lstrip("-").isdigit()}
        state["failed_since"] = failed_since
        in_cooldown = {n for n in failed_at
                       if time.time() - failed_at.get(n, 0) <= RETRY_DELAY_SECONDS}
        cooled = {n for n in failed if time.time() - failed_at.get(n, 0) > RETRY_DELAY_SECONDS}
        candidates = cooled | ({i["number"] for i in issues} - in_cooldown)
    fresh_ids: set[str] = set()  # posts created THIS run — not in the pre-posting snapshot
    post_failures = 0
    sync_failures = 0
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
            # Seed tags from the post's live applied_tags: an empty record
            # would make the sync misread the bot's own tags as human
            # triage (ownership heuristic) and stick them. The name-based
            # map carries ids; a search hit has none, so read them live.
            seed_ids = post.get("applied_tags") or []
            seed = [id_to_name[i] for i in seed_ids if i in id_to_name]
            if not seed:
                live = post_tags(post["id"], id_to_name)
                if live is not None:
                    seed = live
            posts[str(num)] = {"thread": post["id"], "state": "OPEN",
                               "tags": seed,
                               # We don't know who applied the existing
                               # tags (bot or human) — treat them as human
                               # triage so a human 'resolved'/'escalated'
                               # on an open issue is preserved, matching
                               # the adoption path.
                               "state_owner": "human",
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
                state.setdefault("failed_since", {})[num] = time.time()
            failed_at[num] = time.time()
            print(f"post #{num} FAILED (will retry): {type(exc).__name__}: {exc}")
            post_failures += 1
            save_state(state)  # persist NOW — a later success must not orphan it
            continue
        if num in failed:
            failed.remove(num)
            failed_at.pop(num, None)
        posts[str(num)] = {
            "thread": tid,
            "state": "OPEN",
            "tags": [t for t in desired_tags(full) if t in tags],
            "state_owner": "bot",
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
    # run for an issue at/below last_issue, or REOPENED issues whose
    # archived post was pruned when it closed): they would otherwise keep
    # frozen tags forever, never closing/archiving with the issue.
    # Closed issues are skipped — re-adopting their archived posts would
    # reintroduce the per-run gh cost the prune removes; a reopen puts
    # the issue back in open_map and adoption fires then.
    # A number recorded in state["closed"] was bot-pruned as CLOSED: its
    # re-adoption is a genuine reopen, so the record is seeded CLOSED
    # (reopen detection forces pending + unarchive) instead of OPEN/human.
    closed_history = {int(k): v for k, v in state.get("closed", {}).items()
                      if str(k).lstrip("-").isdigit()}
    state["closed"] = closed_history
    # Bound growth: forget close history older than 90 days.
    for num in [n for n, t in closed_history.items() if time.time() - t > 90 * 86400]:
        closed_history.pop(num, None)
    for num, t in list(existing.items()):
        if str(num) in posts:
            continue
        if int(num) not in open_map:
            continue  # closed — leave the archived post alone
        was_closed = int(num) in closed_history
        posts[str(num)] = {
            "thread": t["id"],
            "state": "CLOSED" if was_closed else "OPEN",  # reopen vs unknown
            "tags": [id_to_name[i] for i in (t.get("applied_tags") or [])
                     if i in id_to_name],
            # adopted posts: treat the live state tag as human triage —
            # except a genuine reopen, whose tags were bot-applied at the
            # close (pending must be forced by the reopen path).
            "state_owner": "bot" if was_closed else "human",
            "archived": bool((t.get("thread_metadata") or {}).get("archived")),
        }
        if was_closed:
            closed_history.pop(int(num), None)
        print(f"adopted untracked post #{num} ({t['id']})"
              + (" — reopened" if was_closed else ""))

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
        if not sync_post(state, int(num), rec, tags, id_to_name, issue):
            sync_failures += 1
    save_state(state)

    # A failure summary line — deliberately NOT matching the watchdog's
    # success markers (done:/posted/no new issues) so the watchdog alerts.
    # Post failures are parked in `failed` (auto-retry with cooldown), so
    # they do NOT fail the run — otherwise a permanently failing post
    # would flap the watchdog between FAILED and done every cooldown
    # cycle. A post parked past STUCK_SECONDS (clearly not self-healing)
    # and every sync failure do fail the run.
    stuck = [n for n in failed_since
             if n in failed and time.time() - failed_since.get(n, 0) > STUCK_SECONDS]
    if not listing_ok:
        print("FAILED: forum listing unavailable — posting disabled (fail closed)")
    elif sync_failures:
        print(f"FAILED: {sync_failures} sync failure(s)")
    elif stuck:
        print(f"FAILED: posts stuck in retry longer than {STUCK_SECONDS // 3600}h: "
              f"{sorted(stuck)}")
    else:
        parked = [n for n in failed_at if time.time() - failed_at.get(n, 0) <= RETRY_DELAY_SECONDS]
        if parked:
            print(f"done: {len(issues)} new, {len(posts)} tracked, "
                  f"{len(parked)} post(s) awaiting retry")
        else:
            print(f"done: {len(issues)} new, {len(posts)} tracked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
