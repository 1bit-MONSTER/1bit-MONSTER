#!/usr/bin/env python3
"""post_issue.py — post a GitHub issue to the #issue-tracker FORUM as a post.

Each issue becomes its own Discord forum post (tagged by type / state /
severity), so discussion stays per-issue instead of piling into a flat
channel. This is the only way issues are posted to #issue-tracker — never a
bare message.

The channel is a GUILD_FORUM channel: a post is created in ONE request (the
starter message travels inside the thread-creation call, and applied_tags
stamps the three tag dimensions). This differs from a text channel, where
you would anchor a thread to a separate starter message — forum channels do
not accept bare messages, so the two-step flow is wrong there.

Usage:
    python3 post_issue.py <issue-number>              # from origin repo
    python3 post_issue.py <owner/repo> <issue-number> # from any repo

Requires: DISCORD_TOKEN (env or ~/.secrets/Discord Bot token.txt), and
Network access to the Discord API. GitHub data comes from the `gh` CLI.
When DEEPSEEK_API_KEY is set (and ISSUE_SUMMARY != 0), the starter message
also carries a short LLM summary of the issue.

Config (env, from .env or the environment):
    ISSUE_TRACKER_CHANNEL_ID  Discord FORUM channel id for #issue-tracker
                             (default: 1543724070154145793)
    ISSUE_SUMMARY             "1" (default) adds a DeepSeek 3-line summary to
                             each post; "0" posts without one.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.request

# Absolute bot-directory anchor: cron runs with an arbitrary CWD, so a
# relative ".env" would silently not load (no DEEPSEEK_API_KEY → no LLM
# summary; ISSUE_DIGEST_CHANNEL/ISSUE_SUMMARY overrides ignored).
BOT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv(path: str | None = None) -> None:
    """Minimal .env loader so cron runs see DISCORD_TOKEN/DEEPSEEK_API_KEY etc.

    MUST run before TOKEN / ISSUE_TRACKER_CHANNEL_ID are bound below —
    otherwise a host that keeps secrets only in .env sends an empty
    Authorization header and every Discord call 401s. Anchored to the bot
    directory, not the CWD.
    """
    path = path or os.path.join(BOT_DIR, ".env")
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

API = "https://discord.com/api/v10"
UA = "1bit-docsbot (issue-tracker, 3.0)"
ISSUE_TRACKER_CHANNEL_ID = os.getenv(
    "ISSUE_TRACKER_CHANNEL_ID", "1543724070154145793"
)
TOKEN_FILE = os.path.expanduser("~/.secrets/Discord Bot token.txt")
TOKEN = os.getenv("DISCORD_TOKEN") or (
    open(TOKEN_FILE).read().strip() if os.path.exists(TOKEN_FILE) else ""
)

# ── Forum tags ────────────────────────────────────────────────────────────
# One tag per dimension (type / state / severity). Names must match the
# available_tags configured on the live forum channel (see README.md). Tag
# ids are resolved from the channel at runtime, so reshuffling the tag set
# on Discord never breaks the poster.
TAG_TYPE_TROUBLESHOOTING = "troubleshooting"
TAG_TYPE_FEATURE = "feature"
TAG_TYPE_INQUIRY = "inquiry"
TAG_STATE_PENDING = "pending"
TAG_STATE_RESOLVED = "resolved"
TAG_STATE_ESCALATED = "escalated"
TAG_SEVERITY = {1: "defcon-1", 2: "defcon-2", 3: "defcon-3", 4: "defcon-4", 5: "defcon-5"}

# DEFCON ladder (lower = worse), from the help-desk tag schema. Checked
# top-down; the first matching level wins. Keywords are matched on the
# lowercase title + body.
_DEFCON_KEYWORDS: list[tuple[int, list[str]]] = [
    (1, ["optc hang", "amdgpu hang", "kernel panic", "wayland freeze",
         "hard lock", "power-cycle", "power cycle", "data loss", "security",
         "unusable"]),
    (2, ["crashed", "panic", "segfault", "sigabrt", "sigbus", "oom", "hang",
         "deadlock", "blocker", "regression"]),
    (3, ["broken", "fails", "doesn't work", "does not work", "error:", "bug",
         "not working"]),
    (4, ["slow", "annoying", "quirk", "minor", "would be nice"]),
]

_TYPE_LABEL_KEYWORDS = {
    TAG_TYPE_TROUBLESHOOTING: ("bug", "troubleshoot", "fix"),
    TAG_TYPE_FEATURE: ("feature", "enhancement", "request"),
}

# Labels that escalate a post on the state axis (triage keyword).
ESCALATION_LABEL_KEYWORDS = ("priority", "p0", "p1", "urgent", "critical",
                             "blocker", "hotfix", "severe")


def _headers() -> dict[str, str]:
    return {"Authorization": "Bot " + TOKEN, "User-Agent": UA,
            "Content-Type": "application/json"}


def _api(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(API + path, data=data, headers=_headers(), method=method)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def forum_tags() -> dict[str, str]:
    """Map tag name → tag id from the live forum channel's available_tags."""
    channel = _api("GET", f"/channels/{ISSUE_TRACKER_CHANNEL_ID}")
    return {t["name"]: t["id"] for t in channel.get("available_tags", [])}


def forum_threads() -> tuple[list[dict], bool]:
    """All existing posts (active + archived); returns (threads, complete).

    Used for idempotent posting and deleted-post detection: a post whose
    name starts with "#N " means issue N is already mirrored.

    Pagination notes: the archived-public endpoint honors a `before`
    cursor and is walked page by page; the active endpoint caps at 100
    results and does NOT paginate with `before` (it is fetched once —
    posts auto-archive after 7 days, so >100 active is unlikely).

    ``complete`` is False when any page failed to load — callers must NOT
    treat an absent thread as deleted when the listing itself failed.
    Note: the archived flag lives under thread_metadata.archived, not at
    the top level of the thread object.
    """
    out: list[dict] = []
    complete = True
    for base in ("/threads/active?limit=100", "/threads/archived/public?limit=100"):
        paginate = "archived" in base  # only the archived endpoint takes `before`
        cursor = ""
        while True:
            try:
                path = f"/channels/{ISSUE_TRACKER_CHANNEL_ID}{base}{cursor}"
                data = _api("GET", path)
            except Exception:  # noqa: BLE001 — best-effort; a listing failure
                complete = False  # must not block posting (idempotency is a bonus)
                break
            threads = data.get("threads") or []
            out.extend(threads)
            if not paginate or not data.get("has_more") or not threads:
                break
            cursor = f"&before={threads[-1]['id']}"
    return out, complete


def thread_exists(thread_id: str) -> bool | None:
    """Targeted existence check: True exists, False 404, None other error.

    Used to confirm a deletion before trusting an absent thread id in the
    (possibly truncated) listing scan.
    """
    try:
        _api("GET", f"/channels/{thread_id}")
        return True
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False  # thread does not exist
        return None  # other HTTP errors: unknown
    except Exception:  # noqa: BLE001
        return None


def severity(text: str) -> int:
    """DEFCON severity (1 = worst, 5 = trivial) from a keyword scan."""
    lowered = (text or "").lower()
    for level, keywords in _DEFCON_KEYWORDS:
        if any(k in lowered for k in keywords):
            return level
    return 5


def type_tag(labels: list[dict]) -> str:
    """Pick the type tag from GitHub issue labels (fallback: inquiry)."""
    names = " ".join((label.get("name") or "") for label in (labels or [])).lower()
    for tag, keywords in _TYPE_LABEL_KEYWORDS.items():
        if any(k in names for k in keywords):
            return tag
    return TAG_TYPE_INQUIRY


def is_escalated(labels: list[dict]) -> bool:
    """True when an issue label implies escalation (priority/critical/...)."""
    names = " ".join((label.get("name") or "") for label in (labels or [])).lower()
    return any(k in names for k in ESCALATION_LABEL_KEYWORDS)


def desired_tags(issue: dict) -> list[str]:
    """The three tags a post should carry for this issue's current state.

    Closed issues are resolved; open ones are pending unless an escalation
    label is present. Severity re-derived from title + body each call, so a
    body edit can bump the DEFCON level on the next cron sync.
    """
    ttype = type_tag(issue.get("labels"))
    if (issue.get("state") or "").lower() == "closed":
        state = TAG_STATE_RESOLVED
    else:
        state = TAG_STATE_ESCALATED if is_escalated(issue.get("labels")) else TAG_STATE_PENDING
    body = (issue.get("body") or "") if isinstance(issue.get("body"), str) else ""
    return [ttype, state, TAG_SEVERITY[severity(issue["title"] + "\n" + body)]]


def summarize_issue(issue: dict, api_key: str | None = None) -> str:
    """Optional 3-line LLM summary of the issue (fail-soft: "" on any error)."""
    if not api_key:
        return ""
    if os.getenv("ISSUE_SUMMARY", "1").strip() in ("0", "false", "no", ""):
        return ""
    try:
        import llm
        body = (issue.get("body") or "")[:3000] if isinstance(issue.get("body"), str) else ""
        return llm.chat(
            [
                {"role": "system", "content":
                    "You summarize GitHub issues for the 1bit.MONSTER engine. "
                    "Reply with at most 3 plain-text lines: what the issue is, "
                    "why it matters, and the ask. No markdown headers."},
                {"role": "user", "content":
                    f"Title: {issue['title']}\n\n{body}\n\n{issue['url']}"},
            ],
            api_key,
            max_tokens=180,
            temperature=0.2,
            timeout=45,
        )
    except Exception:  # noqa: BLE001 — a summary must never block posting
        return ""


def gh_issue(repo: str, number: int) -> dict:
    out = subprocess.run(
        ["gh", "issue", "view", str(number), "--repo", repo,
         "--json", "number,title,url,state,labels,author,createdAt,body"],
        capture_output=True, text=True, check=True, timeout=30).stdout
    return json.loads(out)


def update_post(thread_id: str, applied_tags: list[str] | None = None,
                archived: bool | None = None) -> dict:
    """PATCH a forum post's tags and/or archived flag (lifecycle sync)."""
    body: dict = {}
    if applied_tags is not None:
        body["applied_tags"] = applied_tags
    if archived is not None:
        body["archived"] = archived
    return _api("PATCH", f"/channels/{thread_id}", body)


def post_issue_post(issue: dict) -> str:
    """Create a forum post in #issue-tracker for a GitHub issue.

    One request: POST /channels/{forum}/threads with the starter message
    (title + URL + optional LLM summary + body excerpt) and applied_tags.
    Forum post names cap at 100 chars; the compact "#N title" form keeps
    the sidebar readable.
    """
    tags = forum_tags()

    name = f"#{issue['number']} {issue['title']}"
    if len(name) > 100:
        name = name[:97] + "…"

    body = (issue.get("body") or "").strip().replace("\r\n", "\n") if isinstance(issue.get("body"), str) else ""
    excerpt = re.sub(r"\n{2,}", "\n", body)[:600]
    summary = summarize_issue(issue, os.getenv("DEEPSEEK_API_KEY"))

    starter = f"**{issue['title']}** — <{issue['url']}>"
    if summary:
        starter += "\n\n" + summary
    if excerpt and not summary:
        starter += "\n\n" + excerpt
    if len(starter) > 1800:
        starter = starter[:1797] + "…"

    applied = [tags[t] for t in desired_tags(issue) if t in tags]

    post = _api("POST", f"/channels/{ISSUE_TRACKER_CHANNEL_ID}/threads", {
        "name": name,
        # allowed_mentions parse:[] — issue titles/bodies are untrusted
        # public-repo input; "@everyone"/role mentions in them must not
        # ping the server.
        "message": {"content": starter, "allowed_mentions": {"parse": []}},
        "applied_tags": applied,
        "auto_archive_duration": 10080,  # 7 days — issues stay triageable
        "type": 11,                      # GUILD_PUBLIC_THREAD
    })
    return post["id"]


def main() -> int:
    if not TOKEN:
        print("error: no DISCORD_TOKEN (env or ~/.secrets/Discord Bot token.txt)", file=sys.stderr)
        return 2
    repo = "1bit-MONSTER/1bit-MONSTER"
    if len(sys.argv) == 2:
        number = int(sys.argv[1])
    elif len(sys.argv) == 3:
        repo, number = sys.argv[1], int(sys.argv[2])
    else:
        print(__doc__)
        return 2
    issue = gh_issue(repo, number)
    pid = post_issue_post(issue)
    print(f"posted #{issue['number']} '{issue['title']}' to #issue-tracker "
          f"as forum post {pid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
