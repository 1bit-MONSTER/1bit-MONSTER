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

Config (env):
    ISSUE_TRACKER_CHANNEL_ID  Discord FORUM channel id for #issue-tracker
                             (default: 1543724070154145793)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.request

API = "https://discord.com/api/v10"
UA = "1bit-docsbot (issue-tracker, 2.0)"
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


def gh_issue(repo: str, number: int) -> dict:
    out = subprocess.run(
        ["gh", "issue", "view", str(number), "--repo", repo,
         "--json", "number,title,url,state,labels,author,createdAt,body"],
        capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def post_issue_post(issue: dict) -> str:
    """Create a forum post in #issue-tracker for a GitHub issue.

    One request: POST /channels/{forum}/threads with the starter message
    (title + URL + body excerpt) and applied_tags. Forum post names cap at
    100 chars; the compact "#N title" form keeps the sidebar readable.
    """
    tags = forum_tags()

    name = f"#{issue['number']} {issue['title']}"
    if len(name) > 100:
        name = name[:97] + "…"

    body = (issue.get("body") or "").strip().replace("\r\n", "\n")
    excerpt = re.sub(r"\n{2,}", "\n", body)[:600]
    starter = f"**{issue['title']}** — <{issue['url']}>"
    if excerpt:
        starter += "\n\n" + excerpt
    if len(starter) > 1800:
        starter = starter[:1797] + "…"

    applied = [tags[t] for t in (
        type_tag(issue.get("labels")),
        TAG_STATE_PENDING,
        TAG_SEVERITY[severity(issue["title"] + "\n" + body)],
    ) if t in tags]

    post = _api("POST", f"/channels/{ISSUE_TRACKER_CHANNEL_ID}/threads", {
        "name": name,
        "message": {"content": starter},
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
