#!/usr/bin/env python3
"""post_issue.py — post a GitHub issue to Discord #issue-tracker as a THREAD.

Each issue becomes its own Discord public thread (anchored to a starter
message), so discussion stays per-issue instead of piling into a flat
channel. This is the only way issues are posted to #issue-tracker — never a
bare message.

Usage:
    python3 post_issue.py <issue-number>              # from origin repo
    python3 post_issue.py <owner/repo> <issue-number> # from any repo

Requires: DISCORD_TOKEN (env or ~/.secrets/Discord Bot token.txt), and
Network access to the Discord API. GitHub data comes from the `gh` CLI.

Config (env):
    ISSUE_TRACKER_CHANNEL_ID  Discord channel id for #issue-tracker
                             (default: 1542822746986119168)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request

API = "https://discord.com/api/v10"
UA = "1bit-docsbot (issue-tracker, 1.0)"
ISSUE_TRACKER_CHANNEL_ID = os.getenv(
    "ISSUE_TRACKER_CHANNEL_ID", "1542822746986119168"
)
TOKEN_FILE = os.path.expanduser("~/.secrets/Discord Bot token.txt")
TOKEN = os.getenv("DISCORD_TOKEN") or (
    open(TOKEN_FILE).read().strip() if os.path.exists(TOKEN_FILE) else ""
)


def _headers() -> dict[str, str]:
    return {"Authorization": "Bot " + TOKEN, "User-Agent": UA,
            "Content-Type": "application/json"}


def _api(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(API + path, data=data, headers=_headers(), method=method)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def gh_issue(repo: str, number: int) -> dict:
    out = subprocess.run(
        ["gh", "issue", "view", str(number), "--repo", repo,
         "--json", "number,title,url,state,labels,author,createdAt"],
        capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def post_issue_thread(issue: dict) -> str:
    """Starter message in #issue-tracker + a public thread carrying the issue.

    The starter anchors the thread in the channel feed; the issue body is
    posted INTO the thread so the conversation starts with the content.
    """
    labels = ", ".join(l["name"] for l in issue.get("labels", []))
    starter = (
        f"**{issue['title']}** — {issue['url']}\n"
        f"{'`' + labels + '`  ' if labels else ''}· {issue['state']} · "
        f"opened {issue['createdAt'][:10]} by @{issue['author']['login']}"
    )
    msg = _api("POST", f"/channels/{ISSUE_TRACKER_CHANNEL_ID}/messages",
               {"content": starter})

    # Public thread anchored to the starter message (auto-archive 24h).
    thread = _api("POST",
                  f"/channels/{ISSUE_TRACKER_CHANNEL_ID}/messages/{msg['id']}/threads",
                  {"name": f"#{issue['number']} {issue['title'][:90]}",
                   "auto_archive_duration": 1440,
                   "type": 11})  # GUILD_PUBLIC_THREAD
    return thread["id"]


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
    tid = post_issue_thread(issue)
    print(f"posted #{issue['number']} '{issue['title']}' to #issue-tracker "
          f"as thread {tid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
