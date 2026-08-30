#!/usr/bin/env python3
"""discord-issue-poster.py — auto-post new GitHub issues to #issue-tracker as FORUM posts.

Polls the GitHub repo for issues newer than the last-posted one and posts
each to Discord #issue-tracker (a forum channel) as a tagged post (reusing
post_issue.py's forum-post logic). Runs from cron; state (last issue number
handled) persists in ~/.cache/discord-issue-poster-state.json so a re-run
never double-posts.

Cron (strixhalo): */15 * * * * (every 15 min; cheap when nothing new)

Token: ~/.secrets/Discord Bot token.txt (same as the other discord bots).
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "/home/bcloud/1bit-MONSTER/integrations/discord-support-bot")
from post_issue import gh_issue, post_issue_post  # noqa: E402

REPO = "1bit-MONSTER/1bit-MONSTER"
STATE_FILE = os.path.expanduser("~/.cache/discord-issue-poster-state.json")
# Only auto-post issues created from now on — historical issues were posted
# manually. Set to 0 to post every open issue on first run.
BOOTSTRAP_SINCE_DAYS = 1


def last_handled() -> int:
    try:
        return int(json.load(open(STATE_FILE)).get("last_issue", 0))
    except Exception:
        return 0


def save_last(n: int) -> None:
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    json.dump({"last_issue": n, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ")},
              open(STATE_FILE, "w"))


def new_issues(after: int) -> list[dict]:
    """Open issues with number > after, oldest first."""
    out = subprocess.run(
        ["gh", "issue", "list", "--repo", REPO, "--state", "open",
         "--limit", "100", "--json", "number,title,createdAt"],
        capture_output=True, text=True, check=True).stdout
    issues = [i for i in json.loads(out) if i["number"] > after]
    return sorted(issues, key=lambda i: i["number"])


def main() -> int:
    after = last_handled()
    issues = new_issues(after)
    if not issues:
        print(f"no new issues (last handled #{after})")
        return 0
    for issue in issues:
        full = gh_issue(REPO, issue["number"])
        pid = post_issue_post(full)
        print(f"posted #{issue['number']} '{issue['title']}' as forum post {pid}")
        save_last(issue["number"])
        time.sleep(2)  # rate-limit politeness between posts
    print(f"handled {len(issues)} new issue(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
