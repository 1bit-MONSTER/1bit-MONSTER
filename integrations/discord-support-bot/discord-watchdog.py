#!/usr/bin/env python3
"""discord-watchdog.py — alert #general when the 1bit Discord automation is unhealthy.

Runs every 10 minutes via cron and checks the whole bot fleet:

  * systemd user services  docsbot + docsbot-prefix  are active
  * cron log files are fresh (stale logs = a dead cron, which is silent):
      discord-inbox.log             < 40 min   (*/5)
      discord-issue-poster.log      < 70 min   (*/15)
      discord-enterprise-watch.log  < 70 min   (*/15)
      discord-traffic-digest.log    < 26 h     (23:45)
      discord-traffic-report-daily.log < 26 h  (23:50)
  * Context7 retrieval still returns docs (the /docs backbone, smoke.py)
  * .env carries the three required keys (Discord / Context7 / DeepSeek)

Alerts once per failing check, re-alerts after 12 h of continuous failure,
and posts a recovery line when a check comes back. State in
~/.cache/discord-watchdog-state.json.

Cron (strixhalo): */10 * * * *
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, "/home/bcloud/1bit-MONSTER/integrations/discord-support-bot")

# systemctl --user needs the user runtime dir; cron does not set it, so a
# bare cron run would fail every systemctl check with "inactive ()".
os.environ.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")

BOT_DIR = "/home/bcloud/1bit-MONSTER/integrations/discord-support-bot"
ENV_FILE = os.path.join(BOT_DIR, ".env")
API = "https://discord.com/api/v10"
ALERT_CHANNEL = "1542812729272696843"  # #general — default; overridden from .env in main()
TOKEN_FILE = os.path.expanduser("~/.secrets/Discord Bot token.txt")
STATE_FILE = os.path.expanduser("~/.cache/discord-watchdog-state.json")
LOG_DIR = os.path.expanduser("~/.local/share")
RE_ALERT_SECONDS = 12 * 3600

SERVICES = ("docsbot.service", "docsbot-prefix.service")
# name -> (log file, max age seconds, success-marker regex on the last line).
# Freshness alone is not enough: a cron that crashes on every run still
# appends a traceback, keeping the mtime fresh while the job silently
# does nothing. The last non-empty line must also carry a success marker
# (e.g. the poster's "done: N new" / "no new issues").
LOGS = {
    "cron:discord-inbox": ("discord-inbox.log", 40 * 60, r"\bok\b"),
    "cron:discord-issue-poster": ("discord-issue-poster.log", 70 * 60,
                                  r"done:|posted|no new issues|another run is in progress"),
    "cron:discord-enterprise-watch": ("discord-enterprise-watch.log", 70 * 60,
                                      r"no new enterprise|alert|posted"),
    "cron:discord-issue-digest": ("discord-issue-digest.log", 9 * 24 * 3600,
                                  r"posted|already posted|no open issues|found via search"),
    "cron:discord-traffic-digest": ("discord-traffic-digest.log", 26 * 3600,
                                    r"posted"),
    # NOTE: discord-traffic-report-daily (23:50 UTC) was retired — the 23:45
    # digest posts to #general and the #traffic-report embeds come from
    # repo-stats.py on the ryzen host. Keeping the old log check would alert
    # every 12 h about a cron that no longer exists.
}
ENV_KEYS = ("DISCORD_TOKEN", "CONTEXT7_API_KEY", "DEEPSEEK_API_KEY")


def load_dotenv(path: str = ENV_FILE) -> None:
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _run_checks() -> dict[str, str]:
    """Return {check: ok|detail} for every check."""
    out: dict[str, str] = {}

    for svc in SERVICES:
        try:
            r = subprocess.run(["systemctl", "--user", "is-active", svc],
                               capture_output=True, text=True, timeout=20)
            out[svc] = "ok" if r.stdout.strip() == "active" else f"inactive ({r.stdout.strip()})"
        except Exception as exc:  # noqa: BLE001
            out[svc] = f"check error: {type(exc).__name__}: {exc}"

    now = time.time()
    for name, (log, max_age, marker) in LOGS.items():
        path = os.path.join(LOG_DIR, log)
        try:
            age = now - os.path.getmtime(path)
            if age > max_age:
                out[name] = f"stale: last write {int(age // 60)} min ago (>{max_age // 60} min)"
                continue
            tail = ""
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        tail = line
            if re.search(marker, tail):
                out[name] = "ok"
            else:
                out[name] = f"fresh but no success marker: last line {tail[-90:]!r}"
        except OSError:
            out[name] = "missing log file"

    load_dotenv()
    missing = [k for k in ENV_KEYS if not os.getenv(k)]
    out["env:keys"] = "ok" if not missing else f"missing: {', '.join(missing)}"

    try:
        import context7
        data = context7.get_context(
            os.getenv("CONTEXT7_LIBRARY_ID", "/1bit-monster/1bit-monster"),
            "How do I build the engine?",
            os.getenv("CONTEXT7_API_KEY"),
        )
        snippets = len(data.get("infoSnippets", [])) + len(data.get("codeSnippets", []))
        out["context7:retrieval"] = "ok" if snippets > 0 else "empty retrieval"
    except Exception as exc:  # noqa: BLE001
        out["context7:retrieval"] = f"error: {type(exc).__name__}: {exc}"

    return out


def _post(text: str) -> None:
    token = os.getenv("DISCORD_TOKEN") or open(TOKEN_FILE).read().strip()
    req = urllib.request.Request(
        API + f"/channels/{ALERT_CHANNEL}/messages",
        # allowed_mentions parse:[] — alert details can quote log content
        # that includes user-posted text; "@everyone" in it must not ping
        # the channel.
        data=json.dumps({"content": text, "allowed_mentions": {"parse": []}}).encode(),
        headers={"Authorization": "Bot " + token, "Content-Type": "application/json",
                 "User-Agent": "1bit-docsbot (watchdog, 1.0)"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        r.read()


def _sanitize(text: str) -> str:
    """Strip secrets and URLs before anything reaches the public channel.

    Exception messages can embed the full failing request URL — if a
    Context7/DeepSeek key travels as a query parameter, a 429/timeout
    would otherwise post the key to #general.
    """
    for env_key in ("CONTEXT7_API_KEY", "DEEPSEEK_API_KEY", "DISCORD_TOKEN"):
        val = os.getenv(env_key)
        if val and len(val) >= 8:
            text = text.replace(val, "[redacted]")
    text = re.sub(r"https?://[^\s'\"]+", "[url]", text)
    return text[:500]


def main() -> int:
    load_dotenv()
    # WATCHDOG_CHANNEL can live in .env — resolve AFTER load_dotenv() so the
    # import-time default doesn't shadow it.
    global ALERT_CHANNEL
    ALERT_CHANNEL = os.getenv("WATCHDOG_CHANNEL", ALERT_CHANNEL)
    results = _run_checks()
    failing = {k: v for k, v in results.items() if v != "ok"}

    state = {}
    try:
        state = json.load(open(STATE_FILE))
    except Exception:
        pass
    alerts = state.setdefault("alerts", {})  # check -> last alert ts (0 = not alerting)
    now = int(time.time())

    to_post: list[str] = []
    for check, detail in sorted(failing.items()):
        last = alerts.get(check, 0)
        if last == 0 or now - last >= RE_ALERT_SECONDS:
            to_post.append(f"🔴 **{check}** — {_sanitize(detail)}")
            alerts[check] = now
    for check in [c for c in alerts if c not in failing]:
        if alerts.get(check):
            to_post.append(f"🟢 **{check}** — recovered")
        alerts[check] = 0  # reset (keep key)

    if to_post:
        # Multiple simultaneous failures (services + crons + context7 +
        # env keys) can exceed Discord's 2000-char message limit — a 400
        # would mean NO alert is delivered and the same oversized message
        # is retried every 10 min. Chunk into separate posts.
        header = "**1bit Discord watchdog**"
        chunks, cur, size = [], [], len(header) + 1
        for line in to_post:
            if size + len(line) + 1 > 1800 and cur:
                chunks.append(header + "\n" + "\n".join(cur))
                cur, size = [], len(header) + 1
            cur.append(line)
            size += len(line) + 1
        if cur:
            chunks.append(header + "\n" + "\n".join(cur))
        try:
            for chunk in chunks:
                _post(chunk)
            print("\n".join(chunks))
        except Exception as exc:  # noqa: BLE001
            # Send failed — the alert never reached Discord. Do NOT persist
            # the timestamps below, so the next 10-min run retries the
            # alert (only after a successful send does the 12h re-alert
            # cadence start).
            print(f"WARNING: could not post alert: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return 1

    # Persist AFTER a successful send: a delivered alert starts the 12h
    # re-alert cadence; a failed send retries on the next run instead.
    tmp = STATE_FILE + ".tmp"
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)  # ~/.cache may not exist
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"alerts": alerts, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ")},
                  fh, indent=2)
    os.replace(tmp, STATE_FILE)  # atomic — a crash must not reset alert state
    print(f"watchdog ok ({len(failing)} failing)" if not failing
          else f"watchdog: {len(failing)} failing — posted")
    return 0 if not failing else 1


if __name__ == "__main__":
    sys.exit(main())
