# Context7 Discord Support Bot

A `/docs` support command for the existing 1bit.MONSTER Discord bot that
answers users' questions **grounded in the official documentation** — not from
memory, and with source links.

```
user: /docs how do I build the engine?
   └─► docs_slash.py ──► Context7 (/1bit-monster/1bit-monster)   retrieve relevant docs
                     └──► DeepSeek (deepseek-chat)               write a grounded answer
                        └──► replies in Discord (answer + doc links)
```

`Context7` supplies the **facts** (your indexed docs — guides, model families,
wiki, reference). `DeepSeek` supplies the **wording**. No hallucinations.

> Context7 itself does **not** ship a Discord bot. This is a small command you
> attach to your existing bot, using Context7 as the retrieval backend.

---

## What's here

| File | Purpose |
|------|---------|
| **`docs_slash.py`** | **Primary.** A `discord.py` component that adds a `/docs` slash command to the existing bot. |
| `bot.py` | Alternative: a standalone bot that answers a `!docs` prefix command or any message in a support channel (no slash registration). |
| `post_issue.py` | Posts a GitHub issue to **#issue-tracker as a forum post** (each issue = one tagged post in the forum channel, never a flat message). `python3 post_issue.py <number>` — see the docstring. |
| `discord-issue-poster.py` | **Auto-posts new GitHub issues to #issue-tracker as forum posts** — cron poller (every 15 min) that finds open issues newer than the last handled one and posts each with tags. State in `~/.cache/discord-issue-poster-state.json`; never double-posts. |
| `context7.py` | Retrieval client for Context7 `GET /v2/context` (framework-agnostic). |
| `llm.py` | DeepSeek chat-completions client (framework-agnostic). |
| `.env.example` | Copy to `.env` and fill in real credentials. |
| `requirements.txt` | `discord.py`, `requests`. |

The answer pipeline (`context7.py` + `llm.py`) is plain functions, so you can
also drop it into any bot framework (discord.js, py-cord, etc.).

## Setup

### 1. Bot already authorized — just confirm the scope
Your bot token is already on this machine
(`~/.secrets/Discord Bot token.txt`, which `docs_slash.py` reads automatically).
Make sure the bot was invited with the **`applications.commands`** OAuth2 scope
so slash commands are reachable. If it was only invited with `bot`, re-invite:

`https://discord.com/oauth2/authorize?client_id=<APP_ID>&scope=bot+applications.commands&permissions=<perms>`

You can reuse the same invite from the Developer Portal → OAuth2 → URL Generator.

### 2. Keys
- **Context7:** https://context7.com/dashboard → `CONTEXT7_API_KEY`
- **DeepSeek:** your deepseek API key → `DEEPSEEK_API_KEY`

### 3. Configure + run
```bash
cd integrations/discord-support-bot
python3 -m pip install -r requirements.txt
cp .env.example .env      # set CONTEXT7_API_KEY + DEEPSEEK_API_KEY; DISCORD_TOKEN optional
python3 docs_slash.py
```

That starts a gateway connection and registers **`/docs`** with your bot.
The command is registered **globally** (works on every server the bot joins;
propagation can take up to an hour). Set `DISCORD_GUILD_ID` explicitly to
also sync instantly to your primary server — the global registration still
happens, so the command is never guild-scoped (issue #1961).

### Run it as a service (recommended for always-on)
The bot must stay connected to the Discord gateway, so run it as a systemd
**user** service rather than a one-off process:

```bash
./install-docsbot-service.sh
```

This creates `~/.config/systemd/user/docsbot.service` from the committed
`docsbot.service` template, installs the venv, enables linger (start at boot),
and starts it. Manage it with `systemctl --user status/restart docsbot` and
watch logs with `journalctl --user -u docsbot -f`.

### Availability (issue #1962) — do not run only on the dev host

The default install runs as a **systemd user service on the host you run it
on**. If that host reboots, goes offline, or is decommissioned, `/docs` goes
down with it. The bot is self-contained and deployment-agnostic, so run it on
a small VPS / Container app / hosted runner instead:

```bash
# on the target host (any Linux with systemd + python3):
git clone https://github.com/1bit-MONSTER/1bit-MONSTER
cd 1bit-MONSTER/integrations/discord-support-bot
cp .env.example .env            # fill in the three keys (or let the install
                                # script assemble from ~/.secrets/*)
./install-docsbot-service.sh
```

Recommended hardening (from the systemd template, enable as needed):
- `Restart=on-failure` + `RestartSec=5` (already in the template) so a crash
  or a transient gateway disconnect self-heals.
- `WatchdogSec=120` + `NotifyAccess=main` if you add sd_notify heartbeats.
- A basic uptime check: `systemctl --user is-active docsbot` from a cron
  with alerting to the same server.

Secrets: prefer injecting `DISCORD_TOKEN` / `CONTEXT7_API_KEY` /
`DEEPSEEK_API_KEY` via a secret manager or the host's secret store rather
than a checked-in `.env` (see issue #1965 — the install script now assembles
and validates them from `~/.secrets/*`).

### Smoke test
Validate the pipeline (does **not** need the DeepSeek key):

```bash
python3 smoke.py
```

It checks Context7 is retrieval-ready for `/1bit-monster/1bit-monster`, and, if
`DEEPSEEK_API_KEY` is set, that a grounded answer is produced.

## Usage
- **`/docs <question>`** — in any channel, e.g. `/docs how do I build the engine?`
- The bot answers with a grounded reply plus the doc source links it used.

The `/docs` command runs its own gateway connection. Your other 1bit bots
(`discord-inbox.py`, traffic-digest, etc.) are REST pollers and load
separately, so they don't conflict.

## Forum tags (#issue-tracker)

`#issue-tracker` is a Discord **forum channel**: every GitHub issue becomes
one forum **post**, and each post carries exactly one tag from each of three
orthogonal dimensions, so you can filter by clicking any combination:

| Dimension | Tags |
|-----------|------|
| **type** | `troubleshooting` 🐛 · `feature` ✨ · `inquiry` ❓ (chosen from the issue's GitHub labels) |
| **state** | `pending` 🕘 at creation (resolved / escalated are for human triage) |
| **severity** | `defcon-1` 🟥 → `defcon-5` ⬜ (keyword scan of title + body; lower = worse) |

The tag ids are resolved from the live channel at runtime, so reordering the
tag set in Discord never breaks the poster. To point the poster at a
different forum channel, set `ISSUE_TRACKER_CHANNEL_ID` in `.env`.

## How answers stay accurate
1. `context7.get_context(...)` calls
   `GET /api/v2/context?libraryId=/1bit-monster/1bit-monster&query=<question>`
   for the most relevant guide + code snippets.
2. `llm.generate(...)` sends them to DeepSeek with a system prompt that says
   *use ONLY this context*.
3. If nothing relevant is returned, the bot says so rather than guessing.

Because snippets come from your curated docs, you control what it can answer.

---

*MIT — part of 1bit.MONSTER.*
