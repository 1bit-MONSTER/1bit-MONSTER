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
`DISCORD_GUILD_ID` (or auto-discovery) scopes the command to your server so it
appears immediately instead of waiting up to an hour.

## Usage
- **`/docs <question>`** — in any channel, e.g. `/docs how do I build the engine?`
- The bot answers with a grounded reply plus the doc source links it used.

The `/docs` command runs its own gateway connection. Your other 1bit bots
(`discord-inbox.py`, traffic-digest, etc.) are REST pollers and load
separately, so they don't conflict.

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
