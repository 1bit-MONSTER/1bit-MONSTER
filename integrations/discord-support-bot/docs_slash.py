"""docs_slash.py — /docs slash command backed by Context7 + DeepSeek.

A self-contained ``discord.py`` component that adds a ``/docs`` slash command
to the existing authorized 1bit.MONSTER Discord bot. When a user runs
``/docs <question>`` it:
   1. fetches the most relevant docs from Context7 (/1bit-monster/1bit-monster)
   2. has DeepSeek write a grounded answer
   3. replies with the answer + source links

This component runs its own gateway connection. The other 1bit bots
(discord-inbox.py, traffic-digest, etc.) are REST pollers, so they load
separately and do not conflict with this gateway connection. The bot must be
invited with the ``applications.commands`` scope for the slash command to show.

Run:
    python3 docs_slash.py            # reads .env (see .env.example)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
import urllib.request

import discord
from discord import app_commands

import context7
import llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("docsbot")

API = "https://discord.com/api/v10"
# Channel the existing inbox bot watches — used only to auto-discover the guild.
KNOWN_CHANNEL = os.getenv("DISCORD_KNOWN_CHANNEL", "1542589031055888386")

LIBRARY_ID = os.getenv("CONTEXT7_LIBRARY_ID", "/1bit-monster/1bit-monster")
COMMAND_NAME = os.getenv("DOCS_COMMAND_NAME", "docs")
COMMAND_DESC = os.getenv(
    "DOCS_COMMAND_DESC",
    "Ask the 1bit.MONSTER docs (grounded in the official documentation)",
)
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))


def token_from_env_or_files() -> str:
    t = os.getenv("DISCORD_TOKEN")
    if t:
        return t.strip()
    for path in (
        os.path.expanduser("~/.secrets/Discord Bot token.txt"),
        os.path.expanduser("~/Documents/Discord Bot token.txt"),
    ):
        try:
            with open(path, encoding="utf-8") as fh:
                val = fh.read().strip()
            if val:
                return val
        except OSError:
            continue
    return ""


def _discover_guild_id(token: str) -> int | None:
    """Best-effort: find the guild id from a known channel, else /users/@me."""
    try:
        req = urllib.request.Request(
            API + f"/channels/{KNOWN_CHANNEL}",
            headers={"Authorization": "Bot " + token, "User-Agent": "1bit-docsbot (1.0)"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            data = __import__("json").load(r)
        return data.get("guild_id")
    except Exception as exc:  # noqa: BLE001
        log.warning("could not discover guild id from channel %s: %s", KNOWN_CHANNEL, exc)
        return None


def _split(text: str, limit: int = 1993) -> list[str]:
    if len(text) <= limit:
        return [text]
    out, cur = [], ""
    for para in text.split("\n"):
        if len(cur) + len(para) + 1 > limit:
            if cur:
                out.append(cur)
            cur = ""
            while len(para) > limit:
                out.append(para[:limit])
                para = para[limit:]
        cur = (cur + "\n" + para).strip() if cur else para.strip()
    if cur:
        out.append(cur)
    return out


def _compose_answer(question: str) -> tuple[str, str]:
    """Synchronous Context7 + DeepSeek answer (run OFF the event loop).

    Returns (answer, context_block); an empty block means no relevant docs.
    Blocking here would stall the gateway (heartbeats, other commands).
    """
    data = context7.get_context(LIBRARY_ID, question, os.getenv("CONTEXT7_API_KEY"))
    block = context7.format_context(data)
    if not block.strip():
        return "", ""
    answer = llm.generate(
        question, block, os.getenv("DEEPSEEK_API_KEY"), max_tokens=MAX_TOKENS
    )
    links = context7.source_links(data)
    if links:
        answer = answer.rstrip() + "\n\n**Sources:** " + " · ".join(links[:4])
    return answer, block


def _fetch_issue(number: int) -> dict:
    """Synchronous `gh issue view` (run OFF the event loop)."""
    out = subprocess.run(
        ["gh", "issue", "view", str(number),
         "--repo", "1bit-MONSTER/1bit-MONSTER",
         "--json", "number,title,url,state,labels,createdAt"],
        capture_output=True, text=True, timeout=30, check=True)
    return json.loads(out.stdout)


class DocsSlash(discord.Client):
    def __init__(self, token: str, guild_id: int | None) -> None:
        intents = discord.Intents.default()  # message_content not needed for slash
        super().__init__(intents=intents)
        self.token = token
        self.guild_id = guild_id
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self) -> None:
        # Issue #1961: register /docs GLOBALLY so it works on every server the
        # bot joins (a guild-scoped command silently 404s with "Unknown
        # interaction" on a second server). Global propagation can take up to
        # an hour; when DISCORD_GUILD_ID is EXPLICITLY configured we ALSO sync
        # to that guild for instant availability on the primary server.
        @self.tree.command(name=COMMAND_NAME, description=COMMAND_DESC)
        async def docs(interaction: discord.Interaction, question: str) -> None:  # noqa: ANN202
            await self._answer(interaction, question)

        @self.tree.command(name="issue", description="Look up a GitHub issue (1bit-MONSTER/1bit-MONSTER)")
        async def issue(interaction: discord.Interaction, number: int) -> None:  # noqa: ANN202
            await self._lookup_issue(interaction, number)

        await self.tree.sync()
        log.info("synced /%s globally", COMMAND_NAME)
        if self.guild_id:
            guild = self.get_guild(self.guild_id) or await self.fetch_guild(self.guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            log.info("synced /%s to guild %s (instant, primary server)", COMMAND_NAME, guild.id)

    async def on_ready(self) -> None:
        log.info("Logged in as %s (id=%s)", self.user, self.user.id)

    async def _answer(self, interaction: discord.Interaction, question: str) -> None:
        question = (question or "").strip()
        if not question:
            await interaction.response.send_message(
                "Ask me anything about 1bit.MONSTER, e.g. `/docs how do I build the engine?`"
            )
            return
        # Defer so the user sees "thinking" while Context7 + DeepSeek run.
        await interaction.response.defer(thinking=True)
        try:
            # Context7 + DeepSeek are blocking HTTP calls — run them off the
            # gateway loop so heartbeats and other commands stay responsive.
            answer, block = await asyncio.to_thread(_compose_answer, question)
            if not block.strip():
                await interaction.followup.send(
                    "I couldn't find relevant docs for that. Try rephrasing, or check the docs hub: <https://docs.1bit.monster>."
                )
                return
        except Exception as exc:  # noqa: BLE001
            log.exception("answer failed")
            await interaction.followup.send(
                f"Sorry, I hit an error while answering: `{type(exc).__name__}`."
            )
            return
        for chunk in _split(answer):
            await interaction.followup.send(chunk)

    async def _lookup_issue(self, interaction: discord.Interaction, number: int) -> None:
        """/issue <n> — compact GitHub issue card via the gh CLI."""
        await interaction.response.defer(thinking=True)
        try:
            # gh can hang — run it off the event loop.
            d = await asyncio.to_thread(_fetch_issue, number)
        except subprocess.CalledProcessError as exc:
            await interaction.followup.send(
                f"Could not find issue #{number}: {(exc.stderr or '').strip()[:200]}")
            return
        except Exception as exc:  # noqa: BLE001
            log.exception("issue lookup failed")
            await interaction.followup.send(f"Sorry, I hit an error: `{type(exc).__name__}`.")
            return
        labels = ", ".join(l["name"] for l in d.get("labels", [])) or "none"
        closed = (d.get("state") or "").lower() == "closed"
        # Issue titles are untrusted public-repo input — never allow a
        # "@everyone"/role mention in them to ping the server.
        await interaction.followup.send(
            f"{'✅' if closed else '🟡'} **#{d['number']}** {d['title']}\n"
            f"State: {d.get('state', '?')} · Labels: {labels}\n"
            f"Created: {(d.get('createdAt') or '?')[:10]}\n{d['url']}",
            allowed_mentions=discord.AllowedMentions.none(),
        )


def _load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> None:
    _load_dotenv()
    token = token_from_env_or_files()
    if not token:
        log.error("Could not find DISCORD_TOKEN (set it or place a token file).")
        return
    guild_id = os.getenv("DISCORD_GUILD_ID")
    guild_id = int(guild_id) if guild_id and guild_id.strip().isdigit() else None
    if guild_id is None:
        guild_id = _discover_guild_id(token)
    if not os.getenv("DEEPSEEK_API_KEY") or not os.getenv("CONTEXT7_API_KEY"):
        log.warning("DEEPSEEK_API_KEY / CONTEXT7_API_KEY not set — /docs will error until set.")
    bot = DocsSlash(token, guild_id)
    bot.run(token)


if __name__ == "__main__":
    main()
