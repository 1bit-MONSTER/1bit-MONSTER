"""bot.py — Context7-backed Discord support bot for 1bit.MONSTER.

Answers users' support questions in Discord, grounded in the official
1bit.MONSTER documentation retrieved from Context7, and phrased by DeepSeek.

Two ways to trigger it (configurable):
  * an explicit prefix command, e.g. ``!docs how do I build?`` (any channel)
  * any message in a configured support channel (no prefix needed)

Run:
    python3 -m pip install -r requirements.txt
    python3 bot.py            # reads credentials from the .env file

Requires DISCORD_TOKEN, DEEPSEEK_API_KEY, CONTEXT7_API_KEY (see .env.example).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone

import discord

import context7
import llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("docsbot")

# ── config ──────────────────────────────────────────────────────────────── #
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
CONTEXT7_API_KEY = os.getenv("CONTEXT7_API_KEY", "")
LIBRARY_ID = os.getenv("CONTEXT7_LIBRARY_ID", "/1bit-monster/1bit-monster")
COMMAND = os.getenv("DOCS_COMMAND", "!docs").strip().lower()
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))


def _channel_ids() -> set[int]:
    raw = os.getenv("SUPPORT_CHANNEL_IDS", "")
    return {int(x) for x in raw.split(",") if x.strip().isdigit()}


SUPPORT_CHANNELS = _channel_ids()
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "8"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "3"))


def split_msg(text: str, limit: int = 1993) -> list[str]:
    """Split a message into Discord-safe chunks (2000-char limit)."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    cur = ""
    for para in text.split("\n"):
        if len(cur) + len(para) + 1 > limit:
            if cur:
                chunks.append(cur)
            cur = ""
            while len(para) > limit:  # giant single line
                chunks.append(para[:limit])
                para = para[limit:]
        cur = (cur + "\n" + para).strip() if cur else para.strip()
    if cur:
        chunks.append(cur)
    return chunks


class RateLimiter:
    def __init__(self) -> None:
        self._last: dict[int, float] = {}
        self._lock = threading.Lock()

    def allowed(self, user_id: int) -> bool:
        now = time.time()
        with self._lock:
            prev = self._last.get(user_id, 0.0)
            if now - prev < COOLDOWN_SECONDS:
                return False
            self._last[user_id] = now
            return True


class DocsBot(discord.Client):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True  # required in the dev portal too
        super().__init__(intents=intents)
        self._limiter = RateLimiter()
        self._sem = threading.BoundedSemaphore(MAX_WORKERS)

    async def on_ready(self) -> None:
        log.info("Logged in as %s (id=%s)", self.user, self.user.id)
        log.info(
            "support channels=%s command='%s' library=%s",
            sorted(SUPPORT_CHANNELS) or "ALL",
            COMMAND,
            LIBRARY_ID,
        )

    # ── message routing ── #
    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return
        if not message.content:
            # Almost always the Message Content gateway intent being OFF in
            # the Discord Developer Portal (Bot → Message Content Intent).
            log.warning(
                "skipped msg id=%s from %s: empty content "
                "(enable Message Content Intent in the dev portal)",
                message.id, message.author,
            )
            return
        if not message.guild:  # ignore DM-based spam; only support guild channels
            return

        content = message.content.strip()
        lowered = content.lower()
        is_command = lowered == COMMAND or lowered.startswith(COMMAND + " ")
        is_support_channel = message.channel.id in SUPPORT_CHANNELS

        if not (is_command or is_support_channel):
            return

        question = content
        if is_command:
            question = content[len(COMMAND):].strip()
        if not question:
            await message.channel.send(
                f"Hi! Ask me anything about 1bit.MONSTER, e.g. `{COMMAND} how do I build the engine?`"
            )
            return

        await self._answer(message, question)

    async def _answer(self, message: discord.Message, question: str) -> None:
        if not self._limiter.allowed(message.author.id):
            await message.channel.send(
                f"Please wait a moment before asking again ({COOLDOWN_SECONDS}s)."
            )
            return
        if not DEEPSEEK_API_KEY or not CONTEXT7_API_KEY:
            await message.channel.send(
                "The support bot is not configured yet: missing DEEPSEEK_API_KEY / CONTEXT7_API_KEY."
            )
            return

        await message.channel.send("🔎 Looking that up in the docs…")
        try:
            data = context7.get_context(LIBRARY_ID, question, CONTEXT7_API_KEY)
            block = context7.format_context(data)
            if not block.strip():
                await message.channel.send(
                    "I couldn't find relevant docs for that. Try rephrasing, or check the docs hub: <https://docs.1bit.monster>."
                )
                return
            answer = llm.generate(
                question, block, DEEPSEEK_API_KEY, max_tokens=MAX_TOKENS
            )
        except Exception as exc:  # noqa: BLE001 - surface to user, log details
            log.exception("answer failed for question=%r", question)
            await message.channel.send(
                f"Sorry, I hit an error while answering: `{type(exc).__name__}`."
            )
            return

        links = context7.source_links(data)
        if links:
            answer = answer.rstrip() + "\n\n**Sources:** " + " · ".join(links[:4])
        for chunk in split_msg(answer):
            await message.channel.send(chunk)


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no external dependency)."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)


def main() -> None:
    _load_dotenv()
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        log.error("DISCORD_TOKEN is not set (copy .env.example → .env and fill it).")
        return
    client = DocsBot()
    client.run(token)


if __name__ == "__main__":
    main()
