import asyncio
import logging
import time
from typing import Optional

import discord

from discord_bot.config import PROGRESS_UPDATE_INTERVAL

logger = logging.getLogger(__name__)


def _bar(fraction: float, width: int = 20) -> str:
    filled = int(width * fraction)
    return "[" + "=" * filled + ">" * (1 if filled < width else 0) + " " * (width - filled - 1) + "]"


class ProgressTracker:
    """Manages a Discord embed that shows generation progress.

    Uses a regular channel message (not an interaction webhook) so it
    doesn't expire after 15 minutes.
    """

    def __init__(self, channel: discord.abc.Messageable):
        self.channel = channel
        self.message: Optional[discord.Message] = None
        self._last_update = 0.0
        self.total_chapters = 0
        self.current_chapter = 0
        self.total_chunks = 0
        self.current_chunk = 0
        self.status = "Queued"
        self.start_time: Optional[float] = None
        self._lock = asyncio.Lock()
        self._failed = False

    def _build_embed(self) -> discord.Embed:
        if self.total_chunks > 0:
            fraction = self.current_chunk / self.total_chunks
        else:
            fraction = 0.0
        pct = int(fraction * 100)
        bar = _bar(fraction)

        embed = discord.Embed(title="Audiobook Generation", color=discord.Color.blue())
        embed.description = f"```\n{bar} {pct}%\n```"

        detail = f"Chapter: {self.current_chapter}/{self.total_chapters}"
        if self.total_chunks:
            detail += f" | Chunk: {self.current_chunk}/{self.total_chunks}"
        if self.start_time and fraction > 0:
            elapsed = time.time() - self.start_time
            eta_seconds = int(elapsed / fraction * (1 - fraction))
            eta_min = eta_seconds // 60
            detail += f" | ETA: {eta_min}m"

        embed.add_field(name="Status", value=self.status, inline=True)
        embed.add_field(name="Progress", value=detail, inline=False)
        return embed

    async def send_initial(self):
        embed = self._build_embed()
        self.message = await self.channel.send(embed=embed)

    async def update(self, *, force: bool = False):
        """Edit the progress embed. Rate-limited to avoid Discord 429s."""
        if self._failed:
            return
        now = time.time()
        if not force and (now - self._last_update) < PROGRESS_UPDATE_INTERVAL:
            return
        async with self._lock:
            if self.message is None:
                return
            try:
                await self.message.edit(embed=self._build_embed())
                self._last_update = time.time()
            except discord.HTTPException as exc:
                logger.warning("Failed to update progress embed: %s", exc)
                self._failed = True

    async def finish(self, total_chapters: int):
        self.status = "Complete"
        self.current_chapter = total_chapters
        self.total_chapters = total_chapters
        self._failed = False  # try one more time for the final update
        await self.update(force=True)
