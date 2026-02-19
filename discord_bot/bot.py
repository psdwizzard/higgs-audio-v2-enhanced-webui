import logging
from typing import List

import discord
from discord.ext import commands

from discord_bot.config import GUILD_IDS
from discord_bot.engine_bridge import EngineBridge
from discord_bot.job_queue import JobQueue

logger = logging.getLogger(__name__)


class HiggsBot(commands.Bot):
    """Discord bot that wraps the Higgs Audio TTS engine."""

    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        super().__init__(command_prefix="!", intents=intents)

        self.engine = EngineBridge()
        self.job_queue = JobQueue()

    async def setup_hook(self):
        # Load cogs
        cog_extensions = [
            "discord_bot.cogs.tts_commands",
            "discord_bot.cogs.audiobook_commands",
            "discord_bot.cogs.admin_commands",
            "discord_bot.cogs.mention_handler",
        ]
        self._loaded_cogs = []
        for ext in cog_extensions:
            try:
                await self.load_extension(ext)
                self._loaded_cogs.append(ext)
                logger.info("Loaded cog: %s", ext)
            except Exception as exc:
                logger.error("Failed to load cog %s: %s", ext, exc, exc_info=True)

        # Sync slash commands to the configured guilds (instant) or globally
        if GUILD_IDS:
            for gid in GUILD_IDS:
                guild = discord.Object(id=gid)
                self.tree.copy_global_to(guild=guild)
                await self.tree.sync(guild=guild)
                logger.info("Synced commands to guild %d", gid)
        else:
            await self.tree.sync()
            logger.info("Synced commands globally (may take up to 1 hour).")

    async def on_ready(self):
        logger.info("Logged in as %s (ID: %d)", self.user, self.user.id)
        logger.info("Loaded cogs: %s", ", ".join(self._loaded_cogs) or "(none)")
        logger.info("Engine will load on first request (lazy loading).")

        await self.change_presence(
            activity=discord.Activity(type=discord.ActivityType.listening, name="/voices")
        )
