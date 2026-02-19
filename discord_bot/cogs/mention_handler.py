import logging
import re
import time

import discord
from discord.ext import commands

from discord_bot.voice_manager import list_voice_choices
from discord_bot.config import DEBUG_CHANNEL_IDS, MENTION_FALLBACK_CHANNEL_IDS
from discord_bot.google_docs import download_google_doc, extract_doc_id
from discord_bot.audio_delivery import send_audio_chapter
from discord_bot.generation import generate_audiobook
from discord_bot.job_queue import Job, JobStatus
from discord_bot.utils.progress import ProgressTracker
from discord_bot.utils.errors import GoogleDocDownloadFailed, JobCancelled, RateLimitExceeded
from discord_bot.utils.text_processing import clean_google_doc_text
from discord_bot.output_manager import build_output_dir, derive_doc_title, write_document_text

logger = logging.getLogger(__name__)

URL_PATTERN = re.compile(r"https?://\S+")


class VoiceSelect(discord.ui.Select):
    """Dropdown menu of available voices."""

    def __init__(self, doc_url: str, doc_text: str):
        self.doc_url = doc_url
        self.doc_text = doc_text

        voices = list_voice_choices()
        options = [
            discord.SelectOption(label=v, value=v)
            for v in voices[:25]  # Discord max 25 options
        ]

        super().__init__(
            placeholder="Pick a voice...",
            min_values=1,
            max_values=1,
            options=options,
        )

    async def callback(self, interaction: discord.Interaction):
        voice_name = self.values[0]
        bot = interaction.client

        await interaction.response.defer()
        if not bot.engine.ready:
            try:
                await bot.engine.ensure_ready()
            except Exception as exc:
                logger.exception("Failed to load engine on mention callback: %s", exc)
                await interaction.followup.send(
                    f"Failed to load the TTS engine: {exc}", ephemeral=True
                )
                return

        # Create and enqueue the job
        clean_text = clean_google_doc_text(self.doc_text)
        doc_title = derive_doc_title(clean_text)
        output_dir = build_output_dir(
            interaction.user.display_name, interaction.user.id, doc_title
        )
        write_document_text(output_dir, doc_title, clean_text, source_url=self.doc_url)

        job = Job(
            user_id=interaction.user.id,
            user_display_name=interaction.user.display_name,
            channel_id=interaction.channel_id,
            voice_name=voice_name,
            doc_url=self.doc_url,
            doc_title=doc_title,
            text=self.doc_text,
            output_dir=str(output_dir),
        )

        try:
            bot.job_queue.enqueue(job)
        except RateLimitExceeded as exc:
            await interaction.followup.send(str(exc), ephemeral=True)
            return

        job._channel = interaction.channel

        tracker = ProgressTracker(interaction.channel)
        job._tracker = tracker
        await tracker.send_initial()

        await interaction.followup.send(
            f"Audiobook job `{job.id}` queued. Voice: **{voice_name}**. "
            f"Text: {len(self.doc_text):,} characters.",
        )

        # Disable the dropdown after selection
        self.disabled = True
        self.placeholder = f"Selected: {voice_name}"
        await self.view.message.edit(view=self.view)


class VoiceSelectView(discord.ui.View):
    """View containing the voice dropdown. Times out after 2 minutes."""

    def __init__(self, doc_url: str, doc_text: str):
        super().__init__(timeout=120)
        self.message = None
        self.add_item(VoiceSelect(doc_url, doc_text))

    async def on_timeout(self):
        for item in self.children:
            item.disabled = True
        if self.message:
            try:
                await self.message.edit(
                    content="Voice selection timed out. Mention me again with the link to try again.",
                    view=self,
                )
            except discord.HTTPException:
                pass


class MentionHandler(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        # Ignore messages from bots (including self)
        if message.author.bot:
            return

        bot_id = self.bot.user.id if self.bot.user else None
        mention_match = False
        fallback_match = False
        if bot_id is not None:
            mention_match = (
                (self.bot.user in message.mentions)
                or (bot_id in message.raw_mentions)
                or (f"<@{bot_id}>" in message.content)
                or (f"<@!{bot_id}>" in message.content)
            )
            if not mention_match and message.channel and message.channel.id in MENTION_FALLBACK_CHANNEL_IDS:
                display_name = self.bot.user.name
                if message.guild:
                    member = message.guild.get_member(bot_id)
                    if member and member.display_name:
                        display_name = member.display_name
                name_pattern = re.compile(rf"@?\b{re.escape(display_name)}\b", re.IGNORECASE)
                fallback_match = bool(name_pattern.search(message.content))
                mention_match = fallback_match

        if message.guild and message.channel and message.channel.id in DEBUG_CHANNEL_IDS:
            logger.info(
                "DEBUG: Message seen in %s (%d) by %s (%d). Mentioned=%s, raw_mentions=%s, fallback=%s, has_url=%s",
                getattr(message.channel, "name", "unknown"),
                message.channel.id,
                message.author,
                message.author.id,
                mention_match,
                message.raw_mentions,
                fallback_match,
                bool(URL_PATTERN.search(message.content)),
            )

        # Check if the bot is mentioned
        if not mention_match:
            return

        logger.info(
            "Mention received from %s (%d) in channel %s",
            message.author,
            message.author.id,
            getattr(message.channel, "id", "unknown"),
        )

        # Check for a Google Doc or Drive link
        doc_url = None
        for url in URL_PATTERN.findall(message.content):
            if extract_doc_id(url):
                doc_url = url
                break
        if not doc_url:
            # Check if message content is empty (possible missing Message Content Intent)
            stripped = message.content.replace(f"<@{bot_id}>", "").replace(f"<@!{bot_id}>", "").strip()
            if not message.content or not stripped:
                await message.reply(
                    "I can see your mention but couldn't read the message content. "
                    "Make sure **Message Content Intent** is enabled in the Discord Developer Portal.",
                    mention_author=False,
                )
            else:
                await message.reply(
                    "**Usage:** Mention me with a Google Docs link to generate an audiobook.\n"
                    "Example: `@HiggsAudio https://docs.google.com/document/d/.../edit`",
                    mention_author=False,
                )
            return

        # Download the doc first so we can show an error before the dropdown
        async with message.channel.typing():
            try:
                doc_text = await download_google_doc(doc_url)
            except GoogleDocDownloadFailed as exc:
                await message.reply(f"Could not download the document: {exc}", mention_author=False)
                return

        if len(doc_text.strip()) < 10:
            await message.reply("The document appears to be empty or too short.", mention_author=False)
            return

        # Send the voice picker
        view = VoiceSelectView(doc_url, doc_text)
        reply = await message.reply(
            f"Got it! **{len(doc_text):,} characters** downloaded. Pick a voice:",
            view=view,
            mention_author=False,
        )
        view.message = reply


async def setup(bot):
    await bot.add_cog(MentionHandler(bot))
    logger.info("MentionHandler cog loaded and listening for @mentions.")
