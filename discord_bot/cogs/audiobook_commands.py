import logging
import time
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands

from discord_bot.voice_manager import list_voice_choices, voice_exists, is_smart_voice, SMART_VOICE_NAME
from discord_bot.utils.permissions import is_admin_interaction
from discord_bot.google_docs import download_google_doc
from discord_bot.audio_delivery import deliver_chapter
from discord_bot.generation import generate_audiobook
from discord_bot.job_queue import Job, JobStatus
from discord_bot.utils.progress import ProgressTracker
from discord_bot.utils.text_processing import clean_google_doc_text
from discord_bot.output_manager import build_output_dir, build_dropbox_path, derive_doc_title, write_document_text, save_chapter_audio
from discord_bot import dropbox_upload
from discord_bot.utils.errors import (
    GoogleDocDownloadFailed,
    JobCancelled,
    RateLimitExceeded,
)

logger = logging.getLogger(__name__)


async def voice_autocomplete(interaction: discord.Interaction, current: str):
    voices = list_voice_choices()
    filtered = [v for v in voices if current.lower() in v.lower()][:25]
    return [app_commands.Choice(name=v, value=v) for v in filtered]


class AudiobookCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        # Wire up the job handler
        self.bot.job_queue.start(self._run_job)

    @app_commands.command(name="audiobook", description="Generate a full audiobook from a Google Doc (Admin only)")
    @app_commands.autocomplete(voice=voice_autocomplete)
    async def audiobook(self, interaction: discord.Interaction, url: str, voice: str):
        if not is_admin_interaction(interaction):
            await interaction.response.send_message(
                "This command is restricted to admins.", ephemeral=True
            )
            return

        if is_smart_voice(voice):
            voice = SMART_VOICE_NAME
        if not voice_exists(voice):
            await interaction.response.send_message(f"Voice `{voice}` not found.", ephemeral=True)
            return

        await interaction.response.defer()
        if not self.bot.engine.ready:
            try:
                await self.bot.engine.ensure_ready()
            except Exception as exc:
                logger.exception("Failed to load engine for audiobook command: %s", exc)
                await interaction.followup.send(f"Failed to load TTS engine: {exc}", ephemeral=True)
                return

        # Download the Google Doc
        try:
            text = await download_google_doc(url)
        except GoogleDocDownloadFailed as exc:
            await interaction.followup.send(f"Failed to download document: {exc}", ephemeral=True)
            return

        if len(text.strip()) < 10:
            await interaction.followup.send("The document appears to be empty or too short.", ephemeral=True)
            return

        clean_text = clean_google_doc_text(text)
        doc_title = derive_doc_title(clean_text)
        output_dir = build_output_dir(
            interaction.user.display_name, interaction.user.id, doc_title
        )
        write_document_text(output_dir, doc_title, clean_text, source_url=url)

        # Create and enqueue the job
        job = Job(
            user_id=interaction.user.id,
            user_display_name=interaction.user.display_name,
            channel_id=interaction.channel_id,
            voice_name=voice,
            doc_url=url,
            doc_title=doc_title,
            text=text,
            output_dir=str(output_dir),
        )

        try:
            self.bot.job_queue.enqueue(job)
        except RateLimitExceeded as exc:
            await interaction.followup.send(str(exc), ephemeral=True)
            return

        # Store channel on the job so the worker can send updates
        job._channel = interaction.channel

        # Send progress embed as a regular message (not a webhook, so it won't expire)
        tracker = ProgressTracker(interaction.channel)
        job._tracker = tracker
        await tracker.send_initial()

        await interaction.followup.send(
            f"Audiobook job `{job.id}` queued. Voice: **{voice}**. "
            f"Text length: {len(text):,} characters.",
        )

    async def _run_job(self, job: Job):
        """Called by the job queue worker for each audiobook job."""
        tracker: ProgressTracker = getattr(job, "_tracker", None)
        channel = getattr(job, "_channel", None)

        if tracker:
            tracker.status = "Generating"
            tracker.start_time = time.time()

        chapters_delivered = 0

        # Set up Dropbox folder if configured
        dropbox_folder = None
        if dropbox_upload.is_configured():
            dropbox_folder = build_dropbox_path(job.user_display_name, job.doc_title)
            result = await dropbox_upload.create_folder(dropbox_folder)
            if result == "exists":
                # Same user + same title already exists — add timestamp
                dropbox_folder = build_dropbox_path(job.user_display_name, job.doc_title, with_timestamp=True)
                result = await dropbox_upload.create_folder(dropbox_folder)
            if result != "error" and channel:
                folder_link = await dropbox_upload.create_shared_link(dropbox_folder)
                if folder_link:
                    await channel.send(
                        f"Dropbox folder created — your audiobook will be at: {folder_link}"
                    )

        # Resolve the guild for file-limit detection
        guild = None
        if channel and hasattr(channel, "guild"):
            guild = channel.guild
        output_dir = Path(job.output_dir) if job.output_dir else None

        async def on_progress(current_chunk, total_chunks, chapter_num, total_chapters):
            job.progress_current = current_chunk
            job.progress_total = total_chunks
            if tracker:
                tracker.current_chunk = current_chunk
                tracker.total_chunks = total_chunks
                tracker.current_chapter = chapter_num
                tracker.total_chapters = total_chapters
                tracker.status = "Generating"
                await tracker.update()

        async def on_chapter_done(chapter_num, title, audio, sample_rate):
            nonlocal chapters_delivered
            if channel:
                if output_dir:
                    try:
                        save_chapter_audio(output_dir, chapter_num, title, audio, sample_rate)
                    except Exception as exc:
                        logger.warning("Failed to write chapter %d to disk: %s", chapter_num, exc)
                await deliver_chapter(
                    channel, audio, sample_rate, title, chapter_num,
                    dropbox_folder=dropbox_folder,
                    guild=guild,
                    output_dir=output_dir,
                )
            chapters_delivered += 1

        try:
            await generate_audiobook(
                engine=self.bot.engine,
                raw_text=job.text,
                voice_name=job.voice_name,
                cancel_event=job.cancel_event,
                on_chapter_done=on_chapter_done,
                on_progress=on_progress,
            )

            if tracker:
                await tracker.finish(chapters_delivered)

            if channel:
                summary = f"Audiobook complete! {chapters_delivered} chapter(s) delivered. (Job `{job.id}`)"
                if dropbox_folder:
                    folder_link = await dropbox_upload.create_shared_link(dropbox_folder)
                    if folder_link:
                        summary += f"\nAll chapters: {folder_link}"
                await channel.send(summary)

        except JobCancelled:
            job.status = JobStatus.CANCELLED
            if channel:
                await channel.send(f"Audiobook job `{job.id}` was cancelled.")

        except Exception as exc:
            job.status = JobStatus.FAILED
            job.error = str(exc)
            logger.exception("Audiobook job %s failed", job.id)
            if channel:
                await channel.send(f"Audiobook job `{job.id}` failed: {exc}")


async def setup(bot):
    await bot.add_cog(AudiobookCommands(bot))
