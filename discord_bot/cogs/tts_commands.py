import io
import logging

import discord
from discord import app_commands
from discord.ext import commands

import numpy as np
import torch
import torchaudio
import tempfile
import os

from discord_bot.config import MAX_TTS_LENGTH, ADMIN_USERS
from discord_bot.voice_manager import list_voices, voice_exists, get_voice_wav_path, build_voice_embed_fields, load_voice_config
from discord_bot.generation import generate_short_tts
from discord_bot.utils.errors import EngineNotReady, GenerationFailed

logger = logging.getLogger(__name__)


async def voice_autocomplete(interaction: discord.Interaction, current: str):
    """Provide autocomplete for the voice parameter."""
    voices = list_voices()
    filtered = [v for v in voices if current.lower() in v.lower()][:25]
    return [app_commands.Choice(name=v, value=v) for v in filtered]


class TTSCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @app_commands.command(name="voices", description="List all available voices")
    async def voices(self, interaction: discord.Interaction):
        voices = list_voices()
        if not voices:
            await interaction.response.send_message("No voices found in the voice library.", ephemeral=True)
            return

        embed = discord.Embed(title="Available Voices", color=discord.Color.green())
        # Show voices in groups to avoid embed limits
        page_size = 24
        for i in range(0, len(voices), page_size):
            batch = voices[i : i + page_size]
            lines = []
            for v in batch:
                fields = build_voice_embed_fields(v)
                desc = fields["description"][:60] if fields["description"] != "No description" else ""
                tags = fields["tags"]
                entry = f"**{v}**"
                if desc:
                    entry += f" - {desc}"
                if tags != "None":
                    entry += f" [{tags}]"
                lines.append(entry)
            embed.description = "\n".join(lines)

        embed.set_footer(text=f"{len(voices)} voices | Use /preview <voice> to hear a sample")
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="preview", description="Preview a voice sample")
    @app_commands.autocomplete(voice=voice_autocomplete)
    async def preview(self, interaction: discord.Interaction, voice: str):
        if not voice_exists(voice):
            await interaction.response.send_message(f"Voice `{voice}` not found.", ephemeral=True)
            return

        wav_path = get_voice_wav_path(voice)
        fields = build_voice_embed_fields(voice)

        embed = discord.Embed(title=f"Voice Preview: {voice}", color=discord.Color.purple())
        embed.add_field(name="Description", value=fields["description"], inline=False)
        embed.add_field(name="Tags", value=fields["tags"], inline=True)
        embed.add_field(name="Temperature", value=str(fields["temperature"]), inline=True)
        embed.add_field(name="Max Tokens", value=str(fields["max_new_tokens"]), inline=True)

        file = discord.File(wav_path, filename=f"{voice}.wav")
        await interaction.response.send_message(embed=embed, file=file)

    @app_commands.command(name="tts", description="Generate short speech from text")
    @app_commands.autocomplete(voice=voice_autocomplete)
    async def tts(self, interaction: discord.Interaction, voice: str, text: str):
        if not voice_exists(voice):
            await interaction.response.send_message(f"Voice `{voice}` not found.", ephemeral=True)
            return

        if len(text) > MAX_TTS_LENGTH:
            await interaction.response.send_message(
                f"Text too long ({len(text)} chars). Max is {MAX_TTS_LENGTH}. "
                "Use `/audiobook` for longer content.",
                ephemeral=True,
            )
            return

        await interaction.response.defer()

        try:
            if not self.bot.engine.ready:
                await self.bot.engine.ensure_ready()

            audio, sr = await generate_short_tts(self.bot.engine, text, voice)

            # Convert to WAV bytes for upload
            tensor = torch.from_numpy(audio).float()
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                torchaudio.save(tmp_path, tensor, sr)
                file = discord.File(tmp_path, filename=f"tts_{voice}.wav")
                await interaction.followup.send(
                    content=f"**Voice:** {voice} | **Text:** {text[:100]}{'...' if len(text) > 100 else ''}",
                    file=file,
                )
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        except GenerationFailed as exc:
            await interaction.followup.send(f"Generation failed: {exc}", ephemeral=True)
        except Exception as exc:
            logger.exception("TTS generation error")
            await interaction.followup.send(f"An error occurred: {exc}", ephemeral=True)


async def setup(bot):
    await bot.add_cog(TTSCommands(bot))
