import io
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import discord
import numpy as np
import torch
import torchaudio

from discord_bot.config import DEFAULT_DISCORD_FILE_LIMIT_BYTES, MP3_BITRATE, SAMPLE_RATE, get_discord_file_limit
from discord_bot import dropbox_upload

logger = logging.getLogger(__name__)


def wav_array_to_mp3_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert a float32 numpy array to MP3 bytes via ffmpeg."""
    # Write WAV to a temp file, convert to MP3, read back
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_f:
        wav_path = wav_f.name
    mp3_path = wav_path.replace(".wav", ".mp3")

    try:
        tensor = torch.from_numpy(audio).float()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        torchaudio.save(wav_path, tensor, sample_rate)

        subprocess.run(
            [
                "ffmpeg", "-y", "-i", wav_path,
                "-codec:a", "libmp3lame", "-b:a", MP3_BITRATE,
                "-loglevel", "error",
                mp3_path,
            ],
            check=True,
            capture_output=True,
        )

        with open(mp3_path, "rb") as f:
            return f.read()
    finally:
        for p in (wav_path, mp3_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def split_mp3_bytes(data: bytes, max_size: int = DEFAULT_DISCORD_FILE_LIMIT_BYTES) -> List[bytes]:
    """If *data* exceeds *max_size*, split it into roughly equal parts via ffmpeg."""
    if len(data) <= max_size:
        return [data]

    num_parts = (len(data) // max_size) + 1

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(data)
        src_path = tmp.name

    parts: List[bytes] = []
    try:
        # Get duration via ffprobe
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", src_path],
            capture_output=True, text=True, check=True,
        )
        duration = float(result.stdout.strip())
        segment_duration = duration / num_parts

        for i in range(num_parts):
            part_path = src_path.replace(".mp3", f"_part{i}.mp3")
            start = segment_duration * i
            subprocess.run(
                [
                    "ffmpeg", "-y", "-ss", str(start), "-t", str(segment_duration),
                    "-i", src_path, "-codec:a", "copy", "-loglevel", "error",
                    part_path,
                ],
                check=True, capture_output=True,
            )
            with open(part_path, "rb") as f:
                parts.append(f.read())
            try:
                os.unlink(part_path)
            except OSError:
                pass
    finally:
        try:
            os.unlink(src_path)
        except OSError:
            pass

    return parts


def _sanitize_filename(chapter_num: int, chapter_title: str, part_suffix: str = "") -> str:
    base = f"Chapter {chapter_num} - {chapter_title}{part_suffix}.mp3"
    return "".join(c for c in base if c not in '<>:"/\\|?*')


async def deliver_chapter(
    channel: discord.abc.Messageable,
    audio: np.ndarray,
    sample_rate: int,
    chapter_title: str,
    chapter_num: int,
    dropbox_folder: Optional[str] = None,
    guild: Optional[discord.Guild] = None,
    output_dir: Optional[Path] = None,
) -> Optional[str]:
    """Convert audio to MP3 and deliver via Dropbox (preferred) or Discord.

    Returns the Dropbox shared link for the chapter file, or None.
    """
    mp3_data = wav_array_to_mp3_bytes(audio, sample_rate)
    filename = _sanitize_filename(chapter_num, chapter_title)
    size_mb = len(mp3_data) / (1024 * 1024)

    # --- Dropbox delivery ---
    if dropbox_upload.is_configured() and dropbox_folder:
        dropbox_path = f"{dropbox_folder}/{filename}"
        uploaded = await dropbox_upload.upload_file(mp3_data, dropbox_path)
        if uploaded:
            link = await dropbox_upload.create_shared_link(dropbox_path)
            if link:
                await channel.send(f"**{filename}** ({size_mb:.1f} MB) — [Dropbox link]({link})")
                logger.info("Delivered %s via Dropbox (%.1f MB)", filename, size_mb)
                return link
            else:
                await channel.send(f"**{filename}** ({size_mb:.1f} MB) uploaded to Dropbox (link unavailable).")
                return None
        # If Dropbox upload failed, fall through to Discord delivery
        logger.warning("Dropbox upload failed for %s, falling back to Discord upload", filename)

    # --- Discord delivery ---
    file_limit = get_discord_file_limit(guild)
    parts = split_mp3_bytes(mp3_data, max_size=file_limit)

    for idx, part in enumerate(parts):
        if len(parts) == 1:
            part_filename = filename
        else:
            part_filename = _sanitize_filename(chapter_num, chapter_title, f" (Part {idx + 1})")
        part_size_mb = len(part) / (1024 * 1024)

        try:
            file = discord.File(io.BytesIO(part), filename=part_filename)
            await channel.send(content=f"**{part_filename}** ({part_size_mb:.1f} MB)", file=file)
            logger.info("Sent %s via Discord (%.1f MB)", part_filename, part_size_mb)
        except discord.HTTPException as exc:
            logger.warning("Discord upload failed for %s: %s", part_filename, exc)
            # Save locally as last resort
            if output_dir:
                local_path = output_dir / part_filename
                local_path.write_bytes(part)
                await channel.send(
                    f"**{part_filename}** is too large for Discord ({part_size_mb:.1f} MB). "
                    f"Saved locally at: `{local_path}`"
                )
            else:
                await channel.send(
                    f"**{part_filename}** failed to upload ({part_size_mb:.1f} MB): {exc}"
                )
    return None


async def send_audio_chapter(
    channel: discord.abc.Messageable,
    audio: np.ndarray,
    sample_rate: int,
    chapter_title: str,
    chapter_num: int,
) -> None:
    """Legacy function: Convert audio to MP3 and send as Discord attachment(s)."""
    await deliver_chapter(channel, audio, sample_rate, chapter_title, chapter_num)
