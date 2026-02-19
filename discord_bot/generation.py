import asyncio
import logging
import os
import tempfile
import time
from typing import Callable, Optional

import numpy as np
import torch
import torchaudio

from discord_bot.config import DEFAULT_CHUNK_SIZE, SAMPLE_RATE
from discord_bot.engine_bridge import EngineBridge
from discord_bot.voice_manager import (
    SMART_VOICE_NAME,
    get_voice_transcript,
    get_voice_wav_path,
    is_smart_voice,
    load_voice_config,
)
from discord_bot.utils.text_processing import detect_chapters, smart_chunk_text, clean_google_doc_text
from discord_bot.utils.errors import GenerationFailed, JobCancelled

logger = logging.getLogger(__name__)


def _build_messages(system_content, voice_transcript, wav_path, chunk):
    """Build the ChatML message list for a single chunk.

    Imports from boson_multimodal are deferred to here so that the heavy
    model code (triggered by the package __init__) is only loaded when
    the engine is already running, not at cog-import time.
    """
    from boson_multimodal.data_types import AudioContent, Message

    return [
        Message(role="system", content=system_content),
        Message(role="user", content=voice_transcript),
        Message(role="assistant", content=AudioContent(audio_url=wav_path)),
        Message(role="user", content=chunk),
    ]


async def generate_audiobook(
    engine: EngineBridge,
    raw_text: str,
    voice_name: str,
    cancel_event: asyncio.Event,
    on_chapter_done: Optional[Callable] = None,
    on_progress: Optional[Callable] = None,
):
    """Generate an audiobook from text, delivering audio per chapter.

    Parameters
    ----------
    engine : EngineBridge
    raw_text : str - the full book text (from Google Doc)
    voice_name : str - voice library voice name
    cancel_event : asyncio.Event - set to cancel
    on_chapter_done : async callback(chapter_num, chapter_title, audio, sample_rate)
    on_progress : async callback(current_chunk, total_chunks, chapter_num, total_chapters)
    """
    from audio_processing_utils import normalize_audio_volume

    text = clean_google_doc_text(raw_text)
    chapters = detect_chapters(text)

    smart_voice = is_smart_voice(voice_name)
    wav_path = None
    voice_transcript = None
    if not smart_voice:
        wav_path = get_voice_wav_path(voice_name)
        if not wav_path:
            raise GenerationFailed(f"Voice '{voice_name}' not found in voice library.")
        voice_transcript = get_voice_transcript(voice_name)
    else:
        voice_name = SMART_VOICE_NAME
    config = load_voice_config(voice_name)

    total_chapters = len(chapters)
    all_chapter_chunks = []
    for _, body in chapters:
        chunks = smart_chunk_text(body, max_chunk_size=DEFAULT_CHUNK_SIZE)
        all_chapter_chunks.append(chunks)
    total_chunks = sum(len(c) for c in all_chapter_chunks)
    chunks_done = 0

    system_content = "Generate audio following instruction."

    first_chunk_audio_path = None
    first_chunk_text = None

    try:
        for ch_idx, ((title, _body), chunks) in enumerate(zip(chapters, all_chapter_chunks)):
            if cancel_event.is_set():
                raise JobCancelled("Job was cancelled.")

            chapter_audio_parts = []
            sample_rate = SAMPLE_RATE

            for ci, chunk in enumerate(chunks):
                if cancel_event.is_set():
                    raise JobCancelled("Job was cancelled.")

                if smart_voice:
                    if first_chunk_audio_path and first_chunk_text:
                        from boson_multimodal.data_types import AudioContent, Message

                        messages = [
                            Message(role="system", content=system_content),
                            Message(role="user", content=first_chunk_text),
                            Message(role="assistant", content=AudioContent(audio_url=first_chunk_audio_path)),
                            Message(role="user", content=chunk),
                        ]
                    else:
                        from boson_multimodal.data_types import Message

                        messages = [
                            Message(role="system", content=system_content),
                            Message(role="user", content=chunk),
                        ]
                else:
                    messages = _build_messages(system_content, voice_transcript, wav_path, chunk)

                output = await engine.generate_safe(messages, config)

                if output.audio is not None:
                    chapter_audio_parts.append(output.audio)
                    sample_rate = output.sampling_rate

                    if smart_voice and first_chunk_audio_path is None:
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                        tmp_path = tmp.name
                        tmp.close()
                        waveform = torch.from_numpy(output.audio).float()
                        if waveform.dim() == 1:
                            waveform = waveform.unsqueeze(0)
                        torchaudio.save(tmp_path, waveform, output.sampling_rate)
                        first_chunk_audio_path = tmp_path
                        first_chunk_text = chunk
                else:
                    logger.warning("No audio returned for chunk %d of chapter '%s'", ci + 1, title)

                chunks_done += 1
                if on_progress:
                    await on_progress(chunks_done, total_chunks, ch_idx + 1, total_chapters)

            if chapter_audio_parts:
                full_chapter = np.concatenate(chapter_audio_parts, axis=0)
                full_chapter = normalize_audio_volume(full_chapter, sample_rate=sample_rate)

                if on_chapter_done:
                    await on_chapter_done(ch_idx + 1, title, full_chapter, sample_rate)
            else:
                logger.warning("Chapter '%s' produced no audio.", title)
    finally:
        if first_chunk_audio_path and os.path.exists(first_chunk_audio_path):
            try:
                os.unlink(first_chunk_audio_path)
            except OSError:
                pass


async def generate_short_tts(
    engine: EngineBridge,
    text: str,
    voice_name: str,
) -> tuple:
    """Generate a short TTS clip. Returns (audio_array, sample_rate)."""
    from audio_processing_utils import normalize_audio_volume

    wav_path = get_voice_wav_path(voice_name)
    if not wav_path:
        raise GenerationFailed(f"Voice '{voice_name}' not found.")
    voice_transcript = get_voice_transcript(voice_name)
    config = load_voice_config(voice_name)

    messages = _build_messages(
        "Generate audio following instruction.", voice_transcript, wav_path, text
    )

    output = await engine.generate_safe(messages, config)
    if output.audio is None:
        raise GenerationFailed("Engine returned no audio.")
    audio = normalize_audio_volume(output.audio, sample_rate=output.sampling_rate)
    return audio, output.sampling_rate
