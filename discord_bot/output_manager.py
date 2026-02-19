import datetime
import os
from pathlib import Path
from typing import Optional

from discord_bot.audio_delivery import wav_array_to_mp3_bytes

OUTPUT_ROOT = Path("output") / "discord_bot"


def _sanitize_component(value: str, max_len: int = 80) -> str:
    cleaned = "".join(c for c in value if c not in '<>:"/\\|?*')
    cleaned = " ".join(cleaned.split()).strip()
    if not cleaned:
        return "untitled"
    if len(cleaned) > max_len:
        return cleaned[:max_len].rstrip()
    return cleaned


def build_dropbox_path(user_display_name: str, doc_title: str, with_timestamp: bool = False) -> str:
    """Generate a Dropbox folder path: /audiobooks/{user}/{title}/ or with datetime suffix."""
    user_part = _sanitize_component(user_display_name)
    title_part = _sanitize_component(doc_title)
    if with_timestamp:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H-%M")
        title_part = f"{title_part} ({now})"
    return f"/audiobooks/{user_part}/{title_part}"


def derive_doc_title(clean_text: str) -> str:
    for line in clean_text.splitlines():
        line = line.strip()
        if line:
            return _sanitize_component(line.lstrip("# ").strip())
    return "untitled"


def build_output_dir(user_display: str, user_id: int, doc_title: str) -> Path:
    date_str = datetime.date.today().isoformat()
    user_part = _sanitize_component(f"{user_display}-{user_id}")
    title_part = _sanitize_component(doc_title)
    out_dir = OUTPUT_ROOT / date_str / user_part / title_part
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_document_text(
    out_dir: Path,
    doc_title: str,
    text: str,
    source_url: Optional[str] = None,
) -> Path:
    filename = f"{_sanitize_component(doc_title)}.txt"
    path = out_dir / filename
    header = f"{doc_title}\n"
    if source_url:
        header += f"Source: {source_url}\n"
    header += "\n"
    path.write_text(header + text, encoding="utf-8")
    return path


def save_chapter_audio(
    out_dir: Path,
    chapter_num: int,
    chapter_title: str,
    audio,
    sample_rate: int,
) -> Path:
    base = f"Chapter {chapter_num} - {chapter_title}"
    filename = _sanitize_component(base, max_len=120) + ".mp3"
    path = out_dir / filename
    mp3_bytes = wav_array_to_mp3_bytes(audio, sample_rate)
    path.write_bytes(mp3_bytes)
    return path
