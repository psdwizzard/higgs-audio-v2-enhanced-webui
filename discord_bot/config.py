import os
from dotenv import load_dotenv

load_dotenv()

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
GUILD_IDS = [int(g) for g in os.getenv("GUILD_IDS", "").split(",") if g.strip()]
ADMIN_USERS = [int(u) for u in os.getenv("ADMIN_USERS", "").split(",") if u.strip()]
DEBUG_CHANNEL_IDS = [int(c) for c in os.getenv("DEBUG_CHANNEL_IDS", "").split(",") if c.strip()]
MENTION_FALLBACK_CHANNEL_IDS = [
    int(c) for c in os.getenv("MENTION_FALLBACK_CHANNEL_IDS", "").split(",") if c.strip()
]

# Dropbox cloud delivery (OAuth2 refresh token flow)
DROPBOX_APP_KEY = os.getenv("DROPBOX_APP_KEY", "").strip()
DROPBOX_APP_SECRET = os.getenv("DROPBOX_APP_SECRET", "").strip()
DROPBOX_REFRESH_TOKEN = os.getenv("DROPBOX_REFRESH_TOKEN", "").strip()

# Engine paths
MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"

# Generation defaults
DEFAULT_CHUNK_SIZE = 200
MAX_TTS_LENGTH = 500
SAMPLE_RATE = 24000

# Discord limits
DEFAULT_DISCORD_FILE_LIMIT_BYTES = 25 * 1024 * 1024  # 25 MB (non-boosted server default)
MP3_BITRATE = "64k"


def get_discord_file_limit(guild=None) -> int:
    """Return the actual file upload limit for a guild, falling back to 25 MB."""
    if guild is not None:
        try:
            return guild.filesize_limit
        except Exception:
            pass
    return DEFAULT_DISCORD_FILE_LIMIT_BYTES

# Progress update interval (seconds)
PROGRESS_UPDATE_INTERVAL = 10

# Rate limits
MAX_AUDIOBOOKS_PER_USER = 1
MAX_RETRY_ATTEMPTS = 2
