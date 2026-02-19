import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

VOICE_LIBRARY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "voice_library")
SMART_VOICE_NAME = "Smart Voice"


def list_voices() -> List[str]:
    """Return sorted voice names from the voice_library directory."""
    voices = []
    if os.path.exists(VOICE_LIBRARY_DIR):
        for f in os.listdir(VOICE_LIBRARY_DIR):
            if f.endswith(".wav"):
                voices.append(f.replace(".wav", ""))
    voices.sort()
    return voices


def list_voice_choices() -> List[str]:
    """Return voice choices with Smart Voice at the top."""
    return [SMART_VOICE_NAME] + list_voices()


def is_smart_voice(name: str) -> bool:
    return name.strip().lower() in {
        SMART_VOICE_NAME.lower(),
        "smartvoice",
        "smart-voice",
    }


def voice_exists(name: str) -> bool:
    if is_smart_voice(name):
        return True
    return os.path.exists(os.path.join(VOICE_LIBRARY_DIR, f"{name}.wav"))


def get_voice_wav_path(name: str) -> Optional[str]:
    path = os.path.join(VOICE_LIBRARY_DIR, f"{name}.wav")
    return path if os.path.exists(path) else None


def get_voice_txt_path(name: str) -> Optional[str]:
    path = os.path.join(VOICE_LIBRARY_DIR, f"{name}.txt")
    return path if os.path.exists(path) else None


def get_voice_transcript(name: str) -> str:
    txt_path = get_voice_txt_path(name)
    if txt_path:
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "This is a voice sample for cloning."


def load_voice_config(name: str) -> Dict:
    if is_smart_voice(name):
        return _default_config()
    config_path = os.path.join(VOICE_LIBRARY_DIR, f"{name}_config.json")
    default = _default_config()
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            # Merge with defaults so new keys are always present
            merged = {**default, **loaded}
            return merged
        except Exception as e:
            logger.warning("Failed to load config for %s: %s", name, e)
    return default


def get_default_voice_config() -> Dict:
    return _default_config()


def _default_config() -> Dict:
    return {
        "temperature": 0.3,
        "max_new_tokens": 1024,
        "seed": 12345,
        "top_k": 50,
        "top_p": 0.95,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "ras_win_len": 7,
        "ras_win_max_num_repeat": 2,
        "do_sample": True,
        "description": "",
        "tags": [],
    }


def build_voice_embed_fields(name: str) -> Dict:
    """Return a dict of fields suitable for a Discord embed describing a voice."""
    config = load_voice_config(name)
    description = config.get("description", "") or "No description"
    tags = ", ".join(config.get("tags", [])) or "None"
    return {
        "name": name,
        "description": description,
        "tags": tags,
        "temperature": config["temperature"],
        "max_new_tokens": config["max_new_tokens"],
    }
