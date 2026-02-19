import asyncio
import gc
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch

from discord_bot.config import MODEL_PATH, AUDIO_TOKENIZER_PATH

logger = logging.getLogger(__name__)

_serve_engine = None
_executor = ThreadPoolExecutor(max_workers=1)
_lock = asyncio.Lock()
_init_lock = asyncio.Lock()

# How long (seconds) the engine sits idle before it gets unloaded
ENGINE_IDLE_TIMEOUT = 20 * 60  # 20 minutes


def _init_engine_sync():
    global _serve_engine
    if _serve_engine is not None:
        return

    from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading Higgs Audio engine on %s ...", device)
    _serve_engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device=device)
    logger.info("Engine loaded successfully.")


def _unload_engine_sync():
    global _serve_engine
    if _serve_engine is None:
        return
    logger.info("Unloading engine to free VRAM...")
    del _serve_engine
    _serve_engine = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("Engine unloaded. VRAM freed.")


def _generate_sync(messages, config: dict):
    from boson_multimodal.data_types import ChatMLSample

    if _serve_engine is None:
        raise RuntimeError("Engine not initialized")

    sample = ChatMLSample(messages=messages)

    generate_kwargs = {
        "chat_ml_sample": sample,
        "max_new_tokens": config.get("max_new_tokens", 1024),
        "temperature": config.get("temperature", 0.3),
        "top_p": config.get("top_p", 0.95),
        "top_k": config.get("top_k", 50),
        "stop_strings": ["<|end_of_text|>", "<|eot_id|>"],
        "ras_win_len": config.get("ras_win_len", 7),
        "ras_win_max_num_repeat": config.get("ras_win_max_num_repeat", 2),
    }

    seed = config.get("seed")
    if seed and seed > 0:
        generate_kwargs["seed"] = seed

    return _serve_engine.generate(**generate_kwargs)


class EngineBridge:
    """Async wrapper around the synchronous HiggsAudioServeEngine.

    Automatically unloads the model after ENGINE_IDLE_TIMEOUT seconds
    of inactivity, and reloads it on the next generation request.
    """

    def __init__(self):
        self.ready = False
        self._last_used: float = 0.0
        self._idle_task: Optional[asyncio.Task] = None

    async def initialize(self):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(_executor, _init_engine_sync)
        self.ready = True
        self._last_used = time.time()
        self._start_idle_watcher()
        logger.info("EngineBridge ready.")

    async def _ensure_loaded(self):
        """Reload the engine if it was unloaded due to idle timeout."""
        if not self.ready:
            async with _init_lock:
                if not self.ready:
                    logger.info("Engine needed - loading...")
                    await self.initialize()

    async def ensure_ready(self):
        await self._ensure_loaded()

    def _start_idle_watcher(self):
        if self._idle_task and not self._idle_task.done():
            return
        self._idle_task = asyncio.create_task(self._idle_watcher())

    async def _idle_watcher(self):
        """Background loop that unloads the engine after idle timeout."""
        while True:
            await asyncio.sleep(60)  # check every minute
            if not self.ready:
                continue
            idle = time.time() - self._last_used
            if idle >= ENGINE_IDLE_TIMEOUT:
                logger.info("Engine idle for %d minutes, unloading...", int(idle // 60))
                loop = asyncio.get_running_loop()
                async with _lock:
                    await loop.run_in_executor(_executor, _unload_engine_sync)
                self.ready = False
                logger.info("Engine unloaded. Will reload on next request.")

    async def generate(self, messages, config: dict):
        """Run a single generation, serialized by an async lock."""
        await self._ensure_loaded()
        async with _lock:
            self._last_used = time.time()
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(_executor, _generate_sync, messages, config)

    async def generate_safe(self, messages, config: dict, retries: int = 2):
        """Generate with CUDA OOM retry logic."""
        for attempt in range(retries + 1):
            try:
                return await self.generate(messages, config)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if "out of memory" in str(exc).lower() and attempt < retries:
                    logger.warning("CUDA OOM on attempt %d, clearing cache...", attempt + 1)
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                raise
