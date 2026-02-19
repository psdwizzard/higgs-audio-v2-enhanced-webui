import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from discord_bot.config import MAX_AUDIOBOOKS_PER_USER
from discord_bot.utils.errors import RateLimitExceeded

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    QUEUED = "Queued"
    RUNNING = "Running"
    COMPLETE = "Complete"
    CANCELLED = "Cancelled"
    FAILED = "Failed"


@dataclass
class Job:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    user_id: int = 0
    user_display_name: str = ""
    channel_id: int = 0
    voice_name: str = ""
    doc_url: str = ""
    doc_title: str = ""
    text: str = ""
    output_dir: str = ""
    status: JobStatus = JobStatus.QUEUED
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    created_at: float = field(default_factory=time.time)
    progress_current: int = 0
    progress_total: int = 0
    error: Optional[str] = None


class JobQueue:
    """Single-worker async job queue for GPU-bound audiobook generation."""

    def __init__(self):
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._jobs: Dict[str, Job] = {}
        self._worker_task: Optional[asyncio.Task] = None
        self._handler = None  # async callable(job) set by bot

    @property
    def current_job(self) -> Optional[Job]:
        for j in self._jobs.values():
            if j.status == JobStatus.RUNNING:
                return j
        return None

    @property
    def pending_jobs(self) -> List[Job]:
        return [j for j in self._jobs.values() if j.status == JobStatus.QUEUED]

    def user_active_count(self, user_id: int) -> int:
        return sum(
            1 for j in self._jobs.values()
            if j.user_id == user_id and j.status in (JobStatus.QUEUED, JobStatus.RUNNING)
        )

    def enqueue(self, job: Job) -> Job:
        if self.user_active_count(job.user_id) >= MAX_AUDIOBOOKS_PER_USER:
            raise RateLimitExceeded(
                f"You already have {MAX_AUDIOBOOKS_PER_USER} audiobook(s) in the queue. "
                "Please wait for them to finish or cancel them."
            )
        self._jobs[job.id] = job
        self._queue.put_nowait(job)
        logger.info("Job %s enqueued (user %d)", job.id, job.user_id)
        return job

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        if job.status in (JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.CANCELLED):
            return False
        job.cancel_event.set()
        job.status = JobStatus.CANCELLED
        logger.info("Job %s cancelled", job_id)
        return True

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def start(self, handler):
        """Start the worker loop. *handler* is an async callable(Job)."""
        self._handler = handler
        self._worker_task = asyncio.create_task(self._worker())

    async def _worker(self):
        logger.info("Job queue worker started.")
        while True:
            job = await self._queue.get()
            if job.cancel_event.is_set():
                self._queue.task_done()
                continue
            job.status = JobStatus.RUNNING
            try:
                await self._handler(job)
                if job.status == JobStatus.RUNNING:
                    job.status = JobStatus.COMPLETE
            except Exception as exc:
                job.status = JobStatus.FAILED
                job.error = str(exc)
                logger.exception("Job %s failed: %s", job.id, exc)
            finally:
                self._queue.task_done()

    def summary_lines(self) -> List[str]:
        """Human-readable summary for /queue command."""
        lines = []
        current = self.current_job
        if current:
            lines.append(
                f"**Running** `{current.id}` - {current.voice_name} "
                f"({current.progress_current}/{current.progress_total} chunks)"
            )
        pending = self.pending_jobs
        if pending:
            for i, j in enumerate(pending, 1):
                lines.append(f"{i}. `{j.id}` - {j.voice_name} (queued)")
        if not lines:
            lines.append("Queue is empty.")
        return lines
