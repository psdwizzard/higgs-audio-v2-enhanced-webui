class BotError(Exception):
    """Base error for bot operations."""


class EngineNotReady(BotError):
    """Engine has not finished loading."""


class GenerationFailed(BotError):
    """Audio generation failed."""


class GoogleDocDownloadFailed(BotError):
    """Could not download the Google Doc."""


class JobCancelled(BotError):
    """The job was cancelled by a user."""


class ChapterTooLarge(BotError):
    """A chapter's MP3 exceeds Discord's upload limit even after splitting."""


class RateLimitExceeded(BotError):
    """User already has the max number of audiobooks queued."""
