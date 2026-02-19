import logging
import re
from typing import Optional

import aiohttp

from discord_bot.utils.errors import GoogleDocDownloadFailed

logger = logging.getLogger(__name__)

_DOC_ID_PATTERNS = [
    re.compile(r"/document/d/([a-zA-Z0-9_-]+)"),
    re.compile(r"/file/d/([a-zA-Z0-9_-]+)"),
    re.compile(r"[?&]id=([a-zA-Z0-9_-]+)"),
]


def extract_doc_id(url: str) -> Optional[str]:
    """Pull the document ID out of a Google Docs URL."""
    for pattern in _DOC_ID_PATTERNS:
        m = pattern.search(url)
        if m:
            return m.group(1)
    return None


async def download_google_doc(url: str) -> str:
    """Download a public Google Doc as plain text.

    The document must be shared with "Anyone with the link can view".
    """
    doc_id = extract_doc_id(url)
    if not doc_id:
        raise GoogleDocDownloadFailed(
            "Could not extract a document ID from the URL. "
            "Please provide a link like `https://docs.google.com/document/d/ABC123/edit`."
        )

    export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"

    async with aiohttp.ClientSession() as session:
        async with session.get(export_url) as resp:
            if resp.status == 200:
                text = await resp.text()
                if not text.strip():
                    raise GoogleDocDownloadFailed("The document appears to be empty.")
                logger.info("Downloaded Google Doc %s (%d chars)", doc_id, len(text))
                return text
            elif resp.status == 404:
                raise GoogleDocDownloadFailed(
                    "Document not found. Check that the URL is correct."
                )
            else:
                raise GoogleDocDownloadFailed(
                    f"Failed to download document (HTTP {resp.status}). "
                    "Make sure the document is shared as 'Anyone with the link can view'."
                )
