"""Dropbox HTTP API integration with automatic OAuth2 token refresh."""

import json as _json
import logging
import time
from typing import Optional

import aiohttp

from discord_bot.config import DROPBOX_APP_KEY, DROPBOX_APP_SECRET, DROPBOX_REFRESH_TOKEN

logger = logging.getLogger(__name__)

_API_BASE = "https://api.dropboxapi.com/2"
_CONTENT_BASE = "https://content.dropboxapi.com/2"
_TOKEN_URL = "https://api.dropboxapi.com/oauth2/token"

# In-memory token cache
_access_token: Optional[str] = None
_token_expires_at: float = 0


def is_configured() -> bool:
    """Return True if Dropbox OAuth2 credentials are fully set."""
    return bool(DROPBOX_APP_KEY and DROPBOX_APP_SECRET and DROPBOX_REFRESH_TOKEN)


async def _get_access_token() -> str:
    """Return a valid access token, refreshing if expired or about to expire."""
    global _access_token, _token_expires_at

    # Refresh if no token or expiring within 5 minutes
    if _access_token and time.time() < (_token_expires_at - 300):
        return _access_token

    logger.info("Refreshing Dropbox access token...")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            _TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": DROPBOX_REFRESH_TOKEN,
                "client_id": DROPBOX_APP_KEY,
                "client_secret": DROPBOX_APP_SECRET,
            },
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                logger.error("Dropbox token refresh failed (%d): %s", resp.status, body)
                raise RuntimeError(f"Dropbox token refresh failed: {body}")
            data = await resp.json()
            _access_token = data["access_token"]
            _token_expires_at = time.time() + data.get("expires_in", 14400)
            logger.info("Dropbox access token refreshed (expires in %ds)", data.get("expires_in", 14400))
            return _access_token


async def _auth_headers() -> dict:
    token = await _get_access_token()
    return {"Authorization": f"Bearer {token}"}


async def create_folder(dropbox_path: str) -> str:
    """Create a folder in Dropbox. Returns 'created', 'exists', or 'error'."""
    url = f"{_API_BASE}/files/create_folder_v2"
    payload = {"path": dropbox_path, "autorename": False}
    headers = {**(await _auth_headers()), "Content-Type": "application/json"}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status == 200:
                logger.info("Created Dropbox folder: %s", dropbox_path)
                return "created"
            body = await resp.text()
            # 409 with path/conflict means folder already exists
            if resp.status == 409 and "path" in body and "conflict" in body:
                logger.info("Dropbox folder already exists: %s", dropbox_path)
                return "exists"
            logger.error("Dropbox create_folder failed (%d): %s", resp.status, body)
            return "error"


async def upload_file(file_bytes: bytes, dropbox_path: str) -> bool:
    """Upload bytes to a Dropbox path. Returns True on success."""
    url = f"{_CONTENT_BASE}/files/upload"
    dropbox_arg = _json.dumps({
        "path": dropbox_path,
        "mode": "overwrite",
        "autorename": True,
        "mute": False,
    })
    headers = {
        **(await _auth_headers()),
        "Content-Type": "application/octet-stream",
        "Dropbox-API-Arg": dropbox_arg,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=file_bytes, headers=headers) as resp:
            if resp.status == 200:
                logger.info("Uploaded to Dropbox: %s (%.1f MB)", dropbox_path, len(file_bytes) / (1024 * 1024))
                return True
            body = await resp.text()
            logger.error("Dropbox upload failed (%d): %s", resp.status, body)
            return False


async def create_shared_link(dropbox_path: str) -> Optional[str]:
    """Create or retrieve a shared link for a file or folder. Returns the URL or None."""
    url = f"{_API_BASE}/sharing/create_shared_link_with_settings"
    payload = {"path": dropbox_path, "settings": {"requested_visibility": "public"}}
    headers = {**(await _auth_headers()), "Content-Type": "application/json"}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status == 200:
                data = await resp.json()
                link = data.get("url", "")
                logger.info("Created shared link for %s: %s", dropbox_path, link)
                return link

            body = await resp.text()
            # If a shared link already exists, retrieve it
            if resp.status == 409 and "shared_link_already_exists" in body:
                return await _get_existing_shared_link(session, dropbox_path)

            logger.error("Dropbox create_shared_link failed (%d): %s", resp.status, body)
            return None


async def _get_existing_shared_link(session: aiohttp.ClientSession, dropbox_path: str) -> Optional[str]:
    """Retrieve an existing shared link for the given path."""
    url = f"{_API_BASE}/sharing/list_shared_links"
    payload = {"path": dropbox_path, "direct_only": True}
    headers = {**(await _auth_headers()), "Content-Type": "application/json"}

    async with session.post(url, json=payload, headers=headers) as resp:
        if resp.status == 200:
            data = await resp.json()
            links = data.get("links", [])
            if links:
                link = links[0].get("url", "")
                logger.info("Retrieved existing shared link for %s: %s", dropbox_path, link)
                return link
        body = await resp.text()
        logger.error("Dropbox list_shared_links failed (%d): %s", resp.status, body)
        return None
