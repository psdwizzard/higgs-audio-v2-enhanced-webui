"""One-time Dropbox setup: prompts for credentials, writes them to .env, gets refresh token."""

import os
import re
import sys
import webbrowser

import requests
from dotenv import load_dotenv

ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
TOKEN_URL = "https://api.dropboxapi.com/oauth2/token"


def _read_env():
    """Read .env file as text, creating it if it doesn't exist."""
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def _set_env_var(text: str, key: str, value: str) -> str:
    """Set or update a key=value in the .env text."""
    pattern = re.compile(rf"^{re.escape(key)}\s*=.*$", re.MULTILINE)
    new_line = f"{key}={value}"
    if pattern.search(text):
        return pattern.sub(new_line, text)
    # Append with a newline
    if text and not text.endswith("\n"):
        text += "\n"
    return text + new_line + "\n"


def _get_env_var(text: str, key: str) -> str:
    """Read a value from .env text."""
    m = re.search(rf"^{re.escape(key)}\s*=\s*(.*)$", text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def main():
    print()
    print("=" * 60)
    print("  Dropbox Setup for Higgs Audio Bot")
    print("=" * 60)
    print()
    print("  Prerequisites:")
    print("    1. Go to https://www.dropbox.com/developers/apps")
    print("    2. Create app -> Scoped access -> App folder")
    print("    3. Permissions tab -> enable:")
    print("         files.content.write")
    print("         files.content.read")
    print("         sharing.write")
    print("       Then click Submit!")
    print("    4. Settings tab -> copy App key and App secret")
    print()

    env_text = _read_env()

    # --- App Key ---
    existing_key = _get_env_var(env_text, "DROPBOX_APP_KEY")
    if existing_key:
        print(f"  Found existing DROPBOX_APP_KEY: {existing_key[:6]}...")
        use_existing = input("  Keep it? (Y/n): ").strip().lower()
        if use_existing in ("n", "no"):
            existing_key = ""

    if not existing_key:
        existing_key = input("  Enter your Dropbox App Key: ").strip()
        if not existing_key:
            print("  No key entered. Aborting.")
            sys.exit(1)

    app_key = existing_key
    env_text = _set_env_var(env_text, "DROPBOX_APP_KEY", app_key)

    # --- App Secret ---
    existing_secret = _get_env_var(env_text, "DROPBOX_APP_SECRET")
    if existing_secret:
        print(f"  Found existing DROPBOX_APP_SECRET: {existing_secret[:6]}...")
        use_existing = input("  Keep it? (Y/n): ").strip().lower()
        if use_existing in ("n", "no"):
            existing_secret = ""

    if not existing_secret:
        existing_secret = input("  Enter your Dropbox App Secret: ").strip()
        if not existing_secret:
            print("  No secret entered. Aborting.")
            sys.exit(1)

    app_secret = existing_secret
    env_text = _set_env_var(env_text, "DROPBOX_APP_SECRET", app_secret)

    # Save key + secret now
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.write(env_text)
    print()
    print(f"  Saved App Key and Secret to {ENV_PATH}")

    # --- Authorization ---
    auth_url = (
        f"https://www.dropbox.com/oauth2/authorize"
        f"?client_id={app_key}"
        f"&response_type=code"
        f"&token_access_type=offline"
    )

    print()
    print("  Opening your browser to authorize the app...")
    print(f"  (If it doesn't open, go to: {auth_url})")
    print()
    webbrowser.open(auth_url)

    print("  After clicking 'Allow', Dropbox shows you a code.")
    code = input("  Paste the authorization code here: ").strip()
    if not code:
        print("  No code entered. Aborting.")
        sys.exit(1)

    # --- Exchange code for refresh token ---
    print()
    print("  Exchanging code for refresh token...")

    resp = requests.post(
        TOKEN_URL,
        data={
            "code": code,
            "grant_type": "authorization_code",
            "client_id": app_key,
            "client_secret": app_secret,
        },
    )

    if resp.status_code != 200:
        print(f"  ERROR: Dropbox returned {resp.status_code}")
        print(f"  {resp.text}")
        print()
        print("  Common fixes:")
        print("    - Make sure you clicked Submit on the Permissions tab")
        print("    - Make sure the code was copied correctly (single use!)")
        print("    - Try running this setup again for a fresh code")
        sys.exit(1)

    data = resp.json()
    refresh_token = data.get("refresh_token", "")

    if not refresh_token:
        print("  ERROR: No refresh_token in response.")
        print(f"  {data}")
        sys.exit(1)

    # --- Save refresh token to .env ---
    env_text = _read_env()  # Re-read in case it changed
    env_text = _set_env_var(env_text, "DROPBOX_REFRESH_TOKEN", refresh_token)
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.write(env_text)

    print()
    print("=" * 60)
    print("  SUCCESS!")
    print("=" * 60)
    print()
    print(f"  All three values saved to {ENV_PATH}:")
    print(f"    DROPBOX_APP_KEY={app_key}")
    print(f"    DROPBOX_APP_SECRET={app_secret[:6]}...")
    print(f"    DROPBOX_REFRESH_TOKEN={refresh_token[:10]}...")
    print()
    print("  This refresh token never expires.")
    print("  The bot auto-refreshes access tokens as needed.")
    print()
    print("  You can now start the bot!")
    print()


if __name__ == "__main__":
    main()
