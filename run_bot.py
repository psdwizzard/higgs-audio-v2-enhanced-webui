"""Entry point for the Higgs Audio Discord Bot.

Usage:
    python run_bot.py
"""
import logging
import sys

from discord_bot.config import DISCORD_BOT_TOKEN
from discord_bot.bot import HiggsBot


def main():
    if not DISCORD_BOT_TOKEN:
        print("ERROR: DISCORD_BOT_TOKEN not set. Copy .env.example to .env and fill in your token.")
        sys.exit(1)

    token = DISCORD_BOT_TOKEN.strip()
    if len(token) < 50 or "." not in token:
        print("ERROR: DISCORD_BOT_TOKEN looks invalid (too short or missing dots).")
        print(f"       Token length: {len(token)} chars, starts with: {token[:8]}...")
        print()
        print("Make sure you're using the BOT token, not the Client Secret.")
        print("Go to: discord.com/developers/applications > your app > Bot > Reset Token")
        sys.exit(1)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("bot.log", encoding="utf-8"),
        ],
    )
    # Reduce noise from discord.py internals
    logging.getLogger("discord").setLevel(logging.WARNING)
    logging.getLogger("discord.http").setLevel(logging.WARNING)

    bot = HiggsBot()
    bot.run(token, log_handler=None)


if __name__ == "__main__":
    main()
