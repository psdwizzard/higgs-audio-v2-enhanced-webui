"""Print the Application ID extracted from the bot token in .env"""
import base64
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("DISCORD_BOT_TOKEN", "")
if not token:
    sys.exit(1)
try:
    first_part = token.split(".")[0]
    # Add padding
    padded = first_part + "=" * (-len(first_part) % 4)
    app_id = base64.b64decode(padded).decode()
    print(app_id)
except Exception:
    sys.exit(1)
