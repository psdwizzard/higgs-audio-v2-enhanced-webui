import discord
from discord_bot.config import ADMIN_USERS


def is_admin(user: discord.Member | discord.User) -> bool:
    """Check if a user has admin access.

    Returns True if the user is in the ADMIN_USERS list from .env
    OR has the Administrator permission on the current server.
    """
    # Explicit user ID list from .env
    if user.id in ADMIN_USERS:
        return True
    # Discord server Administrator permission
    if isinstance(user, discord.Member) and user.guild_permissions.administrator:
        return True
    return False


def is_admin_interaction(interaction: discord.Interaction) -> bool:
    """Convenience wrapper for slash command interactions."""
    return is_admin(interaction.user)
