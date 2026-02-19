import logging

import discord
from discord import app_commands
from discord.ext import commands

from discord_bot.utils.permissions import is_admin_interaction

logger = logging.getLogger(__name__)


class AdminCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @app_commands.command(name="queue", description="Show current job queue status")
    async def queue(self, interaction: discord.Interaction):
        lines = self.bot.job_queue.summary_lines()
        embed = discord.Embed(title="Job Queue", color=discord.Color.orange())
        embed.description = "\n".join(lines)
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="cancel", description="Cancel a running or queued job (Admin only)")
    async def cancel(self, interaction: discord.Interaction, job_id: str):
        if not is_admin_interaction(interaction):
            await interaction.response.send_message(
                "This command is restricted to admins.", ephemeral=True
            )
            return

        success = self.bot.job_queue.cancel(job_id)
        if success:
            await interaction.response.send_message(f"Job `{job_id}` has been cancelled.")
        else:
            await interaction.response.send_message(
                f"Job `{job_id}` not found or already finished.", ephemeral=True
            )

    @app_commands.command(name="status", description="Show bot and engine status")
    async def status(self, interaction: discord.Interaction):
        embed = discord.Embed(title="Bot Status", color=discord.Color.teal())
        embed.add_field(
            name="Engine",
            value="Ready" if self.bot.engine.ready else "Sleeping (will reload on next request)",
            inline=True,
        )
        current = self.bot.job_queue.current_job
        embed.add_field(
            name="Current Job",
            value=f"`{current.id}` ({current.progress_current}/{current.progress_total})" if current else "Idle",
            inline=True,
        )
        embed.add_field(
            name="Pending",
            value=str(len(self.bot.job_queue.pending_jobs)),
            inline=True,
        )
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="check_channel", description="Check bot permissions for a channel (Admin only)")
    @app_commands.describe(channel="Channel to check (defaults to current channel)")
    async def check_channel(
        self, interaction: discord.Interaction, channel: discord.TextChannel | None = None
    ):
        if not is_admin_interaction(interaction):
            await interaction.response.send_message(
                "This command is restricted to admins.", ephemeral=True
            )
            return

        if channel is None:
            channel = interaction.channel if isinstance(interaction.channel, discord.TextChannel) else None
        if channel is None or interaction.guild is None:
            await interaction.response.send_message(
                "Please run this in a guild text channel or specify a channel.",
                ephemeral=True,
            )
            return

        me = interaction.guild.get_member(self.bot.user.id) if self.bot.user else None
        perms = channel.permissions_for(me) if me else None
        if perms is None:
            await interaction.response.send_message(
                "Could not resolve bot member in this guild.", ephemeral=True
            )
            return

        embed = discord.Embed(
            title="Channel Permission Check",
            color=discord.Color.blurple(),
            description=f"{channel.mention} (`{channel.id}`)",
        )
        embed.add_field(name="View Channel", value=str(perms.view_channel), inline=True)
        embed.add_field(name="Send Messages", value=str(perms.send_messages), inline=True)
        embed.add_field(name="Attach Files", value=str(perms.attach_files), inline=True)
        embed.add_field(name="Read History", value=str(perms.read_message_history), inline=True)
        embed.add_field(name="Send in Threads", value=str(perms.send_messages_in_threads), inline=True)
        await interaction.response.send_message(embed=embed, ephemeral=True)


async def setup(bot):
    await bot.add_cog(AdminCommands(bot))
