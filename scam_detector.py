import asyncio
from io import BytesIO
import logging
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
import discord
from discord import app_commands
from discord.ext import commands
import faiss
from faiss import IndexFlatIP
import numpy as np
import numpy.typing as npt
from transformers import pipeline

from action import (
    Action,
    ActionList,
    BanAction,
    KickAction,
    TimeoutAction,
    PingAction,
    ArchiveAction,
    DeleteAction,
    _format_duration,
)
from bot_config import BotConfig

logger = logging.getLogger(__name__)


class AppCommandsCog(commands.Cog):
    def __init__(
        self,
        bot: "ScamDetector",
    ) -> None:
        super().__init__()
        self.bot: "ScamDetector" = bot

    @app_commands.command(description="Register a new image. Requires you to own the bot!")
    async def register(
        self,
        interaction: discord.Interaction,
        image: discord.Attachment,
    ) -> None:
        if image.content_type is None or not image.content_type.startswith("image/"):
            await interaction.response.send_message(
                "Please upload an image.",
                ephemeral=True,
            )
            return

        if interaction.user.id != self.bot.owner_id:
            await interaction.response.send_message(
                "You need to be the owner of this bot to use this command!",
                ephemeral=True,
            )
            return

        logger.info(f"Registering image {image.filename} from user {interaction.user.id}")
        data: bytes = await image.read()
        img: Image.Image = Image.open(BytesIO(data))
        embedding: npt.NDArray = await self.bot.embed([img])
        self.bot.image_index.add(embedding)
        await interaction.response.send_message(
            "Successfully registered the image",
            ephemeral=True,
        )

    actions = app_commands.Group(name="actions", description="Configure actions for scam detection.")

    async def _check_mod(
        self,
        interaction: discord.Interaction,
    ) -> bool:
        if interaction.guild_id is None:
            await interaction.response.send_message(
                "This command must be used in a server.",
                ephemeral=True,
            )
            return False
        if not interaction.user.guild_permissions.manage_guild:
            await interaction.response.send_message(
                "You need to be a moderator of this server to use this command!",
                ephemeral=True,
            )
            return False
        return True

    @actions.command(
        name="add",
        description="Add an action. Requires server mod.",
    )
    @app_commands.describe(
        action_type="The type of action to add",
        param="Optional parameter as a number (minutes for timeout, ID for ping/archive)",
    )
    @app_commands.choices(action_type=[
        app_commands.Choice(name="ban", value="ban"),
        app_commands.Choice(name="kick", value="kick"),
        app_commands.Choice(name="timeout", value="timeout"),
        app_commands.Choice(name="ping", value="ping"),
        app_commands.Choice(name="archive", value="archive"),
        app_commands.Choice(name="delete", value="delete"),
    ])
    async def actions_add(
        self,
        interaction: discord.Interaction,
        action_type: str,
        param: Optional[str] = None,
    ) -> None:
        if not await self._check_mod(interaction):
            return

        action: Action | None = None
        match action_type:
            case "ban":
                action = BanAction(param)
            case "kick":
                action = KickAction(param)
            case "timeout":
                action = TimeoutAction(param)
            case "ping":
                action = PingAction(param)
            case "archive":
                action = ArchiveAction(param)
            case "delete":
                action = DeleteAction(param)

        if action is None:
            await interaction.response.send_message(
                f"Unknown action type: {action_type}",
                ephemeral=True,
            )
            return

        guild_actions = self.bot.actions_map.setdefault(interaction.guild_id, ActionList())
        result = guild_actions.add_action(action)
        if result:
            await interaction.response.send_message(result, ephemeral=True)
            return

        await self.bot.save_actions()
        if type(action) is BanAction:
            desc = "Ban the message author"
        elif type(action) is KickAction:
            desc = "Kick the message author"
        elif type(action) is TimeoutAction:
            mins = action.param if action.param is not None else 60
            desc = f"Timeout the message author for {_format_duration(mins)}"
        elif type(action) is PingAction:
            if action.param is not None and interaction.guild:
                target: str = ""
                if role := interaction.guild.get_role(action.param):
                    target = role.mention
                elif member := interaction.guild.get_member(action.param):
                    target = member.mention
                else:
                    target = f"ID={action.param}"
                desc = f"Ping {target}"
            else:
                desc = "Ping (no target)"
        elif type(action) is ArchiveAction:
            channel = self.bot.get_channel(action.param) if action.param is not None else None
            ch = f"<#{channel.id}>" if channel else f"ID={action.param}"
            desc = f"Archive to {ch}"
        elif type(action) is DeleteAction:
            desc = "Delete the message"
        else:
            desc = action_type

        await interaction.response.send_message(
            f"Added **{desc}** (ID={action.id}).",
            ephemeral=True,
        )

    @actions.command(name="remove", description="Remove an action by ID. Requires server mod.")
    @app_commands.describe(action_id="The ID of the action to remove (use /actions list)")
    async def actions_remove(
        self,
        interaction: discord.Interaction,
        action_id: int,
    ) -> None:
        if not await self._check_mod(interaction):
            return

        guild_actions = self.bot.actions_map.get(interaction.guild_id)
        if guild_actions is None or not guild_actions.remove_action(action_id):
            await interaction.response.send_message(
                f"No action with ID `{action_id}`.",
                ephemeral=True,
            )
            return

        await self.bot.save_actions()
        await interaction.response.send_message(
            f"Removed action ID=`{action_id}`.",
            ephemeral=True,
        )

    @actions.command(name="list", description="List configured actions. Requires server mod.")
    async def actions_list(
        self,
        interaction: discord.Interaction,
    ) -> None:
        if not await self._check_mod(interaction):
            return

        guild_actions = self.bot.actions_map.get(interaction.guild_id)
        if guild_actions is None or not guild_actions.action_queue:
            await interaction.response.send_message(
                "No actions configured for this server.",
                ephemeral=True,
            )
            return

        def fmt(a: Action) -> str:
            if type(a) is BanAction:
                return f"ID=`{a.id}` | Ban"
            if type(a) is KickAction:
                return f"ID=`{a.id}` | Kick"
            if type(a) is TimeoutAction:
                mins = a.param if a.param is not None else 60
                return f"ID=`{a.id}` | Timeout ({_format_duration(mins)})"
            if type(a) is PingAction:
                target = ""
                if a.param is not None and interaction.guild:
                    if role := interaction.guild.get_role(a.param):
                        target = f" {role.mention}"
                    elif member := interaction.guild.get_member(a.param):
                        target = f" {member.mention}"
                    else:
                        target = f" (ID={a.param})"
                return f"ID=`{a.id}` | Ping {target}"
            if type(a) is ArchiveAction:
                channel = self.bot.get_channel(a.param) if a.param is not None else None
                ch = f" <#{channel.id}>" if channel else f" (ID={a.param})"
                return f"ID=`{a.id}` | Archive {ch}"
            if type(a) is DeleteAction:
                return f"ID=`{a.id}` | Delete"
            return f"ID=`{a.id}` | {a.__class__.__name__}"
        lines = [fmt(a) for a in guild_actions.action_queue]
        await interaction.response.send_message(
            "Configured actions:\n" + "\n".join(lines),
            ephemeral=True,
        )

    @actions.command(name="clear", description="Clear all actions for this server. Requires server mod.")
    async def actions_clear(
        self,
        interaction: discord.Interaction,
    ) -> None:
        if not await self._check_mod(interaction):
            return

        self.bot.actions_map.pop(interaction.guild_id, None)
        await self.bot.save_actions()
        await interaction.response.send_message(
            "Cleared all actions for this server.",
            ephemeral=True,
        )


class ScamDetector(commands.Bot):
    def __init__(
        self,
        command_prefix: str,
        intents: discord.Intents,
        bot_config: BotConfig,
    ) -> None:
        super().__init__(command_prefix=command_prefix, intents=intents)

        self.owner_id: int = bot_config.owner_id
        self.model_name: str = bot_config.model_name
        self.index_path: Path = Path(bot_config.index_path)
        self.db_path: Path = Path(bot_config.db_path)
        self.match_threshold: float = bot_config.match_threshold

        logger.info(f"Loading model {self.model_name}")
        self.pipe = pipeline(
            task="image-feature-extraction",
            model=self.model_name,
        )

        self.actions_map: Dict[int, ActionList] = {}
        self.image_index: IndexFlatIP

    async def on_message(
        self,
        message: discord.Message,
    ) -> None:
        if message.author.bot:
            return
        if message.guild is None:
            return

        image_attachments: List[discord.Attachment] = []
        for attachment in message.attachments:
            if attachment.content_type is not None and attachment.content_type.startswith("image/"):
                image_attachments.append(attachment)

        if not image_attachments:
            return

        logger.info(f"Processing {len(image_attachments)} image(s) from message {message.id}")
        image_data: List[bytes] = await asyncio.gather(*[im.read() for im in image_attachments])
        images: List[Image.Image] = [Image.open(BytesIO(data)) for data in image_data]

        queries: npt.NDArray = await self.embed(images)
        raw_scores, _ = self.image_index.search(queries, k=5)
        raw_scores = [s[0] for s in raw_scores]
        scores = [s for s in raw_scores if s > self.match_threshold]

        if scores:
            logger.warning(f"Match detected (scores={[f'{s:.4f}' for s in scores]}) in message {message.id}")
            actions_list: Optional[ActionList] = self.actions_map.get(message.guild.id, None)
            if actions_list:
                msg: Optional[str] = await actions_list.take_actions(self, message)
                if msg:
                    await message.channel.send(msg)
        else:
            logger.info(f"No matches found in message {message.id} (scores={[f'{s:.4f}' for s in raw_scores]})")

    async def embed(self, images: List[Image.Image]) -> npt.NDArray:
        raw = np.asarray(self.pipe(images))
        embeds = np.ascontiguousarray(raw[:, 0, 0, :], dtype=np.float32)
        faiss.normalize_L2(embeds)
        return embeds

    async def load_index(self) -> None:
        if self.index_path.exists():
            logger.info(f"Loading existing index from {self.index_path}")
            self.image_index: IndexFlatIP = faiss.read_index(str(self.index_path))
        else:
            logger.info(f"Creating new index with dimension {self.pipe.model.config.hidden_size}")
            self.image_index: IndexFlatIP = IndexFlatIP(self.pipe.model.config.hidden_size)

    async def save_index(self) -> None:
        faiss.write_index(self.image_index, str(self.index_path))

    async def load_actions(self) -> None:
        pass

    async def save_actions(self) -> None:
        pass

    async def setup_hook(self) -> None:
        logger.info(f"Bot logged in as {self.user}")
        await self.load_index()
        await self.load_actions()
        await self.add_cog(AppCommandsCog(self))
        await self.tree.sync()

    async def close(self) -> None:
        logger.info(f"Saving index to {self.index_path}")
        await self.save_actions()
        await self.save_index()
        await super().close()
