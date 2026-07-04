from io import BytesIO

import discord
from discord import app_commands
from discord.ext import commands
from bot_config import BotConfig
from faiss import IndexFlatIP
import faiss
from pathlib import Path
from transformers import pipeline
import numpy as np
import numpy.typing as npt
from PIL import Image
from enum import Enum
from typing import List
import asyncio


# I do not recommend using BAN action automatically since models are prone to error
class Action(Enum):
    BAN = "Ban"
    TIMEOUT = "Timeout"
    PING = "Ping"
    DEBUG = "Message"


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

        data: bytes = await image.read()
        img: Image.Image = Image.open(BytesIO(data))
        embedding: npt.NDArray = await self.bot.embed([img])
        self.bot.image_index.add(embedding)


class ScamDetector(commands.Bot):
    def __init__(
        self,
        command_prefix: str,
        intents: discord.Intents,
        bot_config: BotConfig,
    ) -> None:
        super().__init__(command_prefix=command_prefix, intents=intents)
        self.command_tree = app_commands.CommandTree(self)

        self.owner: int = bot_config.owner_id
        self.model_name: str = bot_config.model_name
        self.index_path: Path = Path(bot_config.index_path)
        self.db_path: Path = Path(bot_config.db_path)
        self.match_threshold: float = bot_config.match_threshold

        self.pipe = pipeline(
            task="image-feature-extraction",
            model=self.model_name,
        )

        if self.index_path.exists():
            self.image_index: IndexFlatIP = faiss.read_index(str(self.index_path))
        else:
            self.image_index: IndexFlatIP = IndexFlatL2(d=self.pipe.model.config.hidden_size)

    async def on_message(
        self,
        message: discord.Message,
    ) -> None:
        if message.author.bot:
            return

        image_attachments: List[discord.Attachment] = []
        for attachment in message.attachments:
            if attachment.content_type is not None and attachment.content_type.startswith("image/"):
                image_attachments.append(attachment)

        if not image_attachments:
            return

        image_data: List[bytes] = await asyncio.gather(*[im.read for im in image_attachments])
        images: List[Image.Image] = [Image.open(BytesIO(data)) for data in image_data]

        queries: npt.NDArray = await self.embed(images)
        scores, _ = self.image_index.search(queries, k=5)
        scores = [s[0] for s in scores if s[0] > self.match_threshold]

        if scores.__len__() > 0:
            await message.channel.send("Detected!")

    async def embed(self, images: List[Image.Image]) -> npt.NDArray:
        embeds = self.pipe(
            images,
            return_tensor="np",
        )

        return faiss.normalize_L2(embeds)

    async def setup_hook(self) -> None:
        await self.add_cog(AppCommandsCog(self))
        await self.command_tree.sync()

    async def close(self) -> None:
        faiss.write_index(self.image_index, str(self.index_path))
        await super().close()
