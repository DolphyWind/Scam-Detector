import asyncio
import logging
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import List

import discord
import faiss
import numpy as np
import numpy.typing as npt
from discord import app_commands
from discord.ext import commands
from faiss import IndexFlatIP
from PIL import Image
from transformers import pipeline

from bot_config import BotConfig

logger = logging.getLogger(__name__)


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

        if self.index_path.exists():
            logger.info(f"Loading existing index from {self.index_path}")
            self.image_index: IndexFlatIP = faiss.read_index(str(self.index_path))
        else:
            logger.info(f"Creating new index with dimension {self.pipe.model.config.hidden_size}")
            self.image_index: IndexFlatIP = IndexFlatIP(self.pipe.model.config.hidden_size)

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

        logger.info(f"Processing {len(image_attachments)} image(s) from message {message.id}")
        image_data: List[bytes] = await asyncio.gather(*[im.read() for im in image_attachments])
        images: List[Image.Image] = [Image.open(BytesIO(data)) for data in image_data]

        queries: npt.NDArray = await self.embed(images)
        raw_scores, _ = self.image_index.search(queries, k=5)
        raw_scores = [s[0] for s in raw_scores]
        scores = [s for s in raw_scores if s > self.match_threshold]

        if scores:
            logger.warning(f"Match detected (scores={[f'{s:.4f}' for s in scores]}) in message {message.id}")
            await message.channel.send("Detected!")
        else:
            logger.info(f"No matches found in message {message.id} (scores={[f'{s:.4f}' for s in raw_scores]})")

    async def embed(self, images: List[Image.Image]) -> npt.NDArray:
        raw = np.asarray(self.pipe(images))
        embeds = np.ascontiguousarray(raw[:, 0, 0, :], dtype=np.float32)
        faiss.normalize_L2(embeds)
        return embeds

    async def setup_hook(self) -> None:
        logger.info(f"Bot logged in as {self.user}")
        await self.add_cog(AppCommandsCog(self))
        await self.tree.sync()

    async def close(self) -> None:
        logger.info(f"Saving index to {self.index_path}")
        faiss.write_index(self.image_index, str(self.index_path))
        await super().close()
