import logging

import discord

from scam_detector import ScamDetector

import os
from bot_config import BotConfig
from pathlib import Path


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    intents = discord.Intents.default()
    intents.message_content = True
    token: str = os.environ["SCAM_DETECTOR_BOT_TOKEN"]
    owner_id: int = int(os.environ["OWNER_ID"])
    model_name: str = os.environ["IMAGE_EMBEDDING_MODEL_NAME"]
    index_filename: str = os.environ["INDEX_FILENAME"]
    db_filename: str = os.environ["DB_FILENAME"]
    match_threshold: float = float(os.environ["MATCH_THRESHOLD"])

    bot_config: BotConfig = BotConfig(
        owner_id=owner_id,
        model_name=model_name,
        index_path=index_filename,
        db_path=db_filename,
        match_threshold=match_threshold,
    )

    bot = ScamDetector(
        command_prefix="!",
        intents=intents,
        bot_config=bot_config,
    )
    bot.run(token)


if __name__ == "__main__":
    main()
