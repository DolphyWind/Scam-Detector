from dataclasses import dataclass


@dataclass
class BotConfig:
    owner_id: int
    model_name: str
    index_path: str
    db_path: str
    match_threshold: float
