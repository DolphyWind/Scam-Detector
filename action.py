import datetime
import logging
import time
from abc import ABC
from typing import Any, List, Optional, TYPE_CHECKING, override
import bisect

import discord

if TYPE_CHECKING:
    from scam_detector import ScamDetector

logger = logging.getLogger(__name__)


def _parse_duration(raw: str) -> Optional[int]:
    remaining = raw.strip().lower()
    if not remaining:
        return None
    total = 0
    while remaining:
        buf = ""
        while remaining and remaining[0].isdigit():
            buf += remaining[0]
            remaining = remaining[1:]
        if not buf:
            return None
        n = int(buf)
        if remaining and remaining[0] == "d":
            total += n * 1440
            remaining = remaining[1:]
        elif remaining and remaining[0] == "h":
            total += n * 60
            remaining = remaining[1:]
        elif remaining and remaining[0] == "m":
            total += n
            remaining = remaining[1:]
        elif not remaining:
            total += n
        else:
            return None
    return total


def _format_duration(mins: int) -> str:
    parts: List[str] = []
    d = mins // 1440
    if d:
        parts.append(f"{d}d")
        mins %= 1440
    h = mins // 60
    if h:
        parts.append(f"{h}h")
        mins %= 60
    if mins or not parts:
        parts.append(f"{mins}m")
    return " ".join(parts)


def _extract_user_id(raw: str) -> Optional[int]:
    stripped = raw.strip()
    for prefix, suffix in [("<@!", ">"), ("<@&", ">"), ("<@", ">")]:
        if stripped.startswith(prefix) and stripped.endswith(suffix):
            stripped = stripped[len(prefix):-len(suffix)]
            break
    try:
        return int(stripped)
    except (ValueError, TypeError):
        return None


def _extract_channel_id(raw: str) -> Optional[int]:
    stripped = raw.strip()
    if stripped.startswith("<#") and stripped.endswith(">"):
        stripped = stripped[2:-1]

    try:
        return int(stripped)
    except (ValueError, TypeError):
        return None


class Action(ABC):
    def __init__(self, param: Optional[Any] = None) -> None:
        super().__init__()
        self.param: Optional[Any] = param
        self.priority: int = 0
        self.is_singleton: bool = True
        self.id: int = 0
        self.cooldown_seconds: int = 0
        self._next_allowed: float = 0

    def __lt__(self, other: "Action") -> bool:
        return self.priority < other.priority

    # Take an action and return message(s). ActionList will concatenate these and send it as a single message.
    async def act(
        self,
        bot: "ScamDetector",
        message: discord.Message,
    ) -> Optional[str]:
        ...


class BanAction(Action):
    def __init__(self, param: Optional[Any] = None) -> None:
        super().__init__()
        self.priority: int = 60
        self.is_singleton: bool = True
        self.cooldown_seconds = 10

    @override
    async def act(
        self,
        bot: "ScamDetector",
        message: discord.Message,
    ) -> Optional[str]:
        logger.info(f"Banning {message.author}.")
        if not message.guild.me.guild_permissions.ban_members:
            logger.warning(f"Missing ban_members permission in guild {message.guild}")
            return "I don't have permission to ban members."

        await message.author.ban(reason="Scam detected", delete_message_days=7)


class KickAction(Action):
    def __init__(self, param: Optional[Any] = None) -> None:
        super().__init__()
        self.priority: int = 50
        self.is_singleton: bool = True
        self.cooldown_seconds = 10

    @override
    async def act(
        self,
        bot: "ScamDetector",
        message: discord.Message,
    ) -> Optional[str]:
        logger.info(f"Kicking {message.author}.")
        if not message.guild.me.guild_permissions.kick_members:
            logger.warning(f"Missing kick_members permission in guild {message.guild}")
            return "I don't have permission to kick members."

        await message.author.kick(reason="Scam detected")


# param = duration in minutes (int, default 60)
class TimeoutAction(Action):
    def __init__(self, param: Optional[Any] = None) -> None:
        parsed: Optional[int] = _parse_duration(param) if param is not None else None
        super().__init__(parsed)
        self.priority: int = 40
        self.is_singleton: bool = True
        self.cooldown_seconds = 10

    @override
    async def act(
        self,
        bot: "ScamDetector",
        message: discord.Message,
    ) -> Optional[str]:
        duration = datetime.timedelta(minutes=self.param if self.param is not None else 60)
        logger.info(f"Timing out {message.author} for {duration}.")
        if not message.guild.me.guild_permissions.moderate_members:
            logger.warning(f"Missing moderate_members permission in guild {message.guild}")
            return "I don't have permission to timeout members."

        await message.author.timeout(duration, reason="Scam detected")


# param = user ID or role ID to ping (int)
class PingAction(Action):
    def __init__(self, param: Optional[Any] = None) -> None:
        parsed: Optional[int] = _extract_user_id(param) if param is not None else None
        super().__init__(parsed)
        self.priority: int = 10
        self.is_singleton: bool = False
        self.cooldown_seconds = 10

    @override
    async def act(
        self,
        bot: "ScamDetector",
        message: discord.Message,
    ) -> Optional[str]:
        if self.param is None:
            return "No target configured for ping."
        target_id: int = self.param

        mention: str
        assert message.guild is not None
        if (role := message.guild.get_role(target_id)):
            mention = role.mention
        elif member := message.guild.get_member(target_id):
            mention = member.mention
        else:
            mention = f"<@{target_id}>"

        logger.info(f"Pinging {target_id} for message {message.id} from {message.author}.")
        return mention


# param = archive channel ID (int)
class ArchiveAction(Action):
    def __init__(self, param: Optional[Any] = None) -> None:
        parsed: Optional[int] = _extract_channel_id(param) if param is not None else None
        super().__init__(parsed)
        self.priority: int = 20
        self.is_singleton: bool = False
        self.cooldown_seconds = 10

    @override
    async def act(
        self,
        bot: "ScamDetector",
        message: discord.Message,
    ) -> Optional[str]:
        logger.info(f"Archiving message {message.id} from {message.author}.")
        assert type(self.param) is int, "Channel id must be an integer"
        channel_id: int = self.param
        channel = bot.get_channel(channel_id)

        if channel is None:
            return f"Archive channel not found: {channel_id}"

        perms = channel.permissions_for(message.guild.me)
        if not perms.send_messages:
            logger.warning(f"Missing send_messages permission in archive channel {channel}")
            return "I don't have permission to send messages in the archive channel."

        await channel.send(f"**Archived message from {message.author.mention} (ID: {message.author.id})**\n{message.content}")
        for attachment in message.attachments:
            await channel.send(attachment.url)


class DeleteAction(Action):
    def __init__(self, param: Optional[Any] = None) -> None:
        super().__init__()
        self.priority: int = 30
        self.is_singleton: bool = True

    @override
    async def act(
        self,
        bot: "ScamDetector",
        message: discord.Message,
    ) -> Optional[str]:
        logger.info(f"Deleting message {message.id} from {message.author}.")
        if not message.channel.permissions_for(message.guild.me).manage_messages:
            logger.warning(f"Missing manage_messages permission in channel {message.channel}")
            return "I don't have permission to delete messages."

        await message.delete()


class ActionList():
    def __init__(self) -> None:
        self.action_queue: List[Action] = []
        self._next_id: int = 1
        self._next_allowed_message: float = 0
        self._message_cooldown_seconds: float = 10

    def add_action(self, action: Action) -> Optional[str]:
        for item in self.action_queue:
            if not item.is_singleton:
                continue
            if type(item) is type(action):
                return "Action already exists!"
        action.id = self._next_id
        self._next_id += 1
        bisect.insort(self.action_queue, action)

    def remove_action(self, action_id: int) -> bool:
        for i, a in enumerate(self.action_queue):
            if a.id == action_id:
                self.action_queue.pop(i)
                return True
        return False

    async def take_actions(
        self,
        bot: "ScamDetector",
        message: discord.Message,
    ) -> Optional[str]:
        msgs: List[str] = []
        now = time.time()
        for action in self.action_queue:
            if now < action._next_allowed:
                continue
            try:
                if msg := await action.act(bot, message):
                    msgs.append(msg)
            except Exception as e:
                msgs.append(f"An error occured when performing {action.__class__.__name__}: {e}")
            if action.cooldown_seconds:
                action._next_allowed = now + action.cooldown_seconds

        if msgs and now > self._next_allowed_message:
            self._next_allowed_message = now + self._message_cooldown_seconds
            return "\n".join(msgs)
