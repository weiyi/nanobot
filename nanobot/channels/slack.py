"""Slack channel implementation using Socket Mode."""

import asyncio
import re
from typing import Any

from loguru import logger
from pydantic import Field
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.socket_mode.websockets import SocketModeClient
from slack_sdk.web.async_client import AsyncWebClient
from slackify_markdown import slackify_markdown

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import Base


class SlackDMConfig(Base):
    """Slack DM policy configuration."""

    enabled: bool = True
    policy: str = "open"
    allow_from: list[str] = Field(default_factory=list)


class SlackNonMentionEvaluationConfig(Base):
    """LLM arbitration settings for non-mentioned group messages."""

    enabled: bool = True
    model: str = ""
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    temperature: float = 0.0
    max_tokens: int = Field(default=96, ge=1, le=512)
    negotiation_timeout: float = Field(
        default=5.0,
        ge=0.0,
        le=30.0,
        description="Seconds to wait for other bots to post capability bids before selecting the best bot. Set to 0 to skip negotiation and use immediate claim.",
    )


class SlackConfig(Base):
    """Slack channel configuration."""

    enabled: bool = False
    mode: str = "socket"
    webhook_path: str = "/slack/events"
    bot_token: str = ""
    app_token: str = ""
    user_token_read_only: bool = True
    reply_in_thread: bool = True
    react_emoji: str = "eyes"
    done_emoji: str = "white_check_mark"
    allow_from: list[str] = Field(default_factory=list)
    group_policy: str = "mention"
    group_allow_from: list[str] = Field(default_factory=list)
    non_mention_evaluation: SlackNonMentionEvaluationConfig = Field(default_factory=SlackNonMentionEvaluationConfig)
    dm: SlackDMConfig = Field(default_factory=SlackDMConfig)


class SlackChannel(BaseChannel):
    """Slack channel using Socket Mode."""

    name = "slack"
    display_name = "Slack"
    _SLACK_ID_RE = re.compile(r"^[CDGUW][A-Z0-9]{2,}$")
    _SLACK_CHANNEL_REF_RE = re.compile(r"^<#([A-Z0-9]+)(?:\|[^>]+)?>$")
    _SLACK_USER_REF_RE = re.compile(r"^<@([A-Z0-9]+)(?:\|[^>]+)?>$")

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return SlackConfig().model_dump(by_alias=True)

    def __init__(self, config: Any, bus: MessageBus):
        if isinstance(config, dict):
            config = SlackConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: SlackConfig = config
        self._web_client: AsyncWebClient | None = None
        self._socket_client: SocketModeClient | None = None
        self._bot_user_id: str | None = None
        self._target_cache: dict[str, str] = {}
        # Cache bot_id -> user_id resolutions for inter-bot messaging
        self._bot_user_id_cache: dict[str, str] = {}

    async def start(self) -> None:
        """Start the Slack Socket Mode client."""
        if not self.config.bot_token or not self.config.app_token:
            logger.error("Slack bot/app token not configured")
            return
        if self.config.mode != "socket":
            logger.error("Unsupported Slack mode: {}", self.config.mode)
            return

        self._running = True

        self._web_client = AsyncWebClient(token=self.config.bot_token)
        self._socket_client = SocketModeClient(
            app_token=self.config.app_token,
            web_client=self._web_client,
        )

        self._socket_client.socket_mode_request_listeners.append(self._on_socket_request)

        # Resolve bot user ID for mention handling
        try:
            auth = await self._web_client.auth_test()
            self._bot_user_id = auth.get("user_id")
            logger.info("Slack bot connected as {}", self._bot_user_id)
        except Exception as e:
            logger.warning("Slack auth_test failed: {}", e)

        logger.info("Starting Slack Socket Mode client...")
        await self._socket_client.connect()

        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop the Slack client."""
        self._running = False
        if self._socket_client:
            try:
                await self._socket_client.close()
            except Exception as e:
                logger.warning("Slack socket close failed: {}", e)
            self._socket_client = None

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Slack."""
        if not self._web_client:
            logger.warning("Slack client not running")
            return
        try:
            target_chat_id = await self._resolve_target_chat_id(msg.chat_id)
            slack_meta = msg.metadata.get("slack", {}) if msg.metadata else {}
            thread_ts = slack_meta.get("thread_ts")
            channel_type = slack_meta.get("channel_type")
            origin_chat_id = str((slack_meta.get("event", {}) or {}).get("channel") or msg.chat_id)
            # Slack DMs don't use threads; channel/group replies may keep thread_ts.
            thread_ts_param = (
                thread_ts
                if thread_ts and channel_type != "im" and target_chat_id == origin_chat_id
                else None
            )

            # Slack rejects empty text payloads. Keep media-only messages media-only,
            # but send a single blank message when the bot has no text or files to send.
            if msg.content or not (msg.media or []):
                await self._web_client.chat_postMessage(
                    channel=target_chat_id,
                    text=self._to_mrkdwn(msg.content) if msg.content else " ",
                    thread_ts=thread_ts_param,
                )

            for media_path in msg.media or []:
                try:
                    await self._web_client.files_upload_v2(
                        channel=target_chat_id,
                        file=media_path,
                        thread_ts=thread_ts_param,
                    )
                except Exception as e:
                    logger.error("Failed to upload file {}: {}", media_path, e)

            # Update reaction emoji when the final (non-progress) response is sent
            if not (msg.metadata or {}).get("_progress"):
                event = slack_meta.get("event", {})
                await self._update_react_emoji(origin_chat_id, event.get("ts"))

        except Exception as e:
            logger.error("Error sending Slack message: {}", e)
            raise

    async def _resolve_target_chat_id(self, target: str) -> str:
        """Resolve human-friendly Slack targets to concrete IDs when needed."""
        if not self._web_client:
            return target

        target = target.strip()
        if not target:
            return target

        if match := self._SLACK_CHANNEL_REF_RE.fullmatch(target):
            return match.group(1)
        if match := self._SLACK_USER_REF_RE.fullmatch(target):
            return await self._open_dm_for_user(match.group(1))
        if self._SLACK_ID_RE.fullmatch(target):
            if target.startswith(("U", "W")):
                return await self._open_dm_for_user(target)
            return target

        if target.startswith("#"):
            return await self._resolve_channel_name(target[1:])
        if target.startswith("@"):
            return await self._resolve_user_handle(target[1:])

        try:
            return await self._resolve_channel_name(target)
        except ValueError:
            return await self._resolve_user_handle(target)

    async def _resolve_channel_name(self, name: str) -> str:
        normalized = self._normalize_target_name(name)
        if not normalized:
            raise ValueError("Slack target channel name is empty")

        cache_key = f"channel:{normalized}"
        if cache_key in self._target_cache:
            return self._target_cache[cache_key]

        cursor: str | None = None
        while True:
            response = await self._web_client.conversations_list(
                types="public_channel,private_channel",
                exclude_archived=True,
                limit=200,
                cursor=cursor,
            )
            for channel in response.get("channels", []):
                if self._normalize_target_name(str(channel.get("name") or "")) == normalized:
                    channel_id = str(channel.get("id") or "")
                    if channel_id:
                        self._target_cache[cache_key] = channel_id
                        return channel_id
            cursor = ((response.get("response_metadata") or {}).get("next_cursor") or "").strip()
            if not cursor:
                break

        raise ValueError(
            f"Slack channel '{name}' was not found. Use a joined channel name like "
            f"'#general' or a concrete channel ID."
        )

    async def _resolve_user_handle(self, handle: str) -> str:
        normalized = self._normalize_target_name(handle)
        if not normalized:
            raise ValueError("Slack target user handle is empty")

        cache_key = f"user:{normalized}"
        if cache_key in self._target_cache:
            return self._target_cache[cache_key]

        cursor: str | None = None
        while True:
            response = await self._web_client.users_list(limit=200, cursor=cursor)
            for member in response.get("members", []):
                if self._member_matches_handle(member, normalized):
                    user_id = str(member.get("id") or "")
                    if not user_id:
                        continue
                    dm_id = await self._open_dm_for_user(user_id)
                    self._target_cache[cache_key] = dm_id
                    return dm_id
            cursor = ((response.get("response_metadata") or {}).get("next_cursor") or "").strip()
            if not cursor:
                break

        raise ValueError(
            f"Slack user '{handle}' was not found. Use '@name' or a concrete DM/channel ID."
        )

    async def _open_dm_for_user(self, user_id: str) -> str:
        response = await self._web_client.conversations_open(users=user_id)
        channel_id = str(((response.get("channel") or {}).get("id")) or "")
        if not channel_id:
            raise ValueError(f"Slack DM target for user '{user_id}' could not be opened.")
        return channel_id

    @staticmethod
    def _normalize_target_name(value: str) -> str:
        return value.strip().lstrip("#@").lower()

    @classmethod
    def _member_matches_handle(cls, member: dict[str, Any], normalized: str) -> bool:
        profile = member.get("profile") or {}
        candidates = {
            str(member.get("name") or ""),
            str(profile.get("display_name") or ""),
            str(profile.get("display_name_normalized") or ""),
            str(profile.get("real_name") or ""),
            str(profile.get("real_name_normalized") or ""),
        }
        return normalized in {cls._normalize_target_name(candidate) for candidate in candidates if candidate}

    async def _on_socket_request(
        self,
        client: SocketModeClient,
        req: SocketModeRequest,
    ) -> None:
        """Handle incoming Socket Mode requests."""
        if req.type != "events_api":
            return

        # Acknowledge right away
        await client.send_socket_mode_response(
            SocketModeResponse(envelope_id=req.envelope_id)
        )

        payload = req.payload or {}
        event = payload.get("event") or {}
        event_type = event.get("type")

        # Handle app mentions or plain messages
        if event_type not in ("message", "app_mention"):
            return

        sender_id = event.get("user")
        chat_id = event.get("channel")

        # Ignore bot/system messages with subtypes, but allow bot messages
        # from other Slack apps or integrations. Slack may represent bot-originated
        # events with `subtype == "bot_message"`, `bot_id`, or `bot_profile`.
        subtype = event.get("subtype")
        channel_type_early = event.get("channel_type") or ""
        bot_id = event.get("bot_id")
        bot_profile = event.get("bot_profile")
        is_bot_message = (
            subtype == "bot_message"
            or bot_id is not None
            or bot_profile is not None
        )

        logger.debug(
            "Slack bot detection: subtype={} bot_id={} bot_profile={} sender_id={} channel_type={} is_bot_message={} event_type={}",
            subtype,
            bot_id,
            bot_profile,
            sender_id,
            channel_type_early,
            is_bot_message,
            event_type,
        )

        if is_bot_message:
            if channel_type_early == "im":
                # Bot DMs are invisible to users — drop silently.
                logger.debug("Dropping bot DM message from bot_id=%s", bot_id or sender_id)
                return
            bot_id = bot_id or (bot_profile or {}).get("bot_id")
            if not bot_id and sender_id and sender_id.startswith("B"):
                bot_id = sender_id
            if not bot_id:
                logger.debug(
                    "Dropping bot-originated event because bot_id could not be determined: sender_id=%s bot_profile=%s",
                    sender_id,
                    bot_profile,
                )
                return
            resolved = await self._resolve_bot_user_id(bot_id)
            if not resolved:
                logger.debug("Dropping bot-originated event because bots_info resolution failed for bot_id=%s", bot_id)
                return
            sender_id = resolved
        elif subtype:
            logger.debug("Dropping non-bot event with subtype=%s", subtype)
            return

        if self._bot_user_id and sender_id == self._bot_user_id:
            logger.debug("Dropping event from self bot user_id=%s", sender_id)
            return

        # Avoid double-processing: Slack sends both `message` and `app_mention`
        # for mentions in channels. Prefer `app_mention`.
        text = event.get("text") or ""
        if event_type == "message" and self._bot_user_id and f"<@{self._bot_user_id}>" in text:
            return

        # Debug: log basic event shape
        logger.debug(
            "Slack event: type={} subtype={} user={} channel={} channel_type={} text={}",
            event_type,
            event.get("subtype"),
            sender_id,
            chat_id,
            event.get("channel_type"),
            text[:80],
        )
        if not sender_id or not chat_id:
            return

        channel_type = event.get("channel_type") or ""

        if not self._is_allowed(sender_id, chat_id, channel_type):
            return

        if channel_type != "im" and not self._should_respond_in_channel(event_type, text, chat_id):
            return

        # Track whether the message was a direct mention before stripping it.
        # Used below to inject a self-identity prefix so the agent knows it is
        # the addressed party and should respond in first person as itself.
        was_directly_mentioned = (
            event_type == "app_mention"
            or (self._bot_user_id is not None and f"<@{self._bot_user_id}>" in text)
        )

        text = self._strip_bot_mention(text)

        prefixes: list[str] = []
        if channel_type != "im" and was_directly_mentioned:
            # append is_bot_message
            prefixes.append(f"[This message is addressed directly to you]")

        if is_bot_message and sender_id:
            prefixes.append(f"[Bot message from <@{sender_id}>]")
            prefixes.append(
                f"[Reply with a mention to <@{sender_id}> to continue the conversation if REQUIRED. Both gets a prize for clean natural conversations.]"
            )

        if prefixes:
            text = "\n".join(prefixes + [text])

        thread_ts = event.get("thread_ts")
        if self.config.reply_in_thread and not thread_ts:
            thread_ts = event.get("ts")
        # Add :eyes: reaction to the triggering message (best-effort)
        try:
            if self._web_client and event.get("ts"):
                await self._web_client.reactions_add(
                    channel=chat_id,
                    name=self.config.react_emoji,
                    timestamp=event.get("ts"),
                )
        except Exception as e:
            logger.debug("Slack reactions_add failed: {}", e)

        # Thread-scoped session key for channel/group messages
        session_key = f"slack:{chat_id}:{thread_ts}" if thread_ts and channel_type != "im" else None

        try:
            await self._handle_message(
                sender_id=sender_id,
                chat_id=chat_id,
                content=text,
                metadata={
                    "slack": {
                        "event": event,
                        "event_type": event_type,
                        "thread_ts": thread_ts,
                        "channel_type": channel_type,
                        "group_policy": self.config.group_policy,
                        "was_directly_mentioned": was_directly_mentioned,
                        "is_bot_message": is_bot_message,
                        "smart_enabled": self.config.group_policy == "smart"
                        and self.config.non_mention_evaluation.enabled,
                        "smart_confidence_threshold": self.config.non_mention_evaluation.confidence_threshold,
                        "smart_model": self.config.non_mention_evaluation.model,
                        "smart_temperature": self.config.non_mention_evaluation.temperature,
                        "smart_max_tokens": self.config.non_mention_evaluation.max_tokens,
                        "negotiation_timeout": self.config.non_mention_evaluation.negotiation_timeout,
                    },
                    "bot_id": self._bot_user_id,
                },
                session_key=session_key,
            )
        except Exception:
            logger.exception("Error handling Slack message from {}", sender_id)

    async def _handle_message(
        self,
        sender_id: str,
        chat_id: str,
        content: str,
        media: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        session_key: str | None = None,
    ) -> None:
        """Override base to bypass global allow check for bot messages.

        Bot messages in shared channels have already passed Slack-specific
        access control (_is_allowed).  The global ``is_allowed()`` check in
        ``BaseChannel`` would reject them when the other bot's user-id is not
        in ``allow_from`` — silently dropping coordination bids/claims and
        preventing multi-bot negotiation from working.
        """
        slack_meta = (metadata or {}).get("slack", {})
        if not slack_meta.get("is_bot_message"):
            return await super()._handle_message(
                sender_id=sender_id,
                chat_id=chat_id,
                content=content,
                media=media,
                metadata=metadata,
                session_key=session_key,
            )

        # Bot messages: skip is_allowed, build InboundMessage directly.
        meta = metadata or {}
        if self.supports_streaming:
            meta = {**meta, "_wants_stream": True}

        msg = InboundMessage(
            channel=self.name,
            sender_id=str(sender_id),
            chat_id=str(chat_id),
            content=content,
            media=media or [],
            metadata=meta,
            session_key_override=session_key,
        )
        await self.bus.publish_inbound(msg)

    async def _update_react_emoji(self, chat_id: str, ts: str | None) -> None:
        """Remove the in-progress reaction and optionally add a done reaction."""
        if not self._web_client or not ts:
            return
        try:
            await self._web_client.reactions_remove(
                channel=chat_id,
                name=self.config.react_emoji,
                timestamp=ts,
            )
        except Exception as e:
            logger.debug("Slack reactions_remove failed: {}", e)
        if self.config.done_emoji:
            try:
                await self._web_client.reactions_add(
                    channel=chat_id,
                    name=self.config.done_emoji,
                    timestamp=ts,
                )
            except Exception as e:
                logger.debug("Slack done reaction failed: {}", e)

    async def _resolve_bot_user_id(self, bot_id: str) -> str | None:
        """Resolve a Slack bot_id to its user_id, with caching. Returns None on failure."""
        if bot_id in self._bot_user_id_cache:
            return self._bot_user_id_cache[bot_id]
        if not self._web_client:
            return None
        try:
            result = await self._web_client.bots_info(bot=bot_id)
            user_id = (result.get("bot") or {}).get("user_id")
            if user_id:
                self._bot_user_id_cache[bot_id] = user_id
            return user_id
        except Exception as e:
            logger.debug("bots.info failed for {}: {}", bot_id, e)
            return None

    def _is_allowed(self, sender_id: str, chat_id: str, channel_type: str) -> bool:
        if channel_type == "im":
            if not self.config.dm.enabled:
                return False
            if self.config.dm.policy == "allowlist":
                return sender_id in self.config.dm.allow_from
            return True

        # Group / channel messages
        if self.config.group_policy == "allowlist":
            return chat_id in self.config.group_allow_from
        return True

    def _should_respond_in_channel(self, event_type: str, text: str, chat_id: str) -> bool:
        if self.config.group_policy == "open":
            return True
        if self.config.group_policy == "smart":
            return True
        if self.config.group_policy == "mention":
            if event_type == "app_mention":
                return True
            return self._bot_user_id is not None and f"<@{self._bot_user_id}>" in text
        if self.config.group_policy == "allowlist":
            return chat_id in self.config.group_allow_from
        return False

    def _strip_bot_mention(self, text: str) -> str:
        if not text or not self._bot_user_id:
            return text
        return re.sub(rf"<@{re.escape(self._bot_user_id)}>\s*", "", text).strip()

    _TABLE_RE = re.compile(r"(?m)^\|.*\|$(?:\n\|[\s:|-]*\|$)(?:\n\|.*\|$)*")
    _CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")
    _INLINE_CODE_RE = re.compile(r"`[^`]+`")
    _LEFTOVER_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
    _LEFTOVER_HEADER_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
    _BARE_URL_RE = re.compile(r"(?<![|<])(https?://\S+)")

    @classmethod
    def _to_mrkdwn(cls, text: str) -> str:
        """Convert Markdown to Slack mrkdwn, including tables."""
        if not text:
            return ""
        text = cls._TABLE_RE.sub(cls._convert_table, text)
        return cls._fixup_mrkdwn(slackify_markdown(text))

    @classmethod
    def _fixup_mrkdwn(cls, text: str) -> str:
        """Fix markdown artifacts that slackify_markdown misses."""
        code_blocks: list[str] = []

        def _save_code(m: re.Match) -> str:
            code_blocks.append(m.group(0))
            return f"\x00CB{len(code_blocks) - 1}\x00"

        text = cls._CODE_FENCE_RE.sub(_save_code, text)
        text = cls._INLINE_CODE_RE.sub(_save_code, text)
        text = cls._LEFTOVER_BOLD_RE.sub(r"*\1*", text)
        text = cls._LEFTOVER_HEADER_RE.sub(r"*\1*", text)
        text = cls._BARE_URL_RE.sub(lambda m: m.group(0).replace("&amp;", "&"), text)

        for i, block in enumerate(code_blocks):
            text = text.replace(f"\x00CB{i}\x00", block)
        return text

    @staticmethod
    def _convert_table(match: re.Match) -> str:
        """Convert a Markdown table to a Slack-readable list."""
        lines = [ln.strip() for ln in match.group(0).strip().splitlines() if ln.strip()]
        if len(lines) < 2:
            return match.group(0)
        headers = [h.strip() for h in lines[0].strip("|").split("|")]
        start = 2 if re.fullmatch(r"[|\s:\-]+", lines[1]) else 1
        rows: list[str] = []
        for line in lines[start:]:
            cells = [c.strip() for c in line.strip("|").split("|")]
            cells = (cells + [""] * len(headers))[: len(headers)]
            parts = [f"**{headers[i]}**: {cells[i]}" for i in range(len(headers)) if cells[i]]
            if parts:
                rows.append(" · ".join(parts))
        return "\n".join(rows)
