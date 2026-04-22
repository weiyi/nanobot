"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import random
import re
import time
from contextlib import AsyncExitStack, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.autocompact import AutoCompact
from nanobot.agent.context import ContextBuilder
from nanobot.agent.hook import AgentHook, AgentHookContext, CompositeHook
from nanobot.agent.memory import Consolidator, Dream
from nanobot.agent.runner import _MAX_INJECTIONS_PER_TURN, AgentRunner, AgentRunSpec
from nanobot.agent.skills import BUILTIN_SKILLS_DIR
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.notebook import NotebookEditTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.search import GlobTool, GrepTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.self import MyTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.command import CommandContext, CommandRouter, register_builtin_commands
from nanobot.config.schema import AgentDefaults
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager
from nanobot.utils.document import extract_documents
from nanobot.utils.helpers import image_placeholder_text
from nanobot.utils.helpers import truncate_text as truncate_text_fn
from nanobot.utils.runtime import EMPTY_FINAL_RESPONSE_MESSAGE

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig, ToolsConfig, WebToolsConfig
    from nanobot.cron.service import CronService


UNIFIED_SESSION_KEY = "unified:default"


class _LoopHook(AgentHook):
    """Core hook for the main loop."""

    def __init__(
        self,
        agent_loop: AgentLoop,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        *,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
    ) -> None:
        super().__init__(reraise=True)
        self._loop = agent_loop
        self._on_progress = on_progress
        self._on_stream = on_stream
        self._on_stream_end = on_stream_end
        self._channel = channel
        self._chat_id = chat_id
        self._message_id = message_id
        self._stream_buf = ""

    def wants_streaming(self) -> bool:
        return self._on_stream is not None

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        from nanobot.utils.helpers import strip_think

        prev_clean = strip_think(self._stream_buf)
        self._stream_buf += delta
        new_clean = strip_think(self._stream_buf)
        incremental = new_clean[len(prev_clean) :]
        if incremental and self._on_stream:
            await self._on_stream(incremental)

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        if self._on_stream_end:
            await self._on_stream_end(resuming=resuming)
        self._stream_buf = ""

    async def before_iteration(self, context: AgentHookContext) -> None:
        self._loop._current_iteration = context.iteration

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        if self._on_progress:
            if not self._on_stream:
                thought = self._loop._strip_think(
                    context.response.content if context.response else None
                )
                if thought:
                    await self._on_progress(thought)
            tool_hint = self._loop._strip_think(self._loop._tool_hint(context.tool_calls))
            await self._on_progress(tool_hint, tool_hint=True)
        for tc in context.tool_calls:
            args_str = json.dumps(tc.arguments, ensure_ascii=False)
            logger.info("Tool call: {}({})", tc.name, args_str[:200])
        self._loop._set_tool_context(self._channel, self._chat_id, self._message_id)

    async def after_iteration(self, context: AgentHookContext) -> None:
        u = context.usage or {}
        logger.debug(
            "LLM usage: prompt={} completion={} cached={}",
            u.get("prompt_tokens", 0),
            u.get("completion_tokens", 0),
            u.get("cached_tokens", 0),
        )

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        return self._loop._strip_think(content)


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _RUNTIME_CHECKPOINT_KEY = "runtime_checkpoint"
    _PENDING_USER_TURN_KEY = "pending_user_turn"

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int | None = None,
        context_window_tokens: int | None = None,
        context_block_limit: int | None = None,
        max_tool_result_chars: int | None = None,
        provider_retry_mode: str = "standard",
        web_config: WebToolsConfig | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        timezone: str | None = None,
        session_ttl_minutes: int = 0,
        hooks: list[AgentHook] | None = None,
        unified_session: bool = False,
        disabled_skills: list[str] | None = None,
        tools_config: ToolsConfig | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig, ToolsConfig, WebToolsConfig

        _tc = tools_config or ToolsConfig()
        defaults = AgentDefaults()
        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = (
            max_iterations if max_iterations is not None else defaults.max_tool_iterations
        )
        self.context_window_tokens = (
            context_window_tokens
            if context_window_tokens is not None
            else defaults.context_window_tokens
        )
        self.context_block_limit = context_block_limit
        self.max_tool_result_chars = (
            max_tool_result_chars
            if max_tool_result_chars is not None
            else defaults.max_tool_result_chars
        )
        self.provider_retry_mode = provider_retry_mode
        self.web_config = web_config or WebToolsConfig()
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self._start_time = time.time()
        self._last_usage: dict[str, int] = {}
        self._extra_hooks: list[AgentHook] = hooks or []

        self.context = ContextBuilder(workspace, timezone=timezone, disabled_skills=disabled_skills)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.runner = AgentRunner(provider)
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            web_config=self.web_config,
            max_tool_result_chars=self.max_tool_result_chars,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
            disabled_skills=disabled_skills,
        )
        self._unified_session = unified_session
        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stacks: dict[str, AsyncExitStack] = {}
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._background_tasks: list[asyncio.Task] = []
        self._session_locks: dict[str, asyncio.Lock] = {}
        # Per-session pending queues for mid-turn message injection.
        # When a session has an active task, new messages for that session
        # are routed here instead of creating a new task.
        self._pending_queues: dict[str, asyncio.Queue] = {}
        # NANOBOT_MAX_CONCURRENT_REQUESTS: <=0 means unlimited; default 3.
        _max = int(os.environ.get("NANOBOT_MAX_CONCURRENT_REQUESTS", "3"))
        self._concurrency_gate: asyncio.Semaphore | None = (
            asyncio.Semaphore(_max) if _max > 0 else None
        )
        self.consolidator = Consolidator(
            store=self.context.memory,
            provider=provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=self.context_window_tokens,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
            max_completion_tokens=provider.generation.max_tokens,
        )
        self.auto_compact = AutoCompact(
            sessions=self.sessions,
            consolidator=self.consolidator,
            session_ttl_minutes=session_ttl_minutes,
        )
        self.dream = Dream(
            store=self.context.memory,
            provider=provider,
            model=self.model,
        )
        self._register_default_tools()
        if _tc.my.enable:
            self.tools.register(MyTool(loop=self, modify_allowed=_tc.my.allow_set))
        self._runtime_vars: dict[str, Any] = {}
        self._current_iteration: int = 0
        self.commands = CommandRouter()
        register_builtin_commands(self.commands)

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = (
            self.workspace if (self.restrict_to_workspace or self.exec_config.sandbox) else None
        )
        extra_read = [BUILTIN_SKILLS_DIR] if allowed_dir else None
        self.tools.register(
            ReadFileTool(
                workspace=self.workspace, allowed_dir=allowed_dir, extra_allowed_dirs=extra_read
            )
        )
        for cls in (WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        for cls in (GlobTool, GrepTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(NotebookEditTool(workspace=self.workspace, allowed_dir=allowed_dir))
        if self.exec_config.enable:
            self.tools.register(
                ExecTool(
                    working_dir=str(self.workspace),
                    timeout=self.exec_config.timeout,
                    restrict_to_workspace=self.restrict_to_workspace,
                    sandbox=self.exec_config.sandbox,
                    path_append=self.exec_config.path_append,
                    allowed_env_keys=self.exec_config.allowed_env_keys,
                )
            )
        if self.web_config.enable:
            self.tools.register(
                WebSearchTool(config=self.web_config.search, proxy=self.web_config.proxy)
            )
            self.tools.register(WebFetchTool(proxy=self.web_config.proxy))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(
                CronTool(self.cron_service, default_timezone=self.context.timezone or "UTC")
            )

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers

        try:
            self._mcp_stacks = await connect_mcp_servers(self._mcp_servers, self.tools)
            if self._mcp_stacks:
                self._mcp_connected = True
            else:
                logger.warning("No MCP servers connected successfully (will retry next message)")
        except asyncio.CancelledError:
            logger.warning("MCP connection cancelled (will retry next message)")
            self._mcp_stacks.clear()
        except BaseException as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            self._mcp_stacks.clear()
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        # Compute the effective session key (accounts for unified sessions)
        # so that subagent results route to the correct pending queue.
        effective_key = UNIFIED_SESSION_KEY if self._unified_session else f"{channel}:{chat_id}"
        for name in ("message", "spawn", "cron", "my"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    if name == "spawn":
                        tool.set_context(channel, chat_id, effective_key=effective_key)
                    else:
                        tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        from nanobot.utils.helpers import strip_think

        return strip_think(text) or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hints with smart abbreviation."""
        from nanobot.utils.tool_hints import format_tool_hints

        return format_tool_hints(tool_calls)

    @staticmethod
    def _stringify_gate_content(content: Any) -> str:
        """Flatten persisted message content into plain text for routing decisions."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text)
                elif item is not None:
                    parts.append(str(item))
            return "\n".join(parts)
        if content is None:
            return ""
        return str(content)

    def _needs_reply_gate(self, msg: InboundMessage) -> bool:
        """True when this inbound message should be LLM-arbitrated before replying."""
        if msg.channel != "slack":
            return False
        slack_meta = (msg.metadata or {}).get("slack", {})
        if not isinstance(slack_meta, dict):
            return False
        if (slack_meta.get("channel_type") or "") == "im":
            return False
        if bool(slack_meta.get("was_directly_mentioned")):
            return False
        return bool(slack_meta.get("smart_enabled")) or slack_meta.get("group_policy") == "smart"

    @staticmethod
    def _looks_like_action_request(text: str | None) -> bool:
        """Heuristic: imperative, ask-style, or open-question messages should bias toward replying."""
        cleaned = " ".join((text or "").strip().split())
        if not cleaned:
            return False

        lowered = cleaned.lower()
        prefixes = (
            "check ", "please check ", "look ", "look up ", "please look ",
            "find ", "search ", "fetch ", "get ", "pull ", "review ",
            "verify ", "open ", "show ", "tell me ", "summarize ",
            "list ", "compare ", "inspect ", "investigate ", "debug ",
            "fix ", "update ", "run ", "read ", "write ", "send ",
            "can you ", "could you ", "would you ", "will you ", "please ",
        )
        if lowered.startswith(prefixes):
            return True

        if re.match(r"^(?:hey\s+|hi\s+)?(?:bot\s+)?(?:can|could|would|will)\s+you\b", lowered):
            return True

        # Open questions: what/how/why/where/when/who followed by a word (not just "what?" alone).
        if re.match(r"^(?:hey\s+|hi\s+)?(?:what|how|why|where|when|who|which)\s+\w", lowered):
            return True

        # Questions ending with '?' that contain question words anywhere.
        if cleaned.endswith("?") and re.search(r"\b(?:what|how|why|where|when|who|which)\b", lowered):
            return True

        return False

    @classmethod
    def _parse_reply_gate_decision(cls, content: str | None) -> tuple[bool, float, str]:
        """Parse a reply-gate decision from JSON or a conservative yes/no fallback."""
        text = (cls._strip_think(content) or "").strip()
        if not text:
            return False, 0.0, "empty arbiter response"

        candidates = [text]
        if text.startswith("```"):
            stripped = text.strip("`")
            if "\n" in stripped:
                candidates.append(stripped.split("\n", 1)[1].rsplit("\n", 1)[0].strip())
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(text[start:end + 1])

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except Exception:
                continue
            if isinstance(payload, dict):
                flag = payload.get("should_reply")
                if isinstance(flag, bool):
                    confidence_raw = payload.get("confidence", 1.0 if flag else 0.0)
                    try:
                        confidence = float(confidence_raw)
                    except (TypeError, ValueError):
                        confidence = 1.0 if flag else 0.0
                    reason = str(payload.get("reason") or "").strip()
                    return flag, max(0.0, min(confidence, 1.0)), reason

        lowered = text.lower()
        if any(token in lowered for token in ('"should_reply": true', "'should_reply': true", "should_reply=true", "reply=true")):
            return True, 1.0, "fallback true token"
        if any(token in lowered for token in ('"should_reply": false', "'should_reply': false", "should_reply=false", "reply=false")):
            return False, 1.0, "fallback false token"
        if lowered.startswith("yes"):
            return True, 0.75, "fallback yes"
        if lowered.startswith("no"):
            return False, 0.75, "fallback no"
        return False, 0.0, "unparseable arbiter response"

    async def _should_reply_to_inbound(self, msg: InboundMessage, session: Session) -> tuple[bool, str]:
        """Use a lightweight LLM arbitration step for shared-channel non-mentions.

        Returns (approved, reason) where reason is the arbiter's short explanation.
        """
        if not self._needs_reply_gate(msg):
            return True, ""

        slack_meta = (msg.metadata or {}).get("slack", {})
        threshold_raw = slack_meta.get("smart_confidence_threshold", 0.7)
        try:
            confidence_threshold = float(threshold_raw)
        except (TypeError, ValueError):
            confidence_threshold = 0.7

        recent_history = session.get_history(max_messages=6)
        history_lines: list[str] = []
        for item in recent_history[-6:]:
            role = str(item.get("role") or "unknown")
            content = self._stringify_gate_content(item.get("content"))
            content = " ".join((self._strip_think(content) or "").split())
            if not content and item.get("tool_calls"):
                tool_names = [
                    tc.get("function", {}).get("name", "")
                    for tc in item.get("tool_calls", [])
                    if isinstance(tc, dict)
                ]
                content = f"[tool_calls: {', '.join(name for name in tool_names if name)}]"
            if not content:
                continue
            history_lines.append(f"- {role}: {content[:240]}")

        sender_kind = "bot" if slack_meta.get("is_bot_message") else "human"
        arbiter_messages = [
            {
                "role": "system",
                "content": (
                    "You are a reply arbiter for one Slack bot in a shared channel with humans and other bots. "
                    "Decide whether THIS bot should reply to the newest message. "
                    "Mentions always reply=true, but this request is only for non-mentioned messages. "
                    "Reply true if the message clearly needs this bot, continues an active exchange this bot is already in, "
                    "or specifically benefits from this bot's capabilities. "
                    "IMPORTANT: Open questions or requests from a human that are NOT clearly directed at "
                    "another specific bot should return true — at least one bot must pick up the conversation. "
                    "Only return false when the thread history makes it obvious the user is in an ongoing "
                    "conversation with a different bot, or when the message is pure chatter with no question or request. "
                    "NEGOTIATION PROTOCOL: When multiple bots share a channel, bots discuss openly "
                    "to decide who is best suited for the task. Each interested bot posts a capability bid "
                    "(📋 I could help with this — [reason]) and after a short deliberation window the bots "
                    "collectively select the most suitable one. "
                    "If the thread history already contains a coordination claim ('I'll take care of this' or "
                    "'Based on our discussion, I'll take care of this') from another bot for this same request, "
                    "return false — defer to that bot and do not duplicate the response. "
                    "Your 'reason' field should briefly describe THIS bot's relevant capabilities "
                    "for the request — this reason is used as the capability bid in the negotiation. "
                    "Return strict JSON only: "
                    '{"should_reply": true|false, "confidence": 0.0, "reason": "short capability description"}.'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"channel=slack\n"
                    f"chat_id={msg.chat_id}\n"
                    f"sender_id={msg.sender_id}\n"
                    f"sender_kind={sender_kind}\n"
                    f"was_directly_mentioned=false\n\n"
                    f"Recent thread history:\n"
                    f"{chr(10).join(history_lines) if history_lines else '(no recent thread history)'}\n\n"
                    f"Newest message:\n{msg.content}"
                ),
            },
        ]

        try:
            response = await self.provider.chat_with_retry(
                messages=arbiter_messages,
                tools=None,
                model=str(slack_meta.get("smart_model") or self.model),
                max_tokens=int(slack_meta.get("smart_max_tokens") or 96),
                temperature=float(slack_meta.get("smart_temperature", 0.0) or 0.0),
                reasoning_effort="low",
                retry_mode=self.provider_retry_mode,
            )
        except Exception as exc:
            logger.warning(
                "Reply gate failed for {}:{}; defaulting to no-reply. Error: {}",
                msg.channel,
                msg.chat_id,
                exc,
            )
            return False, ""

        should_reply, confidence, reason = self._parse_reply_gate_decision(response.content)
        approved = should_reply and confidence >= confidence_threshold
        action_request = self._looks_like_action_request(msg.content)

        if action_request and not approved:
            low_signal_reason = reason in {"empty arbiter response", "unparseable arbiter response"}
            if low_signal_reason or confidence < confidence_threshold:
                approved = True
                should_reply = True
                confidence = max(confidence, confidence_threshold)
                reason = f"action-request bias ({reason or 'low confidence'})"

        logger.info(
            "Reply gate for {}:{} -> approved={} should_reply={} confidence={:.2f} threshold={:.2f} action_request={} reason={}",
            msg.channel,
            msg.chat_id,
            approved,
            should_reply,
            confidence,
            confidence_threshold,
            action_request,
            reason or "-",
        )
        return approved, reason

    async def _dispatch_command_inline(
        self,
        msg: InboundMessage,
        key: str,
        raw: str,
        dispatch_fn: Callable[[CommandContext], Awaitable[OutboundMessage | None]],
    ) -> None:
        """Dispatch a command directly from the run() loop and publish the result."""
        ctx = CommandContext(msg=msg, session=None, key=key, raw=raw, loop=self)
        result = await dispatch_fn(ctx)
        if result:
            await self.bus.publish_outbound(result)
        else:
            logger.warning("Command '{}' matched but dispatch returned None", raw)

    async def _cancel_active_tasks(self, key: str) -> int:
        """Cancel and await all active tasks and subagents for *key*.

        Returns the total number of cancelled tasks + subagents.
        """
        tasks = self._active_tasks.pop(key, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        sub_cancelled = await self.subagents.cancel_by_session(key)
        return cancelled + sub_cancelled

    def _effective_session_key(self, msg: InboundMessage) -> str:
        """Return the session key used for task routing and mid-turn injections."""
        if self._unified_session and not msg.session_key_override:
            return UNIFIED_SESSION_KEY
        return msg.session_key

    async def _post_coordination_claim(
        self,
        msg: InboundMessage,
        reason: str = "",
    ) -> None:
        """Post a brief public coordination claim to the open channel.

        When multiple bots share a Slack channel, this announces in the thread
        that this bot is taking on the request so other bots can see the claim
        and stand down rather than duplicating the response.
        """
        slack_meta = (msg.metadata or {}).get("slack", {})
        # Build a human-readable claim, omitting internal arbiter noise
        _internal_reasons = {"empty arbiter response", "unparseable arbiter response"}
        is_internal = not reason or any(ir in reason for ir in _internal_reasons)
        clean_reason = "" if is_internal else reason
        content = f"I'll take care of this. _{clean_reason}_" if clean_reason else "I'll take care of this."
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=content,
            metadata={
                "slack": {
                    "thread_ts": slack_meta.get("thread_ts"),
                    "channel_type": slack_meta.get("channel_type"),
                },
                "_coordination_claim": True,
            },
        ))
        logger.info(
            "Posted coordination claim for {}:{}: {}",
            msg.channel,
            msg.chat_id,
            content[:80],
        )

    # ------------------------------------------------------------------
    # Multi-bot negotiation: bots discuss openly who is best suited
    # ------------------------------------------------------------------

    async def _post_coordination_bid(
        self,
        msg: InboundMessage,
        reason: str = "",
    ) -> None:
        """Post a capability bid to the open channel thread.

        Instead of immediately claiming a task, the bot announces its
        capabilities and willingness so other bots can also bid, enabling
        a transparent discussion about who is best suited.
        """
        slack_meta = (msg.metadata or {}).get("slack", {})
        _internal_reasons = {"empty arbiter response", "unparseable arbiter response"}
        is_internal = not reason or any(ir in reason for ir in _internal_reasons)
        clean_reason = "" if is_internal else reason
        content = (
            f"📋 I could help with this — {clean_reason}"
            if clean_reason
            else "📋 I could help with this."
        )
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=content,
            metadata={
                "slack": {
                    "thread_ts": slack_meta.get("thread_ts"),
                    "channel_type": slack_meta.get("channel_type"),
                },
                "_coordination_bid": True,
            },
        ))
        logger.info(
            "Posted coordination bid for {}:{}: {}",
            msg.channel,
            msg.chat_id,
            content[:80],
        )

    @staticmethod
    def _drain_pending_bids(
        pending_queue: asyncio.Queue | None,
    ) -> tuple[list[InboundMessage], bool]:
        """Non-blocking drain of the pending queue for bid/claim messages.

        Returns ``(bids, claim_detected)`` with any coordination messages
        already waiting in the queue.  Non-coordination messages are put
        back so they are not lost.
        """
        if pending_queue is None:
            return [], False

        bids: list[InboundMessage] = []
        non_bid: list[InboundMessage] = []
        claim_detected = False

        while True:
            try:
                queued_msg = pending_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            slack_meta = (queued_msg.metadata or {}).get("slack", {})
            is_bot = bool(slack_meta.get("is_bot_message"))
            content = queued_msg.content or ""
            has_bid_marker = "📋" in content
            has_claim = is_bot and "I'll take care of this" in content

            if has_claim:
                claim_detected = True
                non_bid.append(queued_msg)
            elif is_bot and has_bid_marker:
                bids.append(queued_msg)
            else:
                non_bid.append(queued_msg)

        for preserved in non_bid:
            try:
                pending_queue.put_nowait(preserved)
            except asyncio.QueueFull:
                pass

        return bids, claim_detected

    async def _collect_negotiation_bids(
        self,
        pending_queue: asyncio.Queue | None,
        timeout: float = 5.0,
    ) -> tuple[list[InboundMessage], bool]:
        """Wait for *timeout* seconds, collecting bot bid messages from the pending queue.

        Non-bid messages are preserved and re-queued so they are not lost.
        Returns ``(bids, claim_detected)`` where *bids* is a list of bid
        messages from other bots and *claim_detected* is True when another
        bot already posted a coordination claim during the collection window.
        """
        if pending_queue is None or timeout <= 0:
            return [], False

        bids: list[InboundMessage] = []
        non_bid_messages: list[InboundMessage] = []
        claim_detected = False
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                queued_msg = await asyncio.wait_for(
                    pending_queue.get(), timeout=min(remaining, 0.5)
                )
            except asyncio.TimeoutError:
                continue

            # Identify bot bid messages by checking for the bid marker in content.
            # When a bot posts a bid via _post_coordination_bid, the content starts
            # with the clipboard emoji. Slack channel prepends "[Bot message from ...]"
            # so we check if the bid marker appears anywhere in the content.
            slack_meta = (queued_msg.metadata or {}).get("slack", {})
            is_bot = bool(slack_meta.get("is_bot_message"))
            content = queued_msg.content or ""
            has_bid_marker = "📋" in content
            # Also detect claim messages from other bots so we can defer
            # immediately instead of duplicating the response.
            has_claim = is_bot and "I'll take care of this" in content
            if has_claim:
                claim_detected = True
                logger.info(
                    "Detected coordination claim from {} during bid collection in {}:{}",
                    queued_msg.sender_id,
                    queued_msg.channel,
                    queued_msg.chat_id,
                )
                non_bid_messages.append(queued_msg)
            elif is_bot and has_bid_marker:
                bids.append(queued_msg)
                logger.info(
                    "Collected negotiation bid from {} in {}:{}",
                    queued_msg.sender_id,
                    queued_msg.channel,
                    queued_msg.chat_id,
                )
            else:
                non_bid_messages.append(queued_msg)

        # Re-queue non-bid messages so they are not lost
        for preserved in non_bid_messages:
            try:
                pending_queue.put_nowait(preserved)
            except asyncio.QueueFull:
                logger.warning(
                    "Could not re-queue non-bid message for session, re-publishing to bus"
                )
                await self.bus.publish_inbound(preserved)

        return bids, claim_detected

    async def _select_from_negotiation(
        self,
        msg: InboundMessage,
        my_bid_reason: str,
        other_bids: list[InboundMessage],
        session: Session,
    ) -> tuple[bool, str]:
        """Use LLM to evaluate all bids and decide if this bot should handle the task.

        Returns (should_proceed, selection_reason).
        """
        slack_meta = (msg.metadata or {}).get("slack", {})
        bot_id = msg.metadata.get("bot_id", "this bot")

        # Build bid summary for the LLM
        bid_lines: list[str] = [f"- This bot ({bot_id}): {my_bid_reason or 'general capabilities'}"]
        for bid_msg in other_bids:
            bid_content = (bid_msg.content or "").strip()
            bid_lines.append(f"- Bot {bid_msg.sender_id}: {bid_content[:300]}")

        selection_messages = [
            {
                "role": "system",
                "content": (
                    "You are a coordination selector for a team of Slack bots that share a channel. "
                    "Multiple bots have posted capability bids for a user's request. "
                    "Your job is to evaluate the bids and determine which SINGLE bot should handle the task. "
                    "Consider: (1) relevance of stated capabilities to the request, "
                    "(2) specificity of the bid — a bot that explains WHY it is suited scores higher, "
                    "(3) if bids are roughly equal, prefer the bot that bid first (listed first). "
                    "Return strict JSON only: "
                    '{"select_self": true|false, "reason": "short explanation"}. '
                    "select_self=true means THIS bot (the first in the list) should proceed."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User request: {msg.content}\n\n"
                    f"Capability bids:\n{chr(10).join(bid_lines)}"
                ),
            },
        ]

        try:
            response = await self.provider.chat_with_retry(
                messages=selection_messages,
                tools=None,
                model=str(slack_meta.get("smart_model") or self.model),
                max_tokens=int(slack_meta.get("smart_max_tokens") or 96),
                temperature=0.0,  # Deterministic for consistent selection across bots
                reasoning_effort="low",
                retry_mode=self.provider_retry_mode,
            )
        except Exception as exc:
            logger.warning(
                "Negotiation selection failed for {}:{}; proceeding as fallback. Error: {}",
                msg.channel,
                msg.chat_id,
                exc,
            )
            return True, "selection fallback (LLM error)"

        text = (self._strip_think(response.content) or "").strip()
        # Parse the selection JSON
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                payload = json.loads(text[start:end + 1])
                select_self = bool(payload.get("select_self", True))
                reason = str(payload.get("reason") or "").strip()
                return select_self, reason
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Fallback: if unparseable, proceed (don't block task execution)
        logger.warning(
            "Unparseable negotiation selection for {}:{}: {}",
            msg.channel,
            msg.chat_id,
            text[:120],
        )
        return True, "selection fallback (unparseable)"

    async def _post_coordination_selection(
        self,
        msg: InboundMessage,
        selected: bool,
        reason: str = "",
    ) -> None:
        """Post the negotiation selection result to the open channel thread.

        If selected, posts a claim referencing the team discussion.
        If not selected, posts a brief deferral.
        """
        slack_meta = (msg.metadata or {}).get("slack", {})
        if selected:
            content = (
                f"Based on our discussion, I'll take care of this. _{reason}_"
                if reason
                else "Based on our discussion, I'll take care of this."
            )
            metadata_extra = {"_coordination_claim": True}
        else:
            content = (
                f"Deferring to the better-suited bot. _{reason}_"
                if reason
                else "Deferring to the better-suited bot on this one."
            )
            metadata_extra = {"_coordination_deferral": True}

        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=content,
            metadata={
                "slack": {
                    "thread_ts": slack_meta.get("thread_ts"),
                    "channel_type": slack_meta.get("channel_type"),
                },
                **metadata_extra,
            },
        ))
        logger.info(
            "Posted coordination selection for {}:{}: selected={} {}",
            msg.channel,
            msg.chat_id,
            selected,
            content[:80],
        )

    async def _post_negotiation_progress(
        self,
        msg: InboundMessage,
        content: str,
    ) -> None:
        """Post a visible progress update to the channel thread during negotiation.

        These messages keep users informed about what the bots are doing so the
        negotiation process is transparent rather than a silent wait.
        """
        slack_meta = (msg.metadata or {}).get("slack", {})
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=content,
            metadata={
                "slack": {
                    "thread_ts": slack_meta.get("thread_ts"),
                    "channel_type": slack_meta.get("channel_type"),
                },
                "_coordination_progress": True,
            },
        ))

    async def _run_coordination_negotiation(
        self,
        msg: InboundMessage,
        arbiter_reason: str,
        session: Session,
        pending_queue: asyncio.Queue | None = None,
    ) -> bool:
        """Run the full multi-bot negotiation flow.

        1. Random jitter to desynchronize competing bots.
        2. Check for early bids/claims that arrived while the arbiter was running.
        3. Post a capability bid to the thread.
        4. Post a progress update so the user knows bots are deliberating.
        5. Wait for other bots to post their bids (negotiation_timeout).
        6. Evaluate all bids via LLM to select the best bot.
        7. Post the selection result (claim or deferral).

        All phases post visible messages to the channel thread so the user
        can follow along instead of waiting without feedback.

        Returns True if this bot should proceed with the task, False to defer.

        When negotiation_timeout is 0, falls back to immediate claim (legacy behavior).
        """
        slack_meta = (msg.metadata or {}).get("slack", {})
        timeout_raw = slack_meta.get("negotiation_timeout", 5.0)
        try:
            negotiation_timeout = float(timeout_raw)
        except (TypeError, ValueError):
            negotiation_timeout = 5.0

        # Legacy behavior: immediate claim when negotiation is disabled
        if negotiation_timeout <= 0:
            await self._post_coordination_claim(msg, arbiter_reason)
            return True

        # Phase 0: Random jitter to desynchronize multiple bots.
        # Without jitter both bots post bids at nearly the same instant and
        # each one's bid may not arrive at the other before the collection
        # window closes.  A small random delay ensures one bot posts first,
        # giving the other a clear signal to detect.
        jitter = random.uniform(0, min(negotiation_timeout * 0.3, 1.5))
        if jitter > 0.05:
            await asyncio.sleep(jitter)

        # Phase 0b: Check for early bids/claims that may have arrived in the
        # pending queue while the arbiter LLM call was running (the other bot
        # may have posted its bid faster due to a shorter jitter or a faster
        # arbiter response).
        early_bids, early_claim = self._drain_pending_bids(pending_queue)
        if early_claim:
            logger.info(
                "Detected early claim for {}:{}; deferring before bidding",
                msg.channel,
                msg.chat_id,
            )
            await self._post_coordination_selection(
                msg, selected=False, reason="another bot already claimed this"
            )
            return False

        # Phase 1: Post capability bid
        await self._post_coordination_bid(msg, arbiter_reason)

        # Phase 2: Notify the user that bots are deliberating
        await self._post_negotiation_progress(
            msg, "🤖 Checking with the team to find the best bot for this…"
        )

        # Phase 3: Wait for other bots' bids (and detect claims).
        # Subtract the jitter from the timeout so the total wall-clock time
        # stays roughly constant regardless of the jitter drawn.
        collect_timeout = max(negotiation_timeout - jitter, negotiation_timeout * 0.5)
        other_bids, claim_detected = await self._collect_negotiation_bids(
            pending_queue, timeout=collect_timeout
        )

        # Include any bids collected in Phase 0b
        other_bids = early_bids + other_bids

        # If another bot already posted a coordination claim during
        # the collection window, defer immediately to avoid duplicating work.
        if claim_detected:
            logger.info(
                "Another bot already claimed {}:{}; deferring",
                msg.channel,
                msg.chat_id,
            )
            await self._post_coordination_selection(
                msg, selected=False, reason="another bot already claimed this"
            )
            return False

        # If no other bots bid, run a short verification window before
        # proceeding.  This catches late-arriving bids/claims that were
        # in-flight when the main collection window closed — e.g. Slack
        # message delivery jitter or event-loop scheduling delays.
        if not other_bids:
            verify_timeout = min(1.5, negotiation_timeout * 0.25)
            late_bids, late_claim = await self._collect_negotiation_bids(
                pending_queue, timeout=verify_timeout
            )
            if late_claim:
                logger.info(
                    "Late claim detected for {}:{}; deferring",
                    msg.channel,
                    msg.chat_id,
                )
                await self._post_coordination_selection(
                    msg, selected=False, reason="another bot claimed during verification"
                )
                return False
            if late_bids:
                other_bids = late_bids
                # Fall through to multi-bid evaluation below
            else:
                logger.info(
                    "No other bids received for {}:{}; proceeding as sole bidder",
                    msg.channel,
                    msg.chat_id,
                )
                return True

        # Phase 4: Notify user that bids are being evaluated
        bid_count = len(other_bids) + 1  # include self
        await self._post_negotiation_progress(
            msg,
            f"🗳️ {bid_count} bots have offered to help — evaluating who's best suited…",
        )

        # Phase 5: Evaluate all bids and select the best bot
        selected, selection_reason = await self._select_from_negotiation(
            msg, arbiter_reason, other_bids, session
        )

        # Phase 6: Post the selection result
        await self._post_coordination_selection(msg, selected=selected, reason=selection_reason)

        if not selected:
            logger.info(
                "Deferring task for {}:{} after negotiation: {}",
                msg.channel,
                msg.chat_id,
                selection_reason,
            )
        return selected

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        on_retry_wait: Callable[[str], Awaitable[None]] | None = None,
        *,
        session: Session | None = None,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
        pending_queue: asyncio.Queue | None = None,
    ) -> tuple[str | None, list[str], list[dict], str, bool]:
        """Run the agent iteration loop.

        *on_stream*: called with each content delta during streaming.
        *on_stream_end(resuming)*: called when a streaming session finishes.
        ``resuming=True`` means tool calls follow (spinner should restart);
        ``resuming=False`` means this is the final response.

        Returns (final_content, tools_used, messages, stop_reason, had_injections).
        """
        loop_hook = _LoopHook(
            self,
            on_progress=on_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            channel=channel,
            chat_id=chat_id,
            message_id=message_id,
        )
        hook: AgentHook = (
            CompositeHook([loop_hook] + self._extra_hooks) if self._extra_hooks else loop_hook
        )

        async def _checkpoint(payload: dict[str, Any]) -> None:
            if session is None:
                return
            self._set_runtime_checkpoint(session, payload)

        async def _drain_pending(*, limit: int = _MAX_INJECTIONS_PER_TURN) -> list[dict[str, Any]]:
            """Non-blocking drain of follow-up messages from the pending queue."""
            if pending_queue is None:
                return []
            items: list[dict[str, Any]] = []
            while len(items) < limit:
                try:
                    pending_msg = pending_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                content = pending_msg.content
                media = pending_msg.media if pending_msg.media else None
                if media:
                    content, media = extract_documents(content, media)
                    media = media or None
                user_content = self.context._build_user_content(content, media)
                runtime_ctx = self.context._build_runtime_context(
                    pending_msg.channel,
                    pending_msg.chat_id,
                    self.context.timezone,
                )
                if isinstance(user_content, str):
                    merged: str | list[dict[str, Any]] = f"{runtime_ctx}\n\n{user_content}"
                else:
                    merged = [{"type": "text", "text": runtime_ctx}] + user_content
                items.append({"role": "user", "content": merged})
            return items

        result = await self.runner.run(AgentRunSpec(
            initial_messages=initial_messages,
            tools=self.tools,
            model=self.model,
            max_iterations=self.max_iterations,
            max_tool_result_chars=self.max_tool_result_chars,
            hook=hook,
            error_message="Sorry, I encountered an error calling the AI model.",
            concurrent_tools=True,
            workspace=self.workspace,
            session_key=session.key if session else None,
            context_window_tokens=self.context_window_tokens,
            context_block_limit=self.context_block_limit,
            provider_retry_mode=self.provider_retry_mode,
            progress_callback=on_progress,
            retry_wait_callback=on_retry_wait,
            checkpoint_callback=_checkpoint,
            injection_callback=_drain_pending,
        ))
        self._last_usage = result.usage
        if result.stop_reason == "max_iterations":
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            # Push final content through stream so streaming channels (e.g. Feishu)
            # update the card instead of leaving it empty.
            if on_stream and on_stream_end:
                await on_stream(result.final_content or "")
                await on_stream_end(resuming=False)
        elif result.stop_reason == "error":
            logger.error("LLM returned error: {}", (result.final_content or "")[:200])
        return result.final_content, result.tools_used, result.messages, result.stop_reason, result.had_injections

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                self.auto_compact.check_expired(
                    self._schedule_background,
                    active_session_keys=self._pending_queues.keys(),
                )
                continue
            except asyncio.CancelledError:
                # Preserve real task cancellation so shutdown can complete cleanly.
                # Only ignore non-task CancelledError signals that may leak from integrations.
                if not self._running or asyncio.current_task().cancelling():
                    raise
                continue
            except Exception as e:
                logger.warning("Error consuming inbound message: {}, continuing...", e)
                continue

            raw = msg.content.strip()
            if self.commands.is_priority(raw):
                await self._dispatch_command_inline(
                    msg, msg.session_key, raw,
                    self.commands.dispatch_priority,
                )
                continue
            effective_key = self._effective_session_key(msg)
            # If this session already has an active pending queue (i.e. a task
            # is processing this session), route the message there for mid-turn
            # injection instead of creating a competing task.
            if effective_key in self._pending_queues:
                # Non-priority commands must not be queued for injection;
                # dispatch them directly (same pattern as priority commands).
                if self.commands.is_dispatchable_command(raw):
                    await self._dispatch_command_inline(
                        msg, effective_key, raw,
                        self.commands.dispatch,
                    )
                    continue
                pending_msg = msg
                if effective_key != msg.session_key:
                    pending_msg = dataclasses.replace(
                        msg,
                        session_key_override=effective_key,
                    )
                try:
                    self._pending_queues[effective_key].put_nowait(pending_msg)
                except asyncio.QueueFull:
                    logger.warning(
                        "Pending queue full for session {}, falling back to queued task",
                        effective_key,
                    )
                else:
                    logger.info(
                        "Routed follow-up message to pending queue for session {}",
                        effective_key,
                    )
                    continue
            # Compute the effective session key before dispatching
            # This ensures /stop command can find tasks correctly when unified session is enabled
            task = asyncio.create_task(self._dispatch(msg))
            self._active_tasks.setdefault(effective_key, []).append(task)
            task.add_done_callback(
                lambda t, k=effective_key: self._active_tasks.get(k, [])
                and self._active_tasks[k].remove(t)
                if t in self._active_tasks.get(k, [])
                else None
            )

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message: per-session serial, cross-session concurrent."""
        session_key = self._effective_session_key(msg)
        if session_key != msg.session_key:
            msg = dataclasses.replace(msg, session_key_override=session_key)
        lock = self._session_locks.setdefault(session_key, asyncio.Lock())
        gate = self._concurrency_gate or nullcontext()

        # Register a pending queue so follow-up messages for this session are
        # routed here (mid-turn injection) instead of spawning a new task.
        pending = asyncio.Queue(maxsize=20)
        self._pending_queues[session_key] = pending

        try:
            async with lock, gate:
                try:
                    on_stream = on_stream_end = None
                    if msg.metadata.get("_wants_stream"):
                        # Split one answer into distinct stream segments.
                        stream_base_id = f"{msg.session_key}:{time.time_ns()}"
                        stream_segment = 0

                        def _current_stream_id() -> str:
                            return f"{stream_base_id}:{stream_segment}"

                        async def on_stream(delta: str) -> None:
                            meta = dict(msg.metadata or {})
                            meta["_stream_delta"] = True
                            meta["_stream_id"] = _current_stream_id()
                            await self.bus.publish_outbound(OutboundMessage(
                                channel=msg.channel, chat_id=msg.chat_id,
                                content=delta,
                                metadata=meta,
                            ))

                        async def on_stream_end(*, resuming: bool = False) -> None:
                            nonlocal stream_segment
                            meta = dict(msg.metadata or {})
                            meta["_stream_end"] = True
                            meta["_resuming"] = resuming
                            meta["_stream_id"] = _current_stream_id()
                            await self.bus.publish_outbound(OutboundMessage(
                                channel=msg.channel, chat_id=msg.chat_id,
                                content="",
                                metadata=meta,
                            ))
                            stream_segment += 1

                    response = await self._process_message(
                        msg, on_stream=on_stream, on_stream_end=on_stream_end,
                        pending_queue=pending,
                    )
                    if response is not None:
                        await self.bus.publish_outbound(response)
                    elif msg.channel == "cli":
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id,
                            content="", metadata=msg.metadata or {},
                        ))
                except asyncio.CancelledError:
                    logger.info("Task cancelled for session {}", session_key)
                    # Preserve partial context from the interrupted turn so
                    # the user does not lose tool results and assistant
                    # messages accumulated before /stop.  The checkpoint was
                    # already persisted to session metadata by
                    # _emit_checkpoint during tool execution; materializing
                    # it into session history now makes it visible in the
                    # next conversation turn.
                    try:
                        key = self._effective_session_key(msg)
                        session = self.sessions.get_or_create(key)
                        if self._restore_runtime_checkpoint(session):
                            self._clear_pending_user_turn(session)
                            self.sessions.save(session)
                            logger.info(
                                "Restored partial context for cancelled session {}",
                                key,
                            )
                    except Exception:
                        logger.debug(
                            "Could not restore checkpoint for cancelled session {}",
                            session_key,
                            exc_info=True,
                        )
                    raise
                except Exception:
                    logger.exception("Error processing message for session {}", session_key)
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="Sorry, I encountered an error.",
                    ))
        finally:
            # Drain any messages still in the pending queue and re-publish
            # them to the bus so they are processed as fresh inbound messages
            # rather than silently lost.
            queue = self._pending_queues.pop(session_key, None)
            if queue is not None:
                leftover = 0
                while True:
                    try:
                        item = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    await self.bus.publish_inbound(item)
                    leftover += 1
                if leftover:
                    logger.info(
                        "Re-published {} leftover message(s) to bus for session {}",
                        leftover, session_key,
                    )

    async def close_mcp(self) -> None:
        """Drain pending background archives, then close MCP connections."""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
        for name, stack in self._mcp_stacks.items():
            try:
                await stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                logger.debug("MCP server '{}' cleanup error (can be ignored)", name)
        self._mcp_stacks.clear()

    def _schedule_background(self, coro) -> None:
        """Schedule a coroutine as a tracked background task (drained on shutdown)."""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        task.add_done_callback(self._background_tasks.remove)

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        pending_queue: asyncio.Queue | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (
                msg.chat_id.split(":", 1) if ":" in msg.chat_id else ("cli", msg.chat_id)
            )
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            if self._restore_runtime_checkpoint(session):
                self.sessions.save(session)
            if self._restore_pending_user_turn(session):
                self.sessions.save(session)

            session, pending = self.auto_compact.prepare_session(session, key)

            await self.consolidator.maybe_consolidate_by_tokens(
                session,
                session_summary=pending,
            )
            # Persist subagent follow-ups into durable history BEFORE prompt
            # assembly. ContextBuilder merges adjacent same-role messages for
            # provider compatibility, which previously caused the follow-up to
            # disappear from session.messages while still being visible to the
            # LLM via the merged prompt. See _persist_subagent_followup.
            is_subagent = msg.sender_id == "subagent"
            if is_subagent and self._persist_subagent_followup(session, msg):
                self.sessions.save(session)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=0)
            current_role = "assistant" if is_subagent else "user"

            # Subagent content is already in `history` above; passing it again
            # as current_message would double-project it into the prompt.
            messages = self.context.build_messages(
                history=history,
                current_message="" if is_subagent else msg.content,
                channel=channel,
                chat_id=chat_id,
                bot_id=msg.metadata.get("bot_id") if msg.metadata else None,
                session_summary=pending,
                current_role=current_role,
            )
            final_content, _, all_msgs, _, _ = await self._run_agent_loop(
                messages, session=session, channel=channel, chat_id=chat_id,
                message_id=msg.metadata.get("message_id"),
            )
            self._save_turn(session, all_msgs, 1 + len(history))
            self._clear_runtime_checkpoint(session)
            self.sessions.save(session)
            self._schedule_background(self.consolidator.maybe_consolidate_by_tokens(session))
            return OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=final_content or "Background task completed.",
            )

        # Extract document text from media at the processing boundary so all
        # channels benefit without format-specific logic in ContextBuilder.
        if msg.media:
            new_content, image_only = extract_documents(msg.content, msg.media)
            msg = dataclasses.replace(msg, content=new_content, media=image_only)

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)
        if self._restore_runtime_checkpoint(session):
            self.sessions.save(session)
        if self._restore_pending_user_turn(session):
            self.sessions.save(session)

        session, pending = self.auto_compact.prepare_session(session, key)

        # Slash commands
        raw = msg.content.strip()
        ctx = CommandContext(msg=msg, session=session, key=key, raw=raw, loop=self)
        if result := await self.commands.dispatch(ctx):
            return result

        approved, arbiter_reason = await self._should_reply_to_inbound(msg, session)
        if not approved:
            logger.info("Skipping reply to {}:{} after smart reply gate", msg.channel, msg.sender_id)
            return None

        # For action requests in a shared channel (smart mode), run multi-bot
        # negotiation: bots post capability bids, discuss openly, and select
        # the best-suited bot before proceeding.  When negotiation_timeout is 0
        # this falls back to the legacy immediate-claim behaviour.
        if self._needs_reply_gate(msg) and self._looks_like_action_request(msg.content):
            should_proceed = await self._run_coordination_negotiation(
                msg, arbiter_reason, session, pending_queue=pending_queue,
            )
            if not should_proceed:
                return None

        await self.consolidator.maybe_consolidate_by_tokens(
            session,
            session_summary=pending,
        )

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=0)

        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            session_summary=pending,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
            bot_id=msg.metadata.get("bot_id") if msg.metadata else None,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=content,
                    metadata=meta,
                )
            )

        async def _on_retry_wait(content: str) -> None:
            meta = dict(msg.metadata or {})
            meta["_retry_wait"] = True
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=content,
                    metadata=meta,
                )
            )

        # Persist the triggering user message immediately, before running the
        # agent loop. If the process is killed mid-turn (OOM, SIGKILL, self-
        # restart, etc.), the existing runtime_checkpoint preserves the
        # in-flight assistant/tool state but NOT the user message itself, so
        # the user's prompt is silently lost on recovery. Saving it up front
        # makes recovery possible from the session log alone.
        user_persisted_early = False
        if isinstance(msg.content, str) and msg.content.strip():
            session.add_message("user", msg.content)
            self._mark_pending_user_turn(session)
            self.sessions.save(session)
            user_persisted_early = True

        final_content, _, all_msgs, stop_reason, had_injections = await self._run_agent_loop(
            initial_messages,
            on_progress=on_progress or _bus_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            on_retry_wait=_on_retry_wait,
            session=session,
            channel=msg.channel,
            chat_id=msg.chat_id,
            message_id=msg.metadata.get("message_id"),
            pending_queue=pending_queue,
        )

        if final_content is None or not final_content.strip():
            final_content = EMPTY_FINAL_RESPONSE_MESSAGE

        # Skip the already-persisted user message when saving the turn
        save_skip = 1 + len(history) + (1 if user_persisted_early else 0)
        self._save_turn(session, all_msgs, save_skip)
        self._clear_pending_user_turn(session)
        self._clear_runtime_checkpoint(session)
        self.sessions.save(session)
        self._schedule_background(self.consolidator.maybe_consolidate_by_tokens(session))

        # When follow-up messages were injected mid-turn, a later natural
        # language reply may address those follow-ups and should not be
        # suppressed just because MessageTool was used earlier in the turn.
        # However, if the turn falls back to the empty-final-response
        # placeholder, suppress it when the real user-visible output already
        # came from MessageTool.
        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            if not had_injections or stop_reason == "empty_final_response":
                return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)

        meta = dict(msg.metadata or {})
        if on_stream is not None and stop_reason != "error":
            meta["_streamed"] = True
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=meta,
        )

    def _sanitize_persisted_blocks(
        self,
        content: list[dict[str, Any]],
        *,
        should_truncate_text: bool = False,
        drop_runtime: bool = False,
    ) -> list[dict[str, Any]]:
        """Strip volatile multimodal payloads before writing session history."""
        filtered: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                filtered.append(block)
                continue

            if (
                drop_runtime
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and block["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG)
            ):
                continue

            if block.get("type") == "image_url" and block.get("image_url", {}).get(
                "url", ""
            ).startswith("data:image/"):
                path = (block.get("_meta") or {}).get("path", "")
                filtered.append({"type": "text", "text": image_placeholder_text(path)})
                continue

            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text = block["text"]
                if should_truncate_text and len(text) > self.max_tool_result_chars:
                    text = truncate_text_fn(text, self.max_tool_result_chars)
                filtered.append({**block, "text": text})
                continue

            filtered.append(block)

        return filtered

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime

        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool":
                if isinstance(content, str) and len(content) > self.max_tool_result_chars:
                    entry["content"] = truncate_text_fn(content, self.max_tool_result_chars)
                elif isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, should_truncate_text=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # Strip the entire runtime-context block (including any session summary).
                    # The block is bounded by _RUNTIME_CONTEXT_TAG and _RUNTIME_CONTEXT_END.
                    end_marker = ContextBuilder._RUNTIME_CONTEXT_END
                    end_pos = content.find(end_marker)
                    if end_pos >= 0:
                        after = content[end_pos + len(end_marker):].lstrip("\n")
                        if after:
                            entry["content"] = after
                        else:
                            continue
                    else:
                        # Fallback: no end marker found, strip the tag prefix
                        after_tag = content[len(ContextBuilder._RUNTIME_CONTEXT_TAG):].lstrip("\n")
                        if after_tag.strip():
                            entry["content"] = after_tag
                        else:
                            continue
                if isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, drop_runtime=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    def _persist_subagent_followup(self, session: Session, msg: InboundMessage) -> bool:
        """Persist subagent follow-ups before prompt assembly so history stays durable.

        Returns True if a new entry was appended; False if the follow-up was
        deduped (same ``subagent_task_id`` already in session) or carries no
        content worth persisting.
        """
        if not msg.content:
            return False
        task_id = msg.metadata.get("subagent_task_id") if isinstance(msg.metadata, dict) else None
        if task_id and any(
            m.get("injected_event") == "subagent_result" and m.get("subagent_task_id") == task_id
            for m in session.messages
        ):
            return False
        session.add_message(
            "assistant",
            msg.content,
            sender_id=msg.sender_id,
            injected_event="subagent_result",
            subagent_task_id=task_id,
        )
        return True

    def _set_runtime_checkpoint(self, session: Session, payload: dict[str, Any]) -> None:
        """Persist the latest in-flight turn state into session metadata."""
        session.metadata[self._RUNTIME_CHECKPOINT_KEY] = payload
        self.sessions.save(session)

    def _mark_pending_user_turn(self, session: Session) -> None:
        session.metadata[self._PENDING_USER_TURN_KEY] = True

    def _clear_pending_user_turn(self, session: Session) -> None:
        session.metadata.pop(self._PENDING_USER_TURN_KEY, None)

    def _clear_runtime_checkpoint(self, session: Session) -> None:
        if self._RUNTIME_CHECKPOINT_KEY in session.metadata:
            session.metadata.pop(self._RUNTIME_CHECKPOINT_KEY, None)

    @staticmethod
    def _checkpoint_message_key(message: dict[str, Any]) -> tuple[Any, ...]:
        return (
            message.get("role"),
            message.get("content"),
            message.get("tool_call_id"),
            message.get("name"),
            message.get("tool_calls"),
            message.get("reasoning_content"),
            message.get("thinking_blocks"),
        )

    def _restore_runtime_checkpoint(self, session: Session) -> bool:
        """Materialize an unfinished turn into session history before a new request."""
        from datetime import datetime

        checkpoint = session.metadata.get(self._RUNTIME_CHECKPOINT_KEY)
        if not isinstance(checkpoint, dict):
            return False

        assistant_message = checkpoint.get("assistant_message")
        completed_tool_results = checkpoint.get("completed_tool_results") or []
        pending_tool_calls = checkpoint.get("pending_tool_calls") or []

        restored_messages: list[dict[str, Any]] = []
        if isinstance(assistant_message, dict):
            restored = dict(assistant_message)
            restored.setdefault("timestamp", datetime.now().isoformat())
            restored_messages.append(restored)
        for message in completed_tool_results:
            if isinstance(message, dict):
                restored = dict(message)
                restored.setdefault("timestamp", datetime.now().isoformat())
                restored_messages.append(restored)
        for tool_call in pending_tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_id = tool_call.get("id")
            name = ((tool_call.get("function") or {}).get("name")) or "tool"
            restored_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": name,
                    "content": "Error: Task interrupted before this tool finished.",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        overlap = 0
        max_overlap = min(len(session.messages), len(restored_messages))
        for size in range(max_overlap, 0, -1):
            existing = session.messages[-size:]
            restored = restored_messages[:size]
            if all(
                self._checkpoint_message_key(left) == self._checkpoint_message_key(right)
                for left, right in zip(existing, restored)
            ):
                overlap = size
                break
        session.messages.extend(restored_messages[overlap:])

        self._clear_pending_user_turn(session)
        self._clear_runtime_checkpoint(session)
        return True

    def _restore_pending_user_turn(self, session: Session) -> bool:
        """Close a turn that only persisted the user message before crashing."""
        from datetime import datetime

        if not session.metadata.get(self._PENDING_USER_TURN_KEY):
            return False

        if session.messages and session.messages[-1].get("role") == "user":
            session.messages.append(
                {
                    "role": "assistant",
                    "content": "Error: Task interrupted before a response was generated.",
                    "timestamp": datetime.now().isoformat(),
                }
            )
            session.updated_at = datetime.now()

        self._clear_pending_user_turn(session)
        return True

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        media: list[str] | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a message directly and return the outbound payload."""
        await self._connect_mcp()
        msg = InboundMessage(
            channel=channel, sender_id="user", chat_id=chat_id,
            content=content, media=media or [],
        )
        return await self._process_message(
            msg,
            session_key=session_key,
            on_progress=on_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
        )
