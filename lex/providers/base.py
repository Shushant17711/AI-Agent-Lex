"""Provider-neutral chat interface with native tool calling."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class Turn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class ProviderError(Exception):
    def __init__(self, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class Chat(Protocol):
    """One agent's conversation. Implementations keep history in their native format."""

    async def send(self, text: str) -> Turn: ...

    async def send_tool_results(self, results: list[tuple[ToolCall, dict[str, Any]]]) -> Turn: ...


class Provider(Protocol):
    name: str
    model: str

    def start_chat(self, role: str, system: str, tools: list[dict[str, Any]]) -> Chat: ...


async def with_retries(call, attempts: int = 5):
    """Retry transient provider failures (rate limits, 5xx, network) with jittered backoff."""
    for attempt in range(attempts):
        try:
            return await call()
        except ProviderError as e:
            if not e.retryable or attempt == attempts - 1:
                raise
        await asyncio.sleep(min(30.0, (2 ** attempt) + random.random()))
    raise AssertionError("unreachable")
