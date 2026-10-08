"""A deterministic provider that replays scripted turns. Used by tests and `--provider scripted` demos."""

from __future__ import annotations

import itertools
from typing import Any, Callable

from lex.providers.base import ToolCall, Turn

_ids = itertools.count(1)


def call(name: str, **args: Any) -> ToolCall:
    return ToolCall(id=f"call-{next(_ids)}", name=name, args=args)


ScriptStep = Turn | Callable[[list[Any]], Turn]


class ScriptedProvider:
    name = "scripted"
    model = "scripted"

    def __init__(self, scripts: dict[str, list[ScriptStep]]) -> None:
        self.scripts = {role: list(steps) for role, steps in scripts.items()}
        self.transcripts: dict[str, list[Any]] = {}

    def start_chat(self, role: str, system: str, tools: list[dict[str, Any]]) -> "ScriptedChat":
        log = self.transcripts.setdefault(role, [])
        return ScriptedChat(self.scripts.setdefault(role, []), log)


class ScriptedChat:
    def __init__(self, script: list[ScriptStep], log: list[Any]) -> None:
        self._script = script
        self.log = log

    async def send(self, text: str) -> Turn:
        self.log.append(("user", text))
        return self._next()

    async def send_tool_results(self, results: list[tuple[ToolCall, dict[str, Any]]]) -> Turn:
        self.log.append(("tools", results))
        return self._next()

    def _next(self) -> Turn:
        if not self._script:
            return Turn(text="(script exhausted)")
        step = self._script.pop(0)
        return step(self.log) if callable(step) else step
