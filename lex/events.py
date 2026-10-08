"""Events emitted during a run. The CLI and the web UI are both just event consumers."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable

# Event types
RUN_STARTED = "run_started"
PHASE = "phase"  # agent handoff: planner -> coder -> reviewer
MESSAGE = "message"  # assistant prose
TOOL_CALL = "tool_call"
TOOL_RESULT = "tool_result"
PLAN = "plan"
STEP = "step"  # plan step status change
FILE_CHANGED = "file_changed"
APPROVAL_REQUEST = "approval_request"
APPROVAL_RESOLVED = "approval_resolved"
REVIEW = "review"
USAGE = "usage"
ERROR = "error"
RUN_FINISHED = "run_finished"


@dataclass
class Event:
    type: str
    agent: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


EventSink = Callable[[Event], Awaitable[None]]


class EventBus:
    """Fans events out to any number of async subscribers and keeps the full log."""

    def __init__(self) -> None:
        self.log: list[Event] = []
        self._subscribers: list[EventSink] = []

    def subscribe(self, sink: EventSink) -> None:
        self._subscribers.append(sink)

    async def emit(self, type: str, agent: str | None = None, **data: Any) -> Event:
        event = Event(type=type, agent=agent, data=data)
        self.log.append(event)
        for sink in list(self._subscribers):
            try:
                await sink(event)
            except Exception:  # a broken consumer must never kill the run
                pass
        return event
