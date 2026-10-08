"""Approval policy for side-effecting tools (file writes and shell commands)."""

from __future__ import annotations

import asyncio
import itertools
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from lex import events as ev

ASK, AUTO, DENY = "ask", "auto", "deny"
MODES = (ASK, AUTO, DENY)

# Commands that always need a human, even in auto mode.
DANGEROUS = [
    r"\bsudo\b", r"\bdoas\b", r"\brm\s+(-\w*\s+)*-\w*[rf]\w*\s+(/|~|\$HOME|\*)(\s|$)", r"\bmkfs",
    r"\bdd\s+.*\bof=/dev/", r">\s*/dev/sd", r"\bshutdown\b", r"\breboot\b", r":\(\)\s*\{",
    r"\b(curl|wget)\b[^|]*\|\s*(sh|bash|zsh|python)", r"\bgit\s+push\b.*(--force|-f\b)",
    r"\bgit\s+reset\s+--hard\b", r"\bgit\s+clean\s+-\w*f", r"\bchmod\s+-R\s+777\b", r"\bchown\s+-R\b",
]
_DANGEROUS_RE = [re.compile(p) for p in DANGEROUS]


def is_dangerous(command: str) -> bool:
    return any(r.search(command) for r in _DANGEROUS_RE)


@dataclass
class ApprovalRequest:
    id: str
    kind: str  # "write" | "exec"
    agent: str
    title: str
    detail: str  # diff preview or command
    reason: str = ""


Decider = Callable[[ApprovalRequest], Awaitable[bool]]


@dataclass
class ApprovalPolicy:
    writes: str = ASK
    commands: str = ASK

    def mode_for(self, kind: str) -> str:
        return self.writes if kind == "write" else self.commands


@dataclass
class Approvals:
    """Applies the policy and, when needed, asks a human through `decider`."""

    bus: ev.EventBus
    policy: ApprovalPolicy = field(default_factory=ApprovalPolicy)
    decider: Decider | None = None
    _ids: itertools.count = field(default_factory=lambda: itertools.count(1))

    async def check(self, kind: str, agent: str, title: str, detail: str) -> tuple[bool, str]:
        """Return (allowed, note). `note` explains a refusal to the model."""
        mode = self.policy.mode_for(kind)
        dangerous = kind == "exec" and is_dangerous(detail)
        if mode == DENY:
            return False, f"{'Shell commands' if kind == 'exec' else 'File changes'} are disabled by the user's policy."
        if mode == AUTO and not dangerous:
            return True, ""
        if self.decider is None:
            return False, "No one is available to approve this action, so it was refused."
        req = ApprovalRequest(
            id=f"a{next(self._ids)}", kind=kind, agent=agent, title=title, detail=detail,
            reason="Flagged as potentially destructive." if dangerous else "",
        )
        await self.bus.emit(ev.APPROVAL_REQUEST, agent, id=req.id, kind=kind, title=title,
                            detail=detail, reason=req.reason)
        allowed = bool(await self.decider(req))
        await self.bus.emit(ev.APPROVAL_RESOLVED, agent, id=req.id, allowed=allowed)
        return allowed, "" if allowed else "The user declined this action. Do not retry it; adapt or explain."


class FutureDecider:
    """A decider whose answers arrive later from elsewhere (the web UI's websocket)."""

    def __init__(self) -> None:
        self._pending: dict[str, asyncio.Future[bool]] = {}

    async def __call__(self, req: ApprovalRequest) -> bool:
        fut: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending[req.id] = fut
        try:
            return await fut
        finally:
            self._pending.pop(req.id, None)

    def resolve(self, request_id: str, allowed: bool) -> bool:
        fut = self._pending.get(request_id)
        if fut is None or fut.done():
            return False
        fut.set_result(allowed)
        return True

    def cancel_all(self) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.set_result(False)
