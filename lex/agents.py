"""The agent loop and the planner -> coder -> reviewer orchestration."""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from lex import events as ev
from lex import prompts
from lex.approval import ApprovalPolicy, Approvals, Decider
from lex.config import Settings
from lex.providers.base import Provider, ProviderError, ToolCall, Turn
from lex.tools import ToolContext, ToolSpec, _clip, _obj, exec_tools, read_tools, write_tools
from lex.workspace import Workspace, WorkspaceError

PLANNER, CODER, REVIEWER = "planner", "coder", "reviewer"
STEP_STATUSES = ("pending", "in_progress", "done", "skipped")


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, turn: Turn) -> None:
        self.input_tokens += turn.input_tokens
        self.output_tokens += turn.output_tokens


class Agent:
    """Drives one chat through tool calls until it calls a terminal tool."""

    def __init__(self, role: str, provider: Provider, system: str, tools: list[ToolSpec],
                 bus: ev.EventBus, usage: Usage, max_turns: int) -> None:
        self.role = role
        self.tools = {t.name: t for t in tools}
        self.terminal_names = [t.name for t in tools if t.terminal]
        self.bus = bus
        self.usage = usage
        self.max_turns = max_turns
        self.chat = provider.start_chat(role, system, [t.schema() for t in tools])
        self._pending: list[tuple[ToolCall, dict[str, Any]]] = []
        self._terminal_call: ToolCall | None = None

    async def start(self, message: str) -> dict[str, Any] | None:
        return await self._loop(await self.chat.send(message))

    async def resume(self, feedback: str) -> dict[str, Any] | None:
        """Continue after a terminal call, delivering `feedback` as that call's result."""
        if not self._pending:
            return await self._loop(await self.chat.send(feedback))
        results = [(c, ({"ok": True, "message": feedback} if c is self._terminal_call else r)) for c, r in self._pending]
        self._pending, self._terminal_call = [], None
        return await self._loop(await self.chat.send_tool_results(results))

    async def _loop(self, turn: Turn) -> dict[str, Any] | None:
        nudges = 0
        for _ in range(self.max_turns):
            self.usage.add(turn)
            await self.bus.emit(ev.USAGE, self.role, input_tokens=self.usage.input_tokens,
                                output_tokens=self.usage.output_tokens)
            if turn.text:
                await self.bus.emit(ev.MESSAGE, self.role, text=turn.text)
            if not turn.tool_calls:
                nudges += 1
                if nudges > 2:
                    return None
                names = " or ".join(f"`{n}`" for n in self.terminal_names)
                turn = await self.chat.send(f"Keep going using your tools. When you are finished, call {names}.")
                continue

            results: list[tuple[ToolCall, dict[str, Any]]] = []
            terminal: tuple[ToolCall, dict[str, Any]] | None = None
            for call in turn.tool_calls:
                result = await self._dispatch(call)
                results.append((call, result))
                spec = self.tools.get(call.name)
                if spec and spec.terminal and result.get("ok") and terminal is None:
                    terminal = (call, call.args)
            if terminal:
                self._pending, self._terminal_call = results, terminal[0]
                return terminal[1]
            turn = await self.chat.send_tool_results(results)
        await self.bus.emit(ev.ERROR, self.role, message=f"{self.role} hit the {self.max_turns}-turn limit.")
        return None

    async def _dispatch(self, call: ToolCall) -> dict[str, Any]:
        call_id = call.id or uuid.uuid4().hex[:8]
        await self.bus.emit(ev.TOOL_CALL, self.role, id=call_id, name=call.name, args=_summarise_args(call.args))
        spec = self.tools.get(call.name)
        if spec is None:
            result = {"ok": False, "error": f"Unknown tool '{call.name}'. Available: {', '.join(self.tools)}"}
        elif "__invalid_json__" in call.args:
            result = {"ok": False, "error": "Tool arguments were not valid JSON. Try again."}
        else:
            try:
                result = await spec.handler(**call.args)
                result.setdefault("ok", True)
            except WorkspaceError as e:
                result = {"ok": False, "error": str(e)}
            except TypeError as e:
                result = {"ok": False, "error": f"Bad arguments for {call.name}: {e}"}
            except (OSError, UnicodeError, ValueError) as e:
                result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        await self.bus.emit(ev.TOOL_RESULT, self.role, id=call_id, name=call.name, ok=bool(result.get("ok")),
                            preview=_clip(_render_result(result), 4000))
        return result


def _summarise_args(args: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 600:
            out[k] = f"{v[:600]}… ({len(v)} chars)"
        else:
            out[k] = v
    return out


def _render_result(result: dict[str, Any]) -> str:
    if not result.get("ok", True) and "error" in result:
        extra = f"\n{result['output']}" if result.get("output") else ""
        return f"error: {result['error']}{extra}"
    for key in ("content", "output"):
        if key in result:
            return str(result[key])
    if "entries" in result:
        return "\n".join(result["entries"])
    if "matches" in result:
        return "\n".join(result["matches"]) or "(no matches)"
    return json.dumps({k: v for k, v in result.items() if k != "ok"}, ensure_ascii=False)


@dataclass
class RunResult:
    status: str  # completed | needs_attention | failed | cancelled
    summary: str
    changes: list[dict[str, Any]] = field(default_factory=list)


class Run:
    """One task, start to finish."""

    def __init__(self, task: str, settings: Settings, workspace: Workspace, provider: Provider,
                 bus: ev.EventBus | None = None, decider: Decider | None = None) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.task = task.strip()
        self.settings = settings
        self.workspace = workspace
        self.provider = provider
        self.bus = bus or ev.EventBus()
        self.approvals = Approvals(self.bus, ApprovalPolicy(settings.approve_writes, settings.approve_commands), decider)
        self.usage = Usage()
        self.steps: list[dict[str, Any]] = []
        self.approach = ""
        self._ctx: list[ToolContext] = []

    # ---- building agents ---------------------------------------------------
    def _context(self, role: str) -> ToolContext:
        ctx = ToolContext(self.workspace, self.approvals, self.bus, role, self.settings.command_timeout)
        self._ctx.append(ctx)
        return ctx

    def _agent(self, role: str, system: str, tools: list[ToolSpec]) -> Agent:
        return Agent(role, self.provider, system, tools, self.bus, self.usage, self.settings.max_turns)

    def _planner(self) -> Agent:
        async def submit_plan(steps: list[str], approach: str = "") -> dict[str, Any]:
            clean = [str(s).strip() for s in steps if str(s).strip()][:12]
            if not clean:
                raise WorkspaceError("The plan needs at least one step.")
            self.approach = approach.strip()
            self.steps = [{"text": s, "status": "pending", "note": ""} for s in clean]
            await self.bus.emit(ev.PLAN, PLANNER, approach=self.approach, steps=self.steps)
            return {"ok": True}

        ctx = self._context(PLANNER)
        tools = read_tools(ctx) + [ToolSpec(
            "submit_plan", "Hand the plan to the Coder. Call once, when you're ready.",
            _obj({"approach": {"type": "string"},
                  "steps": {"type": "array", "items": {"type": "string"}}}, ["approach", "steps"]),
            submit_plan, terminal=True)]
        return self._agent(PLANNER, prompts.PLANNER, tools)

    def _coder(self) -> Agent:
        async def update_step(step: int, status: str, note: str = "") -> dict[str, Any]:
            if status not in STEP_STATUSES:
                raise WorkspaceError(f"status must be one of {', '.join(STEP_STATUSES)}")
            idx = int(step) - 1
            if not 0 <= idx < len(self.steps):
                raise WorkspaceError(f"There is no step {step}; the plan has {len(self.steps)} steps.")
            self.steps[idx].update(status=status, note=note)
            await self.bus.emit(ev.STEP, CODER, index=idx, status=status, note=note)
            return {"ok": True}

        async def finish(summary: str) -> dict[str, Any]:
            return {"ok": True, "summary": summary}

        ctx = self._context(CODER)
        tools = read_tools(ctx) + write_tools(ctx) + exec_tools(ctx) + [
            ToolSpec("update_step", "Mark a plan step's progress (1-based step number).",
                     _obj({"step": {"type": "integer"},
                           "status": {"type": "string", "enum": list(STEP_STATUSES)},
                           "note": {"type": "string"}}, ["step", "status"]),
                     update_step),
            ToolSpec("finish", "Report that the work is complete, with a summary for the user.",
                     _obj({"summary": {"type": "string"}}, ["summary"]), finish, terminal=True),
        ]
        return self._agent(CODER, prompts.CODER, tools)

    def _reviewer(self) -> Agent:
        async def submit_review(approved: bool, summary: str, issues: list[str] | None = None) -> dict[str, Any]:
            return {"ok": True}

        ctx = self._context(REVIEWER)
        tools = read_tools(ctx) + exec_tools(ctx) + [ToolSpec(
            "submit_review", "Deliver your verdict. Call once.",
            _obj({"approved": {"type": "boolean"}, "summary": {"type": "string"},
                  "issues": {"type": "array", "items": {"type": "string"}}}, ["approved", "summary"]),
            submit_review, terminal=True)]
        return self._agent(REVIEWER, prompts.REVIEWER, tools)

    # ---- running -------------------------------------------------------------
    async def execute(self) -> RunResult:
        s = self.settings
        await self.bus.emit(ev.RUN_STARTED, None, id=self.id, task=self.task, workspace=str(self.workspace.root),
                            provider=self.provider.name, model=self.provider.model,
                            options={"plan": s.plan, "review": s.review, "approve_writes": s.approve_writes,
                                     "approve_commands": s.approve_commands})
        try:
            result = await self._execute()
        except asyncio.CancelledError:
            self._kill()
            result = RunResult("cancelled", "The run was stopped.", self._changes())
            await self._finish(result)
            raise
        except ProviderError as e:
            await self.bus.emit(ev.ERROR, None, message=str(e))
            result = RunResult("failed", str(e), self._changes())
        await self._finish(result)
        return result

    async def _finish(self, result: RunResult) -> None:
        await self.bus.emit(ev.RUN_FINISHED, None, status=result.status, summary=result.summary,
                            changes=result.changes, input_tokens=self.usage.input_tokens,
                            output_tokens=self.usage.output_tokens)

    def _kill(self) -> None:
        for ctx in self._ctx:
            ctx.kill_all()

    def _changes(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self.workspace.changes()]

    def _tree(self) -> str:
        entries = self.workspace.walk(".", max_depth=2, limit=150)
        return "\n".join(f"{p}/" if d else p for p, d in entries) or "(empty)"

    async def _execute(self) -> RunResult:
        s = self.settings
        if s.plan:
            await self.bus.emit(ev.PHASE, PLANNER, title="Planning")
            planned = await self._planner().start(
                f"## Request\n{self.task}\n\n## Workspace (top levels)\n{self._tree()}")
            if planned is None:
                await self.bus.emit(ev.ERROR, PLANNER, message="The planner didn't produce a plan; continuing without one.")
        if not self.steps:
            self.steps = [{"text": self.task, "status": "pending", "note": ""}]
            self.approach = "Do the request directly."
            await self.bus.emit(ev.PLAN, None, approach=self.approach, steps=self.steps)

        await self.bus.emit(ev.PHASE, CODER, title="Implementing")
        coder = self._coder()
        texts = [st["text"] for st in self.steps]
        done = await coder.start(prompts.coder_brief(self.task, self.approach, texts, self._tree()))
        if done is None:
            return RunResult("needs_attention", "The coder stopped without finishing. Check the timeline.", self._changes())
        summary = str(done.get("summary", ""))

        status = "completed"
        if s.review and self.workspace.changes():
            for round_no in range(1, s.review_rounds + 2):
                await self.bus.emit(ev.PHASE, REVIEWER, title="Reviewing" if round_no == 1 else f"Re-review #{round_no - 1}")
                diff = _clip("".join(c.diff() for c in self.workspace.changes()), 60_000)
                verdict = await self._reviewer().start(prompts.reviewer_brief(self.task, texts, summary, diff))
                if verdict is None:
                    await self.bus.emit(ev.ERROR, REVIEWER, message="The reviewer gave no verdict.")
                    status = "needs_attention"
                    break
                approved = bool(verdict.get("approved"))
                issues = [str(i) for i in (verdict.get("issues") or [])]
                await self.bus.emit(ev.REVIEW, REVIEWER, approved=approved, summary=str(verdict.get("summary", "")),
                                    issues=issues, round=round_no)
                if approved:
                    status = "completed"
                    break
                if round_no > s.review_rounds:
                    status = "needs_attention"
                    break
                await self.bus.emit(ev.PHASE, CODER, title="Addressing review")
                done = await coder.resume(prompts.review_feedback(str(verdict.get("summary", "")), issues))
                if done is None:
                    return RunResult("needs_attention", "The coder stopped while addressing review feedback.", self._changes())
                summary = str(done.get("summary", summary))
        return RunResult(status, summary, self._changes())
