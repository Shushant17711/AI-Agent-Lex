"""Tools the agents can call. Every handler returns a JSON-serialisable dict."""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import signal
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from lex import events as ev
from lex.approval import Approvals
from lex.workspace import FileChange, Workspace, WorkspaceError

MAX_OUTPUT = 12_000
READ_LINE_LIMIT = 800


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Awaitable[dict[str, Any]]]
    terminal: bool = False  # calling it ends the agent's turn loop

    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def _obj(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required or []}


def _clip(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n… [{len(text) - limit} characters omitted] …\n{text[-half:]}"


@dataclass
class ToolContext:
    workspace: Workspace
    approvals: Approvals
    bus: ev.EventBus
    agent: str = ""
    command_timeout: int = 120
    _running: set[asyncio.subprocess.Process] = field(default_factory=set)

    async def _announce(self, change: FileChange) -> None:
        await self.bus.emit(ev.FILE_CHANGED, self.agent, **change.to_dict())

    # ---- read-only ---------------------------------------------------------
    async def list_files(self, path: str = ".", depth: int = 2) -> dict[str, Any]:
        entries = self.workspace.walk(path, max_depth=max(1, min(int(depth), 6)), limit=600)
        lines = [f"{p}/" if is_dir else p for p, is_dir in entries]
        return {"path": path, "entries": lines, "truncated": len(entries) >= 600}

    async def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> dict[str, Any]:
        lines = self.workspace.read_text(path).splitlines()
        start = max(1, int(start_line))
        end = len(lines) if end_line is None else min(len(lines), int(end_line))
        end = min(end, start + READ_LINE_LIMIT - 1)
        width = len(str(end))
        body = "\n".join(f"{i:>{width}}│{lines[i - 1]}" for i in range(start, end + 1))
        result: dict[str, Any] = {"path": path, "total_lines": len(lines), "content": body}
        if end < len(lines):
            result["note"] = f"Showing lines {start}-{end}. Pass start_line={end + 1} to continue."
        return result

    async def search(self, pattern: str, path: str = ".", glob: str | None = None,
                     ignore_case: bool = False) -> dict[str, Any]:
        try:
            rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as e:
            raise WorkspaceError(f"Invalid regex: {e}")
        base = self.workspace.resolve(path)
        targets = [(self.workspace.rel(base), False)] if base.is_file() else self.workspace.walk(path, 50, 20_000)
        matches: list[str] = []
        for rel, is_dir in targets:
            if is_dir or (glob and not fnmatch.fnmatch(os.path.basename(rel), glob) and not fnmatch.fnmatch(rel, glob)):
                continue
            try:
                text = self.workspace.read_text(rel)
            except WorkspaceError:
                continue
            for n, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    matches.append(f"{rel}:{n}: {line.strip()[:200]}")
                    if len(matches) >= 200:
                        return {"matches": matches, "truncated": True}
        return {"matches": matches, "truncated": False}

    # ---- side effects ------------------------------------------------------
    async def write_file(self, path: str, content: str) -> dict[str, Any]:
        full = self.workspace.resolve(path)
        exists = full.is_file()
        preview = _preview_change(self.workspace, path, full.read_text("utf-8", "replace") if exists else None, content)
        ok, note = await self.approvals.check("write", self.agent, f"{'Overwrite' if exists else 'Create'} {path}", preview)
        if not ok:
            return {"ok": False, "error": note}
        change = self.workspace.write_text(path, content)
        await self._announce(change)
        return {"ok": True, "path": path, "lines": content.count("\n") + 1}

    async def edit_file(self, path: str, old_text: str, new_text: str, replace_all: bool = False) -> dict[str, Any]:
        original = self.workspace.read_text(path)
        count = original.count(old_text) if old_text else 0
        if count == 0:
            raise WorkspaceError("old_text was not found. Re-read the file and copy the exact text, including whitespace.")
        if count > 1 and not replace_all:
            raise WorkspaceError(f"old_text occurs {count} times. Include more surrounding context or set replace_all.")
        updated = original.replace(old_text, new_text) if replace_all else original.replace(old_text, new_text, 1)
        preview = _preview_change(self.workspace, path, original, updated)
        ok, note = await self.approvals.check("write", self.agent, f"Edit {path}", preview)
        if not ok:
            return {"ok": False, "error": note}
        change = self.workspace.write_text(path, updated)
        await self._announce(change)
        return {"ok": True, "path": path, "replacements": count if replace_all else 1}

    async def delete_file(self, path: str) -> dict[str, Any]:
        self.workspace.read_text(path)  # validates existence/path
        ok, note = await self.approvals.check("write", self.agent, f"Delete {path}", f"rm {path}")
        if not ok:
            return {"ok": False, "error": note}
        change = self.workspace.delete(path)
        await self._announce(change)
        return {"ok": True, "path": path}

    async def run_command(self, command: str, timeout: int | None = None) -> dict[str, Any]:
        if not command.strip():
            raise WorkspaceError("Command is empty.")
        ok, note = await self.approvals.check("exec", self.agent, "Run command", command)
        if not ok:
            return {"ok": False, "error": note}
        limit = max(1, min(int(timeout or self.command_timeout), 900))
        proc = await asyncio.create_subprocess_shell(
            command, cwd=self.workspace.root, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, start_new_session=True,
        )
        self._running.add(proc)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=limit)
            timed_out = False
        except asyncio.TimeoutError:
            _kill(proc)
            out, timed_out = b"", True
        except asyncio.CancelledError:
            _kill(proc)
            raise
        finally:
            self._running.discard(proc)
        text = _clip(out.decode("utf-8", errors="replace"))
        if timed_out:
            return {"ok": False, "error": f"Timed out after {limit}s and was killed.", "output": text}
        return {"ok": proc.returncode == 0, "exit_code": proc.returncode, "output": text}

    def kill_all(self) -> None:
        for proc in list(self._running):
            _kill(proc)


def _kill(proc: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _preview_change(ws: Workspace, path: str, before: str | None, after: str) -> str:
    rel = ws.rel(ws.resolve(path))
    return _clip(FileChange(rel, before, after).diff() or "(no textual change)", 20_000)


# ---- tool catalogue ---------------------------------------------------------

def read_tools(ctx: ToolContext) -> list[ToolSpec]:
    return [
        ToolSpec("list_files", "List files and folders in the workspace (noise like .git and node_modules is skipped).",
                 _obj({"path": {"type": "string", "description": "Directory relative to the workspace root. Default '.'"},
                       "depth": {"type": "integer", "description": "How many levels deep to list (1-6). Default 2."}}),
                 ctx.list_files),
        ToolSpec("read_file", "Read a text file. Output lines are prefixed with line numbers and '│' (not part of the file).",
                 _obj({"path": {"type": "string"},
                       "start_line": {"type": "integer", "description": "1-based first line. Default 1."},
                       "end_line": {"type": "integer", "description": "Last line to include (inclusive)."}}, ["path"]),
                 ctx.read_file),
        ToolSpec("search", "Search file contents with a Python regular expression. Returns path:line: text matches.",
                 _obj({"pattern": {"type": "string"},
                       "path": {"type": "string", "description": "File or directory to search. Default '.'"},
                       "glob": {"type": "string", "description": "Only search files matching this glob, e.g. '*.py'."},
                       "ignore_case": {"type": "boolean"}}, ["pattern"]),
                 ctx.search),
    ]


def write_tools(ctx: ToolContext) -> list[ToolSpec]:
    return [
        ToolSpec("write_file", "Create a file or replace its entire contents. Prefer edit_file for small changes to existing files.",
                 _obj({"path": {"type": "string"}, "content": {"type": "string", "description": "The complete new file contents."}},
                      ["path", "content"]),
                 ctx.write_file),
        ToolSpec("edit_file", "Replace an exact snippet in a file. old_text must match exactly once unless replace_all is true.",
                 _obj({"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"},
                       "replace_all": {"type": "boolean"}}, ["path", "old_text", "new_text"]),
                 ctx.edit_file),
        ToolSpec("delete_file", "Delete a file.", _obj({"path": {"type": "string"}}, ["path"]), ctx.delete_file),
    ]


def exec_tools(ctx: ToolContext) -> list[ToolSpec]:
    return [
        ToolSpec("run_command", "Run a shell command in the workspace root (non-interactive; stdin is closed). "
                 "Use it for tests, linters, builds and running scripts.",
                 _obj({"command": {"type": "string"},
                       "timeout": {"type": "integer", "description": "Seconds before the command is killed. Default 120."}},
                      ["command"]),
                 ctx.run_command),
    ]
