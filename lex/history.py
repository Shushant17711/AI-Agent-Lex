"""Persist each run's event log as JSONL so it can be replayed in the UI."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from lex import events as ev


def runs_dir() -> Path:
    base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    path = base / "lex" / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


class Recorder:
    """An event sink that appends to <runs_dir>/<run_id>.jsonl."""

    def __init__(self, run_id: str) -> None:
        self.path = runs_dir() / f"{run_id}.jsonl"

    async def __call__(self, event: ev.Event) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")


def _valid_id(run_id: str) -> bool:
    return run_id.isalnum() and len(run_id) <= 64


def load_run(run_id: str) -> list[dict[str, Any]]:
    if not _valid_id(run_id):
        return []
    path = runs_dir() / f"{run_id}.jsonl"
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def list_runs(workspace: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    files = sorted(runs_dir().glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    out: list[dict[str, Any]] = []
    for path in files:
        if len(out) >= limit:
            break
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            first = json.loads(lines[0])
            last = json.loads(lines[-1])
        except (OSError, IndexError, json.JSONDecodeError):
            continue
        if first.get("type") != ev.RUN_STARTED:
            continue
        data = first["data"]
        if workspace and data.get("workspace") != workspace:
            continue
        finished = last.get("type") == ev.RUN_FINISHED
        out.append({
            "id": path.stem, "task": data.get("task", ""), "workspace": data.get("workspace", ""),
            "model": data.get("model", ""), "started": first.get("ts"),
            "status": last["data"].get("status") if finished else "interrupted",
            "changes": len(last["data"].get("changes", [])) if finished else 0,
        })
    return out
