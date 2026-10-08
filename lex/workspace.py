"""The workspace: the single directory the agents may touch, plus change tracking."""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path

IGNORED_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".tox", "dist", "build", ".next", ".idea", ".vscode", ".lex",
}
MAX_READ_BYTES = 400_000


class WorkspaceError(Exception):
    """Raised for invalid paths or operations; the message is shown to the model."""


@dataclass
class FileChange:
    path: str
    original: str | None  # None means the file did not exist when the run started
    current: str | None  # None means the file is now deleted

    @property
    def status(self) -> str:
        if self.original is None:
            return "added"
        if self.current is None:
            return "deleted"
        return "modified"

    def diff(self) -> str:
        a = (self.original or "").splitlines(keepends=True)
        b = (self.current or "").splitlines(keepends=True)
        lines = difflib.unified_diff(
            a, b,
            fromfile="/dev/null" if self.original is None else f"a/{self.path}",
            tofile="/dev/null" if self.current is None else f"b/{self.path}",
        )
        return "".join(line if line.endswith("\n") else line + "\n" for line in lines)

    def stats(self) -> tuple[int, int]:
        added = removed = 0
        for line in self.diff().splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif line.startswith("-") and not line.startswith("---"):
                removed += 1
        return added, removed

    def to_dict(self) -> dict:
        added, removed = self.stats()
        return {"path": self.path, "status": self.status, "added": added,
                "removed": removed, "diff": self.diff()}


class Workspace:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise WorkspaceError(f"Workspace is not a directory: {self.root}")
        self._changes: dict[str, FileChange] = {}

    # ---- paths -------------------------------------------------------------
    def resolve(self, path: str) -> Path:
        """Resolve a model-supplied path, refusing anything outside the workspace."""
        if not isinstance(path, str) or not path.strip():
            raise WorkspaceError("Path must be a non-empty string.")
        candidate = Path(path.strip())
        full = (candidate if candidate.is_absolute() else self.root / candidate).resolve()
        if full != self.root and self.root not in full.parents:
            raise WorkspaceError(f"Path '{path}' is outside the workspace.")
        return full

    def rel(self, full: Path) -> str:
        r = full.relative_to(self.root).as_posix()
        return r or "."

    # ---- reading -----------------------------------------------------------
    def read_text(self, path: str) -> str:
        full = self.resolve(path)
        if not full.exists():
            raise WorkspaceError(f"File not found: {path}")
        if full.is_dir():
            raise WorkspaceError(f"'{path}' is a directory, not a file.")
        if full.stat().st_size > MAX_READ_BYTES:
            raise WorkspaceError(f"'{path}' is larger than {MAX_READ_BYTES // 1000} KB; read a range or search it.")
        data = full.read_bytes()
        if b"\x00" in data[:8000]:
            raise WorkspaceError(f"'{path}' looks like a binary file.")
        return data.decode("utf-8", errors="replace")

    def walk(self, path: str = ".", max_depth: int = 3, limit: int = 2000) -> list[tuple[str, bool]]:
        """List (relative_path, is_dir) entries, skipping noise directories."""
        base = self.resolve(path)
        if not base.is_dir():
            raise WorkspaceError(f"'{path}' is not a directory.")
        out: list[tuple[str, bool]] = []

        def visit(d: Path, depth: int) -> None:
            try:
                entries = sorted(d.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except OSError:
                return
            for p in entries:
                if len(out) >= limit:
                    return
                if p.is_dir() and p.name in IGNORED_DIRS:
                    continue
                if p.is_symlink() and not p.resolve().is_relative_to(self.root):
                    continue
                out.append((self.rel(p), p.is_dir()))
                if p.is_dir() and depth < max_depth:
                    visit(p, depth + 1)

        visit(base, 1)
        return out

    # ---- writing (tracked) -------------------------------------------------
    def _remember(self, full: Path) -> FileChange:
        key = self.rel(full)
        if key not in self._changes:
            original = None
            if full.is_file():
                try:
                    original = full.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    original = None
            self._changes[key] = FileChange(key, original, original)
        return self._changes[key]

    def write_text(self, path: str, content: str) -> FileChange:
        full = self.resolve(path)
        if full.is_dir():
            raise WorkspaceError(f"'{path}' is a directory.")
        change = self._remember(full)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
        change.current = content
        return change

    def delete(self, path: str) -> FileChange:
        full = self.resolve(path)
        if not full.is_file():
            raise WorkspaceError(f"File not found: {path}")
        change = self._remember(full)
        full.unlink()
        change.current = None
        return change

    def changes(self) -> list[FileChange]:
        return [c for c in self._changes.values() if c.original != c.current]
