"""Command line interface: `lex run`, `lex ui`, `lex history`, `lex show`."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import sys
from pathlib import Path

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from lex import __version__
from lex import events as ev
from lex.approval import AUTO, ApprovalRequest
from lex.config import load_settings
from lex.providers import ProviderError, create_provider
from lex.workspace import Workspace, WorkspaceError

console = Console(highlight=False)
AGENT_STYLE = {"planner": "#f2a93b", "coder": "#5fb3a8", "reviewer": "#d98aa6", None: "#a8a196"}
STATUS_STYLE = {"completed": "bold #8fbf6a", "needs_attention": "bold #f2a93b",
                "failed": "bold #e5604d", "cancelled": "bold #a8a196", "interrupted": "#a8a196"}


def _arg_preview(name: str, args: dict) -> str:
    if name == "run_command":
        return str(args.get("command", ""))
    if "path" in args and name != "search":
        return str(args["path"])
    if name == "search":
        return f"/{args.get('pattern', '')}/ in {args.get('path', '.')}"
    if name == "update_step":
        return f"step {args.get('step')} → {args.get('status')}"
    return ", ".join(f"{k}={str(v)[:40]}" for k, v in args.items())


class Renderer:
    """Turns events into terminal output."""

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.calls: dict[str, str] = {}

    async def __call__(self, e: ev.Event) -> None:
        self.render(e.type, e.agent, e.data)

    def label(self, agent: str | None) -> Text:
        return Text(f"{(agent or 'lex'):>8} ", style=f"bold {AGENT_STYLE.get(agent, AGENT_STYLE[None])}")

    def render(self, kind: str, agent: str | None, d: dict) -> None:
        c = console
        if kind == ev.RUN_STARTED:
            c.print(Panel(Group(Text(d["task"], style="bold"),
                                Text(f"{d['workspace']}  ·  {d['provider']}/{d['model']}", style="dim")),
                          title="[bold #f2a93b]lex[/]", title_align="left", border_style="#3a352e"))
        elif kind == ev.PHASE:
            c.print(Rule(Text(f" {d['title']} ", style=f"bold {AGENT_STYLE.get(agent)}"), style="#3a352e"))
        elif kind == ev.MESSAGE:
            c.print(self.label(agent), end="")
            c.print(Markdown(d["text"]))
        elif kind == ev.TOOL_CALL:
            self.calls[d["id"]] = d["name"]
            if d["name"] in ("finish", "submit_plan", "submit_review", "update_step"):
                return
            c.print(Text("         ↳ ", style="dim") + Text(d["name"], style="bold dim") +
                    Text(f"  {_arg_preview(d['name'], d['args'])}", style="dim"))
        elif kind == ev.TOOL_RESULT:
            if not d["ok"]:
                c.print(Text(f"           ✗ {d['preview'].splitlines()[0][:200] if d['preview'] else 'failed'}",
                             style="#e5604d"))
            elif d["name"] == "run_command" or self.verbose:
                lines = d["preview"].splitlines()
                shown = lines[-12:] if not self.verbose else lines[:60]
                for line in shown:
                    c.print(Text(f"           │ {line}", style="#77706a"))
        elif kind == ev.PLAN:
            if d.get("approach"):
                c.print(Text("         ") + Text(d["approach"], style="italic #c9c2b4"))
            for i, step in enumerate(d["steps"], 1):
                c.print(Text(f"         {i:>2}. ", style="#f2a93b") + Text(step["text"]))
        elif kind == ev.STEP:
            mark = {"done": ("✓", "#8fbf6a"), "in_progress": ("…", "#5fb3a8"), "skipped": ("–", "dim")}.get(d["status"])
            if mark:
                c.print(Text(f"         {mark[0]} step {d['index'] + 1} {d['status'].replace('_', ' ')}", style=mark[1]) +
                        (Text(f" — {d['note']}", style="dim") if d.get("note") else Text()))
        elif kind == ev.FILE_CHANGED:
            c.print(Text(f"         ✎ {d['path']} ", style="#5fb3a8") +
                    Text(f"+{d['added']}", style="#8fbf6a") + Text(f" -{d['removed']}", style="#e5604d"))
        elif kind == ev.REVIEW:
            body = [Text(d["summary"])] + [Text(f"• {i}", style="#e8c39a") for i in d["issues"]]
            ok = d["approved"]
            c.print(Panel(Group(*body), title="approved" if ok else "changes requested",
                          title_align="left", border_style="#8fbf6a" if ok else "#f2a93b"))
        elif kind == ev.ERROR:
            c.print(Text(f"         ! {d['message']}", style="bold #e5604d"))
        elif kind == ev.RUN_FINISHED:
            table = Table.grid(padding=(0, 2))
            for ch in d["changes"]:
                table.add_row(Text(ch["status"], style="dim"), Text(ch["path"]),
                              Text(f"+{ch['added']}", style="#8fbf6a"), Text(f"-{ch['removed']}", style="#e5604d"))
            parts = [Markdown(d["summary"] or "(no summary)")]
            if d["changes"]:
                parts += [Text(""), table]
            parts.append(Text(f"\n{d['input_tokens']:,} in · {d['output_tokens']:,} out tokens", style="dim"))
            c.print(Panel(Group(*parts), title=Text(d["status"].replace("_", " "), style=STATUS_STYLE[d["status"]]),
                          title_align="left", border_style="#3a352e"))


def make_cli_decider(run_ref: list):
    async def decide(req: ApprovalRequest) -> bool:
        detail = (Syntax(req.detail, "diff", theme="ansi_dark", word_wrap=True) if req.kind == "write"
                  else Syntax(req.detail, "bash", theme="ansi_dark", word_wrap=True))
        title = f"[bold]{req.title}[/]" + (f"  [#e5604d]{req.reason}[/]" if req.reason else "")
        console.print(Panel(detail, title=title, title_align="left", border_style="#f2a93b"))
        answer = await asyncio.to_thread(
            Prompt.ask, "  Allow?  [dim](y)es · (n)o · (a)lways for this run[/]",
            choices=["y", "n", "a"], default="y", console=console, show_choices=False)
        if answer == "a" and not req.reason:
            policy = run_ref[0].approvals.policy
            if req.kind == "write":
                policy.writes = AUTO
            else:
                policy.commands = AUTO
        return answer in ("y", "a")

    return decide


async def _run(args: argparse.Namespace) -> int:
    from lex.agents import Run
    from lex.history import Recorder

    workspace = Workspace(args.workspace)
    overrides = dict(provider=args.provider, model=args.model, base_url=args.base_url)
    if args.no_plan:
        overrides["plan"] = False
    if args.no_review:
        overrides["review"] = False
    if args.yes:
        overrides.update(approve_writes="auto", approve_commands="auto")
    elif args.auto_edit:
        overrides["approve_writes"] = "auto"
    if args.read_only:
        overrides.update(approve_writes="deny", approve_commands="deny")
    settings = load_settings(workspace.root, **overrides)

    task = " ".join(args.task).strip()
    if not task:
        if not sys.stdin.isatty():
            task = sys.stdin.read().strip()
        else:
            task = Prompt.ask("[bold #f2a93b]What should Lex do?[/]", console=console).strip()
    if not task:
        console.print("[#e5604d]No task given.[/]")
        return 2

    provider = create_provider(settings)
    run_ref: list = []
    decider = make_cli_decider(run_ref) if sys.stdin.isatty() else None
    run = Run(task, settings, workspace, provider, decider=decider)
    run_ref.append(run)
    run.bus.subscribe(Recorder(run.id))
    run.bus.subscribe(Renderer(verbose=args.verbose))
    try:
        result = await run.execute()
    except asyncio.CancelledError:
        return 130
    return 0 if result.status == "completed" else 1


def _history(args: argparse.Namespace) -> int:
    from lex.history import list_runs

    ws = str(Path(args.workspace).expanduser().resolve()) if args.workspace else None
    runs = list_runs(ws, limit=args.limit)
    if not runs:
        console.print("[dim]No runs yet.[/]")
        return 0
    table = Table(box=None, header_style="dim", pad_edge=False)
    for col in ("id", "when", "status", "files", "task"):
        table.add_column(col)
    for r in runs:
        when = dt.datetime.fromtimestamp(r["started"]).strftime("%b %d %H:%M") if r["started"] else ""
        table.add_row(r["id"], when, Text(r["status"].replace("_", " "), style=STATUS_STYLE.get(r["status"], "")),
                      str(r["changes"]), r["task"][:70].replace("\n", " "))
    console.print(table)
    return 0


def _show(args: argparse.Namespace) -> int:
    from lex.history import load_run

    events = load_run(args.run_id)
    if not events:
        console.print(f"[#e5604d]No run with id {args.run_id}[/]")
        return 1
    r = Renderer(verbose=args.verbose)
    for e in events:
        if e["type"] not in (ev.APPROVAL_REQUEST, ev.APPROVAL_RESOLVED, ev.USAGE):
            r.render(e["type"], e.get("agent"), e["data"])
    return 0


def _ui(args: argparse.Namespace) -> int:
    from lex.server import serve

    serve(Workspace(args.workspace), host=args.host, port=args.port, open_browser=not args.no_open)
    return 0


COMMANDS = {"run", "ui", "history", "show"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lex", description="A planner, a coder and a reviewer that work on your project together.")
    p.add_argument("--version", action="version", version=f"lex {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Run a task in the terminal")
    r.add_argument("task", nargs="*", help="What to do. Omit to be prompted (or pipe it on stdin).")
    r.add_argument("-w", "--workspace", default=".", help="Project directory (default: current directory)")
    r.add_argument("--provider", choices=["gemini", "openai"])
    r.add_argument("-m", "--model")
    r.add_argument("--base-url", help="Base URL for an OpenAI-compatible API")
    r.add_argument("--no-plan", action="store_true", help="Skip the planner")
    r.add_argument("--no-review", action="store_true", help="Skip the reviewer")
    r.add_argument("--auto-edit", action="store_true", help="Apply file changes without asking")
    r.add_argument("-y", "--yes", action="store_true", help="Apply file changes and run commands without asking "
                   "(destructive commands still ask)")
    r.add_argument("--read-only", action="store_true", help="Refuse all writes and commands")
    r.add_argument("-v", "--verbose", action="store_true", help="Show tool output")

    u = sub.add_parser("ui", help="Open the web interface")
    u.add_argument("-w", "--workspace", default=".")
    u.add_argument("--host", default="127.0.0.1")
    u.add_argument("--port", type=int, default=8765)
    u.add_argument("--no-open", action="store_true", help="Don't open a browser")

    h = sub.add_parser("history", help="List past runs")
    h.add_argument("-w", "--workspace", help="Only runs in this workspace")
    h.add_argument("-n", "--limit", type=int, default=20)

    s = sub.add_parser("show", help="Replay a past run")
    s.add_argument("run_id")
    s.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in COMMANDS and not argv[0].startswith("-"):
        argv.insert(0, "run")  # `lex "fix the tests"` is shorthand for `lex run "fix the tests"`
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return asyncio.run(_run(args))
        if args.command == "ui":
            return _ui(args)
        if args.command == "history":
            return _history(args)
        return _show(args)
    except (ProviderError, WorkspaceError, ValueError) as e:
        console.print(f"[bold #e5604d]error:[/] {e}")
        return 2
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/]")
        return 130
