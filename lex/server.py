"""Local web UI: Starlette app + one websocket that streams run events and takes approvals."""

from __future__ import annotations

import asyncio
import secrets
import threading
import webbrowser
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

from lex import __version__
from lex import events as ev
from lex.approval import AUTO, MODES, FutureDecider
from lex.config import load_settings
from lex.history import Recorder, list_runs, load_run
from lex.providers import ProviderError, create_provider
from lex.workspace import Workspace, WorkspaceError

WEB = Path(__file__).parent / "web"
LOOPBACK = {"127.0.0.1", "localhost", "::1", "[::1]"}


class App:
    def __init__(self, workspace: Workspace, host: str, port: int, token: str) -> None:
        self.root = workspace.root
        self.host, self.port, self.token = host, port, token
        self.clients: set[WebSocket] = set()
        self.run = None
        self.task: asyncio.Task | None = None
        self.decider: FutureDecider | None = None

    # ---- guards ------------------------------------------------------------
    def host_ok(self, host_header: str | None) -> bool:
        if self.host not in LOOPBACK:
            return True  # user opted into a non-loopback bind; the token still applies
        name = (host_header or "").rsplit(":", 1)[0] if not (host_header or "").startswith("[") else "[::1]"
        return name in LOOPBACK

    def token_ok(self, value: str | None) -> bool:
        return bool(value) and secrets.compare_digest(value, self.token)

    # ---- broadcasting ------------------------------------------------------
    async def broadcast(self, event: ev.Event) -> None:
        payload = {"kind": "event", "event": event.to_dict()}
        for ws in list(self.clients):
            try:
                await ws.send_json(payload)
            except Exception:
                self.clients.discard(ws)

    def busy(self) -> bool:
        return self.task is not None and not self.task.done()

    # ---- run control -------------------------------------------------------
    async def start(self, msg: dict[str, Any]) -> str | None:
        from lex.agents import Run

        if self.busy():
            return "A run is already in progress."
        task = str(msg.get("task", "")).strip()
        if not task:
            return "Describe a task first."
        overrides: dict[str, Any] = {k: bool(msg[k]) for k in ("plan", "review") if k in msg}
        for k in ("approve_writes", "approve_commands"):
            if msg.get(k) in MODES:
                overrides[k] = msg[k]
        if msg.get("model"):
            overrides["model"] = str(msg["model"])[:100]
        try:
            settings = load_settings(self.root, **overrides)
            provider = create_provider(settings)
        except (ProviderError, ValueError) as e:
            return str(e)
        self.decider = FutureDecider()
        run = Run(task, settings, Workspace(self.root), provider, decider=self.decider)
        run.bus.subscribe(Recorder(run.id))
        run.bus.subscribe(self.broadcast)
        self.run = run
        self.task = asyncio.create_task(self._execute(run))
        return None

    async def _execute(self, run) -> None:
        try:
            await run.execute()
        except asyncio.CancelledError:
            pass
        except Exception as e:  # surface unexpected crashes instead of hanging the UI
            await run.bus.emit(ev.ERROR, None, message=f"Internal error: {type(e).__name__}: {e}")
            await run.bus.emit(ev.RUN_FINISHED, None, status="failed", summary=str(e),
                               changes=[c.to_dict() for c in run.workspace.changes()],
                               input_tokens=run.usage.input_tokens, output_tokens=run.usage.output_tokens)

    def approve(self, msg: dict[str, Any]) -> None:
        if not (self.decider and self.run):
            return
        allow = bool(msg.get("allow"))
        if allow and msg.get("always") and msg.get("scope") in ("write", "exec"):
            policy = self.run.approvals.policy
            if msg["scope"] == "write":
                policy.writes = AUTO
            else:
                policy.commands = AUTO
        self.decider.resolve(str(msg.get("id")), allow)

    def cancel(self) -> None:
        if self.busy():
            if self.decider:
                self.decider.cancel_all()
            self.task.cancel()


def create_app(workspace: Workspace, host: str = "127.0.0.1", port: int = 8765, token: str | None = None) -> Starlette:
    state = App(workspace, host, port, token or secrets.token_urlsafe(18))

    def guard(request: Request) -> Response | None:
        if not state.host_ok(request.headers.get("host")):
            return PlainTextResponse("Bad host", status_code=400)
        if request.url.path.startswith("/api/") and not state.token_ok(request.headers.get("x-lex-token")):
            return PlainTextResponse("Unauthorized", status_code=401)
        return None

    async def index(request: Request) -> Response:
        if (bad := guard(request)) is not None:
            return bad
        if not state.token_ok(request.query_params.get("t")):
            return PlainTextResponse("Open Lex with the link printed in your terminal (it includes an access token).",
                                     status_code=401)
        return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-store"})

    async def api_state(request: Request) -> Response:
        if (bad := guard(request)) is not None:
            return bad
        settings = load_settings(state.root)
        current = None
        if state.run is not None:
            current = {"id": state.run.id, "active": state.busy(), "events": [e.to_dict() for e in state.run.bus.log]}
        return JSONResponse({"version": __version__, "workspace": str(state.root), "settings": settings.public(),
                             "current": current})

    async def api_tree(request: Request) -> Response:
        if (bad := guard(request)) is not None:
            return bad
        ws = Workspace(state.root)
        return JSONResponse({"entries": [{"path": p, "dir": d} for p, d in ws.walk(".", max_depth=8, limit=4000)]})

    async def api_file(request: Request) -> Response:
        if (bad := guard(request)) is not None:
            return bad
        try:
            content = Workspace(state.root).read_text(request.query_params.get("path", ""))
        except (WorkspaceError, OSError) as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse({"content": content})

    async def api_runs(request: Request) -> Response:
        if (bad := guard(request)) is not None:
            return bad
        return JSONResponse({"runs": list_runs(str(state.root), limit=40)})

    async def api_run(request: Request) -> Response:
        if (bad := guard(request)) is not None:
            return bad
        return JSONResponse({"events": load_run(request.path_params["run_id"])})

    async def socket(ws: WebSocket) -> None:
        origin = ws.headers.get("origin", "")
        allowed_origins = {f"http://{h}:{state.port}" for h in ("127.0.0.1", "localhost", "[::1]")}
        if state.host not in LOOPBACK:
            allowed_origins.add(f"http://{ws.headers.get('host', '')}")
        if (origin not in allowed_origins or not state.host_ok(ws.headers.get("host"))
                or not state.token_ok(ws.query_params.get("t"))):
            await ws.close(code=4403)
            return
        await ws.accept()
        state.clients.add(ws)
        try:
            while True:
                msg = await ws.receive_json()
                kind = msg.get("kind")
                if kind == "start":
                    error = await state.start(msg)
                    if error:
                        await ws.send_json({"kind": "error", "message": error})
                elif kind == "approve":
                    state.approve(msg)
                elif kind == "cancel":
                    state.cancel()
        except (WebSocketDisconnect, RuntimeError, ValueError):
            pass
        finally:
            state.clients.discard(ws)

    app = Starlette(routes=[
        Route("/", index),
        Route("/api/state", api_state),
        Route("/api/tree", api_tree),
        Route("/api/file", api_file),
        Route("/api/runs", api_runs),
        Route("/api/runs/{run_id}", api_run),
        WebSocketRoute("/ws", socket),
        Mount("/static", StaticFiles(directory=WEB), name="static"),
    ])
    app.state.lex = state
    return app


def serve(workspace: Workspace, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    import uvicorn
    from rich.console import Console

    token = secrets.token_urlsafe(18)
    app = create_app(workspace, host, port, token)
    shown_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    url = f"http://{shown_host}:{port}/?t={token}"
    console = Console()
    console.print(f"\n  [bold #f2a93b]lex[/] {__version__}  ·  [dim]{workspace.root}[/]\n  → [link={url}]{url}[/link]\n")
    if host not in LOOPBACK:
        console.print("  [#e5604d]Warning: listening beyond localhost. Anyone with the link can run commands here.[/]\n")
    if open_browser:
        threading.Timer(0.8, webbrowser.open, [url]).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
