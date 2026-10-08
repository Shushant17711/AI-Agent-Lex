from starlette.testclient import TestClient

from lex.config import load_settings
from lex.server import create_app
from lex.workspace import Workspace


def test_settings_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    for k in ["LEX_PROVIDER", "LEX_MODEL", "LEX_BASE_URL", "LEX_APPROVE_WRITES", "LEX_APPROVE_COMMANDS"]:
        monkeypatch.delenv(k, raising=False)
    (tmp_path / "lex.toml").write_text('provider = "openai"\nmax-turns = 12\n')
    s = load_settings(tmp_path)
    assert s.provider == "openai" and s.model == "gpt-5-mini" and s.max_turns == 12
    monkeypatch.setenv("LEX_MODEL", "llama")
    assert load_settings(tmp_path).model == "llama"
    assert load_settings(tmp_path, model="cli").model == "cli"


def make_client(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    app = create_app(Workspace(tmp_path), token="tok")
    return TestClient(app, base_url="http://127.0.0.1:8765")


def test_token_and_host_guards(tmp_path):
    c = make_client(tmp_path)
    assert c.get("/").status_code == 401
    assert c.get("/?t=tok").status_code == 200
    assert c.get("/api/state").status_code == 401
    assert c.get("/api/state", headers={"x-lex-token": "tok"}).status_code == 200
    # DNS-rebinding style request: right token, foreign Host header
    assert c.get("/api/state", headers={"x-lex-token": "tok", "host": "evil.example"}).status_code == 400


def test_file_api_stays_in_workspace(tmp_path):
    c = make_client(tmp_path)
    h = {"x-lex-token": "tok"}
    assert c.get("/api/file?path=a.txt", headers=h).json()["content"] == "hello"
    assert c.get("/api/file?path=../../etc/passwd", headers=h).status_code == 400
    assert {"path": "a.txt", "dir": False} in c.get("/api/tree", headers=h).json()["entries"]


def test_websocket_rejects_foreign_origin(tmp_path):
    c = make_client(tmp_path)
    import pytest
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect("/ws?t=tok", headers={"origin": "http://evil.example"}) as ws:
            ws.receive_json()
