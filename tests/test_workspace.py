import pytest

from lex.workspace import Workspace, WorkspaceError


def test_resolve_refuses_escape(tmp_path):
    ws = Workspace(tmp_path)
    for bad in ["../x", "/etc/passwd", "a/../../b", ""]:
        with pytest.raises(WorkspaceError):
            ws.resolve(bad)
    assert ws.resolve("a/b.txt") == tmp_path.resolve() / "a" / "b.txt"


def test_symlink_escape_refused(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("x")
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(outside)
    ws = Workspace(root)
    with pytest.raises(WorkspaceError):
        ws.read_text("link/secret")
    assert ws.walk(".") == []


def test_change_tracking(tmp_path):
    (tmp_path / "keep.txt").write_text("one\n")
    (tmp_path / "gone.txt").write_text("bye\n")
    ws = Workspace(tmp_path)
    ws.write_text("keep.txt", "one\ntwo\n")
    ws.write_text("new/file.py", "print(1)\n")
    ws.delete("gone.txt")
    ws.write_text("temp.txt", "x")
    ws.delete("temp.txt")  # created then removed: not a change
    changes = {c.path: c for c in ws.changes()}
    assert set(changes) == {"keep.txt", "new/file.py", "gone.txt"}
    assert changes["keep.txt"].status == "modified"
    assert changes["keep.txt"].stats() == (1, 0)
    assert changes["new/file.py"].status == "added"
    assert changes["gone.txt"].status == "deleted"


def test_walk_skips_noise(tmp_path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("")
    paths = [p for p, _ in Workspace(tmp_path).walk(".")]
    assert "src/a.py" in paths
    assert not any(p.startswith("node_modules") for p in paths)


def test_binary_refused(tmp_path):
    (tmp_path / "b.bin").write_bytes(b"\x00\x01\x02")
    with pytest.raises(WorkspaceError):
        Workspace(tmp_path).read_text("b.bin")
