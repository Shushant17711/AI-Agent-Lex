"""End-to-end runs driven by the scripted provider (no network)."""

from lex import events as ev
from lex.agents import Run
from lex.config import Settings
from lex.providers.base import Turn
from lex.providers.scripted import ScriptedProvider, call
from lex.workspace import Workspace


def settings(**kw):
    base = dict(provider="openai", model="scripted", approve_writes="auto", approve_commands="auto")
    base.update(kw)
    return Settings(**base)


async def collect(run):
    events = []

    async def sink(e):
        events.append(e)

    run.bus.subscribe(sink)
    result = await run.execute()
    return result, events


async def test_full_cycle_with_review_round(tmp_path):
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    provider = ScriptedProvider({
        "planner": [
            Turn(tool_calls=[call("read_file", path="calc.py")]),
            Turn(tool_calls=[call("submit_plan", approach="Fix the sign.", steps=["Fix add()", "Run it"])]),
        ],
        "coder": [
            Turn(text="Fixing.", tool_calls=[call("update_step", step=1, status="in_progress")]),
            Turn(tool_calls=[call("edit_file", path="calc.py", old_text="a - b", new_text="a + b")]),
            Turn(tool_calls=[call("run_command", command="python3 -c 'import calc; print(calc.add(2, 3))'")]),
            Turn(tool_calls=[call("finish", summary="add() fixed.")]),
            # after review feedback
            Turn(tool_calls=[call("write_file", path="test_calc.py", content="from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n")]),
            Turn(tool_calls=[call("finish", summary="add() fixed and tested.")]),
        ],
        "reviewer": [
            Turn(tool_calls=[call("submit_review", approved=False, summary="No test.", issues=["Add a test"])]),
            Turn(tool_calls=[call("submit_review", approved=True, summary="Good.")]),
        ],
    })
    run = Run("fix add", settings(), Workspace(tmp_path), provider)
    result, events = await collect(run)

    assert result.status == "completed", result.summary
    assert result.summary == "add() fixed and tested."
    assert (tmp_path / "calc.py").read_text().endswith("a + b\n")
    assert {c["path"] for c in result.changes} == {"calc.py", "test_calc.py"}
    types = [e.type for e in events]
    assert types[0] == ev.RUN_STARTED and types[-1] == ev.RUN_FINISHED
    assert types.count(ev.REVIEW) == 2
    cmd = next(e for e in events if e.type == ev.TOOL_RESULT and e.data["name"] == "run_command")
    assert cmd.data["ok"] and "5" in cmd.data["preview"]
    # the reviewer saw the coder's feedback-round summary
    assert run.steps[0]["status"] == "in_progress"


async def test_denied_write_reaches_the_model(tmp_path):
    provider = ScriptedProvider({
        "coder": [
            Turn(tool_calls=[call("write_file", path="x.txt", content="hi")]),
            lambda log: Turn(tool_calls=[call("finish", summary=str(log[-1][1][0][1]))]),
        ],
    })
    run = Run("write x", settings(plan=False, review=False, approve_writes="deny"), Workspace(tmp_path), provider)
    result, _ = await collect(run)
    assert not (tmp_path / "x.txt").exists()
    assert "disabled" in result.summary
    assert result.changes == []


async def test_bad_tool_and_path_escape_are_reported_not_raised(tmp_path):
    provider = ScriptedProvider({
        "coder": [
            Turn(tool_calls=[call("nope"), call("read_file", path="../../etc/passwd"), call("read_file")]),
            Turn(tool_calls=[call("finish", summary="done")]),
        ],
    })
    run = Run("t", settings(plan=False, review=False), Workspace(tmp_path), provider)
    result, events = await collect(run)
    assert result.status == "completed"
    errors = [e.data["preview"] for e in events if e.type == ev.TOOL_RESULT]
    assert "Unknown tool" in errors[0]
    assert "outside the workspace" in errors[1]
    assert "Bad arguments" in errors[2]


async def test_coder_that_never_finishes(tmp_path):
    provider = ScriptedProvider({"coder": [Turn(text="I am thinking.")] * 5})
    run = Run("t", settings(plan=False, review=False), Workspace(tmp_path), provider)
    result, _ = await collect(run)
    assert result.status == "needs_attention"


async def test_command_timeout(tmp_path):
    provider = ScriptedProvider({
        "coder": [
            Turn(tool_calls=[call("run_command", command="sleep 5", timeout=1)]),
            lambda log: Turn(tool_calls=[call("finish", summary=str(log[-1][1][0][1]))]),
        ],
    })
    run = Run("t", settings(plan=False, review=False), Workspace(tmp_path), provider)
    result, _ = await collect(run)
    assert "Timed out" in result.summary
