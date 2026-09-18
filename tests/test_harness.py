"""Tests for the continual harness store, its kernel view, and prompt rendering."""

from __future__ import annotations

import json

import pytest
from conftest import (
    DummyClient,
    DummyMessage,
    DummyToolCall,
    make_runtime_config,
    tool_result,
)

from rlm.config import ExecutionPolicy, HarnessConfig, InvocationContext, RuntimeConfig
from rlm.engine import RLMEngine
from rlm.harness import (
    HarnessStore,
    build_view,
    harness,
    local_dir,
    query_terms,
)
from rlm.prompt import HARNESS_INTRO, render_harness
from rlm.session import Session
from rlm.supervisor import SessionTreeSupervisor
from rlm.types import RLMResult


def test_store_crud_and_versioning(tmp_path):
    store = HarnessStore(tmp_path / "h")
    entry = store.create("memory", "Check git status", "run git status first")
    assert entry.id == "check_git_status" and entry.version == 1
    with pytest.raises(ValueError, match="already exists"):
        store.create("memory", "Check git status", "again")

    updated = store.update("memory", entry.id, "Check git status", "run it twice")
    assert updated.version == 2 and updated.content == "run it twice"
    assert updated.path == "general"  # omitted fields are preserved

    with pytest.raises(ValueError, match="does not exist"):
        store.update("memory", "missing", "t", "c")
    with pytest.raises(ValueError, match="unknown harness kind"):
        store.create("note", "t", "c")  # type: ignore[arg-type]

    data = json.loads((tmp_path / "h" / "harness_state.json").read_text())
    assert data["schema"] == 1
    assert list(data["entries"]) == ["prompt", "memory", "skill", "subagent"]
    assert data["entries"]["memory"]["check_git_status"]["version"] == 2

    assert store.delete("memory", entry.id) is True
    assert store.delete("memory", entry.id) is False
    assert store.list() == []


def test_skill_entries_require_a_python_reference(tmp_path):
    store = HarnessStore(tmp_path / "h")
    with pytest.raises(ValueError, match="Python reference"):
        store.create("skill", "Search", "use websearch")
    with pytest.raises(ValueError, match="callable or call_pattern"):
        store.create(
            "skill", "Search", "x", reference={"type": "python", "import": "websearch"}
        )
    entry = store.create(
        "skill",
        "Search",
        "use websearch",
        reference={"type": "python", "import": "websearch", "callable": "run"},
        arguments={"queries": {"type": "array", "required": True}},
    )
    # A content-only update keeps the reference and argument contract.
    updated = store.update("skill", entry.id, "Search", "use websearch for releases")
    assert updated.reference["import"] == "websearch"
    assert updated.arguments["queries"]["required"] is True


def test_two_stores_on_one_file_never_clobber_each_other(tmp_path):
    first = HarnessStore(tmp_path / "h")
    second = HarnessStore(tmp_path / "h")
    first.create("memory", "From first", "a")
    second.create("memory", "From second", "b")
    assert {e.id for e in first.list("memory")} == {"from_first", "from_second"}
    assert {e.id for e in second.list("memory")} == {"from_first", "from_second"}
    first.record_refinement("test", ["create memory:from_first"])
    assert [e.trigger for e in second.list_refinements()] == ["test"]


def test_view_layers_prefixes_and_ancestor_read_only(tmp_path):
    ancestor = HarnessStore(tmp_path / "parent")
    ancestor.create("memory", "Parent lesson", "use uv run")
    view = build_view(
        tmp_path / "child",
        global_dir=tmp_path / "global",
        ancestor_dirs=[tmp_path / "parent"],
    )
    view.create_memory("Child lesson", "own note")
    view.create_prompt_note("Shared policy", "never guess", global_=True)

    layers = [(layer, e.id) for layer, e in view.entries("memory")]
    assert layers == [("local", "child_lesson"), ("ancestor", "parent_lesson")]
    assert view.get("memory", "ancestor:parent_lesson").content == "use uv run"
    assert view.get("memory", "local:parent_lesson") is None
    assert view.get("prompt", "shared_policy").scope == "global"

    with pytest.raises(PermissionError):
        view.update_memory("ancestor:parent_lesson", "x", "y")
    with pytest.raises(PermissionError):
        view.delete_memory("ancestor:parent_lesson")
    # A global: prefix routes the edit to the global store without global_=True.
    view.update_prompt_note("global:shared_policy", "Shared policy", "verify first")
    assert view.global_.get("prompt", "shared_policy").version == 2

    overview = view.overview()
    assert (
        "[ancestor:parent_lesson]" in overview and "[global:shared_policy]" in overview
    )
    assert view.counts()["ancestors"] == 1
    assert view.counts()["global"]["prompt"] == 1


def test_view_without_global_rejects_global_writes(tmp_path):
    view = build_view(tmp_path / "local")
    with pytest.raises(RuntimeError, match="no global harness store"):
        view.create_memory("x", "y", global_=True)
    assert view.counts()["global"] is None


def test_query_terms_and_search_ranking(tmp_path):
    assert query_terms("worktree? rlm api 修复登录 naïve мир a") == [
        "worktree",
        "rlm",
        "api",
        "修复",
        "复登",
        "登录",
        "naïve",
        "мир",
    ]
    view = build_view(tmp_path / "h")
    view.create_memory("Pytest imports", "run pytest through the project venv")
    view.create_memory("Git status", "check git status before committing")
    view.create_memory("Pytest venv", "pytest and venv both appear here", path="pytest")
    ranked = [e.id for e in view.search("pytest venv")]
    assert ranked[0] == "pytest_venv"
    assert set(ranked) == {"pytest_venv", "pytest_imports"}
    assert view.search("") == []
    with pytest.raises(TypeError):
        view.search("x", limit=0)


def test_render_harness_caps_and_ranks(tmp_path):
    view = build_view(tmp_path / "h")
    assert "No saved harness entries yet." in render_harness(view)

    for index in range(8):
        view.create_memory(f"Note {index}", f"about {'pytest' if index == 6 else 'x'}")
    view.create_subagent("Reviewer", "review a diff for correctness")
    view.record_refinement("kernel: repeated failure", ["create memory:note_6"])

    block = render_harness(view, query="run pytest", can_delegate=True)
    assert block.startswith(HARNESS_INTRO)
    lines = block.splitlines()
    memory_lines = [line for line in lines if line.startswith("- [local:note_")]
    assert len(memory_lines) == 6
    assert memory_lines[0].startswith("- [local:note_6]")
    assert "- +2 more memory entries" in lines
    assert "ranked by relevance" in block
    assert "await rlm.agent.spawn(task, name=...)" in block
    assert "- [local:reviewer] Reviewer" in block
    assert "recent refinements: 1" in block
    assert "kernel: repeated failure: create memory:note_6" in block

    leaf = render_harness(view, can_delegate=False, has_ipython=False)
    assert "rlm.agent.spawn" not in leaf and "rlm.harness.harness()" not in leaf


def test_harness_config_and_ancestor_context():
    config = make_runtime_config()
    assert config.harness == HarnessConfig()
    assert config.harness.enabled and config.harness.global_dir is None

    context = (
        InvocationContext().child("/s/root/harness").child("/s/root/sub-a/harness")
    )
    assert context.depth == 2
    assert context.ancestor_harness_dirs == ("/s/root/sub-a/harness", "/s/root/harness")
    assert InvocationContext().child(None).ancestor_harness_dirs == ()

    with pytest.raises(ValueError):
        HarnessConfig(max_prompt_entries_per_kind=0)
    with pytest.raises(ValueError):
        HarnessConfig(unknown=True)  # type: ignore[call-arg]


def test_harness_env_resolution(tmp_path, monkeypatch):
    monkeypatch.delenv("RLM_HARNESS_LOCAL_DIR", raising=False)
    with pytest.raises(RuntimeError, match="disabled"):
        harness()
    explicit = harness(session_dir=tmp_path / "s")
    assert explicit.local.dir == local_dir(tmp_path / "s").resolve()

    monkeypatch.setenv("RLM_HARNESS_LOCAL_DIR", str(tmp_path / "l"))
    monkeypatch.setenv("RLM_HARNESS_GLOBAL_DIR", "")
    monkeypatch.setenv(
        "RLM_HARNESS_ANCESTOR_DIRS", str(tmp_path / "a1") + ":" + str(tmp_path / "a2")
    )
    view = harness()
    assert view.global_ is None
    assert [s.dir.name for s in view.ancestors] == ["a1", "a2"]


async def test_engine_renders_harness_and_kernel_writes_to_it(session, tmp_path):
    """The kernel's rlm.harness view targets the session's store, and the system
    prompt carries the harness block with what was already recorded."""
    global_dir = tmp_path / "global"
    HarnessStore(global_dir, scope="global").create(
        "prompt", "Global policy", "verify before claiming success"
    )
    code = (
        "h = rlm.harness.harness()\n"
        "h.create_memory('Kernel lesson', 'written from the kernel')\n"
        "print(h.get('prompt', 'global:global_policy').content)\n"
        "print(h.local.dir)"
    )
    client = DummyClient(
        [
            DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": code})]),
            DummyMessage(content="ok"),
        ]
    )
    config = make_runtime_config(harness=HarnessConfig(global_dir=str(global_dir)))
    engine = RLMEngine(client=client, session=session, runtime_config=config)  # type: ignore

    result = await engine.run("remember things")

    output = tool_result(client)
    assert "verify before claiming success" in output
    assert str(local_dir(session.dir).resolve()) in output
    system_prompt = client.calls[0]["messages"][0]["content"]
    assert HARNESS_INTRO in system_prompt
    assert "[global:global_policy] Global policy" in system_prompt
    assert result.answer == "ok"

    stored = json.loads((local_dir(session.dir) / "harness_state.json").read_text())
    assert (
        stored["entries"]["memory"]["kernel_lesson"]["content"]
        == "written from the kernel"
    )
    assert engine.execution_snapshot()["limits"]["harness_global"] is True


async def test_disabled_harness_leaves_prompt_and_kernel_untouched(session):
    client = DummyClient(
        [
            DummyMessage(
                tool_calls=[
                    DummyToolCall(
                        "ipython",
                        {
                            "code": "import os; print(os.environ.get('RLM_HARNESS_LOCAL_DIR'))"
                        },
                    )
                ]
            ),
            DummyMessage(content="ok"),
        ]
    )
    config = make_runtime_config(harness=HarnessConfig(enabled=False))
    engine = RLMEngine(client=client, session=session, runtime_config=config)  # type: ignore

    await engine.run("no harness")

    assert tool_result(client).strip() == "None"
    assert HARNESS_INTRO not in client.calls[0]["messages"][0]["content"]
    assert not local_dir(session.dir).exists()
    assert engine.execution_snapshot()["limits"]["harness_enabled"] is False


def test_runtime_config_copies_harness_to_children():
    config = RuntimeConfig(
        **{
            **make_runtime_config().model_dump(),
            "harness": HarnessConfig(global_dir="/g"),
        }
    )
    child = config.model_copy(update={"invocation": config.invocation.child("/p")})
    assert child.harness.global_dir == "/g"
    assert child.invocation.ancestor_harness_dirs == ("/p",)


async def test_spawned_child_inherits_parent_store_as_ancestor(tmp_path):
    """The supervisor hands each child its parent's local harness directory."""
    seen: list[tuple[int, tuple[str, ...]]] = []

    class Engine:
        def __init__(self, *, runtime_config, session, **kwargs):
            self.runtime_config = runtime_config
            self.session = session

        async def prompt(self, prompt: str) -> RLMResult:
            context = self.runtime_config.invocation
            seen.append((context.depth, context.ancestor_harness_dirs))
            return RLMResult(answer="done", session_dir=self.session.dir)

        async def aclose(self) -> None:
            pass

    session = Session(tmp_path / "root")
    config = make_runtime_config(
        policy=ExecutionPolicy(max_depth=2),
        harness=HarnessConfig(global_dir=str(tmp_path / "global")),
    )
    supervisor = SessionTreeSupervisor(
        root_session=session,
        runtime_config=config,
        cwd=str(tmp_path),
        engine_factory=Engine,
    )
    await supervisor.start()
    scope = await supervisor.open_scope(supervisor.root_id)
    endpoint = supervisor.endpoint_for(supervisor.root_id)
    try:
        child = supervisor._spawn(
            supervisor._caller(endpoint.capability, scope), scope, "task", None, False
        )
        await child.done.wait()
    finally:
        await supervisor.close_scope(scope)
        await supervisor.aclose()
        session.close()

    assert seen == [(1, (str(local_dir(session.dir)),))]
    assert child.runtime_config.harness.global_dir == str(tmp_path / "global")
