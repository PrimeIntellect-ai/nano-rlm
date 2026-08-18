"""Tests for tools: OpenAI-style tool calls dispatched by the engine.

Drives ``rlm.engine.RLMEngine`` with a scripted DummyClient and a dummy
``add(a, b)`` tool (registered per-test, not part of the package) to
exercise how the harness reacts to model tool-call behavior.
"""

from __future__ import annotations

import pytest
from conftest import (
    DummyClient,
    DummyMessage,
    DummyToolCall,
    show_tool_result,
    tool_result,
)

from rlm.engine import RLMEngine
from rlm.tools.ipython import IPythonREPL


async def test_valid_tool(session):
    """Valid tool call: engine dispatches add() and feeds the result back."""
    prompt = "add 2 and 3"
    messages = [
        DummyMessage(tool_calls=[DummyToolCall("add", {"a": 2, "b": 3})]),
        DummyMessage(content="the sum is 5"),
    ]

    client = DummyClient(messages)
    engine = RLMEngine(client=client, session=session)  # type: ignore

    result = await engine.run(prompt)

    show_tool_result(tool_result(client))
    assert tool_result(client) == "5"
    assert result.answer == "the sum is 5"
    assert result.turns == 2


async def test_multiple_tool_calls(session):
    """Each tool_call_id in a multi-call assistant message receives a tool result."""
    prompt = "add things"
    messages = [
        DummyMessage(
            tool_calls=[
                DummyToolCall("add", {"a": 1, "b": 2}, id="call_0"),
                DummyToolCall("add", {"a": 3, "b": 4}, id="call_1"),
            ]
        ),
        DummyMessage(content=""),
    ]

    client = DummyClient(messages)
    engine = RLMEngine(client=client, session=session)  # type: ignore

    await engine.run(prompt)

    tool_messages = [m for m in client.calls[1]["messages"] if m.get("role") == "tool"]
    assert sorted(m["tool_call_id"] for m in tool_messages) == ["call_0", "call_1"]


async def test_tool_raises(session):
    """Tool raising from execute() propagates out of engine.run()."""
    prompt = "set off the boom tool"
    messages = [DummyMessage(tool_calls=[DummyToolCall("boom", {})])]

    client = DummyClient(messages)
    engine = RLMEngine(client=client, session=session)  # type: ignore

    with pytest.raises(RuntimeError, match="boom"):
        await engine.run(prompt)


def test_ipython_kernel_does_not_inherit_parent_stdio(session, capfd):
    """Kernel writes outside IOPub must not corrupt an enclosing ACP transport."""
    repl = IPythonREPL(cwd=str(session.dir), session=session)
    repl.start()
    try:
        repl.execute(
            "import os; "
            "stdout_written = os.write(1, b'123\\n'); "
            "stderr_written = os.write(2, b'kernel stderr\\n')"
        )
        result = repl.execute("print(stdout_written, stderr_written)")
    finally:
        repl.shutdown()

    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert result.strip() == "4 14"


def test_real_kernel_and_subprocess_receive_only_explicit_environment(
    monkeypatch, session
):
    from jupyter_client import KernelManager

    original_init = KernelManager.__init__

    def init_with_host_kernel_environment(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.kernel_spec.env = {
            "KERNELSPEC_SENTINEL": "hidden",
            "RLM_API_KEY": "kernelspec-secret",
        }

    monkeypatch.setattr(KernelManager, "__init__", init_with_host_kernel_environment)
    monkeypatch.setenv("SUPERVISOR_ONLY", "hidden")
    monkeypatch.setenv("SERPER_API_KEY", "search-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "ambient-cloud-secret")
    repl = IPythonREPL(
        cwd=str(session.dir),
        session=session,
        kernel_env={"TASK_VISIBLE": "yes"},
    )
    repl.start()
    code = f"""
import os, subprocess
child_env = subprocess.check_output(['env'], text=True)
print(os.environ.get('TASK_VISIBLE'))
print(all(name not in os.environ for name in ('SUPERVISOR_ONLY', 'SERPER_API_KEY', 'AWS_SECRET_ACCESS_KEY', 'KERNELSPEC_SENTINEL', 'RLM_API_KEY')))
print(all(f'{{name}}=' not in child_env for name in ('SUPERVISOR_ONLY', 'SERPER_API_KEY', 'AWS_SECRET_ACCESS_KEY', 'KERNELSPEC_SENTINEL', 'RLM_API_KEY')))
print(all(os.environ.get(name, '').startswith({repl._ipc_dir!r}) for name in ('IPYTHONDIR', 'JUPYTER_CONFIG_DIR', 'JUPYTER_DATA_DIR', 'JUPYTER_RUNTIME_DIR')))
"""
    try:
        first = repl.execute(code)
        repl.restart_kernel()
        second = repl.execute(code)
    finally:
        repl.shutdown()

    assert first.strip().splitlines() == ["yes", "True", "True", "True"]
    assert second.strip().splitlines() == ["yes", "True", "True", "True"]
