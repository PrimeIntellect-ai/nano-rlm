"""The quote-nesting hint: a nested-quote SyntaxError in an ipython cell earns a delimiter hint."""

from conftest import DummyClient, DummyMessage, DummyToolCall, make_runtime_config

from rlm.engine import MAX_QUOTE_NESTING_HINTS, RLMEngine


async def test_nested_quote_syntax_error_is_hinted_then_muted(session):
    bad = "src = r'''\n'''inner'''\n'''\nprint(1)"  # the inner ''' terminates the outer literal: cell-level SyntaxError
    client = DummyClient(
        [
            DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": bad})]),
            DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": bad})]),
            DummyMessage(
                tool_calls=[DummyToolCall("ipython", {"code": bad})]
            ),  # budget spent: no 3rd hint
            DummyMessage(
                tool_calls=[
                    DummyToolCall(
                        "ipython",
                        {
                            "code": "await rlm.hints.mute('quote-nesting'); print('muted')"
                        },
                    )
                ]
            ),
            DummyMessage(content="done"),
            DummyMessage(
                tool_calls=[
                    DummyToolCall(
                        "ipython", {"code": "await rlm.hints.unmute('quote-nesting')"}
                    )
                ]
            ),
            DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": bad})]),
            DummyMessage(content="done again"),
        ]
    )
    engine = RLMEngine(
        client=client, session=session, runtime_config=make_runtime_config()
    )  # type: ignore
    try:
        result = await engine.prompt("write a module")
        assert engine._quote_nesting_hints == MAX_QUOTE_NESTING_HINTS
        await engine.prompt("write another module")
        assert engine._quote_nesting_hints == 1
    finally:
        await engine.aclose()
    assert result.answer == "done"
    hints = [
        str(m.get("content", ""))
        for call in client.calls
        for m in call["messages"]
        if m.get("role") == "user"
        and "nested inside a Python string literal" in str(m.get("content", ""))
    ]
    last_call_hints = [
        m
        for m in client.calls[-1]["messages"]
        if m.get("role") == "user"
        and "nested inside a Python string literal" in str(m.get("content", ""))
    ]
    assert len(last_call_hints) == MAX_QUOTE_NESTING_HINTS + 1, [
        str(m.get("content"))[:80] for m in last_call_hints
    ]
    assert 'rlm.hints.mute("quote-nesting")' in hints[0]


async def test_multiline_command_in_plain_quotes_gets_the_command_hint(session):
    # A heredoc requires a multiline Python string.
    bad = "r = await rlm.shell.run(\"python3 - <<'EOF'\nprint(1)\nEOF\")"
    plain_mistake = 'name = "ok"\nif True print(name)'
    client = DummyClient(
        [
            DummyMessage(
                tool_calls=[DummyToolCall("ipython", {"code": plain_mistake})]
            ),
            DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": bad})]),
            DummyMessage(content="done"),
        ]
    )
    engine = RLMEngine(
        client=client, session=session, runtime_config=make_runtime_config()
    )  # type: ignore
    try:
        await engine.prompt("run a heredoc")
    finally:
        await engine.aclose()
    users = [
        str(m.get("content", ""))
        for m in client.calls[-1]["messages"]
        if m.get("role") == "user"
    ]
    command_hints = [
        u for u in users if "multi-line command inside a plain-quoted" in u
    ]
    assert len(command_hints) == 1 and "r'''...'''" in command_hints[0]
    assert not any("nested inside a Python string literal" in u for u in users)
    # the ordinary mistake earned nothing: only one hint in the whole conversation
    assert sum("quote-nesting" in u for u in users) == 1
