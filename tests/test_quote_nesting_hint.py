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
        ]
    )
    engine = RLMEngine(
        client=client, session=session, runtime_config=make_runtime_config()
    )  # type: ignore
    try:
        result = await engine.prompt("write a module")
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
    # the hint text is delivered on the turn after each of the first MAX_QUOTE_NESTING_HINTS failures and
    # then stays in the conversation, so count distinct positions by checking the last call saw exactly that many
    last_call_hints = [
        m
        for m in client.calls[-1]["messages"]
        if m.get("role") == "user"
        and "nested inside a Python string literal" in str(m.get("content", ""))
    ]
    assert len(last_call_hints) == MAX_QUOTE_NESTING_HINTS, [
        str(m.get("content"))[:80] for m in last_call_hints
    ]
    assert 'rlm.hints.mute("quote-nesting")' in hints[0]
