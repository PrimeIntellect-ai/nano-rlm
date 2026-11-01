"""The session as a client drives it: several prompts on one kernel, steering into a
waiting agent, cancelling a turn, tool timeouts, and the budgets that end a run."""

from __future__ import annotations

import asyncio
import time

import pytest


async def test_second_prompt_continues_the_session(agent):
    async with agent(files={"secret.txt": "marmalade\n"}) as session:
        await session.prompt(
            "In ipython, store the contents of secret.txt, stripped, in a variable named "
            "word, then delete secret.txt. Do not print word."
        )
        assert not (session.workspace / "secret.txt").exists(), session.describe()
        await session.prompt("Write the value of your variable word to answer.txt.")
    assert session.read("answer.txt") == "marmalade", session.describe()
    assert [turn.stop_reason for turn in session.turns] == ["end_turn", "end_turn"]
    assert "continuation" in session.edges(), session.edges()


async def test_steering_wakes_a_waiting_agent(agent):
    async with agent() as session:
        running = session.start(
            "Before doing anything else, call the wait tool and keep waiting until the user "
            "sends you a message. It contains a code word: write the code word to answer.txt."
        )
        await session.called("wait")
        assert await session.steer("The code word is heron-913.") == {
            "outcome": "injected"
        }
        await running
    assert session.read("answer.txt") == "heron-913", session.describe()
    assert "A new user message is available." in session.tool_results()


async def test_cancelled_turn_leaves_the_session_usable(agent):
    async with agent() as session:
        running = session.start(
            "Run `import time; time.sleep(300)` in one ipython call."
        )
        await session.called("ipython")
        await asyncio.sleep(3)  # let the cell start sleeping
        await session.cancel()
        assert (await running).stop_reason == "cancelled"
        # The kernel is free again only if the cancel interrupted the cell.
        started = time.monotonic()
        await session.prompt("Write the word ok to answer.txt.")
        elapsed = time.monotonic() - started
    assert session.read("answer.txt") == "ok", session.describe()
    assert elapsed < 120, f"second prompt took {elapsed:.0f}s"


async def test_tool_timeout_is_reported_and_work_goes_on(run_agent):
    outcome = await run_agent(
        "Run `import time; time.sleep(60)` in one ipython call, without a timeout "
        "argument. When the call comes back, write the word continued to answer.txt.",
        policy={"exec_timeout": 5},
    )
    assert outcome.read("answer.txt") == "continued", outcome.describe()
    assert any("timed out after 5s" in result for result in outcome.tool_results()), (
        outcome.tool_results()
    )


CAPS = [
    pytest.param(
        {"max_total_turns": 1},
        "max_total_turns",
        "max_turn_requests",
        id="max-total-turns",
    ),
    pytest.param(
        {"max_total_tokens": 1}, "max_total_tokens", "max_tokens", id="max-total-tokens"
    ),
    pytest.param({"max_tokens": 1}, "token_budget", "max_tokens", id="max-tokens"),
]


@pytest.mark.parametrize("cap,last_stop_reason,acp_stop_reason", CAPS)
async def test_budget_stops_the_run(run_agent, cap, last_stop_reason, acp_stop_reason):
    """Each budget ends a task that wants three model calls after the first."""
    outcome = await run_agent(
        "Use three separate ipython calls: write 1 to a.txt, then 2 to b.txt, then 3 "
        "to c.txt.",
        policy=cap,
    )
    assert outcome.snapshot["last_stop_reason"] == last_stop_reason, outcome.describe()
    assert outcome.stop_reason == acp_stop_reason, outcome.describe()
    assert outcome.snapshot["turns"] == 1, outcome.snapshot
    assert not outcome.read("b.txt"), outcome.describe()
