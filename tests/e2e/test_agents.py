"""The supervisor's API from a cell: sub-agents, their history, persistent agents, and
background jobs the agent waits on."""

from __future__ import annotations

import json
import re


async def test_subagent_does_part_of_the_work(run_agent):
    outcome = await run_agent(
        "Spawn a sub-agent with rlm.agent.spawn and have it write the sum of the integers "
        "from 1 to 100 to child.txt. Wait for it to finish, then write the word done to "
        "answer.txt."
    )
    assert outcome.read("child.txt") == "5050", outcome.describe()
    assert outcome.read("answer.txt") == "done", outcome.describe()
    assert outcome.children(), outcome.describe()
    assert "subagent_call" in outcome.edges(), outcome.edges()


async def test_parent_reads_a_childs_history(run_agent):
    outcome = await run_agent(
        "Spawn a sub-agent whose task is: 'Run `import secrets; print(secrets.token_hex(4))` "
        "in ipython, then answer with only the word done.' Its answer will not contain the "
        "token, so once it finishes, read its conversation with `await child.history()` and "
        "write the token it printed to answer.txt."
    )
    (child,) = outcome.children()
    printed = {
        token
        for record in outcome.ledger(child)
        if record.get("type") == "tool_result"
        for token in re.findall(r"\b[0-9a-f]{8}\b", record["content"])
    }
    assert outcome.read("answer.txt") in printed, (printed, outcome.describe())
    assert any(
        "history(" in json.loads(call["function"]["arguments"]).get("code", "")
        for call in outcome.tool_call_records()
        if call["function"]["name"] == "ipython"
    ), outcome.describe()


async def test_persistent_agent_takes_a_second_instruction(run_agent):
    outcome = await run_agent(
        "Spawn a persistent sub-agent with `rlm.agent.spawn(task=..., name='scribe', "
        "persistent=True)` whose task is to write the word alpha to a.txt, and wait for its "
        "result. Then give the same agent a second instruction with `await scribe.send(...)`: "
        "write the word beta to b.txt. Wait for that result too, then write the word done "
        "to answer.txt."
    )
    assert outcome.read("a.txt") == "alpha", outcome.describe()
    assert outcome.read("b.txt") == "beta", outcome.describe()
    assert outcome.read("answer.txt") == "done", outcome.describe()
    assert len(outcome.children()) == 1, outcome.children()


async def test_wait_returns_when_a_background_job_ends(run_agent):
    outcome = await run_agent(
        "Start the command `sleep 10; echo finished` as a background job with "
        "`await rlm.shell.run(command, yield_after=0)`. Do not poll it: call the wait tool, "
        "which returns when the job completes. Then write the job's output to answer.txt."
    )
    assert outcome.read("answer.txt") == "finished", outcome.describe()
    assert (
        "New supervisor events or instructions are available." in outcome.tool_results()
    ), outcome.tool_results()
