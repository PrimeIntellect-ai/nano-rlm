"""Tasks a hosted model completes with nano-rlm, each checked by what it leaves behind."""

from __future__ import annotations


async def test_writes_files(run_agent):
    outcome = await run_agent(
        "Write the prime numbers below 50 to primes.txt, one per line, in ascending order."
    )
    assert outcome.read("primes.txt").split() == [
        "2",
        "3",
        "5",
        "7",
        "11",
        "13",
        "17",
        "19",
        "23",
        "29",
        "31",
        "37",
        "41",
        "43",
        "47",
    ], outcome.describe()


async def test_kernel_keeps_state_between_calls(run_agent):
    outcome = await run_agent(
        "Use two separate ipython tool calls. In the first, read the integer in seed.txt "
        "into a variable named seed and print nothing else. In the second, without "
        "reading any file, write seed * 3 to answer.txt.",
        files={"seed.txt": "1234\n"},
    )
    assert outcome.read("answer.txt") == "3702", outcome.describe()
    assert outcome.tool_calls().count("ipython") >= 2, outcome.describe()


async def test_edit_skill_fixes_a_bug(run_agent):
    outcome = await run_agent(
        "calc.py has a bug: add() should return the sum of its arguments. Fix it with "
        "the edit skill, changing only the broken line.",
        files={
            "calc.py": "def add(a, b):\n    return a - b\n\n\ndef double(x):\n    return 2 * x\n"
        },
    )
    source = outcome.read("calc.py")
    namespace: dict = {}
    exec(source, namespace)
    assert namespace["add"](2, 3) == 5, outcome.describe()
    assert "def double(x):\n    return 2 * x" in source


async def test_native_bash_tool(run_agent):
    outcome = await run_agent(
        "With the bash tool, count the lines of data.csv that contain 'error' and write "
        "the count to answer.txt.",
        files={
            "data.csv": "".join(
                f"{i},{'error' if i % 7 == 0 else 'ok'}\n" for i in range(1, 101)
            )
        },
        builtin_tools=["ipython", "bash", "edit"],
    )
    assert outcome.read("answer.txt") == "14", outcome.describe()
    assert "bash" in outcome.tool_calls(), outcome.describe()


async def test_long_output_shows_its_end(run_agent):
    outcome = await run_agent(
        "Run `seq 1 300000` in a shell and look at its output. Write the last number in "
        "that output to answer.txt."
    )
    assert outcome.read("answer.txt") == "300000", outcome.describe()


async def test_subagent_does_part_of_the_work(run_agent):
    outcome = await run_agent(
        "Spawn a sub-agent with rlm.agent.spawn and have it write the sum of the integers "
        "from 1 to 100 to child.txt. Wait for it to finish, then write the word done to "
        "answer.txt."
    )
    assert outcome.read("child.txt") == "5050", outcome.describe()
    assert outcome.read("answer.txt") == "done", outcome.describe()
    assert any(
        path.name.startswith("sub-") for path in outcome.session_dir.iterdir()
    ), outcome.describe()


async def test_service_answers_requests(run_agent):
    outcome = await run_agent(
        "Serve the working directory over HTTP with `python -m http.server` on a free "
        "port, as a service that keeps running, then fetch /hello.txt from it and write "
        "the response body to answer.txt.",
        files={"hello.txt": "hello from the workspace\n"},
    )
    assert outcome.read("answer.txt") == "hello from the workspace", outcome.describe()


async def test_kernel_recovers_after_it_dies(run_agent):
    outcome = await run_agent(
        "Run `import os; os._exit(1)` in the ipython tool; it kills the kernel. After "
        "the kernel restarts, write the word recovered to answer.txt."
    )
    assert outcome.read("answer.txt") == "recovered", outcome.describe()
    assert any(record.get("type") == "kernel_recovery" for record in outcome.records), (
        outcome.describe()
    )


async def test_work_continues_past_a_compaction(run_agent):
    outcome = await run_agent(
        "Use three separate tool calls. First read the code word in secret.txt. Then run "
        "`seq 1 3000` in a shell and print all of its output. Then write the code word "
        "to answer.txt.",
        files={"secret.txt": "marmalade\n"},
        policy={"summarize_at_tokens": 8000},
    )
    assert outcome.read("answer.txt") == "marmalade", outcome.describe()
    assert outcome.snapshot["metrics"]["num_compactions"] >= 1, outcome.describe()
