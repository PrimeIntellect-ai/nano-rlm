"""Opt-in live API checks: RLM_LIVE_ANTHROPIC=1 ANTHROPIC_API_KEY=... uv run pytest -s tests/test_anthropic_live.py."""

import json
import os
from pathlib import Path
import sys
import uuid

from acp import PROTOCOL_VERSION, spawn_agent_process, text_block
import pytest

from conftest import make_runtime_config
from rlm.acp import RUNTIME_METADATA_KEY, SESSION_METADATA_KEY
from rlm.client import make_client
from rlm.config import ExecutionPolicy, ProviderConfig
from rlm.engine import RLMEngine, _new_tokens
from rlm.session import Session

pytestmark = pytest.mark.skipif(
    os.environ.get("RLM_LIVE_ANTHROPIC") != "1"
    or not os.environ.get("ANTHROPIC_API_KEY"),
    reason="Requires explicit live-test opt-in and ANTHROPIC_API_KEY",
)
MODEL = "claude-haiku-4-5"


def _system_prompt():
    return (
        f"Live cache test {uuid.uuid4().hex}. Follow the user's instructions.\n"
        "The following reference text is irrelevant to the task.\n"
        + "The reference value is stable and may be ignored for this task.\n"
        * 512
    )


def _provider(mode="5m", max_output_tokens=32):
    return ProviderConfig(
        base_url="https://api.anthropic.com/v1",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        prompt_cache=mode,
        max_output_tokens=max_output_tokens,
        max_retries=1,
    )


def _usage(response, usage):
    details = response.usage.prompt_tokens_details
    return {
        "prompt": usage.prompt_tokens,
        "output": usage.completion_tokens,
        "cache_read": details.cached_tokens,
        "cache_write": details.cache_write_tokens,
        "new_tokens": _new_tokens(response, usage),
    }


@pytest.mark.parametrize("mode", ["5m", "1h", "off"])
async def test_live_cache_cold_warm_and_growing_history(mode):
    provider = _provider(mode)
    messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": "Reply with exactly OK."},
    ]
    async with make_client(provider) as client:
        engine = RLMEngine(
            client=client,
            runtime_config=make_runtime_config(model=MODEL, provider=provider),
        )
        engine._active_tool_schemas = []
        cold, cold_usage = await engine._call_model(messages)
        warm, warm_usage = await engine._call_model(messages)
        growing, growing_usage = await engine._call_model(
            [
                *messages,
                warm.choices[0].message.model_dump(exclude_none=True),
                {"role": "user", "content": "Reply with exactly OK again."},
            ]
        )

    rows = [
        _usage(response, usage)
        for response, usage in (
            (cold, cold_usage),
            (warm, warm_usage),
            (growing, growing_usage),
        )
    ]
    print(json.dumps({"mode": mode, "cold_warm_growing": rows}))
    assert rows[0]["prompt"] >= 4096
    if mode == "off":
        assert all(row["cache_read"] == row["cache_write"] == 0 for row in rows)
    else:
        assert rows[0]["cache_read"] == 0
        assert rows[0]["cache_write"] >= 4096
        assert rows[1]["cache_read"] >= 4096
        assert rows[2]["cache_read"] >= rows[1]["cache_read"]
        assert rows[1]["new_tokens"] < rows[0]["new_tokens"] / 10
    assert engine._own_new_tokens == sum(row["new_tokens"] for row in rows)
    assert engine._total_usage.total == sum(
        row["prompt"] + row["output"] for row in rows
    )


async def test_live_tool_execution_and_compaction(monkeypatch, tmp_path):
    provider = _provider(max_output_tokens=1024)
    marker = f"cache-test-{uuid.uuid4().hex}"
    calls = []
    async with make_client(provider) as client:
        engine = RLMEngine(
            client=client,
            session=Session(tmp_path / "session"),
            cwd=str(tmp_path),
            runtime_config=make_runtime_config(
                model=MODEL,
                provider=provider,
                append_to_system_prompt=_system_prompt(),
                builtin_tools=("bash",),
                policy=ExecutionPolicy(
                    max_depth=0,
                    max_total_turns=6,
                    max_total_tokens=30_000,
                    summarize_at_tokens=4096,
                    max_compactions=1,
                    max_compaction_attempts=2,
                ),
            ),
        )
        call_model = engine._call_model

        async def record_call(messages, **kwargs):
            response, usage = await call_model(messages, **kwargs)
            calls.append(
                {
                    "checkpoint": kwargs.get("checkpoint", False),
                    **_usage(response, usage),
                }
            )
            return response, usage

        monkeypatch.setattr(engine, "_call_model", record_call)
        try:
            result = await engine.prompt(
                "Use bash once to run this command: "
                f"printf '{marker}' > cache-result.txt; cat cache-result.txt\n"
                "Then reply with exactly the printed value. "
                "Once the file has been written, do not run the command again."
            )
            assert (tmp_path / "cache-result.txt").read_text() == marker
            assert marker in result.answer
            assert engine._compacted is True
            followup = await engine.prompt(
                "Reply with the same printed value again, without using tools."
            )
            assert marker in followup.answer
        finally:
            await engine.aclose()

    print(json.dumps({"tools_and_compaction": calls}))
    assert calls[0]["cache_write"] >= 4096
    assert any(call["checkpoint"] for call in calls)
    assert all(call["cache_read"] >= 4096 for call in calls[1:])
    assert engine._own_new_tokens == sum(call["new_tokens"] for call in calls)
    assert engine._total_usage.total == sum(
        call["prompt"] + call["output"] for call in calls
    )


class _LiveACPClient:
    def __init__(self):
        self.answers = []

    async def session_update(self, session_id, update, **kwargs):
        if update.session_update == "agent_message_chunk":
            self.answers.append(update.content.text)


async def test_live_acp_stdio_tool_and_persistent_followup(tmp_path):
    system = tmp_path / "system.txt"
    system.write_text(_system_prompt())
    policy = ExecutionPolicy(
        max_depth=0,
        max_total_turns=6,
        max_total_tokens=30_000,
        compaction=False,
    )
    runtime = {
        "session_id": "live-anthropic",
        "model": MODEL,
        "provider": _provider(max_output_tokens=256).model_dump(),
        "policy": policy.model_dump(),
        "system_prompt_path": str(system),
        "append_to_system_prompt": None,
        "skills": [],
        "builtin_tools": ["bash"],
        "kernel_env": {},
        "search_api_key": None,
    }
    client = _LiveACPClient()
    executable = str(Path(sys.executable).parent / "rlm")
    env = {**os.environ, "RLM_HOME": str(tmp_path / "rlm")}
    marker = f"stdio-test-{uuid.uuid4().hex}"
    async with spawn_agent_process(client, executable, "--acp", env=env) as (
        connection,
        _process,
    ):
        initialized = await connection.initialize(PROTOCOL_VERSION)
        created = await connection.new_session(
            cwd=str(tmp_path), mcp_servers=[], **{RUNTIME_METADATA_KEY: runtime}
        )
        first = await connection.prompt(
            session_id=created.session_id,
            prompt=[
                text_block(
                    f"Use bash to run: printf '{marker}' > stdio-result.txt; cat stdio-result.txt\n"
                    "Then reply with exactly the printed value."
                )
            ],
        )
        assert (tmp_path / "stdio-result.txt").read_text() == marker
        assert marker in client.answers[-1]
        second = await connection.prompt(
            session_id=created.session_id,
            prompt=[text_block("Reply with the same printed value, without tools.")],
        )
        assert marker in client.answers[-1]
        closed = await connection.close_session(created.session_id)

    snapshot = closed.field_meta[SESSION_METADATA_KEY]
    print(json.dumps({"acp_stdio": snapshot["usage"], "turns": snapshot["turns"]}))
    assert initialized.agent_info.name == "rlm"
    assert first.stop_reason == second.stop_reason == "end_turn"
    assert snapshot["last_stop_reason"] == "done"
    assert snapshot["turns"] >= 3
    assert snapshot["usage"]["prompt_tokens"] >= 4096 * 3
