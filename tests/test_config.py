import pytest

from rlm.client import make_client, model_call_headers
from rlm.config import (
    KERNEL_ENV_CONFIG_ENV,
    PI_INFERENCE_BASE_URL,
    ExecutionPolicy,
    InvocationContext,
    ProviderConfig,
    RuntimeConfig,
)


def test_provider_config_keeps_keys_paired_with_provider():
    explicit = ProviderConfig.from_env(
        {"RLM_API_KEY": "rlm-secret", "RLM_BASE_URL": "http://interceptor"}
    )
    prime = ProviderConfig.from_env(
        {"PRIME_API_KEY": "prime-secret", "PRIME_TEAM_ID": "team"}
    )
    openai = ProviderConfig.from_env(
        {"OPENAI_API_KEY": "openai-secret", "OPENAI_BASE_URL": "http://openai"}
    )

    assert (explicit.base_url, explicit.api_key) == (
        "http://interceptor",
        "rlm-secret",
    )
    assert (prime.base_url, prime.api_key, prime.headers) == (
        PI_INFERENCE_BASE_URL,
        "prime-secret",
        {"X-Prime-Team-ID": "team"},
    )
    assert (openai.base_url, openai.api_key) == ("http://openai", "openai-secret")

    with pytest.raises(ValueError, match="reserved names"):
        ProviderConfig(
            "http://interceptor",
            "secret",
            headers={"x-RlM-Parent-Invocation-ID": "forged"},
        )


def test_model_call_headers_encode_versioned_lineage():
    headers = model_call_headers(
        session_id="session",
        invocation_id="child",
        parent_invocation_id="parent",
        segment_id="segment",
        call_id="call",
        parent_call_id="parent-call",
        depth=2,
        call_kind="compaction",
    )

    assert headers == {
        "X-RLM-Lineage-Version": "1",
        "X-RLM-Session-ID": "session",
        "X-RLM-Invocation-ID": "child",
        "X-RLM-Parent-Invocation-ID": "parent",
        "X-RLM-Segment-ID": "segment",
        "X-RLM-Call-ID": "call",
        "X-RLM-Parent-Call-ID": "parent-call",
        "X-RLM-Depth": "2",
        "X-RLM-Call-Kind": "compaction",
    }


def test_runtime_config_resolves_environment_once():
    config = RuntimeConfig.from_env(
        model="override",
        summarize_at_tokens=2048,
        environ={
            "RLM_API_KEY": "secret",
            "RLM_BASE_URL": "http://interceptor",
            "RLM_DEPTH": "2",
            "RLM_MAX_DEPTH": "4",
            "RLM_EXEC_TIMEOUT": "30",
            "RLM_MAX_OUTPUT": "1200",
            "RLM_MAX_TOKENS": "500",
            "RLM_MAX_COMPACTIONS": "3",
            "RLM_MAX_TOOL_OUTPUT_CHARS": "700",
            "RLM_ALLOW_GIT": "1",
            "RLM_SKILLS": "search, edit",
            KERNEL_ENV_CONFIG_ENV: '{"TASK_TOKEN": "task-secret"}',
            "SERPER_API_KEY": "search-secret",
        },
    )

    assert config.model == "override"
    assert config.invocation == InvocationContext(depth=2)
    assert config.policy == ExecutionPolicy(
        max_depth=4,
        exec_timeout=30,
        max_output=1200,
        max_tokens=500,
        summarize_at_tokens=2048,
        max_compactions=3,
        max_concurrent_subagents=4,
        max_subagent_calls=64,
        max_tool_output_chars=700,
        allow_git=True,
    )
    assert config.skills == ("search", "edit")
    assert config.kernel_env == (("TASK_TOKEN", "task-secret"),)
    assert config.search_api_key == "search-secret"
    assert "task-secret" not in repr(config)
    assert "search-secret" not in repr(config)
    assert "test-key" not in repr(config.provider)


@pytest.mark.parametrize("value", [0, -1, "bad", True])
def test_runtime_config_rejects_invalid_compaction_threshold(value):
    with pytest.raises(ValueError):
        RuntimeConfig.from_env(summarize_at_tokens=value, environ={})


def test_make_client_uses_explicit_invocation_depth(monkeypatch):
    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("rlm.client.AsyncOpenAI", fake_client)
    provider = ProviderConfig(
        base_url="http://interceptor",
        api_key="secret",
        headers={"X-Test": "yes"},
        max_retries=7,
    )

    make_client(provider, InvocationContext(depth=3))

    assert captured == {
        "base_url": "http://interceptor",
        "api_key": "secret",
        "max_retries": 7,
        "default_headers": {"X-RLM-Depth": "3", "X-Test": "yes"},
    }

    provider.headers["X-RLM-Call-ID"] = "forged-after-construction"
    with pytest.raises(ValueError, match="reserved names"):
        make_client(provider, InvocationContext(depth=3))


def test_runtime_config_validates_recursive_limits():
    with pytest.raises(ValueError, match="at least RLM_MAX_DEPTH"):
        RuntimeConfig.from_env(
            environ={
                "RLM_MAX_DEPTH": "3",
                "RLM_MAX_CONCURRENT_SUBAGENTS": "2",
            }
        )
    with pytest.raises(ValueError, match="RLM_MAX_SUBAGENT_CALLS must be positive"):
        RuntimeConfig.from_env(environ={"RLM_MAX_SUBAGENT_CALLS": "0"})


def test_direct_policy_construction_validates_recursive_limits():
    with pytest.raises(ValueError, match="at least max_depth"):
        ExecutionPolicy(max_depth=3, max_concurrent_subagents=2)
    with pytest.raises(ValueError, match="max_subagent_calls must be positive"):
        ExecutionPolicy(max_subagent_calls=0)
    with pytest.raises(ValueError, match="depth must be non-negative"):
        InvocationContext(depth=-1)


@pytest.mark.parametrize(
    "value",
    ['["not", "an object"]', '{"NUMBER": 1}', "not-json"],
)
def test_runtime_config_rejects_invalid_kernel_environment(value):
    with pytest.raises((ValueError, TypeError)):
        RuntimeConfig.from_env(environ={KERNEL_ENV_CONFIG_ENV: value})
