import pytest

from rlm.client import make_client
from rlm.config import (
    KERNEL_ENV_CONFIG_ENV,
    CompactionConfig,
    ExecutionPolicy,
    InvocationContext,
    ProviderConfig,
    RuntimeConfig,
)


def test_runtime_config_resolves_and_redacts_environment():
    config = RuntimeConfig.from_env(
        environ={
            "RLM_MODEL": "override",
            "RLM_API_KEY": "secret",
            "RLM_BASE_URL": "http://interceptor",
            "RLM_DEPTH": "2",
            "RLM_MAX_DEPTH": "4",
            "RLM_EXEC_TIMEOUT": "30",
            "RLM_MAX_TOKENS": "500",
            "RLM_SUMMARIZE_AT_TOKENS": "2048",
            "RLM_MAX_COMPACTIONS": "3",
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
        max_tokens=500,
        compaction=CompactionConfig(summarize_at_tokens=2048),
        max_compactions=3,
        max_concurrent_subagents=4,
        max_subagent_calls=64,
        allow_git=True,
    )
    assert config.skills == ("search", "edit")
    assert config.kernel_env == (("TASK_TOKEN", "task-secret"),)
    assert config.search_api_key == "search-secret"
    assert "task-secret" not in repr(config)
    assert "search-secret" not in repr(config)
    assert "test-key" not in repr(config.provider)


def test_compaction_is_disabled_by_default():
    config = RuntimeConfig.from_env(environ={})

    assert config.policy.compaction is None


def test_runtime_config_rejects_unsafe_recursive_and_environment_values():
    with pytest.raises(ValueError, match="at least RLM_MAX_DEPTH"):
        RuntimeConfig.from_env(
            environ={
                "RLM_MAX_DEPTH": "3",
                "RLM_MAX_CONCURRENT_SUBAGENTS": "2",
            }
        )
    provider = ProviderConfig(
        base_url="http://interceptor",
        api_key="secret",
    )
    provider.headers["Idempotency-Key"] = "forged-after-construction"
    with pytest.raises(ValueError, match="reserved names"):
        make_client(provider)
