import pytest

from rlm.client import make_client
from rlm.config import (
    ExecutionPolicy,
    InvocationContext,
    ProviderConfig,
    RuntimeConfig,
)


def test_runtime_config_redacts_secrets():
    config = RuntimeConfig(
        model="override",
        provider=ProviderConfig(base_url="http://interceptor", api_key="test-key"),
        invocation=InvocationContext(depth=2),
        policy=ExecutionPolicy(
            max_depth=4,
            exec_timeout=30,
            max_tokens=500,
            summarize_at_tokens=2048,
            max_compactions=3,
            allow_git=True,
        ),
        skills=("search", "edit"),
        kernel_env=(("TASK_TOKEN", "task-secret"),),
        search_api_key="search-secret",
    )

    assert config.invocation.child() == InvocationContext(depth=3)
    assert "task-secret" not in repr(config)
    assert "search-secret" not in repr(config)
    assert "test-key" not in repr(config.provider)


def test_policy_rejects_unsafe_recursive_values_and_reserved_headers():
    with pytest.raises(ValueError, match="at least max_depth"):
        ExecutionPolicy(max_depth=3, max_concurrent_subagents=2)
    provider = ProviderConfig(
        base_url="http://interceptor",
        api_key="secret",
    )
    provider.headers["Idempotency-Key"] = "forged-after-construction"
    with pytest.raises(ValueError, match="reserved names"):
        make_client(provider)


def test_default_policy_enables_compaction_and_recursion():
    policy = ExecutionPolicy()
    assert policy.summarize_at_tokens == 256_000
    assert policy.max_depth == 1
