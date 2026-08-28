"""Public Python API for running rlm agents."""

from rlm import broker
from rlm.types import RLMResult


async def run(prompt: str) -> RLMResult:
    """Run a recursive sub-agent through the session's broker."""
    if not broker.is_configured():
        raise RuntimeError(
            "rlm.run() requires the recursion broker (available inside a "
            "running rlm session). Standalone execution was removed: rlm is "
            "consumed via the ACP runtime contract."
        )
    return await broker.run(prompt)
