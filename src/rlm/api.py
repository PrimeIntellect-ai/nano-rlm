"""Public Python API for running rlm agents."""

from rlm import broker
from rlm.engine import RLMEngine
from rlm.types import RLMResult


async def run(prompt: str, **kwargs) -> RLMResult:
    """Run a single rlm agent."""
    if broker.is_configured():
        return await broker.run(prompt, **kwargs)
    engine = RLMEngine(**kwargs)
    return await engine.run(prompt)
