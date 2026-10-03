"""Run inside a prepared sandbox: uv run examples/replay.py --help."""

import argparse
import asyncio
from pathlib import Path

from rlm.config import RuntimeConfig
from rlm.engine import RLMEngine
from rlm.replay import ExecutionTape
from rlm.session import Session


async def main(args):
    config = RuntimeConfig.model_validate_json(args.config.read_text())
    tape = ExecutionTape(args.tape, mode=args.mode)
    session = None
    engine = None
    try:
        session = Session(args.session)
        engine = RLMEngine(
            runtime_config=config,
            cwd=str(args.cwd.resolve()),
            session=session,
            execution_tape=tape,
        )
        result = await engine.run(args.prompt)
        tape.finish()
        print(result.answer)
    finally:
        if engine is not None:
            await engine.aclose()
        if session is not None:
            session.close()
        tape.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("record", "replay"))
    parser.add_argument("tape", type=Path)
    parser.add_argument("--config", type=Path, required=True, help="RuntimeConfig JSON")
    parser.add_argument(
        "--session", type=Path, required=True, help="New session directory"
    )
    parser.add_argument(
        "--cwd", type=Path, required=True, help="Prepared sandbox workspace"
    )
    parser.add_argument("--prompt", required=True)
    asyncio.run(main(parser.parse_args()))
