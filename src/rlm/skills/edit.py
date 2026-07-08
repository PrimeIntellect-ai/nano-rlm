"""Built-in ``edit`` skill — safe single-occurrence string replacement.

Enabled via ``RLM_SKILLS``; pre-imported into the IPython kernel so the agent calls
``await edit(path=..., old_str=..., new_str=...)``. Ported from the ``edit`` skill in
research-environments/rlm_swe.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["run"]


async def run(path: str, old_str: str, new_str: str) -> str:
    """Replace a unique string in a file.

    Args:
        path: File path, relative to the working directory or absolute.
        old_str: Exact string to find; it must appear exactly once in the file.
        new_str: Replacement string.

    Returns:
        A confirmation message.
    """
    filepath = Path(path)
    if not filepath.is_absolute():
        filepath = Path.cwd() / filepath
    if not filepath.exists():
        raise FileNotFoundError(f"{path} not found")
    content = filepath.read_text()
    count = content.count(old_str)
    if count != 1:
        raise ValueError(f"old_str must appear exactly once in {path} (found {count})")
    filepath.write_text(content.replace(old_str, new_str, 1))
    return f"Edited {path}"
