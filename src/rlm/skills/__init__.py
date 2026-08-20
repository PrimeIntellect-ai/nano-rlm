"""Built-in skills shipped with rlm, enabled per run via ``RLM_SKILLS``.

Each built-in skill is a module here exposing an async ``run(...)`` (the same contract as an
uploaded skill). When enabled, a thin re-export module is written into the session directory
(on the kernel's ``sys.path``) so the kernel pre-imports it by name — the same path MCP-tool
skills take (see ``rlm.mcp``). The agent then calls it from IPython, e.g. ``await edit(...)``.
"""

from __future__ import annotations

from pathlib import Path

# Built-in skill name -> module its ``run`` is re-exported from.
_BUILTIN_SKILLS: dict[str, str] = {
    "edit": "rlm.skills.edit",
    "search": "rlm.skills.search",
    "search_mixedbread": "rlm.skills.search_mixedbread",
}


def available_builtin_skills() -> list[str]:
    """Names of the built-in skills that ``RLM_SKILLS`` can enable."""
    return sorted(_BUILTIN_SKILLS)


def enable_builtin_skills(names: list[str], dest_dir: Path) -> list[str]:
    """Write a re-export module into ``dest_dir`` for each enabled built-in skill.

    Returns the enabled names; unknown names raise.
    """
    unknown = [name for name in names if name not in _BUILTIN_SKILLS]
    if unknown:
        raise ValueError(
            f"RLM_SKILLS contains unknown skill(s): {unknown}. "
            f"Available: {available_builtin_skills()}"
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (dest_dir / f"{name}.py").write_text(
            f"from {_BUILTIN_SKILLS[name]} import run\n"
        )
    return names
