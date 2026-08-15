"""Skill discovery helpers."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

from rlm.mcp import list_skill_modules


TASK_SKILLS_DIR = Path("/task/rlm-skills")


def _find_skills_dir() -> Path | None:
    """Locate the uploaded skills directory when available."""
    return TASK_SKILLS_DIR if TASK_SKILLS_DIR.is_dir() else None


SKILLS_DIR = _find_skills_dir()


def _normalize_skill_name(name: str) -> str:
    """Normalize a discovered skill token to the import/CLI form."""
    return name.replace("-", "_")


def get_installed_skills() -> list[str]:
    """Return installed skill names discovered from distribution metadata."""
    skills: set[str] = set()
    prefix = "rlm-skill-"
    for dist in metadata.distributions():
        name = dist.metadata.get("Name", "")
        if name.startswith(prefix):
            skills.add(_normalize_skill_name(name[len(prefix) :]))
    return sorted(skills)


def discover_skills(session_dir: Path | None = None) -> list[str]:
    """Return unambiguous installed and session-local skill module names."""
    installed = get_installed_skills()
    generated = list_skill_modules(session_dir) if session_dir is not None else []
    collisions = sorted(set(installed) & set(generated))
    if collisions:
        names = ", ".join(collisions)
        raise ValueError(
            f"skill name collision between installed and generated: {names}"
        )
    return [*installed, *generated]
