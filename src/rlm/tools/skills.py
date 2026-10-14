"""Skill discovery helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import yaml

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


@dataclass(frozen=True)
class SkillDoc:
    """A SKILL.md skill: listed in the prompt for the agent to read."""

    name: str
    description: str
    location: str


def _skill_doc(skill_md: Path) -> SkillDoc:
    text = skill_md.read_text()
    meta: dict = {}
    if text.startswith("---"):
        _, front, _ = text.split("---", 2)
        meta = yaml.safe_load(front) or {}
    return SkillDoc(
        name=str(meta.get("name") or skill_md.parent.name),
        description=" ".join(str(meta.get("description") or "").split()),
        location=str(skill_md),
    )


def discover_skill_docs(paths: Iterable[str | Path]) -> list[SkillDoc]:
    """Return the SKILL.md skills under ``paths``: each path is one skill folder
    (holding SKILL.md) or a folder of skill folders."""
    found: dict[Path, SkillDoc] = {}
    for path in map(Path, paths):
        if (path / "SKILL.md").is_file():
            candidates = [path / "SKILL.md"]
        elif path.is_dir():
            candidates = sorted(path.glob("*/SKILL.md"))
        else:
            raise FileNotFoundError(f"skill path {path} holds no SKILL.md")
        for skill_md in candidates:
            found.setdefault(skill_md.resolve(), _skill_doc(skill_md))
    return list(found.values())
