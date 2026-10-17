"""Skill discovery helpers."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import yaml

from rlm.mcp import list_skill_modules

logger = logging.getLogger(__name__)

_FRONTMATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.S)


SKILLS_ROOT = Path(os.environ.get("RLM_HOME") or Path.home() / ".rlm") / "skills"


def _find_skills_dir() -> Path | None:
    """Locate the skills root, ``$RLM_HOME/skills``."""
    return SKILLS_ROOT if SKILLS_ROOT.is_dir() else None


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


def _frontmatter(skill_md: Path, text: str) -> dict:
    """The SKILL.md frontmatter mapping, or ``{}`` when it is absent or unreadable: one
    malformed skill must not keep the agent from starting."""
    match = _FRONTMATTER.match(text)
    if match is None:
        return {}
    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError as error:
        logger.warning("ignoring invalid frontmatter in %s: %s", skill_md, error)
        return {}
    if not isinstance(meta, dict):
        logger.warning("ignoring non-mapping frontmatter in %s", skill_md)
        return {}
    return meta


def _skill_doc(skill_md: Path) -> SkillDoc:
    meta = _frontmatter(skill_md, skill_md.read_text())
    return SkillDoc(
        name=str(meta.get("name") or skill_md.parent.name),
        description=" ".join(str(meta.get("description") or "").split()),
        location=str(skill_md),
    )


def discover_skill_docs(skills_dir: Path) -> list[SkillDoc]:
    """Return the skills under ``skills_dir``: every folder holding a SKILL.md,
    instruction-only skills and Python skill packages alike."""
    return [_skill_doc(skill_md) for skill_md in sorted(skills_dir.glob("*/SKILL.md"))]
