"""Skill discovery helpers."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

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


@dataclass(frozen=True)
class AuthoredSkill:
    """One package under the contract's persistent skills directory.

    An authored skill is an installed skill minus installation: the same layout,
    ``SKILL.md`` and async ``run()`` entrypoint, imported from ``sys.path`` rather than
    a distribution. ``error`` names the first contract violation found without
    importing; the kernel then binds a placeholder that reports it.
    """

    name: str
    path: str  # sys.path entry that makes ``import <name>`` work
    skill_md: str | None
    error: str | None = None


def _check_authored(package_dir: Path, name: str) -> tuple[str | None, str | None]:
    """``(SKILL.md path, error)`` for one authored package directory."""
    if not (package_dir / "src" / name / "__init__.py").is_file():
        return None, f"expected {package_dir.name}/src/{name}/__init__.py"
    skill_md = package_dir / "SKILL.md"
    if not skill_md.is_file():
        return None, f"expected {package_dir.name}/SKILL.md"
    pyproject = package_dir / "pyproject.toml"
    if pyproject.is_file():
        with open(pyproject, "rb") as f:
            project = tomllib.load(f).get("project", {})
        expected = f"rlm-skill-{package_dir.name}"
        if project.get("name") != expected:
            return str(skill_md), (
                f"{package_dir.name}/pyproject.toml must name the distribution "
                f"{expected!r}"
            )
    return str(skill_md), None


def list_authored_skills(skills_dir: str | Path | None) -> list[AuthoredSkill]:
    """Agent-authored packages under ``skills_dir``, one per subdirectory whose name
    is an identifier (hyphens map to underscores, as for installed skills)."""
    if skills_dir is None:
        return []
    root = Path(skills_dir)
    if not root.is_dir():
        return []
    authored: list[AuthoredSkill] = []
    for package_dir in sorted(root.iterdir()):
        name = _normalize_skill_name(package_dir.name)
        if not package_dir.is_dir() or not name.isidentifier():
            continue
        skill_md, error = _check_authored(package_dir, name)
        authored.append(AuthoredSkill(name, str(package_dir / "src"), skill_md, error))
    return authored


def discover_skills(
    session_dir: Path | None = None, skills_dir: str | Path | None = None
) -> list[str]:
    """Return unambiguous installed, session-local and authored skill module names."""
    installed = get_installed_skills()
    generated = list_skill_modules(session_dir) if session_dir is not None else []
    authored = [skill.name for skill in list_authored_skills(skills_dir)]
    for label, names in (("generated", generated), ("authored", authored)):
        collisions = sorted(set(installed) & set(names))
        if collisions:
            raise ValueError(
                f"skill name collision between installed and {label}: "
                + ", ".join(collisions)
            )
    collisions = sorted(set(generated) & set(authored))
    if collisions:
        raise ValueError(
            "skill name collision between generated and authored: "
            + ", ".join(collisions)
        )
    return [*installed, *generated, *authored]
