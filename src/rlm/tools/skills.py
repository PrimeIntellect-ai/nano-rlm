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


def list_authored_skills(skills_dir: str | Path | None) -> list[tuple[str, str]]:
    """Agent-authored packages under the contract's persistent skills directory.

    Each ``<skills_dir>/<name>/`` is a package: either the on-disk skill layout
    ``<name>/src/<name>/__init__.py`` or a flat ``<name>/__init__.py``. Returns
    ``(import name, sys.path entry)`` pairs; nothing is installed.
    """
    if skills_dir is None:
        return []
    root = Path(skills_dir)
    if not root.is_dir():
        return []
    authored: list[tuple[str, str]] = []
    for package_dir in sorted(root.iterdir()):
        name = _normalize_skill_name(package_dir.name)
        if not package_dir.is_dir() or not name.isidentifier():
            continue
        if (package_dir / "src" / name / "__init__.py").is_file():
            authored.append((name, str(package_dir / "src")))
        elif (package_dir / "__init__.py").is_file() and name == package_dir.name:
            authored.append((name, str(root)))
    return authored


def discover_skills(
    session_dir: Path | None = None, skills_dir: str | Path | None = None
) -> list[str]:
    """Return unambiguous installed, session-local and authored skill module names."""
    installed = get_installed_skills()
    generated = list_skill_modules(session_dir) if session_dir is not None else []
    authored = [name for name, _ in list_authored_skills(skills_dir)]
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
