"""Best-effort broad-history policy checks and a PATH-based Git execution guard.

The text checks provide early feedback. The execution guard checks expanded Git
arguments and aliases. Neither is a security boundary against arbitrary code or
filesystem access; see docs/git-guard.md for the precise scope and bypasses.
"""

from __future__ import annotations

import ast
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REFUSAL_TEMPLATE = (
    "Git history option '{cmd}' is not allowed. Use current-branch history only."
)


_RESTRICTED_LOG_OPTIONS = {
    "--all",
    "-all",
    "--alternate-refs",
    "--reflog",
    "--walk-reflogs",
    "-g",
}
_RESTRICTED_LOG_OPTION_PREFIXES = (
    "--branches",
    "--glob",
    "--remotes",
    "--tags",
)

_GIT_GLOBAL_OPTIONS_WITH_VALUE = {
    "-C",
    "-c",
    "--config-env",
    "--exec-path",
    "--git-dir",
    "--namespace",
    "--work-tree",
}


def allow_git() -> bool:
    return os.environ.get("RLM_ALLOW_GIT") == "1"


def _git_allowed(explicit: bool | None) -> bool:
    return allow_git() if explicit is None else explicit


def find_blocked_command(command: str, *, allow_git: bool | None = None) -> str | None:
    """Check literal commands, shell boundaries, prefixes, and nested shell scripts."""
    if _git_allowed(allow_git):
        return None
    lexer = shlex.shlex(
        command.replace("\\\n", ""), posix=True, punctuation_chars=";&|()\n"
    )
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    segment: list[str] = []
    try:
        for token in lexer:
            if token and all(c in ";&|()\n" for c in token):
                blocked = _check_shell_segment(segment)
                if blocked:
                    return blocked
                segment = []
            else:
                segment.append(token)
    except ValueError:
        # Incomplete shell fragments are checked by the execution guard if run.
        return _check_shell_segment(segment)
    return _check_shell_segment(segment)


def _check_shell_segment(argv: list[str]) -> str | None:
    while argv:
        name = argv[0].rsplit("/", 1)[-1]
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", argv[0]) or name in {
            "if",
            "then",
            "elif",
            "else",
            "do",
            "while",
            "until",
            "!",
            "{",
            "command",
            "builtin",
            "exec",
            "time",
            "env",
            "nohup",
        }:
            argv = argv[1:]
            if name in {"command", "exec", "env", "time"}:
                while argv and argv[0].startswith("-"):
                    option, *argv = argv
                    if option in {"-u", "--unset", "-C", "--chdir", "-a"} and argv:
                        argv = argv[1:]
            continue
        if name in {"bash", "sh", "zsh", "dash"}:
            for i, option in enumerate(argv[1:], 1):
                if option.startswith("-") and "c" in option and i + 1 < len(argv):
                    return find_blocked_command(argv[i + 1], allow_git=False)
                if not option.startswith("-"):
                    break
        return find_blocked_git_log_option(argv)
    return None


def refusal(cmd: str) -> str:
    return REFUSAL_TEMPLATE.format(cmd=cmd)


def _is_git_binary(token: str) -> bool:
    return token == "git" or token.rsplit("/", 1)[-1] == "git"


def _skip_git_global_options(argv: list[str], index: int) -> int:
    while index < len(argv):
        token = argv[index]
        if token == "--":
            return index + 1
        if not token.startswith("-"):
            return index

        option = token.split("=", 1)[0]
        if option in _GIT_GLOBAL_OPTIONS_WITH_VALUE and "=" not in token:
            index += 2
        else:
            index += 1
    return index


def _is_restricted_log_option(token: str) -> bool:
    if token in _RESTRICTED_LOG_OPTIONS:
        return True
    option = token.split("=", 1)[0]
    if option.startswith("--") and len(option) > 2:
        restricted = _RESTRICTED_LOG_OPTIONS | set(_RESTRICTED_LOG_OPTION_PREFIXES)
        if any(flag.startswith(option) for flag in restricted):
            return True
    if re.fullmatch(r"-[pug]+", token) and "g" in token:
        return True
    return any(
        token == option or token.startswith(f"{option}=")
        for option in _RESTRICTED_LOG_OPTION_PREFIXES
    )


def find_blocked_git_log_option(argv: list[str]) -> str | None:
    if not argv or not _is_git_binary(argv[0]):
        return None

    subcommand_index = _skip_git_global_options(argv, 1)
    if subcommand_index >= len(argv) or argv[subcommand_index] != "log":
        return None

    for token in argv[subcommand_index + 1 :]:
        if token == "--":
            return None
        if _is_restricted_log_option(token):
            return token
    return None


# IPython shell-escape lines: ``!cmd`` and ``!!cmd``. Leading whitespace
# is allowed (IPython accepts indented shell escapes inside blocks).
_SHELL_ESCAPE_RE = re.compile(r"^\s*!{1,2}(?P<rest>.*)$")
# ``%sx``, ``%system`` line magics and equivalents that shell out.
_SHELL_LINE_MAGIC_RE = re.compile(r"^\s*%(?:sx|system)\s+(?P<rest>.*)$")
# ``%%bash`` / ``%%sh`` cell magic header — the whole cell body is shell.
_SHELL_CELL_MAGIC_RE = re.compile(r"^\s*%%(?:bash|sh)\b")
# Any IPython line magic — used by the AST pre-pass to drop ipython-only
# lines so ``ast.parse`` doesn't choke on them.
_ANY_LINE_MAGIC_RE = re.compile(r"^\s*%[A-Za-z]")
# Any IPython cell magic header — same purpose.
_ANY_CELL_MAGIC_RE = re.compile(r"^\s*%%[A-Za-z]")


def find_blocked_in_ipython(code: str, *, allow_git: bool | None = None) -> str | None:
    """Scan IPython ``code`` for blocked commands.

    Two passes, both honoring the resolved git policy:

    1. Shell-escape scan — ``!cmd`` / ``!!cmd``, ``%sx`` / ``%system``
       line magics, ``%%bash`` / ``%%sh`` cell magic. Each extracted
       bash fragment goes through ``find_blocked_command``.
    2. Pure-Python AST scan via :func:`find_blocked_python` — catches
       restricted literal subprocess / ``os.system`` git-log invocations
       and the obvious aliases. See that function for documented bypasses
       (dynamic ``getattr``, multi-hop reassignment, etc.).
    """
    if _git_allowed(allow_git):
        return None

    lines = code.splitlines()
    for index, line in enumerate(lines):
        if _SHELL_CELL_MAGIC_RE.match(line):
            return find_blocked_command(
                "\n".join(lines[index + 1 :]), allow_git=allow_git
            )
        m = _SHELL_ESCAPE_RE.match(line) or _SHELL_LINE_MAGIC_RE.match(line)
        if m:
            blocked = find_blocked_command(m.group("rest"), allow_git=allow_git)
            if blocked is not None:
                return blocked

    return find_blocked_python(code, allow_git=allow_git)


# Statically-resolved fully-qualified callees that shell out when invoked
# with a literal first positional argument.
_BLOCKED_PY_CALLS = frozenset(
    {
        "subprocess.run",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "subprocess.Popen",
        "os.system",
        "os.popen",
    }
)


def _blocked_option_from_python_call(
    node: ast.Call, *, allow_git: bool | None
) -> str | None:
    if not node.args:
        return None
    arg = node.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return find_blocked_command(arg.value, allow_git=allow_git)
    if isinstance(arg, (ast.List, ast.Tuple)) and arg.elts:
        argv: list[str] = []
        for elt in arg.elts:
            if not isinstance(elt, ast.Constant) or not isinstance(elt.value, str):
                return None
            argv.append(elt.value)
        return find_blocked_git_log_option(argv)
    return None


class _GitCallFinder(ast.NodeVisitor):
    """Single-pass AST walker tracking simple aliases for blocked callees.

    Tracks three alias sources:

    - ``import subprocess as sp`` — module-level name remap.
    - ``from subprocess import run`` — bare-name binding.
    - ``r = subprocess.run`` — single-hop assignment of a known callee.

    Multi-hop chains (``r1 = subprocess.run; r2 = r1; r2(...)``) and
    dynamic forms (``getattr(subprocess, \"run\")(...)``,
    ``__import__(\"subprocess\").run(...)``) are explicitly out of scope.
    """

    def __init__(self, allow_git: bool | None) -> None:
        # Maps local name -> canonical "module.attr" string.
        self.module_aliases: dict[str, str] = {"subprocess": "subprocess", "os": "os"}
        # Maps local name -> blocked callee fqn (e.g. "run" -> "subprocess.run").
        self.callable_aliases: dict[str, str] = {}
        self.found: str | None = None
        self.allow_git = allow_git

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name in {"subprocess", "os"}:
                self.module_aliases[alias.asname or alias.name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module in {"subprocess", "os"}:
            for alias in node.names:
                fqn = f"{node.module}.{alias.name}"
                if fqn in _BLOCKED_PY_CALLS:
                    self.callable_aliases[alias.asname or alias.name] = fqn
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        # Only track single-hop aliases: ``r = subprocess.run``. Chained
        # reassignment (``r2 = r1``) is intentionally not propagated.
        fqn = None
        if isinstance(node.value, ast.Attribute) and isinstance(
            node.value.value, ast.Name
        ):
            module = self.module_aliases.get(node.value.value.id)
            if module is not None:
                fqn = f"{module}.{node.value.attr}"
        if fqn in _BLOCKED_PY_CALLS:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.callable_aliases[target.id] = fqn
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        fqn = self._resolve_callee(node.func)
        if fqn in _BLOCKED_PY_CALLS:
            blocked = _blocked_option_from_python_call(node, allow_git=self.allow_git)
            if blocked is not None:
                self.found = blocked
        self.generic_visit(node)

    def _resolve_callee(self, expr: ast.AST) -> str | None:
        """Return ``module.attr`` if ``expr`` resolves to a tracked callee."""
        if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name):
            module = self.module_aliases.get(expr.value.id)
            if module is not None:
                return f"{module}.{expr.attr}"
        if isinstance(expr, ast.Name):
            return self.callable_aliases.get(expr.id)
        return None


def _strip_ipython_only(code: str) -> str:
    """Drop ipython-only lines so the remainder is pure Python for ``ast.parse``.

    Removes ``!cmd`` / ``!!cmd`` shell escapes, line magics (``%foo``),
    and any ``%%cellmagic`` header plus its body. Trailing ``?`` / ``??``
    object-inspection markers are stripped from the line tail rather
    than dropping the whole line, so ``subprocess.run?`` becomes
    ``subprocess.run`` and still parses. All other Python lines are
    preserved verbatim.
    """
    out: list[str] = []
    for line in code.splitlines():
        # Drop only the cell-magic HEADER, not the body — magics like
        # ``%%timeit`` / ``%%capture`` execute their body as Python and
        # would otherwise hide ``subprocess.run([\"git\", ...])`` calls.
        # Bash-bodied magics (``%%bash`` / ``%%sh``) are caught earlier
        # by the shell-escape pre-pass, so dropping just the header here
        # is safe.
        if _ANY_CELL_MAGIC_RE.match(line):
            continue
        if _SHELL_ESCAPE_RE.match(line) or _ANY_LINE_MAGIC_RE.match(line):
            continue
        stripped = line.rstrip()
        if stripped.endswith("?"):
            line = stripped.rstrip("?")
        out.append(line)
    return "\n".join(out)


def find_blocked_python(code: str, *, allow_git: bool | None = None) -> str | None:
    """Detect restricted pure-Python git invocations via AST walk.

    Returns the offending token (``\"--all\"`` etc.) if a blocked call is found,
    else ``None``. Honors the resolved git policy. Ipython-only syntax
    (``!cmd``, ``%magic``, ``obj?``) is stripped before parsing so
    cells mixing ipython and Python still get scanned. Returns ``None``
    on ``SyntaxError`` so the normal exec path surfaces the parse error.
    """
    if _git_allowed(allow_git):
        return None
    try:
        tree = ast.parse(_strip_ipython_only(code))
    except SyntaxError:
        return None
    finder = _GitCallFinder(allow_git)
    finder.visit(tree)
    return finder.found


def guarded_git_environment(
    env: dict[str, str], directory: Path, *, allow_git: bool | None = None
) -> dict[str, str]:
    """Put an argument-checking Git launcher first on PATH for this runtime."""
    env = dict(env)
    if _git_allowed(allow_git):
        return env
    real_git = shutil.which("git", path=env.get("PATH", os.defpath))
    if real_git is None:
        return env
    if (Path(real_git).parent / ".rlm-git-guard").is_file():
        return env
    directory.mkdir(parents=True, exist_ok=True)
    launcher = (
        "#!/bin/sh\nexec "
        + shlex.join([sys.executable, "-I", str(Path(__file__).resolve()), real_git])
        + ' "$@"\n'
    )
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as stream:
        stream.write(launcher)
        temporary = Path(stream.name)
    temporary.chmod(0o755)
    temporary.replace(directory / "git")
    (directory / ".rlm-git-guard").touch()
    env["PATH"] = str(directory.resolve()) + os.pathsep + env.get("PATH", os.defpath)
    return env


def _check_git_execution(real_git: str, arguments: list[str]) -> str | None:
    """Check expanded arguments and ordinary Git aliases before invoking Git."""
    argv = [real_git, *arguments]
    builtins = None
    for _ in range(10):
        blocked = find_blocked_git_log_option(argv)
        if blocked:
            return refusal(blocked)
        index = _skip_git_global_options(argv, 1)
        if index >= len(argv):
            return None
        command = argv[index]
        if builtins is None:
            result = subprocess.run(
                [real_git, "--list-cmds=builtins"],
                capture_output=True,
                text=True,
                check=True,
            )
            builtins = set(result.stdout.splitlines())
        if command in builtins:
            return None
        config = subprocess.run(
            [real_git, *argv[1:index], "config", "--get", "alias." + command],
            capture_output=True,
            text=True,
        )
        if config.returncode == 1:
            return None
        if config.returncode != 0:
            return "Cannot validate Git alias: " + config.stderr.strip()
        alias = config.stdout.strip()
        if alias.startswith("!"):
            return "Shell Git aliases are disabled by the history guard; run an explicit command."
        argv = [real_git, *argv[1:index], *shlex.split(alias), *argv[index + 1 :]]
    return "Git alias expansion exceeded the history guard limit."


if __name__ == "__main__":
    real_git, *arguments = sys.argv[1:]
    error = _check_git_execution(real_git, arguments)
    if error:
        print(error, file=sys.stderr)
        sys.exit(126)
    os.execv(real_git, [real_git, *arguments])
