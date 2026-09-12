"""Restrict git access to the current branch's history at the tool-call level.

Ordinary git commands are allowed. The guard refuses invocations that reach beyond the
current branch: history-wide ``git log`` options (``--all``, ``--remotes``, ...), history
subcommands given another ref (``git show origin/main``), listing branches/tags/reflog/
remotes, and ``clone``/``fetch``/``pull`` (an audit of 365 SWE-bench Pro rollouts found the
only solution leakage came from cloning an upstream dependency and mining its history).
``git status``, ``git diff``, ``git log`` on the current branch and ``git stash`` stay usable.

- split a bash command on ``&&``, ``||``, ``;`` and ``|``
- if a segment invokes ``git log`` with a restricted history flag, refuse

Standalone execution can disable the guard with ``RLM_ALLOW_GIT=1``. Managed
execution passes the resolved policy explicitly.
"""

from __future__ import annotations

import ast
import os
import re
import shlex

REFUSAL_TEMPLATE = (
    "Git command '{cmd}' is not allowed. Use current-branch history only: no other "
    "branches, tags, remotes, reflog, clones or fetches."
)

# Reuse the mini_swe_agent_plus separators verbatim so behavior matches; command
# substitutions and backticks are split as well so `echo $(git log --all)` is seen.
_SEPARATORS = re.compile(r"&&|\|\||;|\||\$\(|`|\)")

# Subcommands that reach beyond the current branch's history by construction.
_BLOCKED_SUBCOMMANDS = {
    "clone",
    "fetch",
    "pull",
    "ls-remote",
    "remote",
    "reflog",
    "for-each-ref",
    "show-ref",
    "describe",
    "tag",
    "bundle",
    "fsck",
    # object-store enumeration: with refs removed, dangling upstream commits are still
    # discoverable this way (seen in the wild as `cat-file --batch-all-objects`)
    "count-objects",
    "verify-pack",
    "unpack-objects",
    "pack-refs",
    "prune",
    "gc",
    "repack",
}
_BLOCKED_OPTION_ANYWHERE = {
    "--batch-all-objects",
    "--lost-found",
    "--dangling",
    "--unreachable",
}
# Reading the repository's internals directly is the same leak by another route.
_GIT_INTERNALS_RE = re.compile(
    r"\.git/(objects|packed-refs|refs|logs|ORIG_HEAD|FETCH_HEAD|lost-found|info/refs)"
)
# `git branch` is fine for the current branch but not for listing others.
_BLOCKED_BRANCH_OPTIONS = {
    "-a",
    "-r",
    "--all",
    "--remotes",
    "--list",
    "-l",
    "--contains",
    "--merged",
    "--no-merged",
}
# Revision arguments that name other refs; checked on every history-reading subcommand.
_REMOTE_REF_RE = re.compile(
    r"^(?:refs/(?:remotes|tags|heads)/|remotes/|origin/|upstream/|tags/|[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+@\{)"
)
_HISTORY_SUBCOMMANDS = {
    "log",
    "show",
    "rev-list",
    "ls-tree",
    "diff",
    "grep",
    "cat-file",
    "shortlog",
    "whatchanged",
    "format-patch",
    "archive",
    "checkout",
    "restore",
    "switch",
    "worktree",
    "cherry-pick",
    "merge-base",
    "name-rev",
    "rev-parse",
    "blame",
    "annotate",
}
# Shells whose -c/-lc string is itself a command line to scan.
_SHELL_WRAPPERS = {"bash", "sh", "zsh", "dash"}

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
    """Return the offending token if ``command`` asks for broad git history.

    Splits on ``&&``, ``||``, ``;``, ``|`` so chained calls like
    ``cd /repo && git log --all`` are caught. Returns ``None`` if nothing is
    blocked or if the resolved policy allows unrestricted history.
    """
    if _git_allowed(allow_git):
        return None
    # Backslash line continuations are one command line.
    command = command.replace("\\\n", " ")
    # `bash -lc "cd /app && git log --all"`: unwrap the shell wrapper before splitting, so
    # the inner command line is scanned whole instead of being cut at its own separators.
    try:
        argv = shlex.split(command)
    except ValueError:
        argv = []
    if argv and argv[0].rsplit("/", 1)[-1] in _SHELL_WRAPPERS:
        for i, token in enumerate(argv[1:], 1):
            if token in ("-c", "-lc", "-ic", "-lic") and i + 1 < len(argv):
                blocked = find_blocked_command(argv[i + 1], allow_git=False)
                if blocked is not None:
                    return blocked
    internals = _GIT_INTERNALS_RE.search(command)
    if internals:
        return internals.group(0)
    for segment in _SEPARATORS.split(command):
        blocked = find_blocked_git_log_option(_split_segment(segment))
        if blocked is not None:
            return blocked
    return None


def refusal(cmd: str) -> str:
    return REFUSAL_TEMPLATE.format(cmd=cmd)


def _split_segment(segment: str) -> list[str]:
    try:
        return shlex.split(segment)
    except ValueError:
        return segment.strip().split()


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
    return any(
        token == option or token.startswith(f"{option}=")
        for option in (*_RESTRICTED_LOG_OPTION_PREFIXES, *_RESTRICTED_LOG_OPTIONS)
    )


def find_blocked_git_log_option(argv: list[str]) -> str | None:
    """Return the offending token if ``argv`` is a git invocation that reaches beyond the
    current branch (history options, other refs, remotes, clones), else ``None``."""
    if not argv:
        return None
    # `bash -c "git log --all"`, `sh -lc '...'`: scan the wrapped command line.
    if argv[0].rsplit("/", 1)[-1] in _SHELL_WRAPPERS:
        for i, token in enumerate(argv[1:], 1):
            if token in ("-c", "-lc", "-ic", "-lic") and i + 1 < len(argv):
                return find_blocked_command(argv[i + 1], allow_git=False)
        return None
    if not _is_git_binary(argv[0]):
        return None

    subcommand_index = _skip_git_global_options(argv, 1)
    if subcommand_index >= len(argv):
        return None
    subcommand = argv[subcommand_index]
    rest = argv[subcommand_index + 1 :]
    if subcommand in _BLOCKED_SUBCOMMANDS:
        return subcommand
    for token in rest:
        if token.split("=", 1)[0] in _BLOCKED_OPTION_ANYWHERE:
            return token
    if subcommand == "branch":
        for token in rest:
            if token == "--":
                break
            if token in _BLOCKED_BRANCH_OPTIONS:
                return f"branch {token}"
        return None
    if subcommand not in _HISTORY_SUBCOMMANDS:
        return None
    for token in rest:
        if token == "--":
            return None
        # history-wide options (--all, --branches=..., --reflog, -g, ...) are refused on
        # every history subcommand, not only on `log`: `rev-list --objects --all` and
        # `show --all` reach the same commits
        if _is_restricted_log_option(token):
            return token
        if _REMOTE_REF_RE.match(token):
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
    in_bash_cell = False
    for line in lines:
        if in_bash_cell:
            blocked = find_blocked_command(line, allow_git=allow_git)
            if blocked is not None:
                return blocked
            continue
        if _SHELL_CELL_MAGIC_RE.match(line):
            in_bash_cell = True
            continue
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
