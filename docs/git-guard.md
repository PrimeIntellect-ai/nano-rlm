# Git history guard

With `allow_git=False`, Nano-RLM rejects history-wide `git log` options such as
`--all`, `--branches`, `--remotes`, `--tags`, `--glob`, `--alternate-refs`, and
reflog traversal. Ordinary status, diff, add, commit, and current-branch log
operations remain available. `--` still separates path arguments, so a file
named `--all` is permitted. An explicit trusted `allow_git=True` disables the
checks and does not install a launcher.

## Enforcement

The harness checks literal commands before execution, including newline and
background separators, grouping, common command prefixes, and nested shell
scripts. Quoted text and comments are not treated as commands. These checks give
an immediate refusal without starting work when the invocation is recognizable.

A Git launcher at the front of PATH also checks the arguments that reach Git
at execution time. The harness installs it for native Bash, supervisor-owned
Bash jobs, and IPython kernels. Programs launched by those processes inherit it,
including shell magics and Python subprocesses. This catches dynamically built
arguments, loops, command substitution, and scripts passed through stdin without
having to interpret arbitrary Python or Bash in advance.

The launcher checks ordinary Git aliases using the same Git configuration and
global command options. Shell aliases (`alias.name = !...`) are refused because
they execute arbitrary code inside Git's alias machinery. Built-in commands take
precedence over aliases, matching Git. The launcher does not rely on the
model-controlled `RLM_ALLOW_GIT` environment variable to decide whether to enforce
the policy. A refused execution exits with status 126.

The launcher uses the harness Python interpreter and its installed guard module;
Git and that interpreter must remain available for the runtime's lifetime.

## Limits and observed bypasses

This is a guardrail for the stated Git-log policy, not a security sandbox or a
promise that inaccessible history has been removed. Agents are trusted
collaborators with arbitrary Python, Bash, and shared filesystem access.

These simple bypasses were demonstrated against a disposable repository with a
separate `hidden` branch:

```bash
# Bypass PATH lookup and the literal-command check.
real=/usr/bin/git
"$real" log --all

# Read another branch using an operation outside the restricted-log policy.
git show hidden:hidden.txt
```

The first example assumes Git is installed at that path. A model could also
change PATH, invoke a copied Git executable, modify the launcher, or read Git
objects directly with Python. These are not prohibitively difficult attacks.
Absolute literal `git log --all` invocations are caught by the text check, but
that cannot establish a general boundary against dynamically constructed code.

Preventing access to forbidden history requires controlling the data and the
execution environment: supply a checkout whose object database contains only
permitted history, prevent access to other repositories and host executables,
and restrict network access that could retrieve the original repository. Merely
hiding refs is insufficient because objects may remain in packs or reflogs.
Those changes belong in the runtime/sandbox contract. The command guard does not
sanitize or rewrite the user's repository.

## Validation

Tests cover multiline commands, background separators, loops, shell prefixes,
substitution, nested Bash, stdin scripts, dynamic Python subprocess arguments,
Git aliases, abbreviated restricted options, and ordinary Git operations. An
integration test exercises a real IPython kernel, a supervisor Bash job, and the
native Bash path. Tests use temporary repositories and do not read project
history for task solutions.
