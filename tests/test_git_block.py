"""Tests for restricted git-history access."""

from __future__ import annotations

from rlm.tools.base import ToolContext
from rlm.tools.git_block import (
    find_blocked_command,
    find_blocked_git_log_option,
    find_blocked_in_ipython,
    refusal,
)
from rlm.types import RLMMetrics, TokenUsage


REFUSAL = "Git history option '--all' is not allowed. Use current-branch history only."


def _ctx() -> ToolContext:
    return ToolContext(
        messages=[],
        metrics=RLMMetrics(),
        total_usage=TokenUsage(),
        last_prompt_tokens=0,
        exec_timeout=10,
    )


# --- argv-level git log restrictions ---


def test_git_status_allowed():
    assert find_blocked_git_log_option(["git", "status"]) is None


def test_current_branch_git_log_allowed():
    assert find_blocked_git_log_option(["git", "log", "--oneline", "-n", "5"]) is None


def test_git_log_all_blocked():
    assert find_blocked_git_log_option(["git", "log", "--all"]) == "--all"


def test_git_log_single_dash_all_blocked():
    assert find_blocked_git_log_option(["git", "log", "-all"]) == "-all"


def test_git_log_remote_history_blocked():
    assert (
        find_blocked_git_log_option(["git", "log", "--remotes=origin/*"])
        == "--remotes=origin/*"
    )


def test_git_log_reflog_blocked():
    assert find_blocked_git_log_option(["git", "log", "-g"]) == "-g"


def test_git_global_options_before_log_are_supported():
    assert (
        find_blocked_git_log_option(
            ["git", "-C", "/repo", "--no-pager", "log", "--all"]
        )
        == "--all"
    )


def test_git_log_path_separator_stops_option_scan():
    assert find_blocked_git_log_option(["git", "log", "--", "--all"]) is None


def test_absolute_git_binary_is_checked():
    assert find_blocked_git_log_option(["/usr/bin/git", "log", "--all"]) == "--all"


# --- find_blocked_command shell predicate ---


def test_plain_git_status_allowed():
    assert find_blocked_command("git status") is None


def test_plain_git_diff_allowed():
    assert find_blocked_command("git diff") is None


def test_git_log_all_command_blocked():
    assert find_blocked_command("git log --all") == "--all"


def test_chained_git_log_all_after_cd_blocked():
    assert find_blocked_command("cd /testbed && git log --all") == "--all"


def test_pipe_separator_git_log_all_blocked():
    assert find_blocked_command("echo hello | git log --all") == "--all"


def test_or_separator_git_log_all_blocked():
    assert find_blocked_command("false || git log --all") == "--all"


def test_semicolon_git_log_all_blocked():
    assert find_blocked_command("ls; git log --all") == "--all"


def test_quoted_path_named_all_unaffected():
    assert find_blocked_command("git log -- '--all'") is None


def test_command_substring_unaffected():
    assert find_blocked_command("echo github --all") is None


def test_allow_git_env_var_disables_restriction(monkeypatch):
    monkeypatch.setenv("RLM_ALLOW_GIT", "1")
    assert find_blocked_command("git log --all") is None


def test_refusal_message_names_restricted_option():
    assert refusal("--all") == REFUSAL


# --- find_blocked_in_ipython ---


def test_ipython_shell_escape_git_status_allowed():
    assert find_blocked_in_ipython("!git status") is None


def test_ipython_shell_escape_git_log_all_blocked():
    assert find_blocked_in_ipython("!git log --all") == "--all"


def test_ipython_double_shell_escape_git_log_all_blocked():
    assert find_blocked_in_ipython("!!git log --all") == "--all"


def test_ipython_chained_shell_escape_git_log_all_blocked():
    assert find_blocked_in_ipython("!cd /repo && git log --all") == "--all"


def test_ipython_bash_cell_magic_git_log_all_blocked():
    code = "%%bash\ncd /repo\ngit log --all"
    assert find_blocked_in_ipython(code) == "--all"


def test_ipython_sx_line_magic_git_log_all_blocked():
    assert find_blocked_in_ipython("%sx git log --all") == "--all"


def test_ipython_pure_python_unaffected():
    assert find_blocked_in_ipython("x = 'git log --all'\nprint(x)") is None


def test_ipython_allow_git_env_var(monkeypatch):
    monkeypatch.setenv("RLM_ALLOW_GIT", "1")
    assert find_blocked_in_ipython("!git log --all") is None


# --- AST-based detection of Python broad-history invocations ---


def test_python_subprocess_run_git_status_allowed():
    code = "import subprocess\nsubprocess.run(['git', 'status'])"
    assert find_blocked_in_ipython(code) is None


def test_python_subprocess_run_list_git_log_all_blocked():
    code = "import subprocess\nsubprocess.run(['git', 'log', '--all'])"
    assert find_blocked_in_ipython(code) == "--all"


def test_python_subprocess_run_shell_string_git_log_all_blocked():
    code = "import subprocess\nsubprocess.run('git log --all', shell=True)"
    assert find_blocked_in_ipython(code) == "--all"


def test_python_subprocess_popen_current_log_allowed():
    code = "import subprocess\nsubprocess.Popen(['git', 'log', '--oneline'])"
    assert find_blocked_in_ipython(code) is None


def test_python_subprocess_call_git_log_all_blocked():
    code = "import subprocess\nsubprocess.call(['git', 'log', '--all'])"
    assert find_blocked_in_ipython(code) == "--all"


def test_python_subprocess_check_call_git_log_all_blocked():
    code = "import subprocess\nsubprocess.check_call(['git', 'log', '--all'])"
    assert find_blocked_in_ipython(code) == "--all"


def test_python_subprocess_check_output_git_log_all_blocked():
    code = "import subprocess\nsubprocess.check_output(['git', 'log', '--all'])"
    assert find_blocked_in_ipython(code) == "--all"


def test_python_os_system_chained_git_log_all_blocked():
    code = 'import os\nos.system("cd /tmp && git log --all")'
    assert find_blocked_in_ipython(code) == "--all"


def test_python_os_popen_git_log_all_blocked():
    code = 'import os\nos.popen("git log --all")'
    assert find_blocked_in_ipython(code) == "--all"


def test_python_subprocess_module_alias_git_log_all_blocked():
    code = "import subprocess as sp\nsp.run(['git', 'log', '--all'])"
    assert find_blocked_in_ipython(code) == "--all"


def test_python_from_subprocess_import_run_git_log_all_blocked():
    code = "from subprocess import run\nrun(['git', 'log', '--all'])"
    assert find_blocked_in_ipython(code) == "--all"


def test_python_single_hop_assignment_alias_git_log_all_blocked():
    code = "import subprocess\nr = subprocess.run\nr(['git', 'log', '--all'])"
    assert find_blocked_in_ipython(code) == "--all"


def test_python_string_literal_no_call_unaffected():
    assert find_blocked_in_ipython("x = 'git log --all'") is None


def test_python_command_starting_with_git_word_unaffected():
    code = "import subprocess\nsubprocess.run(['github', 'log', '--all'])"
    assert find_blocked_in_ipython(code) is None


def test_python_timeit_cell_magic_with_git_log_all_blocked():
    code = '%%timeit\nimport subprocess\nsubprocess.run(["git", "log", "--all"])'
    assert find_blocked_in_ipython(code) == "--all"


def test_python_shell_escape_with_python_git_log_all_blocked():
    code = '!ls\nimport subprocess\nsubprocess.run(["git", "log", "--all"])'
    assert find_blocked_in_ipython(code) == "--all"


def test_python_line_magic_with_python_git_log_all_blocked():
    code = '%timeit pass\nimport subprocess\nsubprocess.run(["git", "log", "--all"])'
    assert find_blocked_in_ipython(code) == "--all"


def test_python_help_question_mark_with_python_git_log_all_blocked():
    code = "import subprocess\nsubprocess.run?\nsubprocess.run(['git', 'log', '--all'])"
    assert find_blocked_in_ipython(code) == "--all"


def test_python_getattr_documented_bypass():
    code = "import subprocess\ngetattr(subprocess, 'run')(['git', 'log', '--all'])"
    assert find_blocked_in_ipython(code) is None


def test_python_multi_hop_alias_documented_bypass():
    code = (
        "import subprocess\nr1 = subprocess.run\nr2 = r1\nr2(['git', 'log', '--all'])\n"
    )
    assert find_blocked_in_ipython(code) is None


def test_python_syntax_error_does_not_refuse():
    assert find_blocked_in_ipython("def broken(:\n    pass") is None


# --- IpythonTool integration ---


def test_ipython_tool_refuses_git_log_all():
    """IpythonTool.execute short-circuits with the refusal before touching the REPL."""
    from rlm.tools.ipython import IpythonTool

    ctx = _ctx()
    ctx.repl = object()
    outcome = IpythonTool().execute({"code": "!git log --all"}, ctx)
    assert outcome.content == REFUSAL


def test_ipython_tool_allows_git_status(monkeypatch):
    from rlm.tools.ipython import IpythonTool

    class StubRepl:
        def execute(self, code, timeout):
            return f"ran: {code}"

    ctx = _ctx()
    ctx.repl = StubRepl()
    outcome = IpythonTool().execute({"code": "!git status"}, ctx)
    assert outcome.content == "ran: !git status"


def test_ipython_tool_passes_through_non_git():
    from rlm.tools.ipython import IpythonTool

    class StubRepl:
        def execute(self, code, timeout):
            return f"ran: {code}"

    ctx = _ctx()
    ctx.repl = StubRepl()
    outcome = IpythonTool().execute({"code": "print(1+1)"}, ctx)
    assert outcome.content == "ran: print(1+1)"


def test_ipython_tool_uses_explicit_execution_policy(monkeypatch):
    from rlm.tools.ipython import IpythonTool

    class StubRepl:
        def execute(self, code, timeout):
            return "abcdefgh"

    monkeypatch.setenv("RLM_ALLOW_GIT", "1")
    ctx = _ctx()
    ctx.repl = StubRepl()
    ctx.allow_git = False

    refused = IpythonTool().execute({"code": "!git log --all"}, ctx)
    passed = IpythonTool().execute({"code": "print('ignored')"}, ctx)

    assert refused.content == REFUSAL
    assert passed.content == "abcdefgh"
    assert (
        "Default: 17s"
        in IpythonTool(17).schema()["function"]["parameters"]["properties"]["timeout"][
            "description"
        ]
    )


def test_shell_boundaries_and_literal_data():
    blocked = [
        "pwd\ngit log --all",
        "pwd & git log --all",
        "if git log --all; then :; fi",
        "for x in one; do git log --all; done",
        "(git log --all)",
        "env X=1 git log --all",
        "command git log --all",
        "exec git log --all",
        "bash -lc 'git log --all'",
        "git log \\\n --all",
        "git log --al",
        "git log -pg",
        "echo $(git log --all)",
    ]
    for command in blocked:
        assert find_blocked_command(command, allow_git=False), command
    for command in [
        "echo 'git log --all'",
        "printf '%s' 'x; git log --all'",
        "git log -- --all",
        "# git log --all\ngit status",
    ]:
        assert find_blocked_command(command, allow_git=False) is None, command


def _private_git_repo(tmp_path):
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
        )

    git("init", "-q", "-b", "main")
    git("config", "user.name", "Guard test")
    git("config", "user.email", "guard@example.invalid")
    git("config", "commit.gpgsign", "false")
    (repo / "--all").write_text("ordinary file")
    git("add", ".")
    git("commit", "-qm", "ordinary commit")
    git("checkout", "-qb", "hidden")
    (repo / "hidden.txt").write_text("PRIVATE_HISTORY_SENTINEL")
    git("add", ".")
    git("commit", "-qm", "PRIVATE_HISTORY_SENTINEL")
    git("checkout", "-q", "main")
    return repo


def test_execution_guard_expansion_aliases_and_normal_git(tmp_path):
    import os
    import shlex
    import subprocess
    import sys
    from rlm.tools.git_block import guarded_git_environment

    repo = _private_git_repo(tmp_path)
    env = guarded_git_environment(dict(os.environ), tmp_path / "guard", allow_git=False)
    blocked = [
        "pwd\ngit log --all",
        'g=git; option=--all; "$g" log "$option"',
        "for x in one; do git log --all; done",
        "env git log --all",
        "bash -c 'git log --all'",
        "printf '%s\\n' 'git log --all' | bash",
        "git log $(printf -- --all)",
        "git -c alias.rlm-test-history='log --all' rlm-test-history",
        "git -c alias.rlm-test-history='!git log --all' rlm-test-history",
        "git log --al",
        "git log -pg",
        shlex.join(
            [
                sys.executable,
                "-c",
                "import subprocess; a=['g'+'it','log','--'+'all']; raise SystemExit(subprocess.call(a))",
            ]
        ),
    ]
    for command in blocked:
        result = subprocess.run(
            ["/bin/bash", "-c", command],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0, command
        assert "not allowed" in result.stderr or "disabled" in result.stderr, (
            command,
            result.stderr,
        )
        assert "PRIVATE_HISTORY_SENTINEL" not in result.stdout, command
    allowed = [
        "git status --short",
        "git -c alias.status='log --all' status --short",
        "git diff",
        "git log --oneline",
        "git log -- --all",
        "git -c alias.rlm-test-history='log --oneline' rlm-test-history",
        "git add -- --all",
        "git commit --allow-empty -qm 'another ordinary commit'",
    ]
    for command in allowed:
        result = subprocess.run(
            ["/bin/bash", "-c", command],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, (command, result.stderr)
        assert "PRIVATE_HISTORY_SENTINEL" not in result.stdout, command
    # A trusted explicit opt-out does not install the wrapper.
    assert guarded_git_environment(
        dict(os.environ), tmp_path / "allowed", allow_git=True
    ) == dict(os.environ)


async def test_execution_guard_in_kernel_jobs_and_native_bash(tmp_path):
    import json
    from conftest import DummyClient, DummyMessage, DummyToolCall
    from rlm.engine import RLMEngine
    from rlm.session import Session
    from rlm.tools.bash import run_bash
    from test_supervisor import _config

    repo = _private_git_repo(tmp_path)
    command = 'g=git; flag=--all; "$g" log "$flag"'
    assert "not allowed" in run_bash(command, 10, cwd=str(repo), allow_git=False)
    session = Session(tmp_path / "session")
    code = """
import subprocess
argv = ['g' + 'it', 'log', '--' + 'all']
p = subprocess.run(argv, capture_output=True, text=True)
assert p.returncode == 126 and 'not allowed' in p.stderr
job = await rlm.shell.start('g=git; flag=--all; "$g" log "$flag"')
"""
    client = DummyClient(
        [
            DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": code})]),
            DummyMessage(tool_calls=[DummyToolCall("wait", {"timeout": 10})]),
            DummyMessage(
                tool_calls=[
                    DummyToolCall(
                        "ipython",
                        {
                            "code": "assert (await job.info()).exit_code == 126; assert 'not allowed' in (await job.read()).text; print('GUARD_OK')"
                        },
                    )
                ]
            ),
            DummyMessage(content="done"),
        ]
    )
    engine = RLMEngine(
        client=client,
        session=session,
        cwd=str(repo),
        runtime_config=_config(max_depth=0),
    )
    try:
        await engine.prompt("exercise guarded commands")
        records = [
            json.loads(line)
            for line in (session.dir / "messages.jsonl").read_text().splitlines()
        ]
        assert any(r.get("content", "").strip() == "GUARD_OK" for r in records)
    finally:
        await engine.aclose()
