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
