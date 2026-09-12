from rlm.tools.ipython import build_kernel_env


def test_kernel_env_passes_everything_except_the_blocklist():
    environ = {
        "PATH": "/usr/bin",
        "HOME": "/root",
        "PYTHONPATH": "/app/lib:/app",
        "GOMODCACHE": "/root/go/pkg/mod",
        "NODE_OPTIONS": "--max-old-space-size=4096",
        "npm_config_cache": "/root/.npm",
        "LD_LIBRARY_PATH": "/usr/local/lib",
        "PIP_INDEX_URL": "https://pypi.org/simple",
        "RANDOM_APP_SETTING": "x",
        # blocked by name pattern
        "GOOGLE_APPLICATION_CREDENTIALS": "/secrets/gcp.json",
        "AWS_SECRET_ACCESS_KEY": "x",
        "OPENAI_API_KEY": "x",
        "GITHUB_TOKEN": "x",
        "NPM_CONFIG_AUTHTOKEN": "x",
        "DB_PASSWORD": "x",
        "TLS_PRIVATE_PEM": "x",
        "SESSION_ID": "x",
        # blocked by value (URL credentials)
        "PIP_EXTRA_INDEX_URL": "https://user:tok@private.example/simple",
        "DATABASE_URL": "postgres://u:p@db/app",
        # blocked explicitly
        "PYTHONHOME": "/opt/other",
        "UV_PYTHON": "3.12",
        "DOCKER_HOST": "unix:///var/run/docker.sock",
        "SSH_AUTH_SOCK": "/tmp/agent",
        "BUNDLE_GEMS__EXAMPLE__COM": "user:pass",
        "IPYTHONDIR": "/elsewhere",
        "OPENAI_BASE_URL": "https://api.example",
        "PRIME_TEAM_ID": "team",
        "RLM_BASE_URL": "http://broker",
        "AWS_REGION": "eu-west-1",
    }
    env = build_kernel_env({"TASK_FLAG": "1"}, environ=environ)
    for key in (
        "PATH",
        "HOME",
        "PYTHONPATH",
        "GOMODCACHE",
        "NODE_OPTIONS",
        "npm_config_cache",
        "LD_LIBRARY_PATH",
        "PIP_INDEX_URL",
        "RANDOM_APP_SETTING",
    ):
        assert env[key] == environ[key], key
    assert env["TASK_FLAG"] == "1"
    for key in (
        "GOOGLE_APPLICATION_CREDENTIALS",
        "AWS_SECRET_ACCESS_KEY",
        "OPENAI_API_KEY",
        "GITHUB_TOKEN",
        "NPM_CONFIG_AUTHTOKEN",
        "DB_PASSWORD",
        "TLS_PRIVATE_PEM",
        "SESSION_ID",
        "PIP_EXTRA_INDEX_URL",
        "DATABASE_URL",
        "PYTHONHOME",
        "UV_PYTHON",
        "DOCKER_HOST",
        "SSH_AUTH_SOCK",
        "BUNDLE_GEMS__EXAMPLE__COM",
    ):
        assert key not in env, key
    assert env["NO_COLOR"] == "1"


def test_explicit_task_env_and_private_dirs_override_inherited(tmp_path):
    env = build_kernel_env(
        {"PYTHONPATH": "/task"},
        environ={"PYTHONPATH": "/image", "PATH": "/bin", "IPYTHONDIR": "/elsewhere"},
        private_dir=str(tmp_path),
    )
    assert env["PYTHONPATH"] == "/task" and env["PATH"] == "/bin"
    assert env["IPYTHONDIR"].startswith(str(tmp_path))
