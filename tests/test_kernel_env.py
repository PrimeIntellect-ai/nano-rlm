from rlm.tools.ipython import build_kernel_env


def test_kernel_env_passes_toolchain_variables_and_drops_secrets():
    environ = {
        "PATH": "/usr/bin",
        "HOME": "/root",
        "PYTHONPATH": "/app/lib:/app",
        "PYTHONDONTWRITEBYTECODE": "1",
        "GOPATH": "/root/go",
        "GOMODCACHE": "/root/go/pkg/mod",
        "GOFLAGS": "-mod=mod",
        "NODE_OPTIONS": "--max-old-space-size=4096",
        "NPM_CONFIG_CACHE": "/root/.npm",
        "LD_LIBRARY_PATH": "/usr/local/lib",
        "PIP_INDEX_URL": "https://pypi.org/simple",
        # must not pass
        "GOOGLE_APPLICATION_CREDENTIALS": "/secrets/gcp.json",
        "AWS_SECRET_ACCESS_KEY": "x",
        "OPENAI_API_KEY": "x",
        "GITHUB_TOKEN": "x",
        "NPM_CONFIG_AUTHTOKEN": "x",
        "PYTHON_PASSWORD": "x",
        "PIP_EXTRA_INDEX_URL": "https://user:tok@private.example/simple",
        "GOPROXY": "https://user:tok@proxy.example",
        "PYTHONHOME": "/opt/other",
        "UV_PYTHON": "3.12",
        "DOCKER_HOST": "unix:///var/run/docker.sock",
        "BUNDLE_GEMS__EXAMPLE__COM": "user:pass",
        "RANDOM_APP_SETTING": "x",
        "npm_config_cache": "/lower",
    }
    env = build_kernel_env({"TASK_FLAG": "1"}, environ=environ)
    for key in (
        "PATH",
        "HOME",
        "PYTHONPATH",
        "PYTHONDONTWRITEBYTECODE",
        "GOPATH",
        "GOMODCACHE",
        "GOFLAGS",
        "NODE_OPTIONS",
        "NPM_CONFIG_CACHE",
        "LD_LIBRARY_PATH",
        "PIP_INDEX_URL",
    ):
        assert env[key] == environ[key], key
    assert env["TASK_FLAG"] == "1"
    for key in (
        "GOOGLE_APPLICATION_CREDENTIALS",
        "AWS_SECRET_ACCESS_KEY",
        "OPENAI_API_KEY",
        "GITHUB_TOKEN",
        "NPM_CONFIG_AUTHTOKEN",
        "PYTHON_PASSWORD",
        "PIP_EXTRA_INDEX_URL",
        "GOPROXY",
        "PYTHONHOME",
        "UV_PYTHON",
        "DOCKER_HOST",
        "BUNDLE_GEMS__EXAMPLE__COM",
        "RANDOM_APP_SETTING",
        "npm_config_cache",
    ):
        assert key not in env, key
    assert env["NO_COLOR"] == "1"


def test_explicit_task_env_overrides_inherited():
    env = build_kernel_env(
        {"PYTHONPATH": "/task"}, environ={"PYTHONPATH": "/image", "PATH": "/bin"}
    )
    assert env["PYTHONPATH"] == "/task" and env["PATH"] == "/bin"
