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
        "GOOGLE_APPLICATION_CREDENTIALS": "/secrets/gcp.json",
        "AWS_SECRET_ACCESS_KEY": "x",
        "OPENAI_API_KEY": "x",
        "GITHUB_TOKEN": "x",
        "NPM_CONFIG_AUTHTOKEN": "x",
        "PYTHON_PASSWORD": "x",
        "RANDOM_APP_SETTING": "x",
    }
    env = build_kernel_env({"TASK_FLAG": "1"}, environ=environ)
    for key in ("PATH", "HOME", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "GOPATH", "GOMODCACHE",
                "GOFLAGS", "NODE_OPTIONS", "NPM_CONFIG_CACHE", "LD_LIBRARY_PATH", "TASK_FLAG"):
        assert env[key] == environ.get(key, "1"), key
    for key in ("GOOGLE_APPLICATION_CREDENTIALS", "AWS_SECRET_ACCESS_KEY", "OPENAI_API_KEY",
                "GITHUB_TOKEN", "NPM_CONFIG_AUTHTOKEN", "PYTHON_PASSWORD", "RANDOM_APP_SETTING"):
        assert key not in env, key
    assert env["NO_COLOR"] == "1"
