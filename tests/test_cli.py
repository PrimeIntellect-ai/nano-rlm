import resource

from rlm.cli import raise_fd_limit


def test_raise_fd_limit_lifts_soft_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, hard))
    try:
        raise_fd_limit(target=512)
        assert resource.getrlimit(resource.RLIMIT_NOFILE)[0] == 512
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
