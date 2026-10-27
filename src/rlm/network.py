"""Destination rules for search results: domains, wildcard hosts, and URL origins."""

from fnmatch import fnmatchcase
from urllib.parse import urlsplit

from pydantic import AnyHttpUrl, ValidationError

NetworkBlocklist = tuple[str, ...]


def network_url_is_blocked(block: NetworkBlocklist, value: object) -> bool:
    """Exclude matching or unverifiable HTTP URLs when a blocklist is active."""
    if not block:
        return False
    try:
        url = AnyHttpUrl(value)
    except ValidationError:
        return True
    host = url.host.strip("[]").lower().rstrip(".")
    for rule in block:
        try:
            parsed = urlsplit(rule if "://" in rule else f"//{rule}")
            port = parsed.port
            pattern = (parsed.hostname or "").lower().rstrip(".")
            if not pattern.isascii():
                pattern = AnyHttpUrl(f"http://{pattern}").host.rstrip(".")
        except ValueError:
            continue
        if parsed.scheme and port is None:
            port = 443 if parsed.scheme == "https" else 80
        if not pattern or (parsed.scheme and parsed.scheme != url.scheme):
            continue
        if port is not None and port != url.port:
            continue
        if fnmatchcase(host, pattern) or host.endswith(f".{pattern}"):
            return True
    return False
