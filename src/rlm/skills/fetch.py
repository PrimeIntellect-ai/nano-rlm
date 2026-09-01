"""Built-in ``fetch`` skill — retrieve a webpage and return its cleaned text.

Enabled via ``RLM_SKILLS``; pre-imported into the IPython kernel so the agent calls
``await fetch(url="...")``. Gives the model a proper tool for reading a webpage instead of
hand-rolling ``curl``/``requests``/``urllib`` (which tend to return raw HTML, spam, or errors).
Pairs with the ``search`` skill: ``search`` finds URLs, ``fetch`` reads them.

Also exposed as a native builtin tool (``rlm.tools.fetch``) with the same semantics.
"""

from __future__ import annotations

import html as _html
import re

import httpx

DEFAULT_MAX_CHARS = 20_000
REQUEST_TIMEOUT = 45
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; rlm-fetch)"}

_TAG_BLOCKS = re.compile(r"(?is)<(script|style|noscript|template|svg)[^>]*>.*?</\1>")
_TAGS = re.compile(r"(?s)<[^>]+>")
_WS = re.compile(r"\s+")


def html_to_text(body: str) -> str:
    """Strip scripts/styles/tags and collapse whitespace into readable text."""
    body = _TAG_BLOCKS.sub(" ", body)
    body = _TAGS.sub(" ", body)
    return _WS.sub(" ", _html.unescape(body)).strip()


def normalize_url(url: str) -> str:
    """Assume ``https://`` when the URL has no scheme."""
    return url if re.match(r"^https?://", url) else "https://" + url


def fetch_error(url: str, exc: Exception) -> str:
    """A short error string — data for the agent, not a crash."""
    return f"Error fetching {url}: {type(exc).__name__}: {exc}"


def render_response(url: str, response: httpx.Response, max_chars: int) -> str:
    """Decode a response, clean HTML to text, truncate, and prefix the URL."""
    content_type = response.headers.get("content-type", "").lower()
    try:
        body = response.text
    except (UnicodeDecodeError, LookupError):
        body = response.content.decode("utf-8", errors="replace")
    is_html = "html" in content_type or "<html" in body[:1000].lower()
    text = html_to_text(body) if is_html else body.strip()
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [truncated at {max_chars} chars]"
    return f"URL: {url}\n\n{text}"


async def run(url: str, *, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Fetch a webpage and return its cleaned text content (truncated to ``max_chars``).

    Args:
        url: The webpage to fetch (``https://`` assumed if no scheme).
        max_chars: Truncate the returned text to this many characters.

    Returns:
        ``URL: <url>`` followed by the webpage's cleaned text, or a short error string.
    """
    url = normalize_url(url)
    try:
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=REQUEST_TIMEOUT
        ) as client:
            response = await client.get(url, headers=REQUEST_HEADERS)
        response.raise_for_status()
    except Exception as exc:
        return fetch_error(url, exc)
    return render_response(url, response, max_chars)
