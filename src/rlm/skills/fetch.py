"""Built-in ``fetch`` skill — retrieve a webpage and return its cleaned text.

Enabled via ``RLM_SKILLS``; pre-imported into the IPython kernel so the agent calls
``await fetch(url="...")``. Gives the model a proper tool for reading a webpage instead of
hand-rolling ``curl``/``requests``/``urllib`` (which tend to return raw HTML, spam, or errors).
Pairs with the ``search`` skill: ``search`` finds URLs, ``fetch`` reads them.
"""

from __future__ import annotations

import html as _html
import re

import httpx

DEFAULT_MAX_CHARS = 20_000

_TAG_BLOCKS = re.compile(r"(?is)<(script|style|noscript|template|svg)[^>]*>.*?</\1>")
_TAGS = re.compile(r"(?s)<[^>]+>")
_WS = re.compile(r"\s+")


def html_to_text(body: str) -> str:
    """Strip scripts/styles/tags and collapse whitespace into readable text."""
    body = _TAG_BLOCKS.sub(" ", body)
    body = _TAGS.sub(" ", body)
    return _WS.sub(" ", _html.unescape(body)).strip()


async def run(url: str, *, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Fetch a webpage and return its cleaned text content (truncated to ``max_chars``).

    Args:
        url: The webpage to fetch (``https://`` assumed if no scheme).
        max_chars: Truncate the returned text to this many characters.

    Returns:
        ``URL: <url>`` followed by the webpage's cleaned text, or a short error string.
    """
    if not re.match(r"^https?://", url):
        url = "https://" + url
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=45) as client:
            response = await client.get(
                url, headers={"User-Agent": "Mozilla/5.0 (compatible; rlm-fetch)"}
            )
        response.raise_for_status()
    except Exception as exc:
        return f"Error fetching {url}: {type(exc).__name__}: {exc}"
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
