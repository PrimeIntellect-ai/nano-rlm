"""Built-in ``fetch`` skill — read the text of a web page.

Enabled via ``RLM_SKILLS``; pre-imported into the IPython kernel so the agent calls
``await fetch(url="...")``. Pairs with ``search`` (which returns only title/URL/snippet):
``search`` finds candidate URLs, ``fetch`` reads one. Needs live egress (the sandbox must
be allowed to reach the target host).

Returns readable text, not raw HTML: scripts/styles are stripped and tags collapsed to
whitespace, so a page's content enters the REPL without a wall of markup. Output is capped
so a single fetch cannot blow the context; raise ``max_chars`` to read more.
"""

from __future__ import annotations

import re

import httpx

_SCRIPT_STYLE = re.compile(r"<(script|style|noscript)\b[^>]*>.*?</\1>", re.I | re.S)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")
_BLANKLINES = re.compile(r"\n\s*\n\s*")


def _html_to_text(html: str) -> str:
    import html as _htmlmod

    text = _SCRIPT_STYLE.sub(" ", html)
    text = re.sub(r"<(br|/p|/div|/li|/h[1-6]|/tr)\b[^>]*>", "\n", text, flags=re.I)
    text = _TAG.sub(" ", text)
    text = _htmlmod.unescape(text)
    text = _WS.sub(" ", text)
    text = _BLANKLINES.sub("\n\n", text)
    return text.strip()


async def run(url: str, *, max_chars: int = 8000) -> str:
    """Fetch a URL and return its readable text content.

    Args:
        url: The absolute URL to fetch (http/https).
        max_chars: Maximum characters of extracted text to return (default 8000).

    Returns:
        Extracted page text (truncated to ``max_chars``), or an ``Error: ...`` string.
    """
    if not url.lower().startswith(("http://", "https://")):
        return f"Error: url must be http(s), got {url!r}"
    try:
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=45, headers={"User-Agent": "Mozilla/5.0 (rlm-fetch)"}
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
    except Exception as e:  # network/HTTP errors are data for the agent, not a crash
        return f"Error fetching {url}: {type(e).__name__}: {e}"

    ctype = resp.headers.get("content-type", "")
    body = resp.text
    text = _html_to_text(body) if ("html" in ctype or body.lstrip().startswith("<")) else body
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[... truncated at {max_chars} chars; call fetch again with a larger max_chars to read more ...]"
    return text or f"(no extractable text at {url}; content-type {ctype!r})"
