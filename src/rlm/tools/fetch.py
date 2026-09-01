"""Native ``fetch`` builtin tool — retrieve a webpage and return its cleaned text.

Tool twin of the ``fetch`` skill (``rlm.skills.fetch``): same cleaning, truncation,
and error semantics, exposed as a native tool call instead of a REPL function.
Opt-in: outside the default tool set, so it runs only when the runtime
contract's ``builtin_tools`` names it — a network-capable tool stays off by
default.
"""

from __future__ import annotations

from typing import Any

import httpx

from rlm.skills.fetch import (
    DEFAULT_MAX_CHARS,
    REQUEST_HEADERS,
    REQUEST_TIMEOUT,
    fetch_error,
    normalize_url,
    render_response,
)
from rlm.tools.base import ToolContext, ToolOutcome

FETCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "fetch",
        "description": (
            "Fetch a webpage and return its cleaned text (scripts, styles, and tags "
            "stripped). Use it to read URLs, e.g. ones found via search."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The webpage to fetch (https:// assumed if no scheme).",
                },
                "max_chars": {
                    "type": "integer",
                    "description": (
                        "Truncate the returned text to this many characters "
                        f"(default {DEFAULT_MAX_CHARS})."
                    ),
                },
            },
            "required": ["url"],
        },
    },
}


def fetch_page(url: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Synchronous twin of ``rlm.skills.fetch.run`` (tools execute in a worker thread)."""
    url = normalize_url(url)
    try:
        with httpx.Client(follow_redirects=True, timeout=REQUEST_TIMEOUT) as client:
            response = client.get(url, headers=REQUEST_HEADERS)
        response.raise_for_status()
    except Exception as exc:
        return fetch_error(url, exc)
    return render_response(url, response, max_chars)


class FetchTool:
    """One webpage read per call; returns cleaned text or a short error string."""

    name = "fetch"

    def schema(self) -> dict[str, Any]:
        return FETCH_SCHEMA

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolOutcome:
        url = args.get("url")
        if not isinstance(url, str) or not url.strip():
            return ToolOutcome(content="Error: url is required")
        max_chars = args.get("max_chars", DEFAULT_MAX_CHARS)
        if not isinstance(max_chars, int) or max_chars <= 0:
            return ToolOutcome(content="Error: max_chars must be a positive integer")
        return ToolOutcome(content=fetch_page(url.strip(), max_chars=max_chars))
