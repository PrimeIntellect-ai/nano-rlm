"""Built-in ``search`` skill — web search via Serper.

Enabled via ``RLM_SKILLS``; pre-imported into the IPython kernel so the agent calls
``await search(query="...")``. Needs ``SERPER_API_KEY``. Ported from the Serper ``websearch``
skill in research-environments/rlm_browsecomp.
"""

from __future__ import annotations

import os

import httpx

SERPER_URL = "https://google.serper.dev/search"


def format_results(results, query: str) -> str:
    sections: list[str] = []
    for i, result in enumerate(results, 1):
        title = (result.get("title") or "").strip() or "Untitled"
        lines = [f"Result {i}: {title}"]
        link = (result.get("link") or "").strip()
        if link:
            lines.append(f"URL: {link}")
        snippet = (result.get("snippet") or "").strip()
        if snippet:
            lines.append(f"  - {snippet}")
        sections.append("\n".join(lines))
    if not sections:
        return f"No results returned for query: {query}"
    return "\n\n---\n\n".join(sections)


def search(query: str, num_results: int = 5, *, api_key: str | None = None) -> str:
    """Run a synchronous Serper web search and return formatted results."""
    if api_key is None:
        api_key = os.environ.get("SERPER_API_KEY", "")
    if not api_key:
        return "Error: SERPER_API_KEY environment variable is not set"
    try:
        response = httpx.post(
            SERPER_URL,
            json={"q": query},
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            timeout=45,
        )
        response.raise_for_status()
    except Exception:
        raise RuntimeError("web search is unavailable") from None
    organic = response.json().get("organic") or []
    return format_results(organic[:num_results], query)


async def run_with_api_key(
    api_key: str | None, query: str, *, num_results: int = 5
) -> str:
    """Run a web search via Serper and return formatted results.

    Args:
        query: Web search query.
        num_results: Number of results to return.

    Returns:
        One formatted text string containing titles, URLs, and snippets.
    """
    if not api_key:
        return "Error: SERPER_API_KEY environment variable is not set"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                SERPER_URL,
                json={"q": query},
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                timeout=45,
            )
        response.raise_for_status()
        organic = response.json().get("organic") or []
    except Exception:
        raise RuntimeError("web search is unavailable") from None
    return format_results(organic[:num_results], query)


async def run(query: str, *, num_results: int = 5) -> str:
    """Run search directly using ``SERPER_API_KEY`` from the current process."""
    return await run_with_api_key(
        os.environ.get("SERPER_API_KEY"), query, num_results=num_results
    )
