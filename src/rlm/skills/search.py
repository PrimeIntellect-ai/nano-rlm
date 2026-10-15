"""Built-in ``search`` skill — web search via Serper or you.com.

Enabled via ``RLM_SKILLS``; pre-imported into the IPython kernel so the agent calls
``await search(query="...")``. The backend is ``RLM_SEARCH_ENGINE``: ``serper`` (default, ``SERPER_API_KEY``),
``youcom`` (you.com Search API, ``YOU_API_KEY``), or ``youcom-eco`` (you.com's
cheap eco_search endpoint, same key). ``RLM_SEARCH_ENGINE`` also selects the
backend for the supervisor's brokered tool.
"""

from __future__ import annotations

import os

import httpx

SERPER_URL = "https://google.serper.dev/search"
YOUCOM_URL = "https://ydc-index.io/v1/search"
YOUCOM_ECO_URL = "https://ydc-index.io/v1/eco_search"

DEFAULT_NUM_RESULTS = 3


def format_results(results, query: str) -> str:
    sections: list[str] = []
    for i, result in enumerate(results, 1):
        title = (result.get("title") or "").strip() or "Untitled"
        lines = [f"Result {i}: {title}"]
        link = (result.get("link") or result.get("url") or "").strip()
        if link:
            lines.append(f"URL: {link}")
        snippet = (result.get("snippet") or result.get("description") or "").strip()
        if snippet:
            lines.append(f"  - {snippet}")
        sections.append("\n".join(lines))
    if not sections:
        return f"No results returned for query: {query}"
    return "\n\n---\n\n".join(sections)


def _engine() -> str:
    engine = os.environ.get("RLM_SEARCH_ENGINE", "serper").strip().lower()
    if engine not in ("serper", "youcom", "youcom-eco"):
        engine = "serper"
    return engine


def search(query: str, num_results: int = DEFAULT_NUM_RESULTS, *, api_key: str | None = None) -> str:
    """Run a synchronous web search and return formatted results."""
    if api_key is None:
        api_key = os.environ.get("SERPER_API_KEY", "")
    engine = _engine()
    if engine == "youcom-eco":
        api_key = os.environ.get("YOU_API_KEY", "")
        response = httpx.get(
            YOUCOM_ECO_URL,
            params={"query": query},
            headers={"X-API-Key": api_key},
            timeout=45,
        )
        results = (response.json().get("results") or {}).get("web") or []
    elif engine == "youcom":
        api_key = os.environ.get("YOU_API_KEY", "")
        response = httpx.post(
            YOUCOM_URL,
            json={"query": query, "count": num_results},
            headers={"X-API-Key": api_key, "Content-Type": "application/json"},
            timeout=45,
        )
        results = (response.json().get("results") or {}).get("web") or []
    else:
        api_key = api_key or os.environ.get("SERPER_API_KEY", "")
        response = httpx.post(
            SERPER_URL,
            json={"q": query, "num": num_results},
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            timeout=45,
        )
        results = response.json().get("organic") or []
    response.raise_for_status()
    return format_results(results[:num_results], query)


async def run_with_api_key(
    api_key: str | None, query: str, *, num_results: int = DEFAULT_NUM_RESULTS
) -> str:
    """Run a web search and return formatted results.

    ``api_key`` is the Serper key brokered by the runtime contract; when
    ``RLM_SEARCH_ENGINE`` is ``youcom``, the you.com key is read from
    ``YOU_API_KEY`` in the process environment instead.

    Args:
        query: Web search query.
        num_results: Number of results to return.

    Returns:
        One formatted text string containing titles, URLs, and snippets.
    """
    engine = _engine()
    if engine != "serper":
        api_key = os.environ.get("YOU_API_KEY", "")
    if not api_key:
        var = "YOU_API_KEY" if engine != "serper" else "SERPER_API_KEY"
        return f"Error: {var} environment variable is not set"
    try:
        async with httpx.AsyncClient() as client:
            if engine == "youcom-eco":
                response = await client.get(
                    YOUCOM_ECO_URL,
                    params={"query": query},
                    headers={"X-API-Key": api_key},
                    timeout=45,
                )
                results = (response.json().get("results") or {}).get("web") or []
            elif engine == "youcom":
                response = await client.post(
                    YOUCOM_URL,
                    json={"query": query, "count": num_results},
                    headers={"X-API-Key": api_key, "Content-Type": "application/json"},
                    timeout=45,
                )
                results = (response.json().get("results") or {}).get("web") or []
            else:
                response = await client.post(
                    SERPER_URL,
                    json={"q": query, "num": num_results},
                    headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                    timeout=45,
                )
                results = response.json().get("organic") or []
        response.raise_for_status()
    except Exception:
        raise RuntimeError("web search is unavailable") from None
    return format_results(results[:num_results], query)


async def run(query: str, *, num_results: int = DEFAULT_NUM_RESULTS) -> str:
    """Run search directly using ``SERPER_API_KEY`` from the current process."""
    return await run_with_api_key(
        os.environ.get("SERPER_API_KEY"), query, num_results=num_results
    )
