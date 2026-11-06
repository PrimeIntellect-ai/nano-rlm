"""Built-in ``search`` skill — web search via Serper or You.com Eco Search.

Enabled via ``RLM_SKILLS``; pre-imported into the IPython kernel so the agent calls
``await search(query="...")``. Needs the provider's API key. Ported from the Serper
``websearch`` skill in research-environments/rlm_browsecomp.
"""

from __future__ import annotations

import os
from typing import Literal

import httpx

from rlm.network import NetworkBlocklist, network_url_is_blocked

SERPER_URL = "https://google.serper.dev/search"
ECO_URL = "https://ydc-index.io/v1/eco_search"

SearchProvider = Literal["serper", "eco"]


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


def search(
    query: str,
    num_results: int = 5,
    *,
    api_key: str | None = None,
    network_block: NetworkBlocklist = (),
) -> str:
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
    return _format_search_results(organic, query, num_results, network_block)


async def run_with_api_key(
    api_key: str | None,
    query: str,
    *,
    num_results: int = 5,
    network_block: NetworkBlocklist = (),
    provider: SearchProvider = "serper",
) -> str:
    """Run a web search via Serper or You.com Eco Search and return formatted results.

    Args:
        query: Web search query.
        num_results: Number of results to return.
        provider: The search provider; ``api_key`` is its key.

    Returns:
        One formatted text string containing titles, URLs, and snippets.
    """
    if not api_key:
        key_name = "ECO_API_KEY" if provider == "eco" else "SERPER_API_KEY"
        return f"Error: {key_name} environment variable is not set"
    try:
        async with httpx.AsyncClient() as client:
            if provider == "eco":
                response = await client.get(
                    ECO_URL,
                    params={"query": query},
                    headers={"X-API-Key": api_key},
                    timeout=45,
                )
            else:
                response = await client.post(
                    SERPER_URL,
                    json={"q": query},
                    headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                    timeout=45,
                )
        response.raise_for_status()
        if provider == "eco":
            results = _eco_results(response.json())
        else:
            results = response.json().get("organic") or []
    except Exception:
        raise RuntimeError("web search is unavailable") from None
    return _format_search_results(results, query, num_results, network_block)


async def run(
    query: str, *, num_results: int = 5, network_block: NetworkBlocklist = ()
) -> str:
    """Run search directly using ``SERPER_API_KEY`` from the current process."""
    return await run_with_api_key(
        os.environ.get("SERPER_API_KEY"),
        query,
        num_results=num_results,
        network_block=network_block,
    )


def _eco_results(payload: dict) -> list[dict]:
    """Map Eco Search web results onto Serper's ``title``/``link``/``snippet`` fields."""
    return [
        {
            "title": result.get("title"),
            "link": result.get("url"),
            "snippet": result.get("description"),
        }
        for result in payload.get("results", {}).get("web", [])
    ]


def _format_search_results(
    results: list[dict],
    query: str,
    num_results: int,
    network_block: NetworkBlocklist,
) -> str:
    if network_block:
        results = [
            result
            for result in results
            if not network_url_is_blocked(network_block, result.get("link"))
        ]
    return format_results(results[:num_results], query)
