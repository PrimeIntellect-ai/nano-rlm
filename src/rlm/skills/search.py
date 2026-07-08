"""Built-in ``search`` skill — web search via Serper.

Enabled via ``RLM_SKILLS``; pre-imported into the IPython kernel so the agent calls
``await search(query="...")``, or ``await search(query=["...", "..."])`` to batch
several queries into one API call. Needs ``SERPER_API_KEY``. Ported from the Serper
``websearch`` skill in research-environments/rlm_browsecomp.
"""

from __future__ import annotations

import asyncio
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


def search(queries: list[str], num_results: int = 10) -> str:
    """Run one Serper API call for one or more queries and return formatted results."""
    api_key = os.environ.get("SERPER_API_KEY", "")
    if not api_key:
        return "Error: SERPER_API_KEY environment variable is not set"
    payload = [{"q": query, "num": num_results} for query in queries]
    response = httpx.post(
        SERPER_URL,
        json=payload[0] if len(payload) == 1 else payload,
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        timeout=45,
    )
    response.raise_for_status()
    data = response.json()
    responses = data if isinstance(data, list) else [data]
    sections = [
        format_results((r.get("organic") or [])[:num_results], query)
        for query, r in zip(queries, responses, strict=True)
    ]
    if len(sections) == 1:
        return sections[0]
    return "\n\n==========\n\n".join(
        f'Results for query "{query}":\n\n{section}'
        for query, section in zip(queries, sections)
    )


async def run(query: str | list[str], *, num_results: int = 10) -> str:
    """Run web search(es) via Serper and return formatted results.

    Args:
        query: A search query, or a list of queries batched into one API call.
        num_results: Number of results to return per query.

    Returns:
        Formatted results (title, URL, snippet); one section per query when batched.
    """
    queries = [query] if isinstance(query, str) else list(query)
    if not queries:
        return "Error: query must be a non-empty string or list of strings"
    return await asyncio.to_thread(search, queries, num_results)
