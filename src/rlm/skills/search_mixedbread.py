"""Built-in ``search_mixedbread`` skill — Mixedbread Basic Web Search via the ``mixedbread/web`` web store.

Enabled via ``RLM_SKILLS``; pre-imported into the IPython kernel so the agent calls
``await search_mixedbread(query="...")``. Needs ``MIXEDBREAD_API_KEY``. Calls the same
``/v1/stores/search`` endpoint as the rest of Mixedbread's store API with
``store_identifiers=["mixedbread/web"]``; results are reranked and return title, URL,
relevance score, and page-content excerpts per result.
"""

from __future__ import annotations

import asyncio
import os

import httpx

MIXEDBREAD_URL = "https://api.mixedbread.com/v1/stores/search"


def format_results(chunks, query: str) -> str:
    """Format Mixedbread web-store chunks as title/URL/score/excerpt blocks."""
    sections = []
    for i, chunk in enumerate(chunks, 1):
        if not isinstance(chunk, dict):
            continue
        metadata = chunk.get("metadata") or {}
        title = (metadata.get("title") or "").strip() or "Untitled"
        url = (metadata.get("url") or chunk.get("filename") or "").strip()
        lines = [f"Result {i}: {title}"]
        if url:
            lines.append(f"URL: {url}")
        score = chunk.get("score")
        if score is not None:
            lines.append(f"Score: {score}")
        text = (chunk.get("text") or "").strip()
        if text:
            lines.append(f"Content: {text}")
        sections.append("\n".join(lines))
    if not sections:
        return f"No results returned for query: {query}"
    return "\n\n---\n\n".join(sections)


def search(query: str, num_results: int = 5) -> str:
    """Run a synchronous Mixedbread Basic Web Search and return formatted results."""
    api_key = os.environ.get("MIXEDBREAD_API_KEY", "")
    if not api_key:
        return "Error: MIXEDBREAD_API_KEY environment variable is not set"
    try:
        num_results = max(1, int(num_results))
    except (TypeError, ValueError):
        num_results = 5
    response = httpx.post(
        MIXEDBREAD_URL,
        json={
            "query": query,
            "store_identifiers": ["mixedbread/web"],
            "top_k": num_results,
        },
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=45,
    )
    response.raise_for_status()
    chunks = response.json().get("data") or []
    return format_results(chunks[:num_results], query)


async def run(query: str, *, num_results: int = 5) -> str:
    """Run a Mixedbread Basic Web Search (web store ``mixedbread/web``) and return formatted results.

    Args:
        query: Web search query.
        num_results: Number of results to return (maps to ``top_k``).

    Returns:
        Formatted results (title, URL, score, content excerpt).
    """
    return await asyncio.to_thread(search, query, num_results)
