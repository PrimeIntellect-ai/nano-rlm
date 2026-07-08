"""Built-in ``search`` skill — web search via Serper, plus ``search.open`` page reading.

Enabled via ``RLM_SKILLS``; pre-imported into the IPython kernel so the agent calls
``await search(query="...")`` and ``await search.open(url="...")``. Search needs
``SERPER_API_KEY``. ``open`` fetches pages through the Jina Reader API when
``JINA_API_KEY`` is set, falling back to a direct fetch parsed locally. Ported from
the Serper ``websearch`` skill in research-environments/rlm_browsecomp.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re

import httpx

__all__ = ["run", "open"]

SERPER_URL = "https://google.serper.dev/search"
JINA_READER_URL = "https://r.jina.ai"


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


def search(query: str, num_results: int = 10) -> str:
    """Run a synchronous Serper web search and return formatted results."""
    api_key = os.environ.get("SERPER_API_KEY", "")
    if not api_key:
        return "Error: SERPER_API_KEY environment variable is not set"
    response = httpx.post(
        SERPER_URL,
        json={"q": query, "num": num_results},
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        timeout=45,
    )
    response.raise_for_status()
    organic = response.json().get("organic") or []
    return format_results(organic[:num_results], query)


async def run(query: str, *, num_results: int = 10) -> str:
    """Run a web search via Serper and return formatted results.

    Use ``await search.open(url=...)`` to read a result page in full.

    Args:
        query: Web search query.
        num_results: Number of results to return.

    Returns:
        Formatted results (title, URL, snippet).
    """
    return await asyncio.to_thread(search, query, num_results)


def _pdf_to_text(pdf_bytes: bytes) -> str:
    from pdfminer.high_level import extract_text

    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    return extract_text(io.BytesIO(pdf_bytes)) or ""


def _html_to_text(html_text: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_text, "lxml")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _fetch_jina(url: str, api_key: str, timeout: float) -> str:
    response = httpx.get(
        f"{JINA_READER_URL}/{url}",
        timeout=timeout,
        headers={"Authorization": f"Bearer {api_key}", "X-Timeout": str(int(timeout))},
    )
    response.raise_for_status()
    return response.text.strip()


def _fetch_direct(url: str, timeout: float) -> str:
    response = httpx.get(
        url,
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    content_type = (response.headers.get("content-type") or "").lower()
    body = response.content
    if body.startswith(b"%PDF-") or "application/pdf" in content_type:
        return _pdf_to_text(body).strip()
    if "text/html" in content_type or "<html" in response.text[:2048].lower():
        return _html_to_text(response.text).strip()
    return response.text.strip()


def open_page(url: str, timeout: float = 30) -> str:
    """Fetch a URL and return its text content, via Jina Reader when available."""
    api_key = os.environ.get("JINA_API_KEY", "")
    if api_key:
        try:
            return _fetch_jina(url, api_key, timeout)
        except httpx.HTTPError:
            pass  # fall back to direct fetch
    return _fetch_direct(url, timeout)


# Shadows the builtin within this module on purpose: the agent-facing API is `search.open`.
async def open(url: str, *, timeout: float = 30) -> str:
    """Fetch a URL and return its text content. Handles HTML and PDF.

    Uses the Jina Reader API (markdown; handles JS-rendered pages) when
    ``JINA_API_KEY`` is set, falling back to a direct fetch parsed locally.

    Args:
        url: The URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        The page text.
    """
    return await asyncio.to_thread(open_page, url, timeout)
