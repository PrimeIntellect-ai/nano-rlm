"""Built-in ``open_webpage`` skill — fetch a URL and return its text content.

Enabled via ``RLM_SKILLS``; pre-imported into the IPython kernel so the agent calls
``await open_webpage(url="...")``. When ``JINA_API_KEY`` is set, pages are fetched
through the Jina Reader API (markdown output, handles JS-rendered pages and PDFs),
falling back to a direct fetch parsed locally on failure.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re

import httpx

JINA_READER_URL = "https://r.jina.ai"


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


def open_webpage(url: str, timeout: float = 30) -> str:
    """Fetch a URL and return its text content, via Jina Reader when available."""
    api_key = os.environ.get("JINA_API_KEY", "")
    if api_key:
        try:
            return _fetch_jina(url, api_key, timeout)
        except httpx.HTTPError:
            pass  # fall back to direct fetch
    return _fetch_direct(url, timeout)


async def run(url: str, *, timeout: float = 30) -> str:
    """Fetch a URL and return its text content. Handles HTML and PDF.

    Uses the Jina Reader API (markdown; handles JS-rendered pages) when
    ``JINA_API_KEY`` is set, falling back to a direct fetch parsed locally.

    Args:
        url: The URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        The page text.
    """
    return await asyncio.to_thread(open_webpage, url, timeout)
