"""Search omits blocked and unverifiable results before applying the result limit."""

import httpx
import pytest

from rlm.skills.search import run_with_api_key, search


@pytest.mark.parametrize(
    "rules,url,blocked",
    [
        (("example.org",), "https://example.org/a", True),
        (("example.org",), "https://a.b.example.org/a", True),
        (("example.org",), "https://EXAMPLE.ORG./a", True),
        (("example.org",), "https://notexample.org/a", False),
        (("example.org",), "https://example.org.evil.test/a", False),
        (("example.org",), "https://example.org@allowed.test/a", False),
        (("*.example.org",), "https://a.example.org/a", True),
        (("*.example.org",), "https://example.org/a", False),
        (("https://example.org",), "https://example.org/a", True),
        (("http://example.org",), "http://example.org/a", True),
        (("http://example.org",), "https://example.org/a", False),
        (("https://example.org:8443",), "https://example.org:8443/a", True),
        (("https://example.org:8443",), "https://example.org/a", False),
        (("[::1]",), "https://[::1]/a", True),
        (("bücher.de",), "https://bücher.de/a", True),
        (("bücher.de",), "https://xn--bcher-kva.de/a", True),
        (("xn--bcher-kva.de",), "https://bücher.de/a", True),
        (("*.bücher.de",), "https://sub.xn--bcher-kva.de/a", True),
        (("*.bücher.de",), "https://bücher.de/a", False),
        (("BÜCHER.DE.",), "https://xn--bcher-kva.de/a", True),
        (("https://bücher.de",), "https://xn--bcher-kva.de/a", True),
        (("https://bücher.de",), "http://xn--bcher-kva.de/a", False),
        (("https://bücher.de:8443",), "https://xn--bcher-kva.de:8443/a", True),
        (("https://bücher.de:8443",), "https://xn--bcher-kva.de/a", False),
        (("faß.de",), "https://xn--fa-hia.de/a", True),
        (("faß.de",), "https://fass.de/a", False),
        (("example.org",), None, True),
        (("example.org",), "malformed", True),
        ((), "https://example.org/a", False),
        ((), None, False),
    ],
)
async def test_search_filters_destinations_before_limit(
    monkeypatch, rules, url, blocked
):
    response = httpx.Response(
        200,
        request=httpx.Request("POST", "https://google.serper.dev/search"),
        json={
            "organic": [
                {"link": url, "title": "candidate", "snippet": "candidate evidence"},
                {"link": "https://allowed.test/a", "title": "allowed"},
            ]
        },
    )

    async def post(*args, **kwargs):
        return response

    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: response)
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    sync = search("query", 1, api_key="fake", network_block=rules)
    asynchronous = await run_with_api_key(
        "fake", "query", num_results=1, network_block=rules
    )
    assert sync == asynchronous
    assert ("candidate" not in sync) is blocked
    assert ("Result 1: allowed" in sync) is blocked


async def test_search_omits_whole_results_and_handles_block_everything(monkeypatch):
    response = httpx.Response(
        200,
        request=httpx.Request("POST", "https://google.serper.dev/search"),
        json={
            "organic": [
                {"link": "https://example.org", "title": "secret", "snippet": "answer"}
            ]
        },
    )

    async def post(*args, **kwargs):
        return response

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    assert await run_with_api_key("fake", "query", network_block=("*",)) == (
        "No results returned for query: query"
    )
