"""Tests for the Genius lyrics service — search + scrape pipeline.
HTTP is mocked via httpx.MockTransport so no network is hit."""

from __future__ import annotations

import httpx
import pytest

from app.services.lyrics import genius


# --- HTTP mock helper ---------------------------------------------------

_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _patch_async_client(monkeypatch, handler):
    """Redirect every ``httpx.AsyncClient(...)`` in the system under test
    through an in-memory MockTransport driven by ``handler``.

    Captures the real ``httpx.AsyncClient`` up front because the
    replacement lambda itself instantiates it — without this the lambda
    would trampoline into itself when fetch_lyrics issues a second
    request."""
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda **kw: _REAL_ASYNC_CLIENT(transport=transport, **kw),
    )


# --- Search-query builder -----------------------------------------------


def test_build_search_query_strips_parens_and_brackets():
    q = genius._build_search_query("Get Lucky (Radio Edit) [HD]", "Daft Punk")
    assert q == "Get Lucky Daft Punk"


def test_build_search_query_drops_feat_suffix():
    q = genius._build_search_query("One More Time ft. Romanthony", "Daft Punk")
    assert q == "One More Time Daft Punk"


def test_build_search_query_drops_uploader_when_hyphen_in_title():
    # YouTube uploader is often the artist suffix already baked into the
    # title with a hyphen — don't double-count it.
    q = genius._build_search_query("Daft Punk - Get Lucky", "DaftPunkVEVO")
    assert q == "Daft Punk - Get Lucky"


def test_build_search_query_strips_vevo_from_artist():
    q = genius._build_search_query("Get Lucky", "DaftPunkVEVO")
    assert q == "Get Lucky DaftPunk"


def test_build_search_query_handles_none_artist():
    q = genius._build_search_query("Get Lucky", None)
    assert q == "Get Lucky"


# --- Lyrics text cleaner ------------------------------------------------


def test_clean_lyrics_strips_section_headers():
    text = "[Verse 1]\nLine one\n[Chorus]\nLine two"
    cleaned = genius._clean_lyrics(text)
    assert "[Verse 1]" not in cleaned
    assert "[Chorus]" not in cleaned
    assert "Line one" in cleaned
    assert "Line two" in cleaned


def test_clean_lyrics_collapses_excess_blank_lines():
    text = "A\n\n\n\n\nB"
    cleaned = genius._clean_lyrics(text)
    assert "\n\n\n" not in cleaned


# --- search_genius_song -------------------------------------------------


@pytest.mark.asyncio
async def test_search_genius_song_returns_first_song_hit(monkeypatch):
    monkeypatch.setattr(genius.settings, "genius_access_token", "fake-token")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer fake-token"
        return httpx.Response(200, json={
            "response": {
                "hits": [
                    {"type": "song", "result": {"id": 123, "url": "https://genius.com/foo"}},
                ]
            }
        })

    _patch_async_client(monkeypatch, handler)
    result = await genius.search_genius_song("foo", "bar")
    assert result == {"id": 123, "url": "https://genius.com/foo"}


@pytest.mark.asyncio
async def test_search_genius_song_no_token_returns_none(monkeypatch):
    monkeypatch.setattr(genius.settings, "genius_access_token", "")
    result = await genius.search_genius_song("foo", "bar")
    assert result is None


@pytest.mark.asyncio
async def test_search_genius_song_no_hits_returns_none(monkeypatch):
    monkeypatch.setattr(genius.settings, "genius_access_token", "fake-token")

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": {"hits": []}})

    _patch_async_client(monkeypatch, handler)
    result = await genius.search_genius_song("foo", "bar")
    assert result is None


@pytest.mark.asyncio
async def test_search_genius_song_skips_non_song_hits(monkeypatch):
    monkeypatch.setattr(genius.settings, "genius_access_token", "fake-token")

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "response": {
                "hits": [
                    {"type": "artist", "result": {"id": 999}},
                    {"type": "song", "result": {"id": 123, "url": "x"}},
                ]
            }
        })

    _patch_async_client(monkeypatch, handler)
    result = await genius.search_genius_song("foo", "bar")
    assert result == {"id": 123, "url": "x"}


@pytest.mark.asyncio
async def test_search_genius_song_http_error_returns_none(monkeypatch):
    monkeypatch.setattr(genius.settings, "genius_access_token", "fake-token")

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    _patch_async_client(monkeypatch, handler)
    result = await genius.search_genius_song("foo", "bar")
    assert result is None


# --- scrape_genius_lyrics -----------------------------------------------


_LYRICS_HTML = """
<html><body>
<div data-lyrics-container="true">
  Line one<br>Line two<br>Line three
</div>
<div data-lyrics-container="true">
  <h2>Skip me</h2>
  Line four<br>Line five
</div>
</body></html>
"""


@pytest.mark.asyncio
async def test_scrape_genius_lyrics_extracts_text(monkeypatch):
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_LYRICS_HTML)

    _patch_async_client(monkeypatch, handler)
    text = await genius.scrape_genius_lyrics("https://genius.com/foo")
    assert text is not None
    assert "Line one" in text
    assert "Line four" in text
    # The <h2> "Skip me" should not appear.
    assert "Skip me" not in text


@pytest.mark.asyncio
async def test_scrape_genius_lyrics_no_container_returns_none(monkeypatch):
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html><body>Nothing here</body></html>")

    _patch_async_client(monkeypatch, handler)
    text = await genius.scrape_genius_lyrics("https://genius.com/foo")
    assert text is None


@pytest.mark.asyncio
async def test_scrape_genius_lyrics_http_error_returns_none(monkeypatch):
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    _patch_async_client(monkeypatch, handler)
    text = await genius.scrape_genius_lyrics("https://genius.com/foo")
    assert text is None


# --- fetch_lyrics (composition) ---------------------------------------


@pytest.mark.asyncio
async def test_fetch_lyrics_returns_id_and_text(monkeypatch):
    monkeypatch.setattr(genius.settings, "genius_access_token", "fake-token")
    api_host = "api.genius.com"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == api_host:
            return httpx.Response(200, json={
                "response": {"hits": [
                    {"type": "song", "result": {"id": 555, "url": "https://genius.com/foo"}}
                ]}
            })
        return httpx.Response(200, text=_LYRICS_HTML)

    _patch_async_client(monkeypatch, handler)
    result = await genius.fetch_lyrics("foo", "bar")
    assert result is not None
    gid, text = result
    assert gid == 555
    assert "Line one" in text


@pytest.mark.asyncio
async def test_fetch_lyrics_none_when_search_returns_nothing(monkeypatch):
    monkeypatch.setattr(genius.settings, "genius_access_token", "fake-token")

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": {"hits": []}})

    _patch_async_client(monkeypatch, handler)
    result = await genius.fetch_lyrics("foo", "bar")
    assert result is None


@pytest.mark.asyncio
async def test_fetch_lyrics_none_when_scrape_returns_empty(monkeypatch):
    monkeypatch.setattr(genius.settings, "genius_access_token", "fake-token")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.genius.com":
            return httpx.Response(200, json={
                "response": {"hits": [
                    {"type": "song", "result": {"id": 1, "url": "https://genius.com/x"}}
                ]}
            })
        return httpx.Response(200, text="<html></html>")

    _patch_async_client(monkeypatch, handler)
    result = await genius.fetch_lyrics("foo", "bar")
    assert result is None
