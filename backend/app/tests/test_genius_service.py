"""Tests for the Genius lyric service. HTTP mocked via
httpx.MockTransport."""

from __future__ import annotations

import httpx
import pytest

from app.services.lyrics import genius


_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _patch_async_client(monkeypatch, handler):
    """Route every httpx.AsyncClient through MockTransport. Captures
    the real client first so the replacement lambda doesn't trampoline
    into itself."""
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda **kw: _REAL_ASYNC_CLIENT(transport=transport, **kw),
    )


@pytest.mark.asyncio
async def test_search_returns_first_song_hit_with_matching_artist(monkeypatch):
    monkeypatch.setattr(genius.settings, "genius_access_token", "fake")

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "response": {"hits": [
                {"type": "song", "result": {
                    "id": 1, "url": "https://genius.com/wrong-artist",
                    "primary_artist": {"name": "Lionel Richie"},
                }},
                {"type": "song", "result": {
                    "id": 2, "url": "https://genius.com/right-artist",
                    "primary_artist": {"name": "Adele"},
                }},
            ]}
        })

    _patch_async_client(monkeypatch, handler)
    result = await genius.search_genius_song("Hello", "Adele")
    assert result["id"] == 2  # picks the Adele hit, not Lionel Richie


@pytest.mark.asyncio
async def test_search_falls_back_to_first_when_no_artist_overlap(monkeypatch):
    """If none of the hits have an overlapping artist token, take the
    first hit — better than nothing."""
    monkeypatch.setattr(genius.settings, "genius_access_token", "fake")

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "response": {"hits": [
                {"type": "song", "result": {
                    "id": 5, "url": "x",
                    "primary_artist": {"name": "Stranger"},
                }},
                {"type": "song", "result": {
                    "id": 6, "url": "x",
                    "primary_artist": {"name": "Outsider"},
                }},
            ]}
        })

    _patch_async_client(monkeypatch, handler)
    result = await genius.search_genius_song("Hello", "Adele")
    assert result["id"] == 5  # falls back to first


@pytest.mark.asyncio
async def test_search_no_artist_picks_first_hit(monkeypatch):
    """When we don't know the artist, pick the first hit (current
    behavior — no change)."""
    monkeypatch.setattr(genius.settings, "genius_access_token", "fake")

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "response": {"hits": [
                {"type": "song", "result": {"id": 9, "url": "x"}},
            ]}
        })

    _patch_async_client(monkeypatch, handler)
    result = await genius.search_genius_song("Hello", None)
    assert result["id"] == 9
