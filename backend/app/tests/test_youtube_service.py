from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services.youtube import SearchResult, YouTubeService
from app.services.youtube.service import YouTubeDownloadError


def _fake_ydl(extract_return: dict | None = None, download_side_effect=None):
    """Build a MagicMock that mimics yt_dlp.YoutubeDL's context-manager API."""
    instance = MagicMock()
    instance.extract_info.return_value = extract_return
    if download_side_effect is not None:
        instance.download.side_effect = download_side_effect
    cm = MagicMock()
    cm.__enter__.return_value = instance
    cm.__exit__.return_value = False
    factory = MagicMock(return_value=cm)
    return factory, instance


def test_search_returns_normalized_results():
    extract = {
        "entries": [
            {
                "id": "abc123",
                "title": "Song One",
                "artist": "Artist A",
                "duration": 200.0,
                "thumbnails": [{"url": "https://t/small.jpg"}, {"url": "https://t/big.jpg"}],
            },
            {
                "id": "def456",
                "title": "Song Two",
                "uploader": "Channel B",
                "duration": 150,
            },
        ]
    }
    factory, _ = _fake_ydl(extract_return=extract)
    with patch("app.services.youtube.service.yt_dlp.YoutubeDL", factory):
        results = YouTubeService().search("query", limit=5)

    assert results == [
        SearchResult(
            youtube_video_id="abc123",
            title="Song One",
            artist="Artist A",
            duration_seconds=200.0,
            thumbnail_url="https://t/big.jpg",
        ),
        SearchResult(
            youtube_video_id="def456",
            title="Song Two",
            artist="Channel B",
            duration_seconds=150.0,
            thumbnail_url=None,
        ),
    ]


def test_search_skips_entries_without_id():
    extract = {"entries": [None, {"title": "no id"}, {"id": "x", "title": "ok", "duration": 1}]}
    factory, _ = _fake_ydl(extract_return=extract)
    with patch("app.services.youtube.service.yt_dlp.YoutubeDL", factory):
        results = YouTubeService().search("q")
    assert len(results) == 1
    assert results[0].youtube_video_id == "x"


def test_search_handles_no_entries():
    factory, _ = _fake_ydl(extract_return={"entries": []})
    with patch("app.services.youtube.service.yt_dlp.YoutubeDL", factory):
        assert YouTubeService().search("q") == []


def test_download_writes_to_dest_path(tmp_path: Path):
    dest = tmp_path / "audio" / "abc.wav"

    def fake_download(urls):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"RIFF....WAVE")

    factory, _ = _fake_ydl(download_side_effect=fake_download)
    with patch("app.services.youtube.service.yt_dlp.YoutubeDL", factory):
        YouTubeService().download("abc", dest)

    assert dest.read_bytes() == b"RIFF....WAVE"


def test_download_raises_when_file_missing(tmp_path: Path):
    dest = tmp_path / "abc.wav"
    factory, _ = _fake_ydl(download_side_effect=lambda urls: None)
    with patch("app.services.youtube.service.yt_dlp.YoutubeDL", factory):
        with pytest.raises(YouTubeDownloadError, match="was not produced"):
            YouTubeService().download("abc", dest)


def test_download_wraps_ytdlp_errors(tmp_path: Path):
    import yt_dlp

    def raise_error(urls):
        raise yt_dlp.utils.DownloadError("network exploded")

    factory, _ = _fake_ydl(download_side_effect=raise_error)
    with patch("app.services.youtube.service.yt_dlp.YoutubeDL", factory):
        with pytest.raises(YouTubeDownloadError, match="network exploded"):
            YouTubeService().download("abc", tmp_path / "x.wav")
