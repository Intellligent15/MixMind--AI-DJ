from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yt_dlp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    youtube_video_id: str
    title: str
    artist: str | None
    duration_seconds: float
    thumbnail_url: str | None


class YouTubeDownloadError(RuntimeError):
    pass


class YouTubeService:
    """Thin wrapper around yt-dlp for search + best-audio download.

    No DB awareness; callers are responsible for persisting the result.
    """

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "extract_flat": "in_playlist",
            "default_search": f"ytsearch{limit}",
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(query, download=False)

        entries = (info or {}).get("entries") or []
        results: list[SearchResult] = []
        for entry in entries[:limit]:
            if not entry or not entry.get("id"):
                continue
            results.append(_entry_to_result(entry))
        return results

    def download(self, video_id: str, dest_path: Path) -> None:
        """Download `video_id` as WAV to `dest_path`. Overwrites existing file."""
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        # yt-dlp's --extract-audio postprocessor writes to outtmpl with the new
        # extension. We give it a template without an extension to control the
        # final filename precisely.
        out_template = str(dest_path.with_suffix(""))
        opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "format": "bestaudio/best",
            "outtmpl": out_template + ".%(ext)s",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                }
            ],
            "overwrites": True,
        }
        url = f"https://www.youtube.com/watch?v={video_id}"
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
        except yt_dlp.utils.DownloadError as exc:
            raise YouTubeDownloadError(str(exc)) from exc

        if not dest_path.exists():
            raise YouTubeDownloadError(
                f"yt-dlp completed but {dest_path} was not produced"
            )


def _entry_to_result(entry: dict[str, Any]) -> SearchResult:
    duration = entry.get("duration")
    return SearchResult(
        youtube_video_id=entry["id"],
        title=entry.get("title") or entry["id"],
        artist=entry.get("artist") or entry.get("uploader"),
        duration_seconds=float(duration) if duration is not None else 0.0,
        thumbnail_url=_pick_thumbnail(entry),
    )


def _pick_thumbnail(entry: dict[str, Any]) -> str | None:
    thumbs = entry.get("thumbnails") or []
    if thumbs:
        # last thumbnail is typically highest resolution
        return thumbs[-1].get("url")
    return entry.get("thumbnail")
