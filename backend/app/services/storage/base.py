from pathlib import Path
from typing import Protocol


class StorageBackend(Protocol):
    """Abstraction over where generated artifacts (audio, stems, analyses) live.

    `key` is a logical path like "audio/abc123.wav" that the backend resolves
    to its own physical location. Services must go through this protocol so the
    eventual S3 swap stays a config change.

    `path` returns a local filesystem location suitable for handing to
    external tools (yt-dlp, ffmpeg, librosa, demucs). The local backend
    returns its absolute path directly; a future S3 backend would materialize
    a tempfile and sync on close.
    """

    async def write(self, key: str, data: bytes) -> str: ...
    async def read(self, key: str) -> bytes: ...
    async def exists(self, key: str) -> bool: ...
    async def delete(self, key: str) -> None: ...
    async def get_url(self, key: str) -> str: ...
    def path(self, key: str) -> Path: ...
