from pathlib import Path
from typing import AsyncIterator, Protocol


class StorageBackend(Protocol):
    """Abstraction over where generated artifacts (audio, stems, analyses) live.

    `key` is a logical path like "audio/abc123.wav" that the backend resolves
    to its own physical location. Services must go through this protocol so the
    eventual S3 swap stays a config change.
    """

    async def write(self, key: str, data: bytes) -> str: ...
    async def read(self, key: str) -> bytes: ...
    async def exists(self, key: str) -> bool: ...
    async def delete(self, key: str) -> None: ...
    async def get_url(self, key: str) -> str: ...
    async def download_file(self, key: str, dest_path: Path) -> None: ...
    async def upload_file(self, src_path: Path, key: str) -> str: ...
    async def stream(
        self,
        key: str,
        *,
        start: int | None = None,
        end: int | None = None,
    ) -> tuple[AsyncIterator[bytes], int, int]:
        """Stream `key` as chunks, optionally restricted to [start, end].

        Returns (iterator, total_size, content_length). `total_size` is the
        full object size; `content_length` is the size of THIS response
        (== total_size for a full GET, end-start+1 for a Range). The API
        layer uses these to set Content-Length and Content-Range headers
        when answering browser Range requests."""
        ...
