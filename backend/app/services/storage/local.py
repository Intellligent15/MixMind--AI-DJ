from pathlib import Path


class LocalFilesystemStorage:
    """Filesystem-backed StorageBackend rooted at a single directory.

    Keys are interpreted as relative paths beneath `root`. Parent directories
    are created lazily on write. Reads/exists/delete are strict — they do not
    traverse outside the root.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        resolved = (self.root / key).resolve()
        if self.root not in resolved.parents and resolved != self.root:
            raise ValueError(f"key {key!r} escapes storage root")
        return resolved

    async def write(self, key: str, data: bytes) -> str:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return str(p)

    async def read(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    async def exists(self, key: str) -> bool:
        return self._path(key).exists()

    async def delete(self, key: str) -> None:
        p = self._path(key)
        if p.exists():
            p.unlink()

    async def get_url(self, key: str) -> str:
        return str(self._path(key))

    async def list_objects(self, prefix: str = "") -> list[tuple[str, int]]:
        base = self._path(prefix) if prefix else self.root
        # A prefix that names a file (not a dir) lists just that file; a
        # missing path lists nothing.
        if base.is_file():
            return [(prefix, base.stat().st_size)]
        if not base.exists():
            return []
        out: list[tuple[str, int]] = []
        for p in base.rglob("*"):
            if p.is_file():
                # Keys are POSIX-style relative paths from the storage root,
                # matching how they were written.
                key = p.relative_to(self.root).as_posix()
                out.append((key, p.stat().st_size))
        return out

    async def download_file(self, key: str, dest_path: Path) -> None:
        import shutil
        src = self._path(key)
        if not src.exists():
            raise FileNotFoundError(f"Key not found: {key}")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_path)

    async def upload_file(self, src_path: Path, key: str) -> str:
        import shutil
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest)
        return str(dest)

    async def stream(self, key: str, *, start=None, end=None):
        from typing import AsyncIterator
        p = self._path(key)
        if not p.exists():
            raise FileNotFoundError(f"Key not found: {key}")
        size = p.stat().st_size
        lo = 0 if start is None else max(0, start)
        hi = size - 1 if end is None else min(size - 1, end)
        length = hi - lo + 1

        async def gen() -> AsyncIterator[bytes]:
            chunk = 64 * 1024
            with p.open("rb") as f:
                f.seek(lo)
                remaining = length
                while remaining > 0:
                    data = f.read(min(chunk, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return gen(), size, length
