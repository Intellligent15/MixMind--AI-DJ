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

    def path(self, key: str) -> Path:
        resolved = (self.root / key).resolve()
        if self.root not in resolved.parents and resolved != self.root:
            raise ValueError(f"key {key!r} escapes storage root")
        return resolved

    async def write(self, key: str, data: bytes) -> str:
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return str(p)

    async def read(self, key: str) -> bytes:
        return self.path(key).read_bytes()

    async def exists(self, key: str) -> bool:
        return self.path(key).exists()

    async def delete(self, key: str) -> None:
        p = self.path(key)
        if p.exists():
            p.unlink()

    async def get_url(self, key: str) -> str:
        return str(self.path(key))
