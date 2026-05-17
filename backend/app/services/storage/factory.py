from functools import lru_cache

from app.core.config import settings
from app.services.storage.base import StorageBackend
from app.services.storage.local import LocalFilesystemStorage


@lru_cache(maxsize=1)
def get_storage() -> StorageBackend:
    if settings.storage_backend == "local":
        return LocalFilesystemStorage(settings.local_storage_path)
    raise ValueError(f"unknown STORAGE_BACKEND: {settings.storage_backend!r}")
