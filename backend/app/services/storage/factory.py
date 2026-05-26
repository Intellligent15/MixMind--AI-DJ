from functools import lru_cache

from app.core.config import settings
from .base import StorageBackend
from .local import LocalFilesystemStorage
from .s3 import S3Storage


@lru_cache()
def get_storage() -> StorageBackend:
    if settings.storage_backend == "s3":
        return S3Storage(
            endpoint_url=settings.s3_endpoint_url,
            bucket_name=settings.s3_bucket_name,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            region_name=settings.s3_region_name,
        )
    return LocalFilesystemStorage(root=settings.local_storage_path)
