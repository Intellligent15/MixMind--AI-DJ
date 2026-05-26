"""Wipe (or list) every ai-dj object in DO Spaces.

The DELETE /api/songs/{id} endpoint covers the per-song case. This script
is for the "I want a clean slate" path — orphaned blobs, broken state,
etc. Default mode is dry-run: it lists what would be deleted and exits.
Pass --yes to actually delete. Pass --db to also TRUNCATE the relevant
DB tables in the same run so you don't end up with rows pointing at
deleted blobs.

Usage:
    cd backend
    uv run python -m scripts.wipe_storage         # dry-run
    uv run python -m scripts.wipe_storage --yes   # nuke blobs
    uv run python -m scripts.wipe_storage --yes --db  # blobs + DB

The script touches only the keys under the ai-dj prefixes
(audio/, stems/, mixes/, transcriptions/, _smoke/). Anything else in
the bucket is left alone.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.core.config import settings
from app.services.storage.s3 import S3Storage


# Prefixes the app writes to. Includes the legacy `<bucket>` prefix
# (with AND without trailing slash) from a pre-fix era when boto3 was
# misconfigured and double-prefixed every key with the bucket name —
# at least one stray request also produced a key that was just the
# bare bucket name with no separator. Keys outside this list are left
# alone in case the bucket is shared with anything else.
PREFIXES = (
    "audio/",
    "stems/",
    "mixes/",
    "transcriptions/",
    "_smoke/",
    "ai-dj-storage",  # catches both "ai-dj-storage/..." and bare "ai-dj-storage"
)

# Tables to TRUNCATE when --db is passed. Order doesn't matter under
# CASCADE; this list mirrors the test conftest for consistency.
TABLES = (
    "mix_plans",
    "queue_items",
    "queues",
    "transcriptions",
    "stems",
    "analyses",
    "songs",
)


async def _list_keys(s3: S3Storage) -> list[tuple[str, int]]:
    """Return (key, size) for every object under our prefixes."""
    out: list[tuple[str, int]] = []
    async with s3._client() as client:
        paginator = client.get_paginator("list_objects_v2")
        for prefix in PREFIXES:
            async for page in paginator.paginate(
                Bucket=s3.bucket_name, Prefix=prefix
            ):
                for obj in page.get("Contents", []) or []:
                    out.append((obj["Key"], int(obj["Size"])))
    return out


async def _delete_keys(s3: S3Storage, keys: list[str]) -> None:
    """Batch-delete in chunks of 1000 (S3's per-request limit)."""
    async with s3._client() as client:
        for i in range(0, len(keys), 1000):
            chunk = keys[i : i + 1000]
            await client.delete_objects(
                Bucket=s3.bucket_name,
                Delete={"Objects": [{"Key": k} for k in chunk]},
            )


def _truncate_db() -> None:
    from sqlalchemy import text
    from app.core.db import SessionLocal

    with SessionLocal() as db:
        db.execute(text(f"TRUNCATE TABLE {', '.join(TABLES)} CASCADE"))
        db.commit()


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete. Without this flag, the script only lists.",
    )
    parser.add_argument(
        "--db",
        action="store_true",
        help="Also TRUNCATE songs+stems+... in the local DB. Safe with --yes.",
    )
    args = parser.parse_args()

    if settings.storage_backend != "s3":
        print(
            f"STORAGE_BACKEND={settings.storage_backend!r} — this script only "
            f"makes sense against S3-style storage. Aborting.",
            file=sys.stderr,
        )
        return 2

    s3 = S3Storage(
        endpoint_url=settings.s3_endpoint_url,
        bucket_name=settings.s3_bucket_name,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        region_name=settings.s3_region_name,
    )

    entries = await _list_keys(s3)
    if not entries:
        print("Nothing to delete — bucket prefixes are empty.")
        if args.db and args.yes:
            _truncate_db()
            print("DB tables truncated.")
        return 0

    total_bytes = sum(s for _, s in entries)
    print(
        f"Found {len(entries)} object(s), {total_bytes / (1024 * 1024):.1f} MB total."
    )
    by_prefix: dict[str, tuple[int, int]] = {}
    for k, s in entries:
        p = k.split("/", 1)[0] + "/"
        n, b = by_prefix.get(p, (0, 0))
        by_prefix[p] = (n + 1, b + s)
    for p, (n, b) in sorted(by_prefix.items()):
        print(f"  {p:<18}{n:>5} obj  {b / (1024 * 1024):>8.1f} MB")

    if not args.yes:
        print(
            "\nDry-run. Re-run with --yes to actually delete "
            "(--db also wipes DB tables)."
        )
        return 0

    print(f"\nDeleting {len(entries)} objects...")
    await _delete_keys(s3, [k for k, _ in entries])
    print("Done.")

    if args.db:
        print("Truncating DB tables...")
        _truncate_db()
        print("DB tables truncated.")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
