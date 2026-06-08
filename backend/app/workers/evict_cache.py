"""Celery wrapper around the LRU cache evictor.

Dispatched best-effort after events that grow the cache (a completed
download, a finished stitch). Self-gating: a no-op when the cache is under
budget, so firing it on every download is cheap. See
``app.services.cache.eviction`` for the policy.
"""

from __future__ import annotations

import asyncio
import logging

from app.core.config import settings
from app.core.db import SessionLocal
from app.services.cache.eviction import enforce_cache_budget
from app.services.storage import get_storage
from app.workers import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="app.workers.evict_cache.enforce_cache_budget")
def enforce_cache_budget_task() -> dict:
    budget = int(settings.cache_max_size_gb * 1024**3)
    storage = get_storage()
    with SessionLocal() as db:
        return asyncio.run(enforce_cache_budget(db, storage, budget))
