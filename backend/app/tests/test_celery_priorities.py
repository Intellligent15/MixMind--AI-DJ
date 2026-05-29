"""Verify Celery + Redis broker is configured for priority queue ordering.

Without these settings, a 12-song queue lock produces a FIFO stall where
all 12 downloads drain before any analyze starts, then all 12 analyzes
drain before any separate starts, etc. With these settings, a song's
downstream stages can preempt other songs' upstream stages so the first
songs in the queue reach `ready` while later songs are still downloading.
"""

from app.workers import (
    celery_app,
    PRI_DOWNLOAD,
    PRI_ANALYZE,
    PRI_SEPARATE,
    PRI_TRANSCRIBE,
)


def test_priority_constants_strictly_increasing_downstream():
    assert PRI_DOWNLOAD < PRI_ANALYZE < PRI_SEPARATE < PRI_TRANSCRIBE
    assert 0 <= PRI_DOWNLOAD and PRI_TRANSCRIBE <= 9


def test_broker_uses_priority_queue_ordering():
    opts = celery_app.conf.broker_transport_options or {}
    assert opts.get("queue_order_strategy") == "priority"


def test_prefetch_is_one_and_acks_late():
    # prefetch > 1 bypasses priorities (worker grabs N tasks eagerly,
    # then runs them FIFO regardless of priority). acks_late lets a
    # higher-priority arrival redirect the worker after the current
    # task finishes rather than getting blocked behind it.
    assert celery_app.conf.worker_prefetch_multiplier == 1
    assert celery_app.conf.task_acks_late is True
