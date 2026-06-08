"""Phase 11 — LRU cache eviction.

Exercises the song-centric evictor against a real Postgres (the db_session
fixture) + a tmpdir LocalFilesystemStorage. The hazards under test are the
exemptions: queued songs, mid-pipeline songs, and the mix_plan_logs/ plan
cache must never be evicted.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.models import (
    MixPlan,
    MixPlanStatus,
    Queue,
    QueueItem,
    Song,
    SongStatus,
    Stems,
    StemsStatus,
)
from app.services.cache.eviction import (
    collect_song_storage_keys,
    enforce_cache_budget,
    evictable_songs,
)
from app.services.storage.local import LocalFilesystemStorage


@pytest.fixture
def storage(tmp_path):
    return LocalFilesystemStorage(tmp_path)


def _make_song(db, *, vid: str, status=SongStatus.ready, accessed_days_ago=0):
    song = Song(
        youtube_video_id=vid,
        title=f"t-{vid}",
        artist="a",
        duration_seconds=180.0,
        audio_path=f"audio/{vid}.wav",
        status=status,
        last_accessed_at=datetime.now(timezone.utc)
        - timedelta(days=accessed_days_ago),
    )
    db.add(song)
    db.commit()
    db.refresh(song)
    return song


def _write_song_blobs(storage, song, *, audio=5_000_000):
    asyncio.run(storage.write(song.audio_path, b"a" * audio))


def test_evictable_excludes_queued_and_active(db_session):
    db = db_session
    ready = _make_song(db, vid="ready1")
    queued = _make_song(db, vid="queued1")
    active = _make_song(db, vid="active1", status=SongStatus.separating)

    q = Queue(locked=True)
    db.add(q)
    db.commit()
    db.add(QueueItem(queue_id=q.id, song_id=queued.id, position=0))
    db.commit()

    ev = {s.id for s in evictable_songs(db)}
    assert ready.id in ev
    assert queued.id not in ev  # in a queue → exempt
    assert active.id not in ev  # mid-pipeline → exempt


def test_evictable_ordered_lru_first(db_session):
    db = db_session
    newest = _make_song(db, vid="new", accessed_days_ago=1)
    oldest = _make_song(db, vid="old", accessed_days_ago=30)
    mid = _make_song(db, vid="mid", accessed_days_ago=10)

    order = [s.id for s in evictable_songs(db)]
    assert order == [oldest.id, mid.id, newest.id]


def test_collect_song_storage_keys(db_session):
    db = db_session
    song = _make_song(db, vid="keys1")
    db.add(
        Stems(
            song_id=song.id,
            status=StemsStatus.separated,
            vocals_path="stems/keys1/vocals.wav",
            drums_path="stems/keys1/drums.wav",
            bass_path="stems/keys1/bass.wav",
            other_path="stems/keys1/other.wav",
            vocal_envelope_path="stems/keys1/vocal_envelope.json",
        )
    )
    db.commit()
    keys = collect_song_storage_keys(song, db)
    assert "audio/keys1.wav" in keys
    assert "stems/keys1/vocals.wav" in keys
    assert "stems/keys1/vocal_envelope.json" in keys
    assert "transcriptions/keys1.json" in keys


def test_under_budget_is_noop(db_session, storage):
    db = db_session
    song = _make_song(db, vid="small")
    _write_song_blobs(storage, song, audio=1000)

    res = asyncio.run(enforce_cache_budget(db, storage, budget_bytes=10_000_000))
    assert res["evicted"] == []
    assert db.get(Song, song.id) is not None


def test_evicts_lru_until_under_budget(db_session, storage):
    db = db_session
    # Three 5 MB songs = 15 MB; budget 12 MB → evict the single oldest (5 MB)
    # leaves 10 MB, under budget.
    old = _make_song(db, vid="o", accessed_days_ago=30)
    midd = _make_song(db, vid="m", accessed_days_ago=10)
    new = _make_song(db, vid="n", accessed_days_ago=1)
    for s in (old, midd, new):
        _write_song_blobs(storage, s, audio=5_000_000)

    res = asyncio.run(
        enforce_cache_budget(db, storage, budget_bytes=12_000_000)
    )
    assert res["evicted"] == [str(old.id)]
    assert db.get(Song, old.id) is None
    assert db.get(Song, midd.id) is not None
    assert db.get(Song, new.id) is not None
    # Its blob is gone from storage too.
    assert asyncio.run(storage.exists("audio/o.wav")) is False
    assert asyncio.run(storage.exists("audio/m.wav")) is True


def test_never_evicts_queued_song_even_if_oldest(db_session, storage):
    db = db_session
    queued_old = _make_song(db, vid="qo", accessed_days_ago=99)
    free_new = _make_song(db, vid="fn", accessed_days_ago=1)
    for s in (queued_old, free_new):
        _write_song_blobs(storage, s, audio=5_000_000)

    q = Queue(locked=True)
    db.add(q)
    db.commit()
    db.add(QueueItem(queue_id=q.id, song_id=queued_old.id, position=0))
    db.commit()

    # Budget below total (10 MB) — evictor wants to free, but the only
    # truly-old song is queued. It must evict the free one instead.
    res = asyncio.run(enforce_cache_budget(db, storage, budget_bytes=6_000_000))
    assert str(queued_old.id) not in res["evicted"]
    assert db.get(Song, queued_old.id) is not None
    assert str(free_new.id) in res["evicted"]


def test_mix_plan_logs_never_deleted(db_session, storage):
    db = db_session
    # One evictable song + a plan-cache blob. Budget 0 forces eviction of
    # everything evictable, but the plan cache must survive.
    song = _make_song(db, vid="p", accessed_days_ago=10)
    _write_song_blobs(storage, song, audio=5_000_000)
    asyncio.run(storage.write("mix_plan_logs/deadbeef.json", b"{}"))

    asyncio.run(enforce_cache_budget(db, storage, budget_bytes=0))
    assert db.get(Song, song.id) is None
    assert asyncio.run(storage.exists("mix_plan_logs/deadbeef.json")) is True


def test_evicts_song_mix_plan_renders(db_session, storage):
    db = db_session
    a = _make_song(db, vid="a", accessed_days_ago=30)
    b = _make_song(db, vid="b", accessed_days_ago=29)
    for s in (a, b):
        _write_song_blobs(storage, s, audio=1_000_000)

    plan = MixPlan(
        queue_id=uuid.uuid4(),  # orphan queue id is fine; FK is to songs
        from_song_id=a.id,
        to_song_id=b.id,
        status=MixPlanStatus.ready,
        rendered_audio_path="mixes/plan-a-b.wav",
    )
    # The queue_id FK requires a real queue row — make one (unlocked, no
    # items, so it doesn't exempt the songs).
    q = Queue(locked=False)
    db.add(q)
    db.commit()
    plan.queue_id = q.id
    db.add(plan)
    db.commit()
    asyncio.run(storage.write("mixes/plan-a-b.wav", b"r" * 4_000_000))

    keys = collect_song_storage_keys(a, db)
    assert "mixes/plan-a-b.wav" in keys

    asyncio.run(enforce_cache_budget(db, storage, budget_bytes=0))
    assert asyncio.run(storage.exists("mixes/plan-a-b.wav")) is False
