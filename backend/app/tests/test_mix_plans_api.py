"""mix_plans API tests. Real DB, mock the worker dispatch."""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.db import SessionLocal
from app.main import app
from app.models import (
    Analysis,
    MixPlan,
    MixPlanStatus,
    Queue,
    QueueItem,
    Song,
    SongStatus,
    Stems,
    StemsStatus,
)

client = TestClient(app)


@pytest.fixture
def locked_pair():
    """Locked queue with two ready songs + analyses + stems."""
    payload = {}
    with SessionLocal() as db:
        q = Queue(locked=True)
        a = Song(youtube_video_id=f"mpa-{id(object())}",
                 title="A", duration_seconds=180.0, audio_path="audio/a.wav",
                 status=SongStatus.ready)
        b = Song(youtube_video_id=f"mpb-{id(object())}",
                 title="B", duration_seconds=180.0, audio_path="audio/b.wav",
                 status=SongStatus.ready)
        db.add_all([q, a, b])
        db.flush()
        db.add_all([
            Analysis(song_id=a.id, bpm=120.0, key="C", camelot_key="8B",
                     time_signature=4,
                     beat_grid=[i*0.5 for i in range(360)],
                     downbeats=[i*2.0 for i in range(90)],
                     sections=[{"start":0.0,"end":30.0,"label":"intro"},
                               {"start":30.0,"end":150.0,"label":"body"},
                               {"start":150.0,"end":180.0,"label":"outro"}],
                     energy_curve=[0.5]*180, vocal_segments=[]),
            Analysis(song_id=b.id, bpm=128.0, key="D", camelot_key="10B",
                     time_signature=4,
                     beat_grid=[i*(60/128) for i in range(384)],
                     downbeats=[i*(60/128*4) for i in range(96)],
                     sections=[{"start":0.0,"end":30.0,"label":"intro"},
                               {"start":30.0,"end":150.0,"label":"body"},
                               {"start":150.0,"end":180.0,"label":"outro"}],
                     energy_curve=[0.5]*180, vocal_segments=[]),
            Stems(song_id=a.id, model_name="htdemucs", status=StemsStatus.separated,
                  vocals_path=f"stems/{a.youtube_video_id}/vocals.wav",
                  drums_path=f"stems/{a.youtube_video_id}/drums.wav",
                  bass_path=f"stems/{a.youtube_video_id}/bass.wav",
                  other_path=f"stems/{a.youtube_video_id}/other.wav",
                  vocal_rms=0.15),
            Stems(song_id=b.id, model_name="htdemucs", status=StemsStatus.separated,
                  vocals_path=f"stems/{b.youtube_video_id}/vocals.wav",
                  drums_path=f"stems/{b.youtube_video_id}/drums.wav",
                  bass_path=f"stems/{b.youtube_video_id}/bass.wav",
                  other_path=f"stems/{b.youtube_video_id}/other.wav",
                  vocal_rms=0.15),
            QueueItem(queue_id=q.id, song_id=a.id, position=0),
            QueueItem(queue_id=q.id, song_id=b.id, position=1),
        ])
        db.commit()
        payload = {
            "queue_id": str(q.id),
            "a_id": str(a.id), "b_id": str(b.id),
        }
    yield payload
    with SessionLocal() as db:
        for k in ("a_id", "b_id"):
            song = db.get(Song, uuid.UUID(payload[k]))
            if song is not None:
                db.delete(song)
        q = db.get(Queue, uuid.UUID(payload["queue_id"]))
        if q is not None:
            db.delete(q)
        db.commit()


def test_seed_mix_plans_creates_rows(locked_pair):
    r = client.post(f"/api/queues/{locked_pair['queue_id']}/mix_plans")
    assert r.status_code == 201
    body = r.json()
    assert len(body) == 1  # N-1 = 2-1
    row = body[0]
    assert row["from_song_id"] == locked_pair["a_id"]
    assert row["to_song_id"] == locked_pair["b_id"]
    assert row["status"] == "pending"
    assert row["plan_json"] is None  # lazy


def test_seed_mix_plans_is_idempotent(locked_pair):
    r1 = client.post(f"/api/queues/{locked_pair['queue_id']}/mix_plans")
    r2 = client.post(f"/api/queues/{locked_pair['queue_id']}/mix_plans")
    assert r1.status_code == 201
    assert r2.status_code == 201
    assert r1.json() == r2.json()


def test_list_mix_plans(locked_pair):
    client.post(f"/api/queues/{locked_pair['queue_id']}/mix_plans")
    r = client.get(f"/api/queues/{locked_pair['queue_id']}/mix_plans")
    assert r.status_code == 200
    assert len(r.json()) == 1


def test_get_mix_plan_by_id(locked_pair):
    seed = client.post(f"/api/queues/{locked_pair['queue_id']}/mix_plans").json()
    plan_id = seed[0]["id"]
    r = client.get(f"/api/mix_plans/{plan_id}")
    assert r.status_code == 200
    assert r.json()["id"] == plan_id


def test_get_mix_plan_404():
    r = client.get(f"/api/mix_plans/{uuid.uuid4()}")
    assert r.status_code == 404


def test_render_dispatches_task(locked_pair):
    seed = client.post(f"/api/queues/{locked_pair['queue_id']}/mix_plans").json()
    plan_id = seed[0]["id"]
    with patch("app.api.mix_plans.celery_app.send_task") as send:
        r = client.post(f"/api/mix_plans/{plan_id}/render")
    assert r.status_code == 202
    send.assert_called_once()
    args, kwargs = send.call_args
    assert args[0] == "app.workers.render_transition.render_transition"
    assert kwargs["args"] == [plan_id]


def test_render_refuses_if_songs_not_ready(locked_pair):
    seed = client.post(f"/api/queues/{locked_pair['queue_id']}/mix_plans").json()
    plan_id = seed[0]["id"]
    with SessionLocal() as db:
        a = db.get(Song, uuid.UUID(locked_pair["a_id"]))
        a.status = SongStatus.analyzed
        db.commit()
    with patch("app.api.mix_plans.celery_app.send_task") as send:
        r = client.post(f"/api/mix_plans/{plan_id}/render")
    assert r.status_code == 409
    send.assert_not_called()


def test_get_audio_404_when_not_rendered(locked_pair):
    seed = client.post(f"/api/queues/{locked_pair['queue_id']}/mix_plans").json()
    plan_id = seed[0]["id"]
    r = client.get(f"/api/mix_plans/{plan_id}/audio")
    assert r.status_code == 409


def test_get_audio_200_when_rendered(locked_pair, tmp_path: Path):
    seed = client.post(f"/api/queues/{locked_pair['queue_id']}/mix_plans").json()
    plan_id = seed[0]["id"]
    from app.services.storage import get_storage
    storage = get_storage()
    key = f"mixes/{plan_id}.wav"
    
    import soundfile as sf
    import numpy as np
    import io
    import asyncio
    
    buf = io.BytesIO()
    sf.write(buf, np.zeros((100, 2), dtype=np.float32), 44100, format="WAV", subtype="PCM_16")
    asyncio.run(storage.write(key, buf.getvalue()))
    
    with SessionLocal() as db:
        row = db.scalar(
            select(MixPlan).where(MixPlan.id == uuid.UUID(plan_id))
        )
        row.status = MixPlanStatus.ready
        row.rendered_audio_path = key
        db.commit()

    r = client.get(f"/api/mix_plans/{plan_id}/audio")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/")


def test_get_audio_410_when_file_missing(locked_pair):
    seed = client.post(f"/api/queues/{locked_pair['queue_id']}/mix_plans").json()
    plan_id = seed[0]["id"]
    with SessionLocal() as db:
        row = db.scalar(
            select(MixPlan).where(MixPlan.id == uuid.UUID(plan_id))
        )
        row.status = MixPlanStatus.ready
        row.rendered_audio_path = f"mixes/{plan_id}.wav"  # but no file
        db.commit()
    r = client.get(f"/api/mix_plans/{plan_id}/audio")
    assert r.status_code == 410
