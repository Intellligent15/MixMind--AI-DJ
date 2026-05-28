import uuid
from unittest.mock import AsyncMock, patch
import pytest
import soundfile as sf
import numpy as np

from sqlalchemy import select

from app.core.db import SessionLocal
from app.models import (
    Analysis,
    MixPlan,
    MixPlanStatus,
    Queue,
    QueueItem,
    QueueRender,
    QueueRenderStatus,
    Song,
    SongStatus,
)

def _make_analysis(song_id, bpm: float = 120.0, key: str = "C") -> Analysis:
    return Analysis(
        song_id=song_id,
        bpm=bpm,
        key=key,
        camelot_key="8B",
        time_signature=4,
        beat_grid=[i * 0.5 for i in range(360)],
        downbeats=[i * 2.0 for i in range(90)],
        sections=[
            {"start": 0.0, "end": 30.0, "label": "intro"},
            {"start": 30.0, "end": 150.0, "label": "body"},
            {"start": 150.0, "end": 180.0, "label": "outro"},
        ],
        energy_curve=[0.5] * 180,
        vocal_segments=[],
    )

@pytest.fixture
def locked_queue_with_mixes():
    with SessionLocal() as db:
        q = Queue(locked=True)
        db.add(q)
        db.flush()
        
        songs = []
        for i in range(3):
            s = Song(
                youtube_video_id=f"rt{i}-{id(object())}",
                title=f"Song {i}", duration_seconds=180.0, audio_path=f"audio/{i}.wav",
                status=SongStatus.ready,
            )
            db.add(s)
            songs.append(s)
        db.flush()
        
        for i, s in enumerate(songs):
            db.add(_make_analysis(s.id, bpm=120.0 + i * 5))
            db.add(QueueItem(queue_id=q.id, song_id=s.id, position=i))
            
        db.flush()
        
        plan_json_0 = [
            {"tool": "set_transition_window", "from_song_time_start": 150.0, "to_song_time_start": 30.0, "duration_bars": 16},
            {"tool": "set_tempo_ramp", "song": "B", "start_time": 62.0, "end_time": 78.0, "start_bpm": 120.0, "end_bpm": 125.0}
        ]
        plan_json_1 = [
            {"tool": "set_transition_window", "from_song_time_start": 150.0, "to_song_time_start": 30.0, "duration_bars": 16},
        ]
        
        mp0 = MixPlan(queue_id=q.id, from_song_id=songs[0].id, to_song_id=songs[1].id, status=MixPlanStatus.ready, rendered_audio_path="mixes/mp0.wav", plan_json=plan_json_0)
        mp1 = MixPlan(queue_id=q.id, from_song_id=songs[1].id, to_song_id=songs[2].id, status=MixPlanStatus.ready, rendered_audio_path="mixes/mp1.wav", plan_json=plan_json_1)
        db.add_all([mp0, mp1])
        
        qr = QueueRender(queue_id=q.id, status=QueueRenderStatus.pending)
        db.add(qr)
        db.commit()
        
        yield str(q.id)

def test_stitch_queue_happy_path(locked_queue_with_mixes):
    storage = AsyncMock()
    
    async def _download_file(key, dest):
        # Write dummy stereo WAV
        sf.write(str(dest), np.zeros((44100, 2), dtype=np.float32), 44100)
    
    async def _write(key, data):
        return f"/abs/{key}"
        
    storage.download_file = _download_file
    storage.write = _write
    
    with patch("app.workers.stitch_queue.get_storage", return_value=storage):
        from app.workers.stitch_queue import stitch_queue
        res = stitch_queue(locked_queue_with_mixes)
        
    assert res == locked_queue_with_mixes
    
    with SessionLocal() as db:
        qr = db.scalar(select(QueueRender).where(QueueRender.queue_id == uuid.UUID(locked_queue_with_mixes)))
        assert qr.status == QueueRenderStatus.ready
        assert qr.rendered_audio_path == f"queue_mixes/{locked_queue_with_mixes}.flac"
