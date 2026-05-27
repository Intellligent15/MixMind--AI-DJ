"""Tests for the Phase 8.5 song endpoints: GET /api/songs/{id}/lyrics
and GET /api/songs/{id}/vocal_safe_regions."""

from __future__ import annotations

import asyncio
import json
import uuid

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.main import app
from app.models import (
    Lyrics,
    LyricsAlignmentStatus,
    LyricsFetchStatus,
    Song,
    SongStatus,
    Stems,
    StemsStatus,
    Transcription,
    TranscriptionStatus,
)
from app.services.storage import get_storage


def _client(db_session: Session) -> TestClient:
    def override_db():
        yield db_session

    app.dependency_overrides[get_db] = override_db
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def _make_song(db: Session, vid: str = "lyr-test") -> Song:
    s = Song(
        youtube_video_id=f"{vid}-{uuid.uuid4().hex[:8]}",
        title="Test",
        artist="Artist",
        duration_seconds=120.0,
        audio_path=f"audio/{vid}.wav",
        status=SongStatus.ready,
    )
    db.add(s)
    db.flush()
    return s


# --- GET /api/songs/{id}/lyrics ----------------------------------------


def test_get_lyrics_404_song_not_found(db_session: Session):
    client = _client(db_session)
    r = client.get(f"/api/songs/{uuid.uuid4()}/lyrics")
    assert r.status_code == 404


def test_get_lyrics_404_when_no_lyrics_row(db_session: Session):
    song = _make_song(db_session)
    client = _client(db_session)
    r = client.get(f"/api/songs/{song.id}/lyrics")
    assert r.status_code == 404


def test_get_lyrics_returns_row(db_session: Session):
    song = _make_song(db_session)
    db_session.add(Lyrics(
        song_id=song.id,
        genius_id=12345,
        text="Hello world",
        fetch_status=LyricsFetchStatus.success,
        aligned_words=[
            {"word": "Hello", "start": 0.0, "end": 0.5, "confidence": 0.9, "source": "whisper_match"},
            {"word": "world", "start": 0.5, "end": 1.0, "confidence": 0.9, "source": "whisper_match"},
        ],
        alignment_status=LyricsAlignmentStatus.success,
        alignment_quality=1.0,
    ))
    db_session.flush()

    client = _client(db_session)
    r = client.get(f"/api/songs/{song.id}/lyrics")
    assert r.status_code == 200
    body = r.json()
    assert body["genius_id"] == 12345
    assert body["text"] == "Hello world"
    assert body["fetch_status"] == "success"
    assert body["alignment_status"] == "success"
    assert body["alignment_quality"] == 1.0
    assert len(body["aligned_words"]) == 2


def test_get_lyrics_returns_not_found_status_row(db_session: Session):
    """When Genius lookup failed the row still exists with
    fetch_status=not_found; the endpoint surfaces it (not a 404)."""
    song = _make_song(db_session)
    db_session.add(Lyrics(
        song_id=song.id,
        fetch_status=LyricsFetchStatus.not_found,
    ))
    db_session.flush()

    client = _client(db_session)
    r = client.get(f"/api/songs/{song.id}/lyrics")
    assert r.status_code == 200
    body = r.json()
    assert body["fetch_status"] == "not_found"
    assert body["text"] is None


# --- GET /api/songs/{id}/vocal_safe_regions ----------------------------


def _seed_stems_with_envelope(
    db: Session, song: Song, envelope_payload: dict,
) -> Stems:
    """Write a vocal-envelope sidecar via the storage backend and link
    a Stems row to it."""
    key = f"stems/{song.youtube_video_id}/vocal_envelope.json"
    asyncio.run(get_storage().write(key, json.dumps(envelope_payload).encode()))
    stems = Stems(
        song_id=song.id,
        model_name="htdemucs",
        status=StemsStatus.separated,
        vocals_path=f"stems/{song.youtube_video_id}/vocals.wav",
        drums_path=f"stems/{song.youtube_video_id}/drums.wav",
        bass_path=f"stems/{song.youtube_video_id}/bass.wav",
        other_path=f"stems/{song.youtube_video_id}/other.wav",
        vocal_rms=0.15,
        vocal_envelope_path=key,
    )
    db.add(stems)
    db.flush()
    return stems


def _seed_transcription(db: Session, song: Song, segments: list[dict]) -> Transcription:
    t = Transcription(
        song_id=song.id,
        model_name="large-v3",
        status=TranscriptionStatus.success,
        language="en",
        segments=segments,
        duration_seconds=song.duration_seconds,
    )
    db.add(t)
    db.flush()
    return t


def test_vocal_safe_regions_404_song_not_found(db_session: Session):
    client = _client(db_session)
    r = client.get(f"/api/songs/{uuid.uuid4()}/vocal_safe_regions")
    assert r.status_code == 404


def test_vocal_safe_regions_409_when_not_fully_processed(db_session: Session):
    song = _make_song(db_session)
    # No stems, no transcription, no envelope.
    client = _client(db_session)
    r = client.get(f"/api/songs/{song.id}/vocal_safe_regions")
    assert r.status_code == 409


def test_vocal_safe_regions_returns_regions(db_session: Session):
    song = _make_song(db_session)
    # 10-second song, all-quiet envelope.
    _seed_stems_with_envelope(db_session, song, {
        "frame_hz": 10,
        "rms": [0.001] * 100,
        "peak": [0.005] * 100,
    })
    _seed_transcription(db_session, song, segments=[])
    # song.duration_seconds drives the duration the service uses.
    song.duration_seconds = 10.0
    db_session.flush()

    client = _client(db_session)
    r = client.get(f"/api/songs/{song.id}/vocal_safe_regions")
    assert r.status_code == 200
    body = r.json()
    assert "regions" in body
    assert len(body["regions"]) == 1
    assert body["regions"][0]["safe"] is True


def test_vocal_safe_regions_uses_aligned_words_when_available(db_session: Session):
    song = _make_song(db_session)
    _seed_stems_with_envelope(db_session, song, {
        "frame_hz": 10,
        "rms": [0.001] * 50 + [0.05] * 5 + [0.001] * 45,  # hot at 5.0-5.5
        "peak": [0.005] * 50 + [0.15] * 5 + [0.005] * 45,
    })
    _seed_transcription(db_session, song, segments=[])
    db_session.add(Lyrics(
        song_id=song.id,
        text="hello",
        fetch_status=LyricsFetchStatus.success,
        aligned_words=[
            {"word": "hello", "start": 5.0, "end": 5.5, "confidence": 0.9, "source": "whisper_match"},
        ],
        alignment_status=LyricsAlignmentStatus.success,
        alignment_quality=1.0,
    ))
    song.duration_seconds = 10.0
    db_session.flush()

    client = _client(db_session)
    r = client.get(f"/api/songs/{song.id}/vocal_safe_regions")
    assert r.status_code == 200
    regions = r.json()["regions"]
    # Two regions split by the aligned word at 5.0-5.5.
    assert len(regions) == 2


def test_vocal_safe_regions_accepts_threshold_query_params(db_session: Session):
    song = _make_song(db_session)
    _seed_stems_with_envelope(db_session, song, {
        "frame_hz": 10, "rms": [0.001] * 100, "peak": [0.005] * 100,
    })
    _seed_transcription(db_session, song, segments=[])
    song.duration_seconds = 10.0
    db_session.flush()

    client = _client(db_session)
    # Tighter min_safe_region_seconds — should still return one region
    # because the entire 10s is safe.
    r = client.get(
        f"/api/songs/{song.id}/vocal_safe_regions"
        "?min_safe_region_seconds=2.0&word_prob_min=0.5"
    )
    assert r.status_code == 200
    assert len(r.json()["regions"]) == 1
