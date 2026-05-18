from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.main import app
from app.models import Analysis, Song, SongStatus, Stems, StemsStatus
from app.services.storage import LocalFilesystemStorage


def _client(db_session: Session) -> TestClient:
    def override_db():
        yield db_session

    app.dependency_overrides[get_db] = override_db
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def _payload(**overrides):
    base = {
        "youtube_video_id": "abc123",
        "title": "A Song",
        "artist": "An Artist",
        "duration_seconds": 200.0,
        "thumbnail_url": "https://t/thumb.jpg",
    }
    base.update(overrides)
    return base


def test_create_song_persists_and_enqueues(db_session: Session):
    client = _client(db_session)
    with patch("app.api.songs.download_song.delay") as delay:
        r = client.post("/api/songs", json=_payload())
    assert r.status_code == 201
    body = r.json()
    assert body["youtube_video_id"] == "abc123"
    assert body["status"] == "pending"
    delay.assert_called_once_with(body["id"])

    row = db_session.get(Song, body["id"])
    assert row is not None
    assert row.title == "A Song"


def test_create_song_dedupes_on_youtube_id(db_session: Session):
    client = _client(db_session)
    with patch("app.api.songs.download_song.delay") as delay:
        r1 = client.post("/api/songs", json=_payload())
        r2 = client.post("/api/songs", json=_payload(title="different title"))
    assert r1.status_code == 201
    assert r2.status_code == 201
    assert r1.json()["id"] == r2.json()["id"]
    # title stays as the original — dedupe returns existing row unchanged
    assert r2.json()["title"] == "A Song"
    # download enqueued only once
    delay.assert_called_once()


def test_get_song_returns_row(db_session: Session):
    client = _client(db_session)
    with patch("app.api.songs.download_song.delay"):
        created = client.post("/api/songs", json=_payload()).json()
    r = client.get(f"/api/songs/{created['id']}")
    assert r.status_code == 200
    assert r.json()["id"] == created["id"]


def test_get_song_404(db_session: Session):
    client = _client(db_session)
    r = client.get("/api/songs/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404


def test_list_songs(db_session: Session):
    client = _client(db_session)
    with patch("app.api.songs.download_song.delay"):
        client.post("/api/songs", json=_payload())
        client.post("/api/songs", json=_payload(youtube_video_id="def", title="Two"))
    r = client.get("/api/songs")
    assert r.status_code == 200
    titles = [s["title"] for s in r.json()]
    assert set(titles) >= {"A Song", "Two"}


def test_audio_409_when_not_downloaded(db_session: Session):
    client = _client(db_session)
    with patch("app.api.songs.download_song.delay"):
        created = client.post("/api/songs", json=_payload()).json()
    r = client.get(f"/api/songs/{created['id']}/audio")
    assert r.status_code == 409


@pytest.fixture
def tmp_storage(tmp_path: Path):
    storage = LocalFilesystemStorage(tmp_path)
    with patch("app.api.songs.get_storage", return_value=storage):
        yield storage


def test_audio_streams_file_with_range_support(
    db_session: Session, tmp_storage: LocalFilesystemStorage
):
    key = "audio/abc123.wav"
    tmp_storage.path(key).parent.mkdir(parents=True, exist_ok=True)
    tmp_storage.path(key).write_bytes(b"RIFF" + b"\x00" * 100)

    song = Song(
        youtube_video_id="abc123",
        title="A Song",
        artist=None,
        duration_seconds=10.0,
        thumbnail_url=None,
        audio_path=key,
        status=SongStatus.downloaded,
    )
    db_session.add(song)
    db_session.flush()

    client = _client(db_session)
    r = client.get(f"/api/songs/{song.id}/audio")
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.headers.get("accept-ranges") == "bytes"
    assert r.content.startswith(b"RIFF")

    r = client.get(
        f"/api/songs/{song.id}/audio", headers={"Range": "bytes=4-7"}
    )
    assert r.status_code == 206
    assert r.content == b"\x00\x00\x00\x00"


def test_audio_410_when_file_missing(
    db_session: Session, tmp_storage: LocalFilesystemStorage
):
    assert tmp_storage  # fixture patches get_storage in the songs router
    song = Song(
        youtube_video_id="abc",
        title="T",
        artist=None,
        duration_seconds=1.0,
        thumbnail_url=None,
        audio_path="audio/does-not-exist.wav",
        status=SongStatus.downloaded,
    )
    db_session.add(song)
    db_session.flush()

    client = _client(db_session)
    r = client.get(f"/api/songs/{song.id}/audio")
    assert r.status_code == 410


def _downloaded_song(db: Session, vid: str = "downloaded") -> Song:
    song = Song(
        youtube_video_id=vid,
        title="T",
        artist=None,
        duration_seconds=10.0,
        thumbnail_url=None,
        audio_path=f"audio/{vid}.wav",
        status=SongStatus.downloaded,
    )
    db.add(song)
    db.flush()
    return song


def test_trigger_analyze_enqueues_when_downloaded(db_session: Session):
    song = _downloaded_song(db_session)
    client = _client(db_session)
    with patch("app.api.songs.analyze_song.delay") as delay:
        r = client.post(f"/api/songs/{song.id}/analyze")
    assert r.status_code == 202
    delay.assert_called_once_with(str(song.id))


def test_trigger_analyze_404_unknown_song(db_session: Session):
    client = _client(db_session)
    r = client.post("/api/songs/00000000-0000-0000-0000-000000000000/analyze")
    assert r.status_code == 404


def test_trigger_analyze_409_when_not_downloaded(db_session: Session):
    song = Song(
        youtube_video_id="not-yet",
        title="T",
        artist=None,
        duration_seconds=10.0,
        thumbnail_url=None,
        status=SongStatus.pending,
    )
    db_session.add(song)
    db_session.flush()
    client = _client(db_session)
    with patch("app.api.songs.analyze_song.delay") as delay:
        r = client.post(f"/api/songs/{song.id}/analyze")
    assert r.status_code == 409
    delay.assert_not_called()


def test_get_analysis_returns_row(db_session: Session):
    song = _downloaded_song(db_session, vid="hasanalysis")
    db_session.add(
        Analysis(
            song_id=song.id,
            bpm=120.0,
            key="C",
            camelot_key="8B",
            time_signature=4,
            beat_grid=[0.0, 0.5],
            downbeats=[0.0],
            sections=[{"start": 0.0, "end": 10.0, "label": "section_1"}],
            energy_curve=[0.1, 0.2],
            vocal_segments=[],
        )
    )
    db_session.flush()
    client = _client(db_session)
    r = client.get(f"/api/songs/{song.id}/analysis")
    assert r.status_code == 200
    body = r.json()
    assert body["bpm"] == 120.0
    assert body["camelot_key"] == "8B"
    assert body["sections"] == [
        {"start": 0.0, "end": 10.0, "label": "section_1"}
    ]


def test_get_analysis_404_song_not_found(db_session: Session):
    client = _client(db_session)
    r = client.get("/api/songs/00000000-0000-0000-0000-000000000000/analysis")
    assert r.status_code == 404


def test_get_analysis_404_when_not_yet_analyzed(db_session: Session):
    song = _downloaded_song(db_session, vid="no-analysis")
    client = _client(db_session)
    r = client.get(f"/api/songs/{song.id}/analysis")
    assert r.status_code == 404
    assert "analysis" in r.json()["detail"].lower()


def _analyzed_song(db: Session, vid: str = "analyzed") -> Song:
    song = _downloaded_song(db, vid=vid)
    song.status = SongStatus.analyzed
    db.flush()
    return song


def _stems_row(db: Session, song: Song, **overrides) -> Stems:
    vid = song.youtube_video_id
    base = dict(
        song_id=song.id,
        model_name="htdemucs",
        status=StemsStatus.separated,
        vocals_path=f"stems/{vid}/vocals.wav",
        drums_path=f"stems/{vid}/drums.wav",
        bass_path=f"stems/{vid}/bass.wav",
        other_path=f"stems/{vid}/other.wav",
        vocal_rms=0.2,
    )
    base.update(overrides)
    row = Stems(**base)
    db.add(row)
    db.flush()
    return row


def test_trigger_separate_enqueues_when_analyzed(db_session: Session):
    song = _analyzed_song(db_session, vid="sep-ok")
    client = _client(db_session)
    with patch("app.api.songs.celery_app.send_task") as send:
        r = client.post(f"/api/songs/{song.id}/separate")
    assert r.status_code == 202
    send.assert_called_once_with(
        "app.workers.separate.separate_stems", args=[str(song.id)]
    )


def test_trigger_separate_404_unknown_song(db_session: Session):
    client = _client(db_session)
    r = client.post("/api/songs/00000000-0000-0000-0000-000000000000/separate")
    assert r.status_code == 404


def test_trigger_separate_409_when_not_analyzed(db_session: Session):
    song = _downloaded_song(db_session, vid="sep-too-early")
    client = _client(db_session)
    with patch("app.api.songs.celery_app.send_task") as send:
        r = client.post(f"/api/songs/{song.id}/separate")
    assert r.status_code == 409
    send.assert_not_called()


def test_get_stems_returns_row(db_session: Session):
    song = _analyzed_song(db_session, vid="hasstems")
    _stems_row(db_session, song)
    client = _client(db_session)
    r = client.get(f"/api/songs/{song.id}/stems")
    assert r.status_code == 200
    body = r.json()
    assert body["model_name"] == "htdemucs"
    assert body["status"] == "separated"
    assert body["vocals_path"].endswith("/vocals.wav")
    assert body["vocal_rms"] == 0.2


def test_get_stems_404_song_not_found(db_session: Session):
    client = _client(db_session)
    r = client.get("/api/songs/00000000-0000-0000-0000-000000000000/stems")
    assert r.status_code == 404


def test_get_stems_404_when_not_separated(db_session: Session):
    song = _analyzed_song(db_session, vid="no-stems-yet")
    client = _client(db_session)
    r = client.get(f"/api/songs/{song.id}/stems")
    assert r.status_code == 404


def test_get_stem_audio_streams_file(
    db_session: Session, tmp_storage: LocalFilesystemStorage
):
    song = _analyzed_song(db_session, vid="streamstems")
    _stems_row(db_session, song)
    key = f"stems/{song.youtube_video_id}/vocals.wav"
    tmp_storage.path(key).parent.mkdir(parents=True, exist_ok=True)
    tmp_storage.path(key).write_bytes(b"RIFF" + b"\x00" * 50)

    client = _client(db_session)
    r = client.get(f"/api/songs/{song.id}/stems/vocals")
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.content.startswith(b"RIFF")


def test_get_stem_audio_404_unknown_stem_name(db_session: Session):
    song = _analyzed_song(db_session, vid="badname")
    _stems_row(db_session, song)
    client = _client(db_session)
    r = client.get(f"/api/songs/{song.id}/stems/cowbell")
    assert r.status_code == 404


def test_get_stem_audio_410_when_file_missing(
    db_session: Session, tmp_storage: LocalFilesystemStorage
):
    assert tmp_storage
    song = _analyzed_song(db_session, vid="ghost-stems")
    _stems_row(db_session, song)
    client = _client(db_session)
    r = client.get(f"/api/songs/{song.id}/stems/drums")
    assert r.status_code == 410


def test_get_stem_audio_409_when_path_null(db_session: Session):
    song = _analyzed_song(db_session, vid="partial-stems")
    _stems_row(db_session, song, vocals_path=None)
    client = _client(db_session)
    r = client.get(f"/api/songs/{song.id}/stems/vocals")
    assert r.status_code == 409
