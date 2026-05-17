from pathlib import Path

import pytest

from app.services.storage import LocalFilesystemStorage


@pytest.fixture
def storage(tmp_path: Path) -> LocalFilesystemStorage:
    return LocalFilesystemStorage(tmp_path)


async def test_write_creates_parent_dirs_and_returns_path(
    storage: LocalFilesystemStorage, tmp_path: Path
) -> None:
    returned = await storage.write("audio/abc.wav", b"hello")
    expected = tmp_path / "audio" / "abc.wav"
    assert Path(returned) == expected.resolve()
    assert expected.read_bytes() == b"hello"


async def test_read_round_trip(storage: LocalFilesystemStorage) -> None:
    await storage.write("k", b"data")
    assert await storage.read("k") == b"data"


async def test_exists(storage: LocalFilesystemStorage) -> None:
    assert not await storage.exists("missing")
    await storage.write("present", b"x")
    assert await storage.exists("present")


async def test_delete_is_idempotent(storage: LocalFilesystemStorage) -> None:
    await storage.write("k", b"x")
    await storage.delete("k")
    assert not await storage.exists("k")
    await storage.delete("k")  # second delete must not raise


async def test_get_url_returns_absolute_path(
    storage: LocalFilesystemStorage, tmp_path: Path
) -> None:
    await storage.write("k", b"x")
    url = await storage.get_url("k")
    assert url == str((tmp_path / "k").resolve())


async def test_path_rejects_traversal(storage: LocalFilesystemStorage) -> None:
    with pytest.raises(ValueError, match="escapes storage root"):
        storage.path("../escape")
