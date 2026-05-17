from collections.abc import Generator

from sqlalchemy.orm import Session

from app.core.db import SessionLocal
from app.services.youtube import YouTubeService


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_youtube_service() -> YouTubeService:
    return YouTubeService()
