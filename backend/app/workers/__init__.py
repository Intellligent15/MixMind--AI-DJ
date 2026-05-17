from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "ai_dj",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.workers.ping", "app.workers.download"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)
