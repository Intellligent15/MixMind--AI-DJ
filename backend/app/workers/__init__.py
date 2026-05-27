from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "ai_dj",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "app.workers.ping",
        "app.workers.download",
        "app.workers.analyze",
        "app.workers.separate",
        "app.workers.transcribe",
        "app.workers.render_transition",
        "app.workers.stitch_queue",
        "app.workers.fetch_lyrics",
        "app.workers.align_lyrics",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)
