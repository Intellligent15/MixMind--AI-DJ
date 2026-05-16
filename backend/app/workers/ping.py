from app.workers import celery_app


@celery_app.task(name="ping")
def ping() -> str:
    return "pong"
