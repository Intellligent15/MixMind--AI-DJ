import redis

from app.core.config import settings

redis_client = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=2)


def check_redis() -> bool:
    try:
        return bool(redis_client.ping())
    except Exception:
        return False
