from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.db import check_db
from app.core.redis_client import check_redis

app = FastAPI(title="AI DJ Backend", version="0.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    db_ok = check_db()
    redis_ok = check_redis()
    overall = "ok" if (db_ok and redis_ok) else "degraded"
    return {
        "status": overall,
        "db": "ok" if db_ok else "down",
        "redis": "ok" if redis_ok else "down",
    }
