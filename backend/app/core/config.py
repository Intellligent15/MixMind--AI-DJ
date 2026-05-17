from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://aidj:aidj@localhost:5432/aidj"
    redis_url: str = "redis://localhost:6379/0"
    log_level: str = "INFO"

    storage_backend: str = "local"
    local_storage_path: Path = Path("./cache")


settings = Settings()
