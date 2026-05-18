from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://aidj:aidj@localhost:5432/aidj"
    redis_url: str = "redis://localhost:6379/0"
    log_level: str = "INFO"

    storage_backend: str = "local"
    local_storage_path: Path = Path("./cache")

    # Phase 6: skip Whisper if the Stems row's `vocal_rms` is below this
    # threshold. 0.005 sits well below an audible monologue (~0.04 in our
    # smoke-test clip) and well above silence — only true instrumentals
    # and ambient tracks land underneath.
    whisper_vocal_rms_threshold: float = 0.005


settings = Settings()
