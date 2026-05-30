from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://aidj:aidj@localhost:5432/aidj"
    redis_url: str = "redis://localhost:6379/0"
    log_level: str = "INFO"

    storage_backend: str = "local"
    local_storage_path: Path = Path("./cache")
    s3_endpoint_url: str = ""
    s3_bucket_name: str = ""
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_region_name: str = "auto"
    
    modal_token_id: str = ""
    modal_token_secret: str = ""
    
    genius_access_token: str = ""
    gemini_api_key: str = ""
    llm_provider: str = "gemini"
    use_llm_planner: bool = True

    # Path to a Netscape-format cookies.txt that yt-dlp passes to YouTube.
    # Required on cloud hosts (the droplet) — YouTube's anti-bot system
    # rejects datacenter IPs unless an authenticated session is presented.
    # On macOS dev the residential IP is unblocked and this can stay empty.
    yt_dlp_cookies_file: str = ""

    # Phase 6: skip Whisper if the Stems row's `vocal_rms` is below this
    # threshold. 0.005 sits well below an audible monologue (~0.04 in our
    # smoke-test clip) and well above silence — only true instrumentals
    # and ambient tracks land underneath.
    whisper_vocal_rms_threshold: float = 0.005

settings = Settings()
