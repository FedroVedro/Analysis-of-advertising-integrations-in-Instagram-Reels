from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    database_url: str = f"sqlite:///{(BASE_DIR / 'data' / 'app.db').as_posix()}"

    apify_token: str | None = None

    # Любой OpenAI-совместимый шлюз (NeuroAPI, OpenRouter, сам OpenAI):
    # меняются только base_url, ключ и названия моделей
    ai_api_key: str | None = None
    ai_base_url: str | None = None
    ai_text_model: str = "gpt-4o-mini"
    ai_vision_model: str = "gpt-4o-mini"
    ai_transcribe_model: str = "whisper-1"

    # Если шлюз не поддерживает /audio/transcriptions — отдельный провайдер только для транскрипции
    ai_transcribe_api_key: str | None = None
    ai_transcribe_base_url: str | None = None

    max_urls_per_request: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()
