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
    ai_text_model: str = "gpt-5.4-mini"
    ai_vision_model: str = "gemini-3.8-flash"  # лучше всех размечает рамку баннера (сравнение в docs/05)
    ai_transcribe_model: str = "whisper-1"

    # Если шлюз не поддерживает /audio/transcriptions — отдельный провайдер только для транскрипции
    ai_transcribe_api_key: str | None = None
    ai_transcribe_base_url: str | None = None

    max_urls_per_request: int = 20

    # Анализ видео (этапы 4–6)
    ffmpeg_path: str | None = None  # по умолчанию ffmpeg из PATH или из пакета imageio-ffmpeg
    analysis_max_frames: int = 60  # 1 кадр/с; длинные ролики — равномерная выборка
    analysis_concurrency: int = 3  # сколько роликов анализируется одновременно
    vision_batch_size: int = 8  # кадров в одном запросе к vision-модели
    vision_concurrency: int = 2  # параллельных запросов к vision-модели на ролик (квота ключа NeuroAPI)
    max_transcribe_seconds: int = 900  # речь дальше 15 минут не транскрибируем
    max_video_mb: int = 300

    # Защита публичного URL от расходов на Apify/AI
    rate_limit_requests: int = 20  # POST /api/jobs с одного IP ...
    rate_limit_window_secs: int = 600  # ... за это окно
    max_new_reels_per_day: int = 300  # новых роликов в сутки на весь сервис


@lru_cache
def get_settings() -> Settings:
    return Settings()
