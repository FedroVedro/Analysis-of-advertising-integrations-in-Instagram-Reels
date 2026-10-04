"""Клиенты к OpenAI-совместимому AI-шлюзу (по умолчанию NeuroAPI) и общие повторы запросов."""

import logging
import time
from collections.abc import Callable
from functools import lru_cache
from typing import TypeVar

import openai
from openai import OpenAI

from app.config import get_settings
from app.errors import ConfigError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Таймаут по умолчанию в OpenAI SDK — 600 с: один зависший запрос держал бы всю пачку роликов.
CHAT_TIMEOUT = 120.0
TRANSCRIBE_TIMEOUT = 300.0  # транскрипция у шлюза бывает больше минуты
# Повторы делаем сами (с паузами и с учётом 403-квоты NeuroAPI), поэтому внутренний повтор SDK — один
SDK_MAX_RETRIES = 1
RETRY_DELAYS = (5, 15, 30)


@lru_cache
def get_ai_client() -> OpenAI:
    """Клиент для текстовых и vision-моделей."""
    settings = get_settings()
    if not settings.ai_api_key:
        raise ConfigError("Сервис не настроен: не задан AI_API_KEY")
    # Без base_url ключ NeuroAPI ушёл бы на api.openai.com
    if not settings.ai_base_url:
        raise ConfigError("Сервис не настроен: не задан AI_BASE_URL (адрес шлюза NeuroAPI)")
    return OpenAI(api_key=settings.ai_api_key, base_url=settings.ai_base_url,
                  timeout=CHAT_TIMEOUT, max_retries=SDK_MAX_RETRIES)


@lru_cache
def get_transcribe_client() -> OpenAI:
    """Клиент для транскрипции: отдельный провайдер, если задан, иначе тот же шлюз."""
    settings = get_settings()
    if settings.ai_transcribe_api_key:
        client = OpenAI(api_key=settings.ai_transcribe_api_key, base_url=settings.ai_transcribe_base_url,
                        max_retries=SDK_MAX_RETRIES)
    else:
        client = get_ai_client()
    return client.with_options(timeout=TRANSCRIBE_TIMEOUT)


def is_retryable(exc: Exception) -> bool:
    """429, 5xx, обрыв связи и 403 «нет квоты на резерв» (NeuroAPI при параллельных запросах) — повторяем."""
    if isinstance(exc, openai.APIConnectionError):  # включая таймаут
        return True
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        return status == 429 or status >= 500 or (status == 403 and "quota" in str(exc).lower())
    return False


def with_retry(call: Callable[[], T], label: str, delays: tuple[int, ...] = RETRY_DELAYS) -> T:
    for attempt, delay in enumerate((*delays, None), start=1):
        try:
            return call()
        except Exception as exc:
            if delay is None or not is_retryable(exc):
                raise
            logger.warning("%s: попытка %d не удалась (%s), повтор через %d с",
                           label, attempt, getattr(exc, "status_code", type(exc).__name__), delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


def chat(**kwargs):
    """chat.completions.create с повторами."""
    return with_retry(lambda: get_ai_client().chat.completions.create(**kwargs), f"AI {kwargs.get('model')}")


if __name__ == "__main__":
    # Проверка ключа и шлюза: python -m app.ai
    settings = get_settings()
    client = get_ai_client()
    models = sorted(m.id for m in client.models.list())
    print(f"Доступно моделей: {len(models)}")
    for name in (settings.ai_text_model, settings.ai_vision_model, settings.ai_transcribe_model):
        print(f"  {name}: {'есть' if name in models else 'НЕТ в списке'}")
    reply = client.chat.completions.create(
        model=settings.ai_text_model,
        messages=[{"role": "user", "content": "Ответь одним словом: ок"}],
        max_tokens=5,
    )
    print("Тестовый запрос:", reply.choices[0].message.content)
