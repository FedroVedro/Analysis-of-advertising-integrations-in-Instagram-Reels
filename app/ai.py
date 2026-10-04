"""Клиенты к OpenAI-совместимому AI-шлюзу (по умолчанию NeuroAPI)."""

from functools import lru_cache

from openai import OpenAI

from app.config import get_settings


@lru_cache
def get_ai_client() -> OpenAI:
    """Клиент для текстовых и vision-моделей."""
    settings = get_settings()
    if not settings.ai_api_key:
        raise RuntimeError("AI_API_KEY не задан")
    # Без base_url ключ NeuroAPI ушёл бы на api.openai.com
    if not settings.ai_base_url:
        raise RuntimeError("AI_BASE_URL не задан: укажите адрес API шлюза (NeuroAPI)")
    return OpenAI(api_key=settings.ai_api_key, base_url=settings.ai_base_url, max_retries=3)


@lru_cache
def get_transcribe_client() -> OpenAI:
    """Клиент для транскрипции: отдельный провайдер, если задан, иначе тот же шлюз."""
    settings = get_settings()
    if settings.ai_transcribe_api_key:
        return OpenAI(
            api_key=settings.ai_transcribe_api_key,
            base_url=settings.ai_transcribe_base_url,
            max_retries=3,
        )
    return get_ai_client()


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
