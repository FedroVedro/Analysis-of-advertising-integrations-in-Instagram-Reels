import httpx
import openai
import pytest

from app import ai


def status_error(code, body="error"):
    return openai.APIStatusError(body, response=httpx.Response(code, request=httpx.Request("POST", "http://x")), body=None)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(ai.time, "sleep", lambda s: None)


@pytest.mark.parametrize("exc", [status_error(403, "pre_consume_token_quota_failed"), status_error(503), status_error(429)])
def test_retryable_errors_are_retried(exc):
    calls = iter([exc, "ok"])
    def call():
        v = next(calls)
        if isinstance(v, Exception):
            raise v
        return v
    assert ai.with_retry(call, "test") == "ok"


@pytest.mark.parametrize("exc", [status_error(400, "context_length_exceeded"), status_error(403, "forbidden"), ValueError("x")])
def test_other_errors_are_not_retried(exc):
    calls = []
    def call():
        calls.append(1)
        raise exc
    with pytest.raises(type(exc)):
        ai.with_retry(call, "test")
    assert len(calls) == 1


def test_gives_up_after_all_delays():
    calls = []
    def call():
        calls.append(1)
        raise status_error(503)
    with pytest.raises(openai.APIStatusError):
        ai.with_retry(call, "test", delays=(1, 1))
    assert len(calls) == 3


def test_clients_have_sane_timeouts(monkeypatch):
    from app.config import get_settings
    monkeypatch.setattr(get_settings(), "ai_api_key", "k")
    monkeypatch.setattr(get_settings(), "ai_base_url", "https://example.invalid/v1")
    ai.get_ai_client.cache_clear(); ai.get_transcribe_client.cache_clear()
    try:
        assert ai.get_ai_client().timeout == ai.CHAT_TIMEOUT  # было 600 с по умолчанию SDK
        assert ai.get_transcribe_client().timeout == ai.TRANSCRIBE_TIMEOUT
    finally:
        ai.get_ai_client.cache_clear(); ai.get_transcribe_client.cache_clear()


def test_missing_config_is_non_retryable(monkeypatch):
    from app.config import get_settings
    from app.errors import NonRetryableError
    monkeypatch.setattr(get_settings(), "ai_api_key", None)
    ai.get_ai_client.cache_clear()
    with pytest.raises(NonRetryableError):
        ai.get_ai_client()
    ai.get_ai_client.cache_clear()
