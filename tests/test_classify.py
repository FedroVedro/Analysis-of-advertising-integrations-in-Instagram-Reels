from app.pipeline import classify as cl
from app.pipeline.scoring import BannerFacts


def facts(visible=True, codes=("VALFUN",)):
    f = BannerFacts(frames_total=10, step_sec=1, duration=10)
    if visible:
        f.banner_frames, f.texts, f.promo_codes = 5, ["SKYCOACH CODE: VALFUN"], list(codes)
    return f


def fake_llm(**over):
    def fake(*_args):
        data = dict(integration_class=1, caption_mention=True, caption_cta=True, used_llm=True)
        data.update(over)
        return cl.Classification(**data)
    return fake


def test_nothing_about_skycoach_is_class_0_without_llm(monkeypatch):
    monkeypatch.setattr(cl, "_ask_llm", lambda *a: (_ for _ in ()).throw(AssertionError("LLM не нужен")))
    result = cl.classify("Гайд по боссу", None, facts(visible=False))
    assert result.integration_class == 0 and not result.used_llm


def test_promo_code_on_banner_forces_class_2(monkeypatch):
    monkeypatch.setattr(cl, "_ask_llm", fake_llm(integration_class=1))
    assert cl.classify("", None, facts()).integration_class == 2


def test_caption_flags_are_checked_against_caption_text(monkeypatch):
    # модель ошибочно говорит «в подписи призыв», а в подписи про Skycoach ничего нет
    monkeypatch.setattr(cl, "_ask_llm", fake_llm(caption_mention=True, caption_cta=True))
    result = cl.classify("They will cower! Reyna guide #valorant", None, facts())
    assert not result.caption_mention and not result.caption_cta


def test_promo_code_in_caption_is_caption_cta(monkeypatch):
    monkeypatch.setattr(cl, "_ask_llm", fake_llm(caption_mention=False, caption_cta=False))
    result = cl.classify("use code valfun!", None, facts())
    assert result.caption_mention and result.caption_cta


def test_skycoach_misspelling_in_speech_counts_as_mention(monkeypatch):
    called = []
    monkeypatch.setattr(cl, "_ask_llm", lambda *a: called.append(a) or cl.Classification(1, used_llm=True))

    class T:
        def with_timestamps(self):
            return "[0:14] go to sky coach dot gg"
    cl.classify("", T(), facts(visible=False))
    assert called


def _llm_json(monkeypatch, payload):
    import json, types
    resp = types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=json.dumps(payload)))])
    monkeypatch.setattr(cl, "chat", lambda **kw: resp)


def test_promo_code_as_string_is_not_split_into_letters(monkeypatch):
    _llm_json(monkeypatch, {"integration_class": 2, "promo_codes": "VALFUN", "caption_cta": True})
    result = cl.classify("Reyna guide, love this agent", None, facts())
    assert result.promo_codes == ["VALFUN"]
    assert not result.caption_mention and not result.caption_cta  # раньше буквы «находились» в подписи


def test_class_as_string_is_respected(monkeypatch):
    _llm_json(monkeypatch, {"integration_class": "2", "voice_cta": True})

    class T:
        def with_timestamps(self):
            return "[0:05] go to skycoach and use my code"
    assert cl.classify("", T(), facts(visible=False)).integration_class == 2
