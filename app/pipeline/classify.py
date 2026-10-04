"""Класс интеграции 0/1/2.

Жёсткие правила там, где ответ однозначен, LLM — только для смысла речи и подписи:
- нигде нет Skycoach (кадры, подпись, речь, текст на экране) → 0 без вызова модели;
- на баннере есть промокод → 2 (промокод — прямой признак рекламы по заданию);
- иначе модель читает подпись, транскрипт и текст с экрана и решает 1 или 2.
"""

import json
import logging
import re
from dataclasses import dataclass, field

from app.ai import get_ai_client
from app.config import get_settings
from app.pipeline.scoring import BannerFacts
from app.pipeline.transcribe import Transcript

logger = logging.getLogger(__name__)

# Учитываем типичные ошибки распознавания речи и OCR: «sky coach», «скайкоуч», «skycoach.gg»
SKYCOACH_RE = re.compile(r"sky\s*-?\s*c[o0]a?ch|ска[йи]\s*-?\s*ко[уа]?ч", re.IGNORECASE)

PROMPT = """You classify a sponsored-content integration of Skycoach (a marketplace of gaming services: boosting, coaching, leveling) in a short video.

Classes:
0 — Skycoach is not present at all.
1 — Skycoach is mentioned or visible (logo, name in speech or text) but its product is NOT advertised.
2 — Skycoach product IS advertised: a call to action ("go to", "use", "link in bio", "order"), a description of the service, a promo code, a discount, a price.

Evidence (may be partial; speech is machine-transcribed and can contain errors):
CAPTION:
{caption}

SPEECH TRANSCRIPT (with timestamps):
{transcript}

TEXT ON THE BANNER (from video frames):
{banner}

OTHER ON-SCREEN TEXT:
{other}

Return JSON only:
{{"integration_class": 0|1|2,
 "voice_mentions": [{{"time": "m:ss", "quote": "exact words from the transcript about Skycoach"}}],
 "voice_cta": true|false,      // the speaker calls to use/buy Skycoach or names a promo code
 "caption_mention": true|false,
 "caption_cta": true|false,    // caption contains a call to action / promo code for Skycoach
 "promo_codes": ["..."],
 "reasoning": "1–2 sentences in Russian"}}"""


@dataclass
class Classification:
    integration_class: int
    voice_mentions: list[dict] = field(default_factory=list)
    voice_cta: bool = False
    caption_mention: bool = False
    caption_cta: bool = False
    promo_codes: list[str] = field(default_factory=list)
    reasoning: str = ""
    used_llm: bool = False

    @property
    def text_cta(self) -> bool:
        return self.caption_cta or bool(self.promo_codes)


def classify(caption: str | None, transcript: Transcript | None, facts: BannerFacts) -> Classification:
    caption = caption or ""
    speech = transcript.with_timestamps() if transcript else ""
    banner_text = "\n".join(facts.texts)
    other_text = "\n".join(facts.other_texts)

    mentioned = facts.visible or any(SKYCOACH_RE.search(s) for s in (caption, speech, banner_text, other_text))
    if not mentioned:
        return Classification(0, reasoning="Skycoach не найден ни в кадре, ни в речи, ни в подписи.")

    result = _ask_llm(caption, speech, banner_text, other_text)
    codes = {c.upper() for c in facts.promo_codes} | {c.upper() for c in result.promo_codes}
    result.promo_codes = sorted(codes)

    # Флаги подписи проверяем по тексту, а не верим модели: она путает текст баннера с подписью.
    # Призыв в подписи засчитываем, только если там реально есть Skycoach или промокод.
    caption_codes = [c for c in codes if c and c.lower() in caption.lower()]
    result.caption_mention = bool(SKYCOACH_RE.search(caption)) or bool(caption_codes)
    result.caption_cta = result.caption_mention and (result.caption_cta or bool(caption_codes))
    if facts.promo_codes and result.integration_class < 2:
        result.integration_class = 2
        result.reasoning = (result.reasoning + " На баннере промокод — это реклама.").strip()
    if facts.visible and result.integration_class == 0:
        result.integration_class = 1  # баннер в кадре — минимум упоминание
    return result


def _ask_llm(caption: str, speech: str, banner: str, other: str) -> Classification:
    prompt = PROMPT.format(
        caption=caption[:3000] or "(none)",
        transcript=speech[:6000] or "(no speech)",
        banner=banner[:1000] or "(no banner detected)",
        other=other[:1000] or "(none)",
    )
    response = get_ai_client().chat.completions.create(
        model=get_settings().ai_text_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=2000,  # без явного лимита шлюз резервирует максимум модели (см. vision.py)
    )
    text = (response.choices[0].message.content or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        data = json.loads(text[text.index("{"): text.rindex("}") + 1], strict=False)
    except ValueError as exc:
        raise RuntimeError(f"Текстовая модель вернула невалидный JSON: {text[:200]}") from exc

    cls = data.get("integration_class")
    return Classification(
        integration_class=cls if cls in (0, 1, 2) else 1,
        voice_mentions=[m for m in data.get("voice_mentions") or [] if isinstance(m, dict) and m.get("quote")],
        voice_cta=bool(data.get("voice_cta")),
        caption_mention=bool(data.get("caption_mention")),
        caption_cta=bool(data.get("caption_cta")),
        promo_codes=[str(c) for c in data.get("promo_codes") or [] if c],
        reasoning=str(data.get("reasoning") or "").strip(),
        used_llm=True,
    )
