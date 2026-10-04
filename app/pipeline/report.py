"""Текст обоснования для менеджера — собирается из фактов, без LLM.

Первая строка — сводка (её видно в таблице), дальше — детали с цифрами.
"""

from app.pipeline.classify import Classification
from app.pipeline.scoring import BannerFacts, Issue, deduction
from app.pipeline.transcribe import Transcript, fmt_time

CLASS_NAMES = {0: "Нет интеграции", 1: "Упоминание", 2: "Реклама"}
MAX_INTERVALS = 5


def build_justification(
    *,
    cls: Classification,
    facts: BannerFacts,
    score: int | None,
    score_reasons: list[str],
    issues: list[Issue],
    transcript: Transcript | None,
    audio_note: str | None,
    sampled: bool,
) -> str:
    if cls.integration_class == 0:
        lines = [cls.reasoning or "Skycoach не найден ни в кадре, ни в речи, ни в подписи."]
        if audio_note:
            lines.append(audio_note)
        return "\n".join(lines)

    pct, verdict = deduction(issues)
    summary = [f"{score}/5", CLASS_NAMES[cls.integration_class]]
    if facts.visible:
        summary.append(f"баннер {_secs(facts.seconds)} с ({facts.share:.0%}), ≈{facts.area_pct:g} % кадра")
    else:
        summary.append("баннера в кадре нет")
    summary.append(verdict)
    lines = [" · ".join(summary)]

    if facts.visible:
        intervals = ", ".join(f"{fmt_time(a)}–{fmt_time(b)}" for a, b in facts.intervals[:MAX_INTERVALS])
        if len(facts.intervals) > MAX_INTERVALS:
            intervals += f" и ещё {len(facts.intervals) - MAX_INTERVALS}"
        banner = (
            f"Баннер: в кадре ≈{_secs(facts.seconds)} с из {facts.duration:.0f} ({facts.share:.0%}), {intervals}; "
            f"{facts.position} (≈{facts.top_pct:g}–{facts.bottom_pct:g} % высоты); ширина ≈{facts.width_pct:g} % кадра, площадь ≈{facts.area_pct:g} %."
        )
        if facts.texts:
            banner += f" Текст: «{facts.texts[0][:120]}»."
        lines.append(banner)
    else:
        lines.append(f"Баннер: не найден ни на одном из {facts.frames_total} кадров.")

    if cls.promo_codes:
        lines.append("Промокод: " + ", ".join(cls.promo_codes) + ".")

    if audio_note:
        lines.append(audio_note)
    elif cls.voice_mentions:
        quotes = "; ".join(f"{m.get('time', '?')} «{m['quote'][:100]}»" for m in cls.voice_mentions[:3])
        lines.append(f"Голос: {quotes}" + ("; есть призыв." if cls.voice_cta else "; призыва нет."))
    elif transcript and transcript.text:
        lines.append("Голос: Skycoach не упоминается.")
    else:
        lines.append("Голос: речи нет.")

    if cls.caption_cta:
        lines.append("Подпись: есть призыв или промокод Skycoach.")
    elif cls.caption_mention:
        lines.append("Подпись: Skycoach упоминается без призыва.")
    else:
        lines.append("Подпись: Skycoach не упоминается.")

    lines.append(f"Заметность {score}/5: " + "; ".join(score_reasons) + ".")

    if issues:
        details = "; ".join(f"{i.title.lower()} ({i.detail})" for i in issues)
        lines.append(f"Размещение по правилам Skycoach: {details} → {verdict}.")
    else:
        lines.append("Размещение по правилам Skycoach: без замечаний.")

    if cls.reasoning:
        lines.append(f"Класс: {cls.reasoning}")
    if sampled:
        lines.append(f"Длинный ролик: проанализирована выборка из {facts.frames_total} кадров (шаг {facts.step_sec:g} с).")
    return "\n".join(lines)


def _secs(value: float) -> str:
    """Короткие ролики — с точностью до десятых, длинные (выборка) — до целых."""
    return f"{value:g}" if value < 60 else f"{value:.0f}"
