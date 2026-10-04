"""Транскрипция через OpenAI-совместимый API (whisper-1 с таймкодами)."""

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import openai

from app.ai import get_transcribe_client
from app.config import get_settings

# Подсказка повышает шанс, что Whisper напишет название бренда правильно
PROMPT = "Skycoach, SkyCoach, promo code, промокод, бустинг"
# Сегменты, которые Whisper сам считает «не речью» (музыка, шум), — частый источник галлюцинаций
NO_SPEECH_PROB = 0.6
MIN_AVG_LOGPROB = -1.0
# Типовые галлюцинации Whisper на музыке и тишине (из субтитров, на которых он обучен)
HALLUCINATION_RE = re.compile(
    r"продолжение следует|субтитр|редактор|спасибо за просмотр|подписывайтесь|"
    r"thanks? (you )?for watching|subscribe|amara\.org",
    re.IGNORECASE,
)
# Шлюз под нагрузкой отвечает 503 «слишком большая нагрузка» — ждём и повторяем
RETRY_DELAYS = (10, 30)

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    segments: list[Segment] = field(default_factory=list)
    language: str | None = None

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments).strip()

    def with_timestamps(self) -> str:
        return "\n".join(f"[{fmt_time(s.start)}] {s.text}" for s in self.segments)


def fmt_time(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def transcribe(audio: Path) -> Transcript:
    for attempt, delay in enumerate((*RETRY_DELAYS, None), start=1):
        try:
            with audio.open("rb") as f:
                response = get_transcribe_client().audio.transcriptions.create(
                    model=get_settings().ai_transcribe_model,
                    file=f,
                    response_format="verbose_json",
                    prompt=PROMPT,
                    temperature=0,
                )
            break
        except (openai.APIStatusError, openai.APIConnectionError) as exc:
            status = getattr(exc, "status_code", None)
            retryable = status is None or status == 429 or status >= 500
            if not retryable or delay is None:
                raise
            logger.warning("Транскрипция: попытка %d не удалась (%s), повтор через %d с", attempt, status, delay)
            time.sleep(delay)
    data = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    segments = [
        Segment(float(s["start"]), float(s["end"]), s["text"].strip())
        for s in data.get("segments") or []
        if s.get("text", "").strip()
        and s.get("no_speech_prob", 0) < NO_SPEECH_PROB
        and s.get("avg_logprob", 0) > MIN_AVG_LOGPROB
        and not HALLUCINATION_RE.search(s["text"])
    ]
    if not data.get("segments") and data.get("text", "").strip() and not HALLUCINATION_RE.search(data["text"]):
        # Шлюз вернул текст без сегментов — сохраняем хотя бы его
        segments = [Segment(0.0, float(data.get("duration") or 0), data["text"].strip())]
    return Transcript(segments, data.get("language"))
