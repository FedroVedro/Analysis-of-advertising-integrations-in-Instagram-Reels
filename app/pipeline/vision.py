"""Поиск баннера Skycoach на кадрах vision-моделью.

Модель отвечает только на вопросы «что на кадре»: есть ли баннер, где его рамка, что на нём
написано. Все выводы (секунды, размер, обрезка, перекрытие интерфейсом) считает код в scoring.py —
так цифры в обосновании воспроизводимы и не зависят от фантазии модели.
"""

import base64
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import openai

from app.ai import get_ai_client
from app.config import get_settings

logger = logging.getLogger(__name__)

VISION_MAX_TOKENS_BASE = 2000
VISION_MAX_TOKENS_PER_FRAME = 300
# 429/5xx и 403 «нет квоты на резерв» у NeuroAPI проходят при повторе, когда параллельные запросы завершатся
RETRY_DELAYS = (5, 15, 30)

PROMPT = """You inspect frames from short vertical videos (Instagram Reels) for a Skycoach sponsorship banner.
Skycoach is a gaming services marketplace (boosting, coaching). Its banner is usually a rectangular plate
with the SKYCOACH wordmark (the O may look like a ring/gear), a slogan and often a promo code
("CODE: ...", "COUPON: ..."). It can be large or very small. Ignore other brands, game UI and stream overlays.

For EACH frame return one object, in the same order as the frames:
{"frame": <frame number>,
 "banner": true|false,          // Skycoach banner or wordmark visible in this frame
 "confidence": 0..1,
 "bbox": [x0, y0, x1, y1] | null, // integers 0..1000 relative to frame width/height; the WHOLE banner incl. its background
 "clipped": true|false,         // part of the banner or wordmark is cut off by the frame edge
 "text": "",                    // text readable on the banner, as written
 "promo_code": null,            // promo code from the banner, if any
 "other_text": ""}              // any other on-screen text mentioning Skycoach or a call to action, else ""
Answer with a JSON object {"frames": [...]} only."""


@dataclass
class FrameDetection:
    t: float
    banner: bool
    confidence: float
    bbox: tuple[float, float, float, float] | None  # доли кадра 0..1
    clipped: bool
    text: str
    promo_code: str | None
    other_text: str


def detect(frames: list[tuple[float, Path]]) -> list[FrameDetection]:
    settings = get_settings()
    size = settings.vision_batch_size
    batches = [frames[i:i + size] for i in range(0, len(frames), size)]
    with ThreadPoolExecutor(max_workers=settings.vision_concurrency) as pool:
        results = list(pool.map(_detect_batch, batches))
    return [d for batch in results for d in batch]


def _detect_batch(batch: list[tuple[float, Path]]) -> list[FrameDetection]:
    content: list[dict] = [{"type": "text", "text": PROMPT}]
    for i, (t, path) in enumerate(batch, start=1):
        b64 = base64.b64encode(path.read_bytes()).decode()
        content.append({"type": "text", "text": f"frame {i} (t={t:.0f}s)"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    last_error: Exception | None = None
    for _ in range(2):  # один повтор, если модель вернула невалидный JSON
        response = _create_with_retry(
            model=get_settings().ai_vision_model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            # Явный лимит обязателен: без него NeuroAPI резервирует максимум модели на каждый
            # запрос и параллельные запросы упираются в квоту ключа (403 pre_consume_token_quota_failed).
            # Запас — на «размышления» Gemini, которые тоже считаются выходными токенами.
            max_tokens=VISION_MAX_TOKENS_BASE + VISION_MAX_TOKENS_PER_FRAME * len(batch),
        )
        try:
            items = _parse(response.choices[0].message.content or "")
            break
        except ValueError as exc:
            last_error = exc
            logger.warning("Vision: невалидный ответ, повторяю: %s", exc)
    else:
        raise RuntimeError(f"Vision-модель вернула невалидный ответ: {last_error}")

    by_frame = {int(it.get("frame", 0)): it for it in items if isinstance(it, dict)}
    detections = []
    for i, (t, _path) in enumerate(batch, start=1):
        # Если модель пропустила кадр — считаем, что баннера на нём нет
        it = by_frame.get(i) or (items[i - 1] if len(items) == len(batch) else {})
        detections.append(_to_detection(t, it))
    return detections


def _create_with_retry(**kwargs):
    for attempt, delay in enumerate((*RETRY_DELAYS, None), start=1):
        try:
            return get_ai_client().chat.completions.create(**kwargs)
        except (openai.APIStatusError, openai.APIConnectionError) as exc:
            status = getattr(exc, "status_code", None)
            quota = status == 403 and "quota" in str(exc).lower()
            retryable = status is None or status == 429 or status >= 500 or quota
            if not retryable or delay is None:
                raise
            logger.warning("Vision: попытка %d не удалась (%s), повтор через %d с", attempt, status, delay)
            time.sleep(delay)


def _parse(text: str) -> list:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
    if start < 0:
        raise ValueError("в ответе нет JSON")
    # strict=False: модель иногда переносит строку прямо внутри текста с баннера
    data, _ = json.JSONDecoder(strict=False).raw_decode(text[start:])
    if isinstance(data, dict):
        data = data.get("frames", [])
    if not isinstance(data, list):
        raise ValueError("ожидался список кадров")
    return data


def _to_detection(t: float, it: dict) -> FrameDetection:
    bbox = None
    raw = it.get("bbox")
    if isinstance(raw, list) and len(raw) == 4:
        try:
            x0, y0, x1, y1 = (min(max(float(v), 0.0), 1000.0) / 1000 for v in raw)
            if x1 > x0 and y1 > y0:
                bbox = (x0, y0, x1, y1)
        except (TypeError, ValueError):
            pass
    try:
        confidence = float(it.get("confidence") or 0)
    except (TypeError, ValueError):
        confidence = 0.0
    code = it.get("promo_code")
    return FrameDetection(
        t=t,
        banner=bool(it.get("banner")) and bbox is not None,
        confidence=confidence,
        bbox=bbox,
        clipped=bool(it.get("clipped")),
        text=str(it.get("text") or "").strip(),
        promo_code=str(code).strip() if code else None,
        other_text=str(it.get("other_text") or "").strip(),
    )
