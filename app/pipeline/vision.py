"""Поиск баннера Skycoach на кадрах vision-моделью.

Модель отвечает только на вопросы «что на кадре»: есть ли баннер, где его рамка, что на нём
написано. Все выводы (секунды, размер, обрезка, перекрытие интерфейсом) считает код в scoring.py —
так цифры в обосновании воспроизводимы и не зависят от фантазии модели.
"""

import base64
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from app.ai import chat
from app.config import get_settings

logger = logging.getLogger(__name__)

VISION_MAX_TOKENS_BASE = 2000
VISION_MAX_TOKENS_PER_FRAME = 300

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
        response = chat(
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

    return [_to_detection(t, it) for (t, _path), it in zip(batch, _align(items, len(batch)))]


def _align(items: list, n: int) -> list[dict]:
    """Сопоставляет ответы модели кадрам 1..n.

    Модель может написать номер строкой («frame 3»), пропустить его или посчитать кадры с нуля —
    тогда сдвигаем нумерацию, иначе каждый кадр получил бы ответ соседнего.
    Пропущенный кадр считается кадром без баннера.
    """
    objs = [it for it in items if isinstance(it, dict)]
    numbers = [_frame_number(it.get("frame")) for it in objs]
    known = [x for x in numbers if x is not None]
    if known and min(known) == 0:
        numbers = [x + 1 if x is not None else None for x in numbers]
    by_number = {x: it for x, it in zip(numbers, objs) if x is not None and 1 <= x <= n}
    if len(by_number) < len(objs) and len(objs) == n:
        return objs  # номера не разобрать, но ответов ровно столько, сколько кадров — берём по порядку
    return [by_number.get(i, {}) for i in range(1, n + 1)]


def _frame_number(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and (m := re.search(r"\d+", value)):
        return int(m.group())
    return None


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
