"""Поиск логотипа SKYCOACH на кадрах по эталону (OpenCV, multi-scale template matching).

Логотип одинаков на всех баннерах Skycoach (меняются слоган и промокод), поэтому его можно
искать классическим компьютерным зрением: бесплатно, детерминированно, с точностью до пикселя.
На примерах Skycoach: кадры с баннером дают совпадение 0.87–0.97, без баннера — не выше 0.73.

Эталоны — PNG в assets/ (`skycoach_wordmark*.png`); новый вариант логотипа = ещё один файл.
Обрезанный краем кадра логотип целиком не совпадёт ни с чем, поэтому дополнительно ищем его левую
и правую части: если часть найдена так, что остальное логотипа выходит за край кадра, — логотип
обрезан. Требование «вылета за край» отсекает случайный похожий текст посреди кадра.
Если логотип не найден и так (новый дизайн), анализ переходит на vision-модель.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ASSETS = Path(__file__).parent / "assets"
MATCH_THRESHOLD = 0.80

COARSE_WIDTH = 480  # грубый поиск по всем масштабам на уменьшенном кадре
FINE_WIDTH = 720  # уточнение масштаба и положения рядом с найденным местом
# Ширина логотипа от 7 % до 45 % ширины кадра — с запасом за пределы встреченного (10.8–24.9 %).
# Мельче 7 % шаблон на уменьшенном кадре — пара десятков пикселей и совпадает с любым шумом.
MIN_WIDTH_PCT, MAX_WIDTH_PCT = 0.07, 0.45
COARSE_STEPS = 24
FINE_STEPS = 9
FINE_SPAN = 0.12  # ± к найденному масштабу при уточнении
PART = 0.6  # части логотипа для поиска обрезанного: левые и правые 60 % ширины
EDGE_PX = 2  # насколько полный логотип должен выходить за край, чтобы считаться обрезанным


@dataclass
class LogoHit:
    t: float
    score: float
    bbox: tuple[float, float, float, float] | None  # доли кадра 0..1, только если score ≥ порога
    width_pct: float | None  # ширина логотипа, % ширины кадра (у обрезанного — оценка полной ширины)
    clipped: bool = False  # логотип выходит за край кадра

    @property
    def found(self) -> bool:
        return self.bbox is not None


@lru_cache
def _templates() -> tuple[np.ndarray, ...]:
    paths = sorted(ASSETS.glob("skycoach_wordmark*.png"))
    if not paths:
        raise FileNotFoundError(f"Нет эталонов логотипа в {ASSETS}")
    return tuple(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths)


def detect(frames: list[tuple[float, Path]], workers: int = 4) -> list[LogoHit]:
    # OpenCV отпускает GIL — потоки дают реальное ускорение
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda f: detect_frame(*f), frames))


def detect_frame(t: float, path: Path) -> LogoHit:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return LogoHit(t, 0.0, None, None)
    H, W = gray.shape
    best_score = 0.0
    for tmpl in _templates():
        score, x, y, w, h = _match(gray, tmpl)
        best_score = max(best_score, score)
        if score >= MATCH_THRESHOLD:
            return LogoHit(t, round(score, 3), _norm(x, y, w, h, W, H), round(w / W * 100, 1))

    # Целиком не найден — ищем обрезанный: левую часть у правого края и правую часть у левого
    for tmpl in _templates():
        tw = tmpl.shape[1]
        cut = round(tw * PART)
        for part, offset in ((tmpl[:, :cut], 0), (tmpl[:, tw - cut:], tw - cut)):
            score, x, y, w, h = _match(gray, part)
            if score < MATCH_THRESHOLD:
                continue
            full_w = w * tw / cut
            full_x0 = x - offset * full_w / tw
            if full_x0 < -EDGE_PX or full_x0 + full_w > W + EDGE_PX:
                x0, x1 = max(0.0, full_x0), min(float(W), full_x0 + full_w)
                return LogoHit(t, round(score, 3), _norm(x0, y, x1 - x0, h, W, H),
                               round(full_w / W * 100, 1), clipped=True)
    return LogoHit(t, round(best_score, 3), None, None)


def _norm(x, y, w, h, W, H) -> tuple[float, float, float, float]:
    return (x / W, y / H, (x + w) / W, (y + h) / H)


def _match(gray: np.ndarray, tmpl: np.ndarray) -> tuple[float, float, float, float, float]:
    """Возвращает (score, x, y, w, h) лучшего совпадения в пикселях исходного кадра."""
    H, W = gray.shape
    th, tw = tmpl.shape

    # 1. Грубо: все масштабы на кадре шириной COARSE_WIDTH
    k = COARSE_WIDTH / W
    small = cv2.resize(gray, (COARSE_WIDTH, round(H * k)), interpolation=cv2.INTER_AREA)
    widths = np.geomspace(MIN_WIDTH_PCT, MAX_WIDTH_PCT, COARSE_STEPS) * COARSE_WIDTH
    score, loc, width = _best_over_widths(small, tmpl, widths)
    if score < MATCH_THRESHOLD - 0.15:
        # Явно нет логотипа — уточнять нечего
        return score, 0, 0, 0, 0

    # 2. Точно: кадр шириной FINE_WIDTH, только окрестность найденного места, масштабы ±FINE_SPAN
    k2 = min(1.0, FINE_WIDTH / W)
    fine = cv2.resize(gray, (round(W * k2), round(H * k2)), interpolation=cv2.INTER_AREA) if k2 < 1 else gray
    ratio = fine.shape[1] / COARSE_WIDTH
    cw, ch = width * ratio, width * th / tw * ratio
    cx, cy = loc[0] * ratio, loc[1] * ratio
    pad_x, pad_y = cw * 0.3, ch * 0.6
    x0, y0 = max(0, int(cx - pad_x)), max(0, int(cy - pad_y))
    x1 = min(fine.shape[1], int(cx + cw * (1 + FINE_SPAN) + pad_x))
    y1 = min(fine.shape[0], int(cy + ch * (1 + FINE_SPAN) + pad_y))
    roi = fine[y0:y1, x0:x1]
    widths = np.linspace(cw * (1 - FINE_SPAN), cw * (1 + FINE_SPAN), FINE_STEPS)
    score2, loc2, width2 = _best_over_widths(roi, tmpl, widths)
    # Решение принимаем только по точному проходу: на грубом кадре мелкий шаблон даёт ложные совпадения
    scale = 1 / k2
    return score2, (x0 + loc2[0]) * scale, (y0 + loc2[1]) * scale, width2 * scale, width2 * th / tw * scale


def _best_over_widths(img: np.ndarray, tmpl: np.ndarray, widths) -> tuple[float, tuple[int, int], float]:
    th, tw = tmpl.shape
    best = (-1.0, (0, 0), 0.0)
    for w in widths:
        s = w / tw
        tw2, th2 = max(1, round(tw * s)), max(1, round(th * s))
        if tw2 < 12 or th2 < 4 or tw2 >= img.shape[1] or th2 >= img.shape[0]:
            continue
        t = cv2.resize(tmpl, (tw2, th2), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        res = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        if mx > best[0]:
            best = (float(mx), loc, float(tw2))
    return best
