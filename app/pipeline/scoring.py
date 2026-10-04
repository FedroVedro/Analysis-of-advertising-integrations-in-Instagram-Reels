"""Факты о баннере, замечания по размещению и балл заметности — детерминированно, из детекций.

Пороги откалиброваны на примерах Skycoach (banner_review_examples):
- хороший баннер valorant_funzone: ширина ≈68 % кадра, площадь ≈7.4 %;
- «Banner too small» (−30 %): площадь ≈2.9 %;
- «Banner small and partly covered» (−20 %): площадь ≈5.7 %.
Правила удержаний — из того же документа: обрезан 20 %, маленький 30 % (чуть маленький 20 %),
перекрыт интерфейсом / слишком низко 20 %, не виден — исключение. При нескольких замечаниях
берём самое большое удержание (так в примере «small AND partly covered» → 20 %, а не сумма).
"""

from dataclasses import asdict, dataclass, field
from statistics import median

from app.pipeline.vision import FrameDetection

MIN_CONFIDENCE = 0.6
TOO_SMALL_AREA_PCT = 4.0
SLIGHTLY_SMALL_AREA_PCT = 6.0
EDGE = 0.005  # рамка ближе к краю кадра, чем 0.5 % — касается края
ISSUE_SHARE = 0.3  # замечание засчитывается, если проблема есть на ≥30 % кадров с баннером
UI_OVERLAP = 0.25  # кадр считается перекрытым, если интерфейс закрывает ≥25 % площади баннера

# Зоны, которые в Instagram Reels закрывает интерфейс (доли экрана 9:16):
# шапка с «Reels» и камерой, подпись/автор/музыка снизу, колонка кнопок справа.
# Нижняя граница откалибрована на принятых Skycoach роликах: баннеры на 74–85 %, 78–88 % и даже 82–93 %
# высоты (DcUEhNgx8Il, DbRI307xMEy, Dblv_p8RP4T) прошли без замечаний — перекрытием считаем только
# самую нижнюю полосу со строкой музыки и подписью
IG_UI_ZONES = {
    "шапка Reels": (0.0, 0.0, 1.0, 0.07),
    "подпись и музыка": (0.0, 0.93, 1.0, 1.0),
    "кнопки справа": (0.85, 0.50, 1.0, 0.80),
}


@dataclass
class BannerFacts:
    frames_total: int
    step_sec: float
    duration: float
    banner_frames: int = 0
    seconds: float = 0.0
    share: float = 0.0
    intervals: list[tuple[float, float]] = field(default_factory=list)
    width_pct: float | None = None
    height_pct: float | None = None
    area_pct: float | None = None
    max_area_pct: float | None = None
    position: str | None = None
    top_pct: float | None = None  # верхняя граница баннера, % высоты кадра (медиана)
    bottom_pct: float | None = None
    clipped_share: float = 0.0
    ui_overlap_share: float = 0.0
    ui_zones: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    promo_codes: list[str] = field(default_factory=list)
    other_texts: list[str] = field(default_factory=list)

    @property
    def visible(self) -> bool:
        return self.banner_frames > 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Issue:
    code: str
    title: str
    detail: str
    deduction_pct: int | None  # None — ролик исключается из выплаты


def aggregate(detections: list[FrameDetection], duration: float, step: float) -> BannerFacts:
    facts = BannerFacts(frames_total=len(detections), step_sec=round(step, 2), duration=round(duration, 2))
    hits = [d for d in detections if d.banner and d.bbox and d.confidence >= MIN_CONFIDENCE]
    # Тексты вне баннера (призывы на экране) собираем со всех кадров
    facts.other_texts = _unique(d.other_text for d in detections if d.other_text)
    if not hits:
        return facts

    facts.banner_frames = len(hits)
    facts.seconds = round(min(len(hits) * step, duration), 1)
    facts.share = round(facts.seconds / duration, 3) if duration else 0.0
    facts.intervals = _intervals([d.t for d in hits], step, duration)

    widths = [(d.bbox[2] - d.bbox[0]) * 100 for d in hits]
    heights = [(d.bbox[3] - d.bbox[1]) * 100 for d in hits]
    areas = [w * h / 100 for w, h in zip(widths, heights)]
    facts.width_pct = round(median(widths), 1)
    facts.height_pct = round(median(heights), 1)
    facts.area_pct = round(median(areas), 1)
    facts.max_area_pct = round(max(areas), 1)
    facts.top_pct = round(median(d.bbox[1] for d in hits) * 100, 1)
    facts.bottom_pct = round(median(d.bbox[3] for d in hits) * 100, 1)
    facts.position = _position(median((d.bbox[1] + d.bbox[3]) / 2 for d in hits),
                               median((d.bbox[0] + d.bbox[2]) / 2 for d in hits))

    facts.clipped_share = round(sum(_is_clipped(d) for d in hits) / len(hits), 2)
    overlaps = [_ui_overlap(d.bbox) for d in hits]
    facts.ui_overlap_share = round(sum(o >= UI_OVERLAP for o, _ in overlaps) / len(hits), 2)
    facts.ui_zones = _unique(z for o, zones in overlaps if o >= UI_OVERLAP for z in zones)

    facts.texts = _unique(d.text for d in hits if d.text)
    facts.promo_codes = _unique(d.promo_code.upper() for d in hits if d.promo_code)
    return facts


def placement_issues(facts: BannerFacts) -> list[Issue]:
    if not facts.visible:
        return [Issue("not_visible", "Баннер не виден",
                      "баннер Skycoach не найден ни на одном кадре", None)]
    issues = []
    if facts.area_pct is not None and facts.area_pct < TOO_SMALL_AREA_PCT:
        issues.append(Issue("too_small", "Баннер слишком маленький",
                            f"площадь ≈{facts.area_pct:g} % кадра (норма от {SLIGHTLY_SMALL_AREA_PCT:g} %)", 30))
    elif facts.area_pct is not None and facts.area_pct < SLIGHTLY_SMALL_AREA_PCT:
        issues.append(Issue("slightly_small", "Баннер немного мелкий",
                            f"площадь ≈{facts.area_pct:g} % кадра (норма от {SLIGHTLY_SMALL_AREA_PCT:g} %)", 20))
    if facts.clipped_share >= ISSUE_SHARE:
        issues.append(Issue("clipped", "Баннер обрезан краем кадра",
                            f"обрезан на {facts.clipped_share:.0%} кадров с баннером", 20))
    if facts.ui_overlap_share >= ISSUE_SHARE:
        zones = ", ".join(facts.ui_zones) or "интерфейс"
        issues.append(Issue("covered", "Баннер перекрыт интерфейсом Instagram",
                            f"заходит в зону «{zones}» на {facts.ui_overlap_share:.0%} кадров с баннером", 20))
    return issues


def deduction(issues: list[Issue]) -> tuple[int | None, str]:
    """Итог по правилам Skycoach: (удержание %, вердикт). None — ролик исключается."""
    if any(i.deduction_pct is None for i in issues):
        return None, "исключить из выплаты"
    pct = max((i.deduction_pct for i in issues), default=0)
    return pct, ("без замечаний" if pct == 0 else f"удержание {pct} %")


def visibility_score(facts: BannerFacts, voice_cta: bool, caption_cta: bool) -> tuple[int, list[str]]:
    """Балл 1–5: основа — размер баннера, поправки за время в кадре и за призыв вне баннера.

    Размер, % площади кадра (пороги согласованы с удержаниями Skycoach):
    <4 → 1, <6 → 2, <9 → 3, <13 → 4, иначе 5.
    Время: <10 % ролика → −1, ≥50 % → +1.
    Голосовой призыв или призыв/промокод в подписи → +1. Промокод на самом баннере бонуса
    не даёт: он есть почти на всех баннерах и уже учтён в размере.
    Без баннера балл определяется только призывом: 2 при призыве, иначе 1.
    """
    cta = voice_cta or caption_cta
    reasons = []
    if not facts.visible:
        reasons.append("баннера в кадре нет")
        if cta:
            reasons.append("+1 за призыв голосом или в подписи")
        return (2 if cta else 1), reasons

    area = facts.area_pct or 0
    score = 1 if area < 4 else 2 if area < 6 else 3 if area < 9 else 4 if area < 13 else 5
    reasons.append(f"размер ≈{area:g} % кадра → {score}")
    if facts.share < 0.10:
        score -= 1
        reasons.append(f"−1: в кадре лишь {facts.share:.0%} ролика")
    elif facts.share >= 0.50:
        score += 1
        reasons.append(f"+1: в кадре {facts.share:.0%} ролика")
    if voice_cta:
        score += 1
        reasons.append("+1: голосовой призыв")
    elif caption_cta:
        score += 1
        reasons.append("+1: призыв в подписи")
    return max(1, min(5, score)), reasons


def _is_clipped(d: FrameDetection) -> bool:
    if d.clipped:
        return True
    x0, y0, x1, y1 = d.bbox
    # Баннер во всю ширину касается обоих краёв — это дизайн, а не обрезка; обрезка — касание одного края
    left, right = x0 <= EDGE, x1 >= 1 - EDGE
    return left != right or y0 <= EDGE or y1 >= 1 - EDGE


def _ui_overlap(bbox: tuple[float, float, float, float]) -> tuple[float, list[str]]:
    x0, y0, x1, y1 = bbox
    area = (x1 - x0) * (y1 - y0)
    total, zones = 0.0, []
    for name, (zx0, zy0, zx1, zy1) in IG_UI_ZONES.items():
        w = max(0.0, min(x1, zx1) - max(x0, zx0))
        h = max(0.0, min(y1, zy1) - max(y0, zy0))
        if w * h > 0:
            total += w * h
            zones.append(name)
    return (total / area if area else 0.0), zones


def _intervals(times: list[float], step: float, duration: float) -> list[tuple[float, float]]:
    times = sorted(times)
    result: list[list[float]] = []
    for t in times:
        start, end = max(0.0, t - step / 2), min(duration, t + step / 2)
        if result and start - result[-1][1] <= step * 0.6:
            result[-1][1] = end
        else:
            result.append([start, end])
    return [(round(a, 1), round(b, 1)) for a, b in result]


def _position(cy: float, cx: float) -> str:
    vertical = "вверху" if cy < 0.33 else "внизу" if cy > 0.66 else "посередине"
    horizontal = "слева" if cx < 0.35 else "справа" if cx > 0.65 else "по центру"
    return f"{vertical} {horizontal}"


def _unique(values) -> list[str]:
    seen, out = set(), []
    for v in values:
        key = v.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(v.strip())
    return out
