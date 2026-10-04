"""Оркестратор этапов 4–6 для одного ролика: медиа → (транскрипт ‖ кадры) → класс → оценка → обоснование.

Транскрипция у шлюза медленная (до минуты на ролик), поэтому идёт параллельно с анализом кадров.
Все временные файлы живут в TemporaryDirectory и удаляются после анализа.
"""

import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.errors import NonRetryableError
from app.pipeline import logo, media
from app.pipeline.classify import classify
from app.pipeline.report import build_justification
from app.pipeline.scoring import (
    aggregate, aggregate_hybrid, deduction, placement_issues, review_reasons, visibility_score,
)
from app.pipeline.transcribe import Transcript, transcribe
from app.pipeline.vision import FrameDetection, detect
from app.scraper.apify_reels import download_video

logger = logging.getLogger(__name__)

SILENCE_DB = -50.0  # средняя громкость ниже — считаем, что звука нет


@dataclass
class AnalysisResult:
    duration_sec: float | None
    has_audio: bool | None  # None — звук получить не удалось
    transcript: str | None
    integration_class: int
    visibility_score: int | None
    justification: str
    analysis: dict[str, Any]


def analyze_reel(*, shortcode: str, video_url: str | None, audio_url: str | None,
                 caption: str | None, duration_hint: float | None) -> AnalysisResult:
    if not video_url:
        raise NonRetryableError("Apify не вернул ссылку на видео — анализ невозможен")
    settings = get_settings()

    with tempfile.TemporaryDirectory(prefix=f"reel_{shortcode}_") as tmp:
        tmp_dir = Path(tmp)
        video = download_video(video_url, tmp_dir / "video.mp4", max_bytes=settings.max_video_mb * 1024 * 1024)
        info = media.probe(video)
        duration = info.duration or duration_hint
        if not duration:
            raise NonRetryableError("Не удалось определить длительность видео")

        sampled = duration > settings.analysis_max_frames
        with ThreadPoolExecutor(max_workers=2) as pool:
            audio_future = pool.submit(_process_audio, video, info.has_audio_stream, audio_url, tmp_dir)
            facts, vision_failed = _analyze_frames(video, tmp_dir, duration)
            has_audio, transcript, audio_note, audio_meta = audio_future.result()

    cls = classify(caption, transcript, facts)
    review = review_reasons(facts, vision_failed=vision_failed) if cls.integration_class else []

    if cls.integration_class == 0:
        score, score_reasons, issues = None, [], []
    else:
        score, score_reasons = visibility_score(facts, cls.voice_cta, cls.caption_cta)
        issues = placement_issues(facts)
    pct, verdict = deduction(issues) if cls.integration_class else (0, "не применимо")

    justification = build_justification(
        cls=cls, facts=facts, score=score, score_reasons=score_reasons, issues=issues,
        transcript=transcript, audio_note=audio_note, sampled=sampled, review=review,
    )
    analysis = {
        "banner": facts.to_dict(),
        "placement": {
            "issues": [i.__dict__ for i in issues],
            "deduction_pct": pct,
            "verdict": verdict,
        },
        "classification": {
            "voice_mentions": cls.voice_mentions,
            "voice_cta": cls.voice_cta,
            "caption_mention": cls.caption_mention,
            "caption_cta": cls.caption_cta,
            "promo_codes": cls.promo_codes,
            "reasoning": cls.reasoning,
            "used_llm": cls.used_llm,
        },
        "score_reasons": score_reasons,
        "review": {"needed": bool(review), "reasons": review},
        "audio": audio_meta,
        "sampled": sampled,
        "models": {
            "vision": settings.ai_vision_model,
            "text": settings.ai_text_model if cls.used_llm else None,
            "transcribe": settings.ai_transcribe_model if transcript else None,
        },
    }
    return AnalysisResult(
        duration_sec=round(duration, 2),
        has_audio=has_audio,
        transcript=transcript.with_timestamps() if transcript and transcript.segments else None,
        integration_class=cls.integration_class,
        visibility_score=score,
        justification=justification,
        analysis=analysis,
    )


def _analyze_frames(video: Path, tmp_dir: Path, duration: float):
    """Логотип по эталону на 4 кадрах/с; vision-модели — только характерные кадры.

    Логотип не найден → полный проход vision-модели (новый дизайн баннера, сильная обрезка).
    Возвращает (факты, vision_failed).
    """
    settings = get_settings()
    vision_frames = media.extract_frames(video, tmp_dir / "frames", duration, settings.analysis_max_frames)
    logo_frames = media.extract_frames(video, tmp_dir / "logo", duration, settings.logo_max_frames,
                                       fps=settings.logo_fps, long_side=1280)
    hits = logo.detect(logo_frames)
    logo_step = duration / len(logo_frames) if duration * settings.logo_fps > settings.logo_max_frames \
        else 1 / settings.logo_fps

    if not any(h.found for h in hits):
        step = duration / len(vision_frames) if duration > settings.analysis_max_frames else 1.0
        return aggregate(detect(vision_frames), duration, step), False

    plate: list[FrameDetection] = []
    vision_failed = False
    try:
        plate = detect(_key_frames(vision_frames, hits, settings.vision_key_frames))
    except Exception as exc:
        # Без модели всё равно есть время и размер логотипа — результат полезен, но помечается к проверке
        if isinstance(exc, NonRetryableError):
            raise
        logger.warning("Vision-модель недоступна, анализ только по логотипу: %s", exc)
        vision_failed = True
    return aggregate_hybrid(hits, plate, duration, logo_step), vision_failed


def _key_frames(frames: list[tuple[float, Path]], hits: list[logo.LogoHit], k: int) -> list[tuple[float, Path]]:
    """Кадры 1/с, ближайшие к самым уверенным находкам логотипа, разнесённые по времени."""
    found = sorted((h for h in hits if h.found), key=lambda h: -h.score)
    chosen: list[tuple[float, Path]] = []
    for h in found:
        frame = min(frames, key=lambda f: abs(f[0] - h.t))
        if frame not in chosen and all(abs(frame[0] - c[0]) >= 1.5 for c in chosen):
            chosen.append(frame)
        if len(chosen) == k:
            break
    return sorted(chosen) or [min(frames, key=lambda f: abs(f[0] - found[0].t))]


def _process_audio(video: Path, has_stream: bool, audio_url: str | None, tmp_dir: Path):
    """Возвращает (has_audio, transcript, примечание для обоснования, метаданные).

    Любой сбой со звуком не валит ролик: кадры уже проанализированы, а речь — лишь часть картины.
    """
    meta: dict[str, Any] = {"source": None, "mean_volume_db": None, "transcribe_error": None, "audio_error": None}
    try:
        return _process_audio_inner(video, has_stream, audio_url, tmp_dir, meta)
    except Exception as exc:
        logger.warning("Звук не обработан: %s", exc)
        meta["audio_error"] = str(exc)[:300]
        return None, None, "Звук получить не удалось — анализ только по изображению и подписи.", meta


def _process_audio_inner(video: Path, has_stream: bool, audio_url: str | None, tmp_dir: Path, meta: dict):
    settings = get_settings()

    # Instagram часто отдаёт видео и звук отдельными потоками: тогда звук берём по audioUrl
    if has_stream:
        source, meta["source"] = video, "video"
    elif audio_url:
        source = download_video(audio_url, tmp_dir / "audio_src.mp4",
                                max_bytes=settings.max_video_mb * 1024 * 1024)
        meta["source"] = "separate"
    else:
        meta["source"] = "none"
        return False, None, "Звука нет — анализ только по изображению и подписи.", meta

    audio = media.extract_audio(source, tmp_dir / "audio.mp3", settings.max_transcribe_seconds)
    volume = media.mean_volume_db(audio)
    meta["mean_volume_db"] = None if volume is None or volume == float("-inf") else round(volume, 1)
    if volume is not None and volume < SILENCE_DB:
        return False, None, "Звука нет (дорожка беззвучная) — анализ только по изображению и подписи.", meta

    try:
        transcript = transcribe(audio)
    except Exception as exc:
        # Без транскрипта анализ кадров всё ещё полезен — не валим весь ролик
        logger.warning("Транскрипция не удалась: %s", exc)
        meta["transcribe_error"] = str(exc)[:300]
        return True, None, "Голос: транскрипция не удалась, речь не учтена.", meta
    return True, transcript, None, meta
