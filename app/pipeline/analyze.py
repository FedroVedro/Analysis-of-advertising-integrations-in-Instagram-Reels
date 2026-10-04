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
from app.pipeline import media
from app.pipeline.classify import classify
from app.pipeline.report import build_justification
from app.pipeline.scoring import aggregate, deduction, placement_issues, visibility_score
from app.pipeline.transcribe import Transcript, transcribe
from app.pipeline.vision import detect
from app.scraper.apify_reels import download_video

logger = logging.getLogger(__name__)

SILENCE_DB = -50.0  # средняя громкость ниже — считаем, что звука нет


@dataclass
class AnalysisResult:
    duration_sec: float | None
    has_audio: bool
    transcript: str | None
    integration_class: int
    visibility_score: int | None
    justification: str
    analysis: dict[str, Any]


def analyze_reel(*, shortcode: str, video_url: str | None, audio_url: str | None,
                 caption: str | None, duration_hint: float | None) -> AnalysisResult:
    if not video_url:
        raise RuntimeError("Apify не вернул ссылку на видео — анализ невозможен")
    settings = get_settings()

    with tempfile.TemporaryDirectory(prefix=f"reel_{shortcode}_") as tmp:
        tmp_dir = Path(tmp)
        video = download_video(video_url, tmp_dir / "video.mp4", max_bytes=settings.max_video_mb * 1024 * 1024)
        info = media.probe(video)
        duration = info.duration or duration_hint
        if not duration:
            raise RuntimeError("Не удалось определить длительность видео")

        sampled = duration > settings.analysis_max_frames
        with ThreadPoolExecutor(max_workers=2) as pool:
            audio_future = pool.submit(_process_audio, video, info.has_audio_stream, audio_url, tmp_dir)
            frames = media.extract_frames(video, tmp_dir / "frames", duration, settings.analysis_max_frames)
            detections = detect(frames)
            has_audio, transcript, audio_note, audio_meta = audio_future.result()

    step = duration / len(frames) if sampled else 1.0
    facts = aggregate(detections, duration, step)
    cls = classify(caption, transcript, facts)

    if cls.integration_class == 0:
        score, score_reasons, issues = None, [], []
    else:
        score, score_reasons = visibility_score(facts, cls.voice_cta, cls.caption_cta)
        issues = placement_issues(facts)
    pct, verdict = deduction(issues) if cls.integration_class else (0, "не применимо")

    justification = build_justification(
        cls=cls, facts=facts, score=score, score_reasons=score_reasons, issues=issues,
        transcript=transcript, audio_note=audio_note, sampled=sampled,
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


def _process_audio(video: Path, has_stream: bool, audio_url: str | None, tmp_dir: Path):
    """Возвращает (has_audio, transcript, примечание для обоснования, метаданные)."""
    settings = get_settings()
    meta: dict[str, Any] = {"source": None, "mean_volume_db": None, "transcribe_error": None}

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
