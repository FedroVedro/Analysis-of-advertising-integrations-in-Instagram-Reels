"""Фоновый воркер: забирает ролики из очереди в SQLite и проводит их по пайплайну.

Запуск: python -m app.worker

Воркер рассчитан на один экземпляр (ограничение SQLite). Долгие внешние вызовы (Apify, AI)
выполняются вне транзакций, чтобы не блокировать запись для web.
"""

import logging
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update

from app.config import get_settings
from app.db import SessionLocal, init_db
from app.models import Reel, ReelStatus
from app.pipeline.analyze import analyze_reel
from app.scraper.apify_reels import (
    MAX_BATCH_SIZE,
    ApifyReelsScraper,
    ApifyScraperError,
    ReelResult,
)
from app.scraper.apify_reels import ReelStatus as ScrapeStatus

logger = logging.getLogger("worker")

POLL_INTERVAL_SECS = 2
MAX_ATTEMPTS = 3
STALE_AFTER = timedelta(minutes=15)
IN_PROGRESS = (ReelStatus.FETCHING, ReelStatus.ANALYZING)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def claim_batch(limit: int = MAX_BATCH_SIZE) -> list[Reel]:
    """Атомарно переводит до `limit` роликов из queued в fetching и возвращает их.

    В SQLite одна пишущая транзакция за раз, поэтому UPDATE ... RETURNING не отдаст
    один ролик дважды. Для нескольких воркеров нужен PostgreSQL + FOR UPDATE SKIP LOCKED.
    """
    queued = (
        select(Reel.id)
        .where(Reel.status == ReelStatus.QUEUED)
        .order_by(Reel.created_at)
        .limit(limit)
    )
    with SessionLocal() as session:
        ids = session.scalars(
            update(Reel)
            .where(Reel.id.in_(queued))
            .values(status=ReelStatus.FETCHING, locked_at=utcnow(), updated_at=utcnow())
            .returning(Reel.id)
        ).all()
        session.commit()
        if not ids:
            return []
        return list(session.scalars(select(Reel).where(Reel.id.in_(ids)).order_by(Reel.created_at)))


def requeue_stale(older_than: timedelta | None = STALE_AFTER) -> int:
    """Возвращает в очередь ролики, зависшие в обработке (воркер упал или был перезапущен).

    older_than=None — вернуть все незавершённые: так делаем при старте единственного воркера.
    """
    stmt = update(Reel).where(Reel.status.in_(IN_PROGRESS))
    if older_than is not None:
        stmt = stmt.where(Reel.locked_at < utcnow() - older_than)
    with SessionLocal() as session:
        count = session.execute(
            stmt.values(status=ReelStatus.QUEUED, locked_at=None, updated_at=utcnow())
        ).rowcount
        session.commit()
    if count:
        logger.warning("Возвращено в очередь зависших роликов: %d", count)
    return count


def process_batch(scraper: ApifyReelsScraper) -> int:
    """Обрабатывает одну пачку. Возвращает число взятых роликов (0 — очередь пуста)."""
    reels = claim_batch()
    if not reels:
        return 0
    logger.info("Взято в работу: %s", ", ".join(r.shortcode for r in reels))

    try:
        results = scraper.fetch([r.canonical_url for r in reels])
    except ApifyScraperError as exc:
        logger.error("Apify недоступен: %s", exc)
        for reel in reels:
            _fail_or_retry(reel.id, f"Сервис получения данных недоступен: {exc}")
        return len(reels)

    by_shortcode = {r.shortcode: r for r in results}
    to_analyze = []
    for reel in reels:
        result = by_shortcode.get(reel.shortcode)
        try:
            if result is None:
                raise RuntimeError("Apify не вернул результат для ролика")
            _save_fetch_result(reel.id, result)
            if result.status == ScrapeStatus.OK and not result.video_url:
                # Ссылка /p/ на фото или карусель: метрики есть, анализировать нечего — повтор не поможет
                _mark_unavailable(reel.id, "Публикация не содержит видео (фото или карусель)")
            elif result.status == ScrapeStatus.OK:
                to_analyze.append(reel)
        except Exception as exc:
            logger.exception("Ошибка сохранения %s", reel.shortcode)
            _fail_or_retry(reel.id, f"Внутренняя ошибка: {exc}")

    # Анализ долгий (скачивание, транскрипция, vision) — несколько роликов параллельно
    with ThreadPoolExecutor(max_workers=get_settings().analysis_concurrency) as pool:
        list(pool.map(_analyze_safely, to_analyze))
    return len(reels)


def _analyze_safely(reel: Reel) -> None:
    try:
        _analyze(reel.id)
    except Exception as exc:
        # Ошибка одного ролика не должна валить воркер и остальные ролики
        logger.exception("Ошибка анализа %s", reel.shortcode)
        _fail_or_retry(reel.id, f"Анализ не выполнен: {exc}")


def _save_fetch_result(reel_id: str, result: ReelResult) -> None:
    with SessionLocal() as session:
        reel = session.get(Reel, reel_id)
        if result.status != ScrapeStatus.OK:
            reel.status = ReelStatus.UNAVAILABLE
            reel.error = result.error
            reel.locked_at = None
            session.commit()
            logger.info("%s: недоступен (%s)", reel.shortcode, result.error)
            return

        reel.views = result.views
        reel.views_source = result.views_source
        reel.likes = result.likes
        reel.comments = result.comments
        reel.published_at = (
            result.published_at.astimezone(timezone.utc) if result.published_at else None
        )
        reel.author = result.author
        reel.duration_sec = result.duration_sec
        reel.caption = result.caption
        reel.video_url = result.video_url
        reel.audio_url = result.audio_url
        reel.error = None
        reel.status = ReelStatus.ANALYZING
        reel.locked_at = utcnow()
        session.commit()
        logger.info("%s: метрики получены (views=%s)", reel.shortcode, reel.views)


def _mark_unavailable(reel_id: str, error: str) -> None:
    with SessionLocal() as session:
        reel = session.get(Reel, reel_id)
        reel.status, reel.error, reel.locked_at = ReelStatus.UNAVAILABLE, error, None
        session.commit()
        logger.info("%s: недоступен (%s)", reel.shortcode, error)


def _analyze(reel_id: str) -> None:
    """Этапы 4–6: транскрипция, анализ кадров, класс, заметность, обоснование."""
    with SessionLocal() as session:
        reel = session.get(Reel, reel_id)
        params = dict(
            shortcode=reel.shortcode, video_url=reel.video_url, audio_url=reel.audio_url,
            caption=reel.caption, duration_hint=reel.duration_sec,
        )
    started = time.monotonic()
    result = analyze_reel(**params)  # долгие внешние вызовы — вне сессии БД

    with SessionLocal() as session:
        reel = session.get(Reel, reel_id)
        reel.duration_sec = result.duration_sec or reel.duration_sec
        reel.has_audio = result.has_audio
        reel.transcript = result.transcript
        reel.integration_class = result.integration_class
        reel.visibility_score = result.visibility_score
        reel.justification = result.justification
        reel.analysis = result.analysis
        reel.status = ReelStatus.DONE
        reel.error = None
        reel.locked_at = None
        session.commit()
    logger.info(
        "%s: анализ готов за %.0f с — класс %s, заметность %s",
        params["shortcode"], time.monotonic() - started, result.integration_class, result.visibility_score,
    )


def _fail_or_retry(reel_id: str, error: str) -> None:
    with SessionLocal() as session:
        reel = session.get(Reel, reel_id)
        reel.attempts += 1
        reel.locked_at = None
        if reel.attempts >= MAX_ATTEMPTS:
            reel.status = ReelStatus.FAILED
            reel.error = f"{error} (попыток: {reel.attempts})"
            logger.error("%s: failed после %d попыток", reel.shortcode, reel.attempts)
        else:
            reel.status = ReelStatus.QUEUED
            reel.error = f"Повторная попытка {reel.attempts + 1} из {MAX_ATTEMPTS}: {error}"
        session.commit()


class Worker:
    def __init__(self, scraper: ApifyReelsScraper | None = None) -> None:
        self.scraper = scraper or ApifyReelsScraper()
        self._stopping = False

    def stop(self, *_args) -> None:
        logger.info("Получен сигнал остановки, завершаю после текущей пачки")
        self._stopping = True

    def run(self) -> None:
        init_db()
        requeue_stale(older_than=None)
        last_stale_check = time.monotonic()
        logger.info("Воркер запущен")

        while not self._stopping:
            if time.monotonic() - last_stale_check > 60:
                requeue_stale()
                last_stale_check = time.monotonic()
            try:
                taken = process_batch(self.scraper)
            except Exception:
                # Например, БД временно заблокирована: не падаем, пробуем снова
                logger.exception("Сбой цикла воркера")
                taken = 0
            if not taken:
                time.sleep(POLL_INTERVAL_SECS)
        logger.info("Воркер остановлен")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    worker = Worker()
    signal.signal(signal.SIGTERM, worker.stop)
    signal.signal(signal.SIGINT, worker.stop)
    worker.run()


if __name__ == "__main__":
    main()
