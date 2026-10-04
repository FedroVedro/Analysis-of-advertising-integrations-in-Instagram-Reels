"""Приём ссылок: создание задач, дедупликация роликов, сериализация результатов."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.orm import Session, selectinload

from app.models import Job, Reel, ReelStatus, new_id
from app.scraper.apify_reels import canonical_reel_url, extract_shortcode

INVALID_URL = "invalid_url"
INVALID_URL_ERROR = "Это не ссылка на Instagram Reel. Пример: https://www.instagram.com/reel/ABC123/"


def normalize_urls(urls: list[str]) -> list[str]:
    """Убирает пустые строки и точные дубли, сохраняя порядок."""
    seen: set[str] = set()
    result = []
    for url in urls:
        url = url.strip()
        if url and url not in seen:
            seen.add(url)
            result.append(url)
    return result


def count_new_reels(session: Session, urls: list[str]) -> int:
    """Сколько роликов из запроса ещё нет в БД (только они стоят денег)."""
    codes = {c for c in map(extract_shortcode, urls) if c}
    if not codes:
        return 0
    existing = session.scalars(select(Reel.shortcode).where(Reel.shortcode.in_(codes))).all()
    return len(codes - set(existing))


def reels_created_since(session: Session, since: datetime) -> int:
    return session.scalar(select(func.count()).select_from(Reel).where(Reel.created_at >= since))


def create_jobs(session: Session, urls: list[str]) -> list[Job]:
    """Создаёт по задаче на каждую ссылку. Ролик с тем же shortcode повторно не ставится в очередь."""
    jobs = []
    for url in urls:
        shortcode = extract_shortcode(url)
        if shortcode is None:
            job = Job(input_url=url, status=INVALID_URL, error=INVALID_URL_ERROR)
        else:
            reel = _get_or_create_reel(session, shortcode)
            if reel.status == ReelStatus.FAILED:
                # Внутренняя ошибка: при повторной отправке пробуем ещё раз
                reel.status, reel.error, reel.attempts = ReelStatus.QUEUED, None, 0
            cached = reel.status in (ReelStatus.DONE, ReelStatus.UNAVAILABLE)
            job = Job(input_url=url, reel=reel, cached=cached)
        session.add(job)
        jobs.append(job)
    session.commit()
    return jobs


def _get_or_create_reel(session: Session, shortcode: str) -> Reel:
    # INSERT ... ON CONFLICT DO NOTHING: две одновременные отправки одной ссылки
    # не создадут дубль и не упадут на уникальном индексе
    dialect = session.get_bind().dialect.name
    insert = postgresql.insert if dialect == "postgresql" else sqlite.insert
    session.execute(
        insert(Reel)
        .values(
            id=new_id(),
            shortcode=shortcode,
            canonical_url=canonical_reel_url(shortcode),
            status=ReelStatus.QUEUED,
            attempts=0,
            created_at=_utcnow(),
            updated_at=_utcnow(),
        )
        .on_conflict_do_nothing(index_elements=["shortcode"])
    )
    return session.scalars(select(Reel).where(Reel.shortcode == shortcode)).one()


def get_jobs(session: Session, ids: list[str]) -> list[Job]:
    jobs = session.scalars(
        select(Job).where(Job.id.in_(ids)).options(selectinload(Job.reel))
    ).all()
    by_id = {job.id: job for job in jobs}
    return [by_id[i] for i in ids if i in by_id]


def get_recent_reels(session: Session, limit: int = 50) -> list[Reel]:
    return list(session.scalars(select(Reel).order_by(Reel.created_at.desc()).limit(limit)))


def job_to_dict(job: Job) -> dict[str, Any]:
    return {
        "job_id": job.id,
        "input_url": job.input_url,
        "status": job.status or job.reel.status.value,
        "error": job.error or (job.reel.error if job.reel else None),
        "cached": job.cached,
        "created_at": _iso(job.created_at),
        "reel": reel_to_dict(job.reel) if job.reel else None,
    }


def reel_to_dict(reel: Reel) -> dict[str, Any]:
    return {
        "id": reel.id,
        "shortcode": reel.shortcode,
        "url": reel.canonical_url,
        "status": reel.status.value,
        "error": reel.error,
        "author": reel.author,
        "published_at": _iso(reel.published_at),
        "views": reel.views,
        "views_source": reel.views_source,
        "likes": reel.likes,
        "comments": reel.comments,
        "duration_sec": reel.duration_sec,
        "caption": reel.caption,
        "has_audio": reel.has_audio,
        "transcript": reel.transcript,
        "integration_class": reel.integration_class,
        "visibility_score": reel.visibility_score,
        "justification": reel.justification,
        "analysis": reel.analysis,
        "updated_at": _iso(reel.updated_at),
    }


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    """SQLite возвращает naive datetime; всё хранится в UTC, поэтому явно помечаем зону."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()
