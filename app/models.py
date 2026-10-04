import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, Enum, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


class ReelStatus(StrEnum):
    QUEUED = "queued"
    FETCHING = "fetching"
    ANALYZING = "analyzing"
    DONE = "done"
    UNAVAILABLE = "unavailable"  # приватный, удалён или не найден
    FAILED = "failed"  # внутренняя ошибка после всех ретраев

    @property
    def is_final(self) -> bool:
        return self in (ReelStatus.DONE, ReelStatus.UNAVAILABLE, ReelStatus.FAILED)


# В SQLite enum хранится как VARCHAR; values_callable — чтобы в БД лежало "queued", а не "QUEUED"
_status_type = Enum(
    ReelStatus,
    native_enum=False,
    length=20,
    values_callable=lambda e: [m.value for m in e],
)


class Reel(Base):
    __tablename__ = "reels"
    __table_args__ = (Index("ix_reels_status_created", "status", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    shortcode: Mapped[str] = mapped_column(String(64), unique=True)
    canonical_url: Mapped[str] = mapped_column(Text)

    status: Mapped[ReelStatus] = mapped_column(_status_type, default=ReelStatus.QUEUED)
    error: Mapped[str | None] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(default=0)
    locked_at: Mapped[datetime | None]

    # Метрики: None = недоступно, никогда не подменяем нулём
    views: Mapped[int | None]
    views_source: Mapped[str | None] = mapped_column(String(32))
    likes: Mapped[int | None]
    comments: Mapped[int | None]
    published_at: Mapped[datetime | None]
    author: Mapped[str | None] = mapped_column(String(255))
    duration_sec: Mapped[float | None]
    caption: Mapped[str | None] = mapped_column(Text)
    video_url: Mapped[str | None] = mapped_column(Text)
    audio_url: Mapped[str | None] = mapped_column(Text)

    # Анализ
    has_audio: Mapped[bool | None]
    transcript: Mapped[str | None] = mapped_column(Text)
    integration_class: Mapped[int | None]
    visibility_score: Mapped[int | None]
    justification: Mapped[str | None] = mapped_column(Text)
    analysis: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)

    jobs: Mapped[list["Job"]] = relationship(back_populates="reel")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    input_url: Mapped[str] = mapped_column(Text)
    reel_id: Mapped[str | None] = mapped_column(ForeignKey("reels.id"), index=True)
    # Заполняется только для задач без ролика (invalid_url); иначе статус берём из reel
    status: Mapped[str | None] = mapped_column(String(20))
    error: Mapped[str | None] = mapped_column(Text)
    cached: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    reel: Mapped[Reel | None] = relationship(back_populates="jobs")
