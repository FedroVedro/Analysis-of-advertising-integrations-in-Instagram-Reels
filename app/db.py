from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


def _create_engine(database_url: str) -> Engine:
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite":
        return create_engine(database_url, pool_pre_ping=True)

    if url.database and url.database != ":memory:":
        Path(url.database).parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(
        database_url,
        # Сессии открываются в потоках FastAPI и в воркере
        connect_args={"check_same_thread": False, "timeout": 5},
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_conn, _record) -> None:
        cursor = dbapi_conn.cursor()
        # WAL: web читает, пока воркер пишет, без блокировок
        cursor.execute("PRAGMA journal_mode=WAL")
        # При конкурентной записи ждём до 5 с вместо ошибки "database is locked"
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

    return engine


engine = _create_engine(get_settings().database_url)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    from app import models  # noqa: F401  регистрирует таблицы в Base.metadata

    # web и воркер стартуют одновременно: create_all одного может проверить таблицы
    # до того, как второй их создаст, и упасть с "table already exists" — тогда повторяем
    for attempt in range(3):
        try:
            Base.metadata.create_all(engine)
            _add_missing_columns()
            return
        except OperationalError as exc:
            if "already exists" not in str(exc) or attempt == 2:
                raise


def _add_missing_columns() -> None:
    """Мини-миграция: create_all не добавляет новые колонки в существующие таблицы.

    Хватает для добавления nullable-колонок; для чего-то сложнее — Alembic (пункт «что дальше»).
    """
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            existing = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name not in existing and column.nullable:
                    col_type = column.type.compile(dialect=engine.dialect)
                    conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {col_type}'))


def get_session() -> Iterator[Session]:
    """Зависимость FastAPI: одна сессия на запрос."""
    with SessionLocal() as session:
        yield session
