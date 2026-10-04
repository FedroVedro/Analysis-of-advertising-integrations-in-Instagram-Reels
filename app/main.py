import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_session, init_db
from app.jobs import (
    count_new_reels,
    create_jobs,
    get_jobs,
    get_recent_reels,
    job_to_dict,
    normalize_urls,
    reel_to_dict,
    reels_created_since,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Skycoach Reels Analyzer", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def revalidate_ui(request: Request, call_next):
    # После деплоя браузер должен сразу получить новые index.html/app.js/app.css,
    # а не держать старые из кэша; ETag при этом экономит трафик (ответ 304)
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


class RateLimiter:
    """Скользящее окно в памяти: хватает для одного инстанса (ограничение SQLite и так одно)."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window_secs: int) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= now - window_secs:
                hits.popleft()
            if len(hits) >= limit:
                return False
            hits.append(now)
            return True


rate_limiter = RateLimiter()


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


class CreateJobsRequest(BaseModel):
    urls: list[str]


@app.post("/api/jobs")
def post_jobs(
    body: CreateJobsRequest,
    request: Request,
    session: Session = Depends(get_session),
) -> dict:
    """Ставит ссылки в обработку и сразу возвращает ID задач."""
    settings = get_settings()
    urls = normalize_urls(body.urls)
    limit = settings.max_urls_per_request
    if not urls:
        raise HTTPException(422, "Не передано ни одной ссылки")
    if len(urls) > limit:
        raise HTTPException(422, f"Не больше {limit} ссылок за раз, передано {len(urls)}")

    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.allow(client_ip, settings.rate_limit_requests, settings.rate_limit_window_secs):
        minutes = settings.rate_limit_window_secs // 60
        raise HTTPException(429, f"Слишком много отправок: не больше {settings.rate_limit_requests} за {minutes} мин. Попробуйте позже")

    # Уже обработанные ролики бесплатны (кэш), считаем только новые
    new_reels = count_new_reels(session, urls)
    if new_reels:
        day_ago = datetime.now(timezone.utc) - timedelta(days=1)
        used = reels_created_since(session, day_ago)
        if used + new_reels > settings.max_new_reels_per_day:
            raise HTTPException(
                429,
                f"Достигнут дневной лимит проверки новых роликов ({settings.max_new_reels_per_day}). "
                "Уже проверенные ссылки по-прежнему доступны",
            )

    jobs = create_jobs(session, urls)
    return {"jobs": [job_to_dict(job) for job in jobs]}


@app.get("/api/jobs")
def list_jobs(
    ids: str = Query(..., description="ID задач через запятую"),
    session: Session = Depends(get_session),
) -> dict:
    id_list = [i.strip() for i in ids.split(",") if i.strip()][:100]
    return {"jobs": [job_to_dict(job) for job in get_jobs(session, id_list)]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, session: Session = Depends(get_session)) -> dict:
    jobs = get_jobs(session, [job_id])
    if not jobs:
        raise HTTPException(404, "Задача не найдена")
    return job_to_dict(jobs[0])


@app.get("/api/reels")
def list_reels(
    limit: int = Query(50, ge=1, le=200),
    session: Session = Depends(get_session),
) -> dict:
    return {"reels": [reel_to_dict(r) for r in get_recent_reels(session, limit)]}


@app.get("/health")
def health(session: Session = Depends(get_session)) -> JSONResponse:
    try:
        session.execute(text("SELECT 1"))
    except Exception as exc:
        return JSONResponse({"status": "error", "db": str(exc)}, status_code=503)
    return JSONResponse({"status": "ok", "db": "ok"})
