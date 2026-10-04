from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_session, init_db
from app.jobs import (
    create_jobs,
    get_jobs,
    get_recent_reels,
    job_to_dict,
    normalize_urls,
    reel_to_dict,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Skycoach Reels Analyzer", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


class CreateJobsRequest(BaseModel):
    urls: list[str]


@app.post("/api/jobs")
def post_jobs(body: CreateJobsRequest, session: Session = Depends(get_session)) -> dict:
    """Ставит ссылки в обработку и сразу возвращает ID задач."""
    urls = normalize_urls(body.urls)
    limit = get_settings().max_urls_per_request
    if not urls:
        raise HTTPException(422, "Не передано ни одной ссылки")
    if len(urls) > limit:
        raise HTTPException(422, f"Не больше {limit} ссылок за раз, передано {len(urls)}")
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
