from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db import get_session, init_db


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Skycoach Reels Analyzer", lifespan=lifespan)


@app.get("/health")
def health(session: Session = Depends(get_session)) -> JSONResponse:
    try:
        session.execute(text("SELECT 1"))
    except Exception as exc:
        return JSONResponse({"status": "error", "db": str(exc)}, status_code=503)
    return JSONResponse({"status": "ok", "db": "ok"})
