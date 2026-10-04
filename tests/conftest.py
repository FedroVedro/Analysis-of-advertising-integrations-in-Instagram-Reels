import os
import tempfile
from pathlib import Path

# Отдельная БД для тестов; задаётся до импорта app, т.к. engine создаётся при импорте
_tmp = Path(tempfile.mkdtemp()) / "test.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp.as_posix()}"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.db import Base, engine  # noqa: E402
from app.main import app, rate_limiter  # noqa: E402


@pytest.fixture(autouse=True)
def clean_db():
    from app import models  # noqa: F401

    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    rate_limiter._hits.clear()
    yield


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
