from app.db import SessionLocal
from app.models import Reel, ReelStatus

REEL = "https://www.instagram.com/reel/DceO7gsR0w-/"


def test_post_returns_job_ids_immediately(client):
    resp = client.post("/api/jobs", json={"urls": [REEL, "not a link", "  ", REEL]})
    assert resp.status_code == 200
    jobs = resp.json()["jobs"]
    # пустая строка и точный дубль отброшены
    assert [j["status"] for j in jobs] == ["queued", "invalid_url"]
    assert jobs[1]["reel"] is None
    assert "не ссылка" in jobs[1]["error"]


def test_same_reel_in_other_format_is_one_reel(client):
    urls = [REEL, "https://instagram.com/valorant_funzone/reel/DceO7gsR0w-/?igsh=abc"]
    jobs = client.post("/api/jobs", json={"urls": urls}).json()["jobs"]
    assert jobs[0]["job_id"] != jobs[1]["job_id"]
    assert jobs[0]["reel"]["id"] == jobs[1]["reel"]["id"]
    with SessionLocal() as s:
        assert s.query(Reel).count() == 1


def test_resubmit_done_reel_is_cached(client):
    client.post("/api/jobs", json={"urls": [REEL]})
    with SessionLocal() as s:
        reel = s.query(Reel).one()
        reel.status, reel.views = ReelStatus.DONE, 100
        s.commit()
    job = client.post("/api/jobs", json={"urls": [REEL]}).json()["jobs"][0]
    assert job["cached"] is True
    assert job["status"] == "done"
    assert job["reel"]["views"] == 100


def test_resubmit_failed_reel_is_requeued(client):
    client.post("/api/jobs", json={"urls": [REEL]})
    with SessionLocal() as s:
        reel = s.query(Reel).one()
        reel.status, reel.attempts, reel.error = ReelStatus.FAILED, 3, "boom"
        s.commit()
    job = client.post("/api/jobs", json={"urls": [REEL]}).json()["jobs"][0]
    assert job["status"] == "queued" and job["cached"] is False and job["error"] is None


def test_too_many_urls(client):
    urls = [f"https://www.instagram.com/reel/CODE{i:04d}/" for i in range(21)]
    resp = client.post("/api/jobs", json={"urls": urls})
    assert resp.status_code == 422
    assert "20" in resp.json()["detail"]


def test_empty_request(client):
    assert client.post("/api/jobs", json={"urls": ["", " "]}).status_code == 422


def test_get_jobs(client):
    ids = [j["job_id"] for j in client.post("/api/jobs", json={"urls": [REEL, "bad"]}).json()["jobs"]]
    jobs = client.get("/api/jobs", params={"ids": ",".join(ids)}).json()["jobs"]
    assert [j["job_id"] for j in jobs] == ids
    assert client.get(f"/api/jobs/{ids[0]}").json()["job_id"] == ids[0]
    assert client.get("/api/jobs/nope").status_code == 404
    assert len(client.get("/api/reels").json()["reels"]) == 1
