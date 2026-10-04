from app.config import get_settings

REEL = "https://www.instagram.com/reel/{}/"


def test_rate_limit_per_ip(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "rate_limit_requests", 2)
    for _ in range(2):
        assert client.post("/api/jobs", json={"urls": ["bad"]}).status_code == 200
    resp = client.post("/api/jobs", json={"urls": ["bad"]})
    assert resp.status_code == 429
    assert "Слишком много" in resp.json()["detail"]


def test_daily_cap_counts_only_new_reels(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "max_new_reels_per_day", 2)
    assert client.post("/api/jobs", json={"urls": [REEL.format("AAAAA1"), REEL.format("AAAAA2")]}).status_code == 200
    # новый ролик сверх лимита — отказ
    resp = client.post("/api/jobs", json={"urls": [REEL.format("AAAAA3")]})
    assert resp.status_code == 429
    assert "дневной лимит" in resp.json()["detail"]
    # уже известные ролики и невалидные ссылки проходят
    assert client.post("/api/jobs", json={"urls": [REEL.format("AAAAA1"), "bad"]}).status_code == 200


def test_ui_is_served_and_revalidated(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Reels Analyzer" in resp.text
    assert resp.headers["cache-control"] == "no-cache"
    assert client.get("/static/app.js").headers["cache-control"] == "no-cache"
