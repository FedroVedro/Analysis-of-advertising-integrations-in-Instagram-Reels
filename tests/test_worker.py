from datetime import datetime, timedelta, timezone

from app import worker
from app.db import SessionLocal
from app.jobs import create_jobs
from app.models import Reel, ReelStatus
from app.scraper.apify_reels import ApifyScraperError, ReelResult
from app.scraper.apify_reels import ReelStatus as ScrapeStatus


class FakeScraper:
    def __init__(self, results=None, error=None):
        self.results = results or {}
        self.error = error
        self.calls = []

    def fetch(self, urls):
        self.calls.append(list(urls))
        if self.error:
            raise self.error
        out = []
        for url in urls:
            code = url.rstrip("/").rsplit("/", 1)[-1]
            out.append(self.results.get(code) or ReelResult(
                input_url=url, status=ScrapeStatus.UNAVAILABLE, shortcode=code,
                error="Ролик недоступен"))
        return out


def ok(code, **kw):
    return ReelResult(input_url="", status=ScrapeStatus.OK, shortcode=code, views=10,
                      likes=None, author="bob",
                      published_at=datetime(2026, 9, 1, tzinfo=timezone.utc), **kw)


def enqueue(*codes):
    with SessionLocal() as s:
        create_jobs(s, [f"https://www.instagram.com/reel/{c}/" for c in codes])


def reels():
    with SessionLocal() as s:
        return {r.shortcode: r for r in s.query(Reel)}


def test_batch_ok_and_unavailable():
    enqueue("AAAAA1", "BBBBB2")
    scraper = FakeScraper({"AAAAA1": ok("AAAAA1")})
    assert worker.process_batch(scraper) == 2
    assert len(scraper.calls) == 1  # один запуск Apify на пачку
    r = reels()
    assert r["AAAAA1"].status == ReelStatus.DONE
    assert r["AAAAA1"].views == 10 and r["AAAAA1"].likes is None
    assert r["BBBBB2"].status == ReelStatus.UNAVAILABLE
    assert r["BBBBB2"].error == "Ролик недоступен"
    assert worker.process_batch(scraper) == 0  # очередь пуста


def test_apify_failure_retries_then_fails():
    enqueue("AAAAA1")
    scraper = FakeScraper(error=ApifyScraperError("402 no credits"))
    for attempt in range(1, worker.MAX_ATTEMPTS):
        worker.process_batch(scraper)
        r = reels()["AAAAA1"]
        assert r.status == ReelStatus.QUEUED and r.attempts == attempt
    worker.process_batch(scraper)
    r = reels()["AAAAA1"]
    assert r.status == ReelStatus.FAILED
    assert "402 no credits" in r.error


def test_error_in_one_reel_does_not_affect_others(monkeypatch):
    enqueue("AAAAA1", "BBBBB2")
    original = worker._analyze

    def flaky(reel_id):
        with SessionLocal() as s:
            if s.get(Reel, reel_id).shortcode == "AAAAA1":
                raise ValueError("boom")
        original(reel_id)

    monkeypatch.setattr(worker, "_analyze", flaky)
    worker.process_batch(FakeScraper({"AAAAA1": ok("AAAAA1"), "BBBBB2": ok("BBBBB2")}))
    r = reels()
    assert r["AAAAA1"].status == ReelStatus.QUEUED and "boom" in r["AAAAA1"].error
    assert r["AAAAA1"].views == 10  # метрики сохранились, несмотря на сбой анализа
    assert r["BBBBB2"].status == ReelStatus.DONE


def test_requeue_stale():
    enqueue("AAAAA1", "BBBBB2")
    with SessionLocal() as s:
        old = datetime.now(timezone.utc) - timedelta(hours=1)
        for r in s.query(Reel):
            r.status = ReelStatus.FETCHING
            r.locked_at = old if r.shortcode == "AAAAA1" else datetime.now(timezone.utc)
        s.commit()
    assert worker.requeue_stale() == 1
    assert reels()["AAAAA1"].status == ReelStatus.QUEUED
    assert reels()["BBBBB2"].status == ReelStatus.FETCHING
    assert worker.requeue_stale(older_than=None) == 1


def test_claim_does_not_take_same_reel_twice():
    enqueue("AAAAA1", "BBBBB2")
    first = worker.claim_batch()
    assert len(first) == 2
    assert worker.claim_batch() == []
