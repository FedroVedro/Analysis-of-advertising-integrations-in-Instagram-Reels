from pathlib import Path

import httpx

from app.pipeline import analyze


def test_failed_separate_audio_does_not_fail_analysis(monkeypatch, tmp_path):
    def expired(url, dest, **kw):
        raise httpx.HTTPStatusError("403", request=httpx.Request("GET", url), response=httpx.Response(403))
    monkeypatch.setattr(analyze, "download_video", expired)
    has_audio, transcript, note, meta = analyze._process_audio(Path("v.mp4"), False, "https://cdn/expired", tmp_path)
    assert has_audio is None and transcript is None
    assert "не удалось" in note and meta["audio_error"]


def test_no_audio_at_all(tmp_path):
    has_audio, _, note, meta = analyze._process_audio(Path("v.mp4"), False, None, tmp_path)
    assert has_audio is False and meta["source"] == "none" and "Звука нет" in note


def test_download_retries_transport_errors(monkeypatch, tmp_path):
    from app.scraper import apify_reels
    calls = []
    def flaky(url, dest, timeout, max_bytes):
        calls.append(1)
        if len(calls) < 3:
            raise httpx.ConnectError("[SSL: UNEXPECTED_EOF_WHILE_READING]")
        return dest
    monkeypatch.setattr(apify_reels, "_download_once", flaky)
    monkeypatch.setattr(apify_reels.time, "sleep", lambda s: None)
    assert apify_reels.download_video("https://cdn/x.mp4", tmp_path / "v.mp4") == tmp_path / "v.mp4"
    assert len(calls) == 3
