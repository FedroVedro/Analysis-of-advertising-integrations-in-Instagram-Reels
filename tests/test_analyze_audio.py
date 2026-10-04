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

