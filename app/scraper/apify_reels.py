"""Получение метрик Instagram Reels через Apify (актёр apify/instagram-reel-scraper).

Один запуск актёра обрабатывает весь батч ссылок (до 20): так дешевле и быстрее,
чем запускать актёр на каждую ссылку, потому что cold start актёра ~10–20 секунд.

Контракт:
    - невалидная ссылка -> ReelResult.status = INVALID_URL (в Apify не отправляется);
    - ролик приватный / удалён / не найден -> status = UNAVAILABLE с пояснением;
    - сбой самого Apify (таймаут, нет кредитов, сеть) -> исключение ApifyScraperError,
      чтобы воркер мог повторить задачу;
    - метрика недоступна -> None, а не 0.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import httpx
from apify_client import ApifyClient

from app.config import get_settings

logger = logging.getLogger(__name__)

ACTOR_ID = "apify/instagram-reel-scraper"
RUN_TIMEOUT_SECS = 300
MAX_BATCH_SIZE = 20

# instagram.com/reel/<code>, /reels/<code>, /p/<code>, /tv/<code>,
# в том числе с username в пути: instagram.com/<user>/reel/<code>
_REEL_URL_RE = re.compile(
    r"^https?://(?:www\.|m\.)?instagram\.com/"
    r"(?:[\w.]+/)?(?:reels?|p|tv)/(?P<code>[\w-]{5,})/?",
    re.IGNORECASE,
)


class ReelStatus(str, Enum):
    OK = "ok"
    INVALID_URL = "invalid_url"
    UNAVAILABLE = "unavailable"  # приватный, удалён или не найден


class ApifyScraperError(RuntimeError):
    """Сбой на стороне Apify: задачу имеет смысл повторить позже."""


@dataclass
class ReelResult:
    input_url: str
    status: ReelStatus
    shortcode: str | None = None
    canonical_url: str | None = None
    error: str | None = None

    views: int | None = None
    views_source: str | None = None  # videoPlayCount или videoViewCount
    likes: int | None = None
    comments: int | None = None
    published_at: datetime | None = None
    author: str | None = None

    caption: str | None = None
    duration_sec: float | None = None
    video_url: str | None = None  # ссылка на CDN Instagram, живёт несколько часов
    audio_url: str | None = None  # отдельная звуковая дорожка, если видео отдано без звука
    hashtags: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["published_at"] = self.published_at.isoformat() if self.published_at else None
        return data


def extract_shortcode(url: str) -> str | None:
    """Достаёт shortcode ролика из ссылки. None, если ссылка не похожа на Reel/пост."""
    match = _REEL_URL_RE.match(url.strip())
    return match.group("code") if match else None


def canonical_reel_url(shortcode: str) -> str:
    """Единый вид ссылки: по нему дедуплицируем повторные отправки."""
    return f"https://www.instagram.com/reel/{shortcode}/"


class ApifyReelsScraper:
    def __init__(self, token: str | None = None, actor_id: str = ACTOR_ID) -> None:
        token = token or get_settings().apify_token
        if not token:
            raise ValueError("APIFY_TOKEN не задан")
        self._client = ApifyClient(token)
        self._actor_id = actor_id

    def fetch(self, urls: Iterable[str]) -> list[ReelResult]:
        """Возвращает результат для каждой входной ссылки в исходном порядке."""
        urls = list(urls)
        if len(urls) > MAX_BATCH_SIZE:
            raise ValueError(f"Не больше {MAX_BATCH_SIZE} ссылок за раз, передано {len(urls)}")

        results: list[ReelResult] = []
        to_scrape: dict[str, ReelResult] = {}  # shortcode -> результат-заготовка

        for url in urls:
            shortcode = extract_shortcode(url)
            if shortcode is None:
                results.append(ReelResult(
                    input_url=url,
                    status=ReelStatus.INVALID_URL,
                    error="Это не ссылка на Instagram Reel или пост",
                ))
                continue
            result = to_scrape.get(shortcode) or ReelResult(
                input_url=url,
                status=ReelStatus.UNAVAILABLE,
                shortcode=shortcode,
                canonical_url=canonical_reel_url(shortcode),
                # Если Apify ничего не вернёт по ссылке, останется это объяснение
                error="Ролик недоступен: приватный аккаунт, удалён или не существует",
            )
            to_scrape[shortcode] = result
            results.append(result)

        if to_scrape:
            items = self._run_actor([r.canonical_url for r in to_scrape.values()])
            for item in items:
                self._apply_item(item, to_scrape)

        # Одинаковые ссылки в одном батче делят один объект; копируем, чтобы сохранить input_url
        return [
            r if r.input_url == url else ReelResult(**{**r.__dict__, "input_url": url})
            for url, r in zip(urls, results)
        ]

    def _run_actor(self, reel_urls: list[str]) -> list[dict[str, Any]]:
        run_input = {
            "username": reel_urls,
            "resultsLimit": 1,  # для прямых ссылок на ролики игнорируется
            "includeSharesCount": False,
            "includeTranscript": False,  # транскрипцию делаем сами через Whisper
            "includeDownloadedVideo": False,
        }
        logger.info("Apify: запуск %s для %d ссылок", self._actor_id, len(reel_urls))
        try:
            run = self._client.actor(self._actor_id).call(
                run_input=run_input,
                run_timeout=timedelta(seconds=RUN_TIMEOUT_SECS),
                wait_duration=timedelta(seconds=RUN_TIMEOUT_SECS + 30),
                logger=None,  # не дублировать лог актёра в наш stdout
            )
        except Exception as exc:  # сеть, 401, 402 (кончились кредиты) и т.п.
            raise ApifyScraperError(f"Не удалось запустить актёр Apify: {exc}") from exc

        status = str(getattr(run.status, "value", run.status)) if run else "UNKNOWN"
        if run is None or status != "SUCCEEDED":
            message = f": {run.status_message}" if run and run.status_message else ""
            raise ApifyScraperError(f"Запуск Apify завершился со статусом {status}{message}")

        items = list(self._client.dataset(run.default_dataset_id).iterate_items())
        logger.info(
            "Apify: получено %d записей, run_id=%s, стоимость $%s",
            len(items), run.id, run.usage_total_usd,
        )
        return items

    @staticmethod
    def _apply_item(item: dict[str, Any], by_shortcode: dict[str, ReelResult]) -> None:
        shortcode = item.get("shortCode") or extract_shortcode(
            item.get("inputUrl") or item.get("url") or ""
        )
        result = by_shortcode.get(shortcode) if shortcode else None
        if result is None:
            logger.warning("Apify вернул запись, которую не удалось сопоставить: %s", item.get("url"))
            return

        # Для недоступных роликов Apify возвращает запись с полем error вместо данных
        if item.get("error"):
            result.status = ReelStatus.UNAVAILABLE
            description = item.get("errorDescription") or item["error"]
            result.error = f"Instagram не отдал данные: {description}"
            return

        play_count = _count(item.get("videoPlayCount"))
        view_count = _count(item.get("videoViewCount"))
        if play_count is not None:
            result.views, result.views_source = play_count, "videoPlayCount"
        elif view_count is not None:
            result.views, result.views_source = view_count, "videoViewCount"

        result.status = ReelStatus.OK
        result.error = None
        result.likes = _count(item.get("likesCount"))  # скрытые лайки -> None
        result.comments = _count(item.get("commentsCount"))
        result.published_at = _parse_datetime(item.get("timestamp"))
        result.author = item.get("ownerUsername")
        result.caption = item.get("caption")
        result.duration_sec = item.get("videoDuration")
        result.video_url = item.get("videoUrl")
        result.audio_url = item.get("audioUrl")
        result.hashtags = item.get("hashtags") or []
        result.mentions = item.get("mentions") or []


def download_video(video_url: str, dest: Path, timeout: float = 60.0, max_bytes: int | None = None) -> Path:
    """Скачивает файл по ссылке CDN. Делать сразу после скрапинга: ссылки быстро истекают."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with httpx.stream("GET", video_url, timeout=timeout, follow_redirects=True) as response:
        response.raise_for_status()
        with dest.open("wb") as f:
            for chunk in response.iter_bytes():
                size += len(chunk)
                if max_bytes and size > max_bytes:
                    raise ValueError(f"Видео больше {max_bytes // (1024 * 1024)} МБ — анализ не выполняется")
                f.write(chunk)
    return dest


def _count(value: Any) -> int | None:
    """Apify отдаёт -1 или null, когда счётчик скрыт; оба варианта -> None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return int(value)


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO)
    scraper = ApifyReelsScraper()
    for r in scraper.fetch(sys.argv[1:]):
        print(json.dumps(r.to_dict(), ensure_ascii=False, indent=2))
