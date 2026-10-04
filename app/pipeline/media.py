"""Работа с видео через ffmpeg: проба, кадры, аудио, громкость.

ffprobe не используем: всё нужное (длительность, наличие аудио) есть в выводе `ffmpeg -i`,
а значит хватает одного бинарника — системного ffmpeg (Docker) или из пакета imageio-ffmpeg (локально).
"""

import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.config import get_settings
from app.errors import NonRetryableError

FRAME_LONG_SIDE = 768  # логотип ещё читается, токенов у vision-модели заметно меньше


class MediaError(NonRetryableError):
    """Файл не обрабатывается ffmpeg — повтор того же файла не поможет."""


@lru_cache
def ffmpeg_exe() -> str:
    configured = get_settings().ffmpeg_path
    if configured:
        return configured
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise MediaError("ffmpeg не найден: установите ffmpeg или пакет imageio-ffmpeg") from exc


def _run(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        [ffmpeg_exe(), "-hide_banner", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
    )


@dataclass
class ProbeResult:
    duration: float | None
    has_audio_stream: bool
    width: int | None
    height: int | None


def probe(path: Path) -> ProbeResult:
    info = _run(["-i", str(path)]).stderr  # без выходного файла ffmpeg печатает сведения и выходит с ошибкой
    if "Invalid data found" in info or "Stream #" not in info:
        raise MediaError(f"Файл не похож на видео: {info.strip().splitlines()[-1] if info.strip() else 'пусто'}")
    duration = None
    if m := re.search(r"Duration: (\d+):(\d+):([\d.]+)", info):
        h, mnt, s = m.groups()
        duration = int(h) * 3600 + int(mnt) * 60 + float(s)
    width = height = None
    if m := re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", info):
        width, height = int(m.group(1)), int(m.group(2))
    return ProbeResult(duration, "Audio:" in info, width, height)


def extract_frames(video: Path, out_dir: Path, duration: float, max_frames: int) -> list[tuple[float, Path]]:
    """Кадры раз в секунду; для длинных роликов — равномерно `max_frames` штук.

    Возвращает (секунда кадра, путь). Шаг между кадрами = duration / число кадров.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = 1.0 if duration <= max_frames else max_frames / duration
    scale = f"scale='if(gt(iw,ih),{FRAME_LONG_SIDE},-2)':'if(gt(iw,ih),-2,{FRAME_LONG_SIDE})'"
    result = _run([
        "-v", "error", "-y", "-i", str(video),
        "-vf", f"fps={fps:.6f},{scale}", "-q:v", "4",
        "-frames:v", str(max_frames), str(out_dir / "f_%04d.jpg"),
    ], timeout=600)
    files = sorted(out_dir.glob("f_*.jpg"))
    if result.returncode != 0 or not files:
        raise MediaError(f"Не удалось извлечь кадры: {result.stderr.strip()[-300:]}")
    step = 1.0 / fps
    # Фильтр fps берёт кадр из середины каждого интервала
    return [(round(i * step + step / 2, 2), f) for i, f in enumerate(files)]


def extract_audio(src: Path, out: Path, max_seconds: int) -> Path:
    """Моно 16 кГц, 32 кбит/с: 25 МБ лимита API хватает на ~100 минут."""
    result = _run([
        "-v", "error", "-y", "-i", str(src), "-vn",
        "-ac", "1", "-ar", "16000", "-b:a", "32k", "-t", str(max_seconds), str(out),
    ], timeout=600)
    if result.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        raise MediaError(f"Не удалось извлечь аудио: {result.stderr.strip()[-300:]}")
    return out


def mean_volume_db(audio: Path) -> float | None:
    info = _run(["-i", str(audio), "-af", "volumedetect", "-f", "null", "-"]).stderr
    m = re.search(r"mean_volume: (-?[\d.]+|-inf) dB", info)
    if not m:
        return None
    return float("-inf") if m.group(1) == "-inf" else float(m.group(1))
