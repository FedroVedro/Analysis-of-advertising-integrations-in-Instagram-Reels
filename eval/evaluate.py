"""Оценка точности анализа на размеченном наборе (eval/labels.csv).

    python eval/evaluate.py [--logo-only] [--only ID ...]

Сравнивает класс интеграции и удержание за размещение с эталоном, печатает таблицу
и сводку по split (calibration / holdout / synthetic), сохраняет eval/results/<время>.json.
"""

import argparse
import csv
import json
import logging
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from app.pipeline import analyze, media, vision  # noqa: E402
from app.scraper.apify_reels import ApifyReelsScraper, download_video  # noqa: E402

CACHE = ROOT / "cache"
RESULTS = ROOT / "results"

# Синтетика из реальных роликов: имя -> (исходный shortcode, аргументы ffmpeg)
SYNTHETIC = {
    "clipped": ("Db3C60btk6E", ["-vf", "crop=iw*0.75:ih:iw*0.25:0,pad=iw/0.75:ih:0:0:black", "-c:a", "copy"]),
    "covered": ("Db3C60btk6E", ["-vf", "pad=iw:ih*1.74:0:ih*0.74:black,crop=iw:ih/1.74:0:0,scale=720:1280", "-c:a", "copy"]),
    "no_audio": ("Db3C60btk6E", ["-an", "-c:v", "copy"]),
    "silent_track": ("Db3C60btk6E", ["SILENT"]),
    "long": ("Db3C60btk6E", ["LOOP"]),
    "no_skycoach": ("Db3C60btk6E", ["-t", "9", "-c:v", "libx264", "-c:a", "aac"]),
    "no_skycoach_poe": ("DbQGs9HMcOQ", ["-t", "8", "-c:v", "libx264", "-c:a", "aac"]),
    "no_skycoach_poe2": ("DbpiVrpMa1j", ["-t", "6", "-c:v", "libx264", "-an"]),
}


def load_labels(only: list[str] | None) -> list[dict]:
    with (ROOT / "labels.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if not only or r["id"] in only]


def ensure_cache(labels: list[dict]) -> None:
    """Скачивает реальные ролики (с метаданными) и собирает синтетику — один раз."""
    CACHE.mkdir(exist_ok=True)
    needed = {r["id"]: r["source"] for r in labels if not r["source"].startswith("synthetic:")}
    for r in labels:
        if r["source"].startswith("synthetic:"):
            src = SYNTHETIC[r["source"].split(":", 1)[1]][0]
            needed.setdefault(src, f"https://www.instagram.com/reel/{src}/")
    missing = {k: v for k, v in needed.items() if not (CACHE / f"{k}.json").exists()}
    for chunk in [list(missing.items())[i:i + 20] for i in range(0, len(missing), 20)]:
        print(f"Apify: скачиваю {len(chunk)} роликов в кэш…")
        for res in ApifyReelsScraper().fetch([url for _, url in chunk]):
            if res.status.value != "ok" or not res.video_url:
                print(f"  {res.shortcode}: недоступен — {res.error}")
                continue
            try:
                download_video(res.video_url, CACHE / f"{res.shortcode}.mp4")
                if res.audio_url:
                    download_video(res.audio_url, CACHE / f"{res.shortcode}.audio.mp4")
            except Exception as exc:
                print(f"  {res.shortcode}: не скачался — {exc}")
                continue
            meta = {"caption": res.caption, "duration": res.duration_sec, "has_separate_audio": bool(res.audio_url)}
            (CACHE / f"{res.shortcode}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    for r in labels:
        if r["source"].startswith("synthetic:"):
            make_synthetic(r["source"].split(":", 1)[1])


def make_synthetic(name: str) -> Path:
    out = CACHE / f"syn_{name}.mp4"
    if out.exists():
        return out
    src_code, args = SYNTHETIC[name]
    src = CACHE / f"{src_code}.mp4"
    ff = media.ffmpeg_exe()
    if args == ["SILENT"]:
        cmd = [ff, "-v", "error", "-y", "-i", str(src), "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
               "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)]
    elif args == ["LOOP"]:
        cmd = [ff, "-v", "error", "-y", "-stream_loop", "19", "-i", str(src), "-c", "copy", str(out)]
    else:
        cmd = [ff, "-v", "error", "-y", "-i", str(src), *args, str(out)]
    subprocess.run(cmd, check=True)
    meta = json.loads((CACHE / f"{src_code}.json").read_text(encoding="utf-8"))
    meta["duration"] = None
    meta["has_separate_audio"] = meta["has_separate_audio"] and name not in ("no_audio", "silent_track")
    (CACHE / f"syn_{name}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return out


def run_one(label: dict) -> dict:
    key = f"syn_{label['source'].split(':', 1)[1]}" if label["source"].startswith("synthetic:") else label["id"]
    meta = json.loads((CACHE / f"{key}.json").read_text(encoding="utf-8"))
    audio = CACHE / f"{key}.audio.mp4"
    started = time.time()
    try:
        result = analyze.analyze_reel(
            shortcode=label["id"], video_url=str(CACHE / f"{key}.mp4"),
            audio_url=str(audio) if meta.get("has_separate_audio") and audio.exists() else None,
            caption=meta.get("caption"), duration_hint=meta.get("duration"),
        )
    except Exception as exc:
        return {**label, "error": f"{type(exc).__name__}: {exc}"[:200], "seconds": round(time.time() - started)}
    placement = result.analysis["placement"]
    banner = result.analysis["banner"]
    got_deduction = "na" if not result.integration_class else \
        "excluded" if placement["deduction_pct"] is None else str(placement["deduction_pct"])
    return {
        **label,
        "got_class": result.integration_class,
        "got_deduction": got_deduction,
        "score": result.visibility_score,
        "method": banner["method"],
        "logo_width_pct": banner["logo_width_pct"],
        "plate_area_pct": banner["area_pct"],
        "banner_seconds": banner["seconds"],
        "review": result.analysis["review"]["reasons"],
        "has_audio": result.has_audio,
        "summary": result.justification.splitlines()[0],
        "seconds": round(time.time() - started),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logo-only", action="store_true", help="без vision-модели (только логотип по эталону)")
    parser.add_argument("--only", nargs="*", help="id из labels.csv")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    labels = load_labels(args.only)
    ensure_cache(labels)

    # Видео уже в кэше: «скачивание» = копирование локального файла
    analyze.download_video = lambda url, dest, **kw: Path(shutil.copy(url, dest))
    if args.logo_only:
        # Отключённая модель «ничего не видит»: ролики без логотипа получают класс 0, а не ошибку
        analyze.detect = lambda _frames: []
        # Без модели нет и текстового классификатора: класс по правилам (логотип → минимум 1)
        from app.pipeline import classify as cl
        cl._ask_llm = lambda *a: cl.Classification(1, reasoning="(--logo-only: LLM не вызывался)")
        # Транскрипция тоже ходит в NeuroAPI — отключаем
        analyze.transcribe = lambda audio: None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_one, labels))

    print(f"\n{'id':<18} {'split':<11} {'класс':>9} {'удержание':>15} {'вис':>3} {'логотип':>8} {'плашка':>7} метод  проверка")
    stats = defaultdict(lambda: {"n": 0, "cls": 0, "ded": 0, "err": 0, "review": 0})
    for r in results:
        s = stats[r["split"]]
        s["n"] += 1
        if "error" in r:
            s["err"] += 1
            print(f"{r['id']:<18} {r['split']:<11} ОШИБКА: {r['error']}")
            continue
        cls_ok = str(r["got_class"]) == r["expected_class"]
        ded_ok = r["got_deduction"] == r["expected_deduction"]
        s["cls"] += cls_ok
        s["ded"] += ded_ok
        s["review"] += bool(r["review"])
        mark = lambda ok: "✓" if ok else "✗"
        print(f"{r['id']:<18} {r['split']:<11} {mark(cls_ok)} {r['got_class']}/{r['expected_class']:<5} "
              f"{mark(ded_ok)} {r['got_deduction']:>5}/{r['expected_deduction']:<7} {str(r['score']):>3} "
              f"{str(r['logo_width_pct']):>7}% {str(r['plate_area_pct']):>6}% {r['method']:<6} "
              f"{'; '.join(r['review'])[:70]}")
    print("\nСводка:")
    for split, s in stats.items():
        ok = s["n"] - s["err"]
        print(f"  {split:<11} роликов {s['n']}: класс {s['cls']}/{ok}, удержание {s['ded']}/{ok}, "
              f"к ручной проверке {s['review']}, ошибок {s['err']}")
    if "calibration" in stats:
        print("  Внимание: calibration — пороги подбирались на этих роликах, оценка на них оптимистична.")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{datetime.now():%Y%m%d-%H%M%S}{'-logo' if args.logo_only else ''}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nПодробно: {out.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
