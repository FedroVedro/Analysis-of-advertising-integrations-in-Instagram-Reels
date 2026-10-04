import cv2
import numpy as np
import pytest

from app.pipeline import logo
from app.pipeline.logo import LogoHit
from app.pipeline.scoring import aggregate_hybrid, placement_issues, review_reasons, visibility_score
from app.pipeline.vision import FrameDetection

W, H = 720, 1280


def frame_with_logo(tmp_path, width_pct, x_pct=0.3, y_pct=0.15, name="f.jpg", seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(40, 90, (H, W), dtype=np.uint8)  # «шумный» фон, как игровая картинка
    tmpl = logo._templates()[0]
    tw = round(W * width_pct / 100)
    th = round(tmpl.shape[0] * tw / tmpl.shape[1])
    t = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_AREA)
    x, y = round(W * x_pct), round(H * y_pct)
    # кладём с учётом выхода за края (обрезка)
    sx0, sy0 = max(0, -x), max(0, -y)
    dx0, dy0 = max(0, x), max(0, y)
    w = min(tw - sx0, W - dx0)
    img[dy0:dy0 + th, dx0:dx0 + w] = t[sy0:sy0 + th, sx0:sx0 + w]
    path = tmp_path / name
    cv2.imwrite(str(path), img)
    return path


@pytest.mark.parametrize("width_pct", [10.8, 15.6, 24.3, 35.0])
def test_measures_logo_width(tmp_path, width_pct):
    hit = logo.detect_frame(1.0, frame_with_logo(tmp_path, width_pct))
    assert hit.found and not hit.clipped
    assert abs(hit.width_pct - width_pct) < 0.8


def test_no_logo_on_noise(tmp_path):
    rng = np.random.default_rng(1)
    path = tmp_path / "n.jpg"
    cv2.imwrite(str(path), rng.integers(0, 255, (H, W), dtype=np.uint8))
    assert not logo.detect_frame(1.0, path).found


@pytest.mark.parametrize("x_pct", [-0.08, 0.84])  # срезан слева / справа
def test_clipped_logo_is_found_and_flagged(tmp_path, x_pct):
    hit = logo.detect_frame(1.0, frame_with_logo(tmp_path, 24.3, x_pct=x_pct))
    assert hit.found and hit.clipped
    assert abs(hit.width_pct - 24.3) < 1.5  # оценка полной ширины


def hits(width, n=10, clipped=False, total=40, step=0.25):
    return [LogoHit(t=i * step + 0.125, score=0.95 if i < n else 0.5,
                    bbox=(0.3, 0.15, 0.3 + width / 100, 0.18) if i < n else None,
                    width_pct=width if i < n else None, clipped=clipped and i < n) for i in range(total)]


PLATE = [FrameDetection(1.0, True, 0.9, (0.16, 0.12, 0.84, 0.23), False, "SKYCOACH CODE: VALFUN", "VALFUN", "")]


@pytest.mark.parametrize("width, expected", [(24.3, None), (15.6, 20), (10.7, 30)])
def test_logo_width_drives_deduction(width, expected):
    facts = aggregate_hybrid(hits(width), PLATE, duration=10, step=0.25)
    assert facts.method == "logo" and facts.seconds == 2.5 and facts.plate_measured
    deductions = [i.deduction_pct for i in placement_issues(facts)]
    assert deductions == ([] if expected is None else [expected])
    assert facts.promo_codes == ["VALFUN"]


def test_logo_width_drives_score():
    assert visibility_score(aggregate_hybrid(hits(24.3), PLATE, 10, 0.25), False, False)[0] == 3
    assert visibility_score(aggregate_hybrid(hits(10.7), PLATE, 10, 0.25), False, False)[0] == 1


def test_clipped_logo_without_plate():
    facts = aggregate_hybrid(hits(24.3, clipped=True), [], duration=10, step=0.25)
    assert "clipped" in [i.code for i in placement_issues(facts)]


@pytest.mark.parametrize("width, needs", [(24.3, False), (19.0, True), (13.9, True), (16.0, False)])
def test_review_near_thresholds(width, needs):
    facts = aggregate_hybrid(hits(width), PLATE, duration=10, step=0.25)
    assert bool(review_reasons(facts)) is needs


def test_review_when_vision_failed_or_disagrees():
    facts = aggregate_hybrid(hits(24.3), [], duration=10, step=0.25)
    assert "модель недоступна" in review_reasons(facts, vision_failed=True)[0]
    assert "не увидела" in review_reasons(facts)[0]
