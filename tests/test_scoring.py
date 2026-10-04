from app.pipeline.scoring import (
    aggregate, deduction, placement_issues, visibility_score,
)
from app.pipeline.vision import FrameDetection


def det(t, bbox=None, clipped=False, code=None, conf=0.95):
    return FrameDetection(t=t, banner=bbox is not None, confidence=conf, bbox=bbox, clipped=clipped,
                          text="SKYCOACH CODE: X" if bbox else "", promo_code=code, other_text="")


GOOD = (0.159, 0.122, 0.841, 0.230)    # хороший баннер из примеров Skycoach: ≈7.4 % кадра
SMALL = (0.216, 0.118, 0.755, 0.172)   # «too small»: ≈2.9 %
SMALLISH = (0.100, 0.105, 0.878, 0.178)  # «small and partly covered»: ≈5.7 %


def video(bbox, seconds_with_banner, total=30, **kw):
    return [det(t + 0.5, bbox if 10 <= t < 10 + seconds_with_banner else None, **kw) for t in range(total)]


def test_good_banner_no_issues():
    facts = aggregate(video(GOOD, 8, code="VALFUN"), duration=30, step=1)
    assert facts.banner_frames == 8 and facts.seconds == 8
    assert facts.intervals == [(10.0, 18.0)]
    assert 7.0 <= facts.area_pct <= 7.8 and facts.position == "вверху по центру"
    assert facts.promo_codes == ["VALFUN"]
    assert placement_issues(facts) == []
    assert deduction([]) == (0, "без замечаний")
    assert visibility_score(facts, voice_cta=False, caption_cta=False)[0] == 3  # ≈7.4 % → 3
    assert visibility_score(facts, voice_cta=True, caption_cta=False)[0] == 4


def test_too_small_banner_matches_skycoach_30pct():
    facts = aggregate(video(SMALL, 8), duration=30, step=1)
    issues = placement_issues(facts)
    assert [i.code for i in issues] == ["too_small"]
    assert deduction(issues) == (30, "удержание 30 %")


def test_slightly_small_banner_matches_skycoach_20pct():
    issues = placement_issues(aggregate(video(SMALLISH, 8), duration=30, step=1))
    assert [i.code for i in issues] == ["slightly_small"]
    assert deduction(issues)[0] == 20


def test_clipped_banner():
    left_cut = (0.0, 0.12, 0.6, 0.23)  # касается только левого края
    issues = placement_issues(aggregate(video(left_cut, 8), duration=30, step=1))
    assert "clipped" in [i.code for i in issues]


def test_full_width_banner_is_not_clipped():
    full = (0.0, 0.12, 1.0, 0.23)
    assert "clipped" not in [i.code for i in placement_issues(aggregate(video(full, 8), 30, 1))]


def test_banner_low_but_above_caption_is_ok():
    # как в принятом Skycoach ролике DbRI307xMEy
    assert placement_issues(aggregate(video((0.16, 0.78, 0.84, 0.88), 8), 30, 1)) == []


def test_banner_just_above_music_line_is_ok():
    # как в принятом Skycoach ролике Dblv_p8RP4T: 82–93 % высоты
    assert placement_issues(aggregate(video((0.16, 0.818, 0.84, 0.927), 8), 30, 1)) == []


def test_banner_in_bottom_ui_zone_is_covered():
    low = (0.2, 0.90, 0.8, 0.99)
    issues = placement_issues(aggregate(video(low, 8), duration=30, step=1))
    assert "covered" in [i.code for i in issues]


def test_several_issues_take_max_deduction():
    issues = placement_issues(aggregate(video((0.0, 0.94, 0.3, 0.98), 8), duration=30, step=1))
    codes = {i.code for i in issues}
    assert {"too_small", "clipped", "covered"} <= codes
    assert deduction(issues)[0] == 30


def test_no_banner_is_excluded():
    facts = aggregate(video(None, 0), duration=30, step=1)
    issues = placement_issues(facts)
    assert deduction(issues) == (None, "исключить из выплаты")
    assert visibility_score(facts, voice_cta=True, caption_cta=False)[0] == 2


def test_low_confidence_detections_ignored():
    facts = aggregate(video(GOOD, 8, conf=0.3), duration=30, step=1)
    assert not facts.visible


def test_score_bounds_and_duration_bonus():
    huge = (0.0, 0.0, 1.0, 0.5)  # 50 % кадра, весь ролик
    facts = aggregate([det(t + 0.5, huge) for t in range(10)], duration=10, step=1)
    assert visibility_score(facts, True, True)[0] == 5
    tiny_flash = aggregate(video(SMALL, 1), duration=30, step=1)
    assert visibility_score(tiny_flash, False, False)[0] == 1
    smallish = aggregate(video(SMALLISH, 10), duration=18, step=1)  # как DbpiVrpMa1j: мелкий, но долго
    assert visibility_score(smallish, False, False)[0] == 3


def test_long_video_justification_is_compact():
    from app.pipeline.classify import Classification
    from app.pipeline.report import build_justification
    dets = [det(i * 10.35 + 5, GOOD if i % 3 == 0 else None) for i in range(60)]
    facts = aggregate(dets, duration=621, step=10.35)
    text = build_justification(cls=Classification(2), facts=facts, score=3, score_reasons=["x"], issues=[],
                               transcript=None, audio_note=None, sampled=True)
    assert "и ещё 15" in text and "≈207 с" in text and "выборка из 60 кадров" in text
