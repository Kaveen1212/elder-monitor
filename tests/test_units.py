import json

import cv2
import numpy as np
import pytest

from elder_monitor.evaluation import class_metrics, evaluate, load_annotations, match_events, parse_time
from elder_monitor.features import Scene, measure
from elder_monitor.reporting import fmt_clock, fmt_hms, fmt_human, read_observations, write_observations
from elder_monitor.schemas import (
    LYING_IN_BED, LYING_ON_FLOOR, OUT_OF_BED, SITTING_ON_BED, SITTING_OUTSIDE_BED, STANDING, UNKNOWN, WALKING,
    Proposal,
)
from elder_monitor.temporal import build_timeline, leg_posture, propose, smooth
from elder_monitor.video import VideoReader
from elder_monitor.vision import Person, TargetSelector
from elder_monitor.vlm import answer_state, parse_answer

from conftest import make_obs


def person(points, track_id=1):
    kp = np.zeros((17, 3))
    for i, (x, y) in points.items():
        kp[i] = [x, y, 0.9]
    xs, ys = kp[list(points), 0], kp[list(points), 1]
    return Person(track_id, [xs.min() - 20, ys.min() - 40, xs.max() + 20, ys.max() + 10], kp.tolist(), 0.9)


STANDING_BESIDE = {5: (180, 300), 6: (220, 300), 11: (185, 450), 12: (215, 450),
                   13: (185, 580), 14: (215, 580), 15: (185, 700), 16: (215, 700)}
LYING_ON_BED = {5: (350, 500), 6: (350, 520), 11: (500, 505), 12: (500, 525),
                13: (580, 505), 14: (580, 525), 15: (660, 505), 16: (660, 525)}
SITTING_ON_CHAIR = {5: (150, 400), 6: (190, 400), 11: (155, 540), 12: (185, 540),
                    13: (100, 545), 14: (130, 545), 15: (100, 660), 16: (130, 660)}
EDGE_SIT_FACING_CAMERA = {5: (440, 300), 6: (560, 300), 11: (450, 480), 12: (550, 480),
                          13: (450, 560), 14: (550, 560), 15: (450, 800), 16: (550, 800)}
LYING_TOWARDS_CAMERA = {5: (430, 450), 6: (570, 450), 11: (460, 510), 12: (540, 510)}


def test_measure_standing_lying_sitting(cfg):
    scene, c = Scene(cfg, 1000, 1000), cfg["posture"]
    s = measure(0.0, 0, person(STANDING_BESIDE), scene, cfg)
    assert s.visible and s.torso_angle < 10 and leg_posture(s, c) == "extended" and not s.hip_in_bed
    lying = measure(0.0, 0, person(LYING_ON_BED), scene, cfg)
    assert lying.torso_angle > 80 and lying.hip_in_bed and lying.body_in_bed > 0.9
    chair = measure(0.0, 0, person(SITTING_ON_CHAIR), scene, cfg)
    assert leg_posture(chair, c) == "seated"
    assert propose(chair, None, cfg).label == SITTING_OUTSIDE_BED
    assert propose(lying, None, cfg).label == LYING_IN_BED
    assert propose(s, None, cfg).label == STANDING


def test_edge_sitting_facing_camera_is_seated(cfg):
    o = measure(0.0, 0, person(EDGE_SIT_FACING_CAMERA), Scene(cfg, 1000, 1000), cfg)
    assert o.thigh_ratio < 0.5 and leg_posture(o, cfg["posture"]) == "seated"
    assert propose(o, None, cfg).label == SITTING_ON_BED


def test_standing_without_shoulders_uses_leg_axis(cfg):
    legs_only = {k: v for k, v in STANDING_BESIDE.items() if k >= 11}
    o = measure(0.0, 0, person(legs_only), Scene(cfg, 1000, 1000), cfg)
    assert o.torso_angle is None and o.leg_angle < 10
    assert propose(o, None, cfg).label == STANDING


def test_lying_along_camera_axis_is_lying(cfg):
    o = measure(0.0, 0, person(LYING_TOWARDS_CAMERA), Scene(cfg, 1000, 1000), cfg)
    assert o.torso_angle < 30 and o.hip_in_bed
    assert propose(o, None, cfg).label == LYING_IN_BED


def test_measure_low_keypoints_is_not_visible(cfg):
    o = measure(0.0, 0, person({5: (100, 100), 6: (120, 100)}), Scene(cfg, 1000, 1000), cfg)
    assert not o.visible and o.reason == "low_keypoints"
    assert propose(o, None, cfg).label == UNKNOWN


def test_edge_margin_is_capped_by_bed_size(cfg):
    cfg["scene"]["bed_polygon"] = [[0.2, 0.5], [0.8, 0.5], [0.8, 0.56], [0.2, 0.56]]
    scene = Scene(cfg, 1000, 1000)
    assert scene.edge_margin < 30 * 0.5


def test_propose_rules(cfg):
    assert propose(make_obs(0, "lying", "away"), None, cfg).label == LYING_ON_FLOOR
    assert propose(make_obs(0, "walking", "away"), None, cfg).label == WALKING
    assert propose(make_obs(0, "sitting", "bed"), None, cfg).label == SITTING_ON_BED
    standing_overlap = make_obs(0, "standing", "bed")
    standing_overlap.feet_in_bed = False
    assert propose(standing_overlap, None, cfg).label == STANDING
    long_sitting = make_obs(0, "standing", "bed")
    long_sitting.feet_in_bed = True
    assert propose(long_sitting, None, cfg).label == SITTING_ON_BED
    unresolved = make_obs(0, "sitting", "away")
    unresolved.knee_angle = unresolved.thigh_ratio = None
    assert propose(unresolved, None, cfg).label == OUT_OF_BED
    cut_off = make_obs(0, "sitting", "away", truncated=True)
    cut_off.torso_angle = cut_off.knee_angle = cut_off.thigh_ratio = None
    cut_off.bbox = [0, 0, 900, 500]
    assert propose(cut_off, None, cfg).label == UNKNOWN


def test_lying_hysteresis(cfg):
    o = make_obs(0, "lying", "bed")
    o.torso_angle = 52
    assert propose(o, LYING_IN_BED, cfg).label == LYING_IN_BED
    assert propose(o, SITTING_ON_BED, cfg).label == SITTING_ON_BED


def test_dwell_backdates_boundary(cfg):
    labels = ["A"] * 10 + ["B"] * 10
    props = [Proposal(i * 0.2, lab, 1.0) for i, lab in enumerate(labels)]
    cfg["temporal"]["smooth_samples"] = 1
    segs = build_timeline(props, 4.0, cfg)
    assert [(s.label, s.start, s.end) for s in segs] == [("A", 0.0, pytest.approx(2.0)), ("B", pytest.approx(2.0), 4.0)]


def test_dwell_ignores_single_frame_flicker(cfg):
    labels = ["A"] * 10 + ["B"] + ["A"] * 10
    props = [Proposal(i * 0.2, lab, 1.0) for i, lab in enumerate(labels)]
    assert [s.label for s in build_timeline(props, 4.2, cfg)] == ["A"]
    assert smooth(["A", "B", "A"], 3) == ["A", "A", "A"]


def test_time_formats():
    assert fmt_clock(272) == "04:32" and fmt_clock(3725) == "1:02:05"
    assert fmt_hms(308) == "00:05:08"
    assert fmt_human(702) == "11m 42s" and fmt_human(45) == "45s" and fmt_human(1200) == "20m 00s"
    assert parse_time("04:32") == 272 and parse_time("00:05:08") == 308 and parse_time(12.5) == 12.5
    with pytest.raises(ValueError):
        parse_time("9:99")


def test_vlm_answer_parsing():
    ans = parse_answer('```json\n{"person_visible": true, "posture": "lying", "support": "bed"}\n```')
    assert answer_state(ans) == LYING_IN_BED
    assert parse_answer('{"person_visible": "yes", "posture": "lying", "support": "bed"}') is None
    assert parse_answer("no json here") is None
    assert answer_state({"person_visible": True, "posture": "lying", "support": "unclear"}) is None


def test_event_matching_is_one_to_one_and_maximal():
    tp, fp, fn, errs = match_events([10.0, 50.0], [10.5, 11.0, 80.0], tol=2.0)
    assert tp == 1 and sorted(fp) == [11.0, 80.0] and fn == [50.0] and errs == [0.5]
    tp, fp, fn, _ = match_events([10.0, 12.5], [11.5, 14.0], tol=2.0)
    assert tp == 2 and not fp and not fn


def test_macro_f1_ignores_abstention_without_support():
    cm = np.array([[50.0, 0, 0], [0, 49.9, 0.1], [0, 0, 0]])
    _, macro = class_metrics(cm, ("IN_BED", "OUT_OF_BED", "UNKNOWN"))
    assert macro > 0.99


def write_video(path, n, fps):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    for k in range(n):
        writer.write(np.full((48, 64, 3), k * 8 % 255, np.uint8))
    writer.release()
    return VideoReader(path)


def test_video_reader_timestamps(tmp_path):
    reader = write_video(tmp_path / "clip.avi", 30, 10)
    samples = list(reader.sample(5))
    times = [t for t, _, _ in samples]
    assert len(samples) == 15 and all(f is not None for _, _, f in samples)
    assert times == sorted(times) and times[1] == pytest.approx(0.2, abs=0.05)
    assert reader.duration == pytest.approx(3.0, abs=0.05)
    assert reader.frame_at(1.0) is not None


def test_video_reader_low_fps_source_has_no_fake_gaps(tmp_path):
    reader = write_video(tmp_path / "slow.avi", 9, 3)
    samples = list(reader.sample(5))
    times = [t for t, _, _ in samples]
    assert all(f is not None for _, _, f in samples) and times == sorted(times) and len(samples) == 9


def colour_frame():
    frame = np.zeros((1000, 1000, 3), np.uint8)
    frame[:, :500] = (200, 40, 40)
    frame[:, 500:] = (40, 40, 200)
    return frame


def test_selector_never_transfers_identity_to_covisible_person(cfg):
    scene, frame = Scene(cfg, 1000, 1000), colour_frame()
    resident, carer = person(LYING_ON_BED, track_id=1), person(STANDING_BESIDE, track_id=2)
    sel = TargetSelector(cfg, scene)
    assert sel.select(0.0, frame, [resident, carer])[2] == "selected"
    assert sel.select(0.2, frame, [resident, carer])[2] == "tracked"
    for t in (1.0, 5.0, 20.0):
        p, ok, reason = sel.select(t, frame, [carer])
        assert p is None and not ok and reason == "identity_uncertain"


def test_selector_keeps_lock_through_untracked_detection(cfg):
    scene, frame = Scene(cfg, 1000, 1000), colour_frame()
    sel = TargetSelector(cfg, scene)
    sel.select(0.0, frame, [person(LYING_ON_BED, track_id=1)])
    assert sel.select(0.2, frame, [person(LYING_ON_BED, track_id=None)])[2] == "reassociated"
    assert sel.selected and sel.track_id == 1
    stranger = person(LYING_ON_BED, track_id=9)
    stranger.conf = 0.3
    assert sel.select(0.4, frame, [stranger])[0] is None


def write_prediction(d, activity, events, T):
    d.mkdir()
    bed = [(a, b, "IN_BED" if lab in (LYING_IN_BED, SITTING_ON_BED) else "OUT_OF_BED") for a, b, lab in activity]
    for name, segs in (("timeline.csv", activity), ("bed_timeline.csv", bed)):
        rows = ["start,end,start_sec,end_sec,duration_sec,label"] + [f"x,x,{a},{b},{b - a},{lab}" for a, b, lab in segs]
        (d / name).write_text("\n".join(rows) + "\n")
    (d / "events.json").write_text(json.dumps({"bed_events": [
        {"event": e, "start_sec": t, "confirmed_sec": t + 2} for e, t in events]}))
    (d / "summary.json").write_text(json.dumps({"observation_duration_sec": T}))


def test_evaluate_known_example(tmp_path):
    ann = tmp_path / "gt.json"
    ann.write_text(json.dumps({
        "duration": 100,
        "activity": [{"start": 0, "end": 50, "label": "LYING_IN_BED"}, {"start": 50, "end": 60, "label": "STANDING"},
                     {"start": 60, "end": 100, "label": "WALKING"}],
        "events": [{"event": "bed_exit", "time": 50}],
    }))
    pred = tmp_path / "clip"
    write_prediction(pred, [(0, 45, LYING_IN_BED), (45, 60, STANDING), (60, 100, WALKING)],
                     [("bed_exit", 45.5), ("bed_exit", 90)], 100)
    m = evaluate([str(pred)], [str(ann)], str(tmp_path / "eval"))
    assert m["activity"]["accuracy"] == pytest.approx(0.95)
    ex = m["events"]["bed_exit"]
    assert ex["tp"] == 0 and ex["fp"] == 2 and ex["fn"] == 1 and ex["precision"] == 0.0
    m2 = evaluate([str(pred)], [str(ann)], str(tmp_path / "eval2"), tolerance=5.0)
    assert m2["events"]["bed_exit"]["tp"] == 1 and m2["events"]["bed_exit"]["recall"] == 1.0
    assert (tmp_path / "eval" / "confusion_matrix.png").exists()
    report = (tmp_path / "eval" / "failure_cases.md").read_text(encoding="utf-8")
    assert "missed_bed_exit" in report and "false_bed_exit" in report and "Case 4" in report


EDGE_SIT_JUST_OUTSIDE = {5: (330, 150), 6: (370, 150), 11: (335, 290), 12: (365, 290),
                         13: (250, 300), 14: (280, 300), 15: (250, 420), 16: (280, 420)}


def test_seated_at_bed_edge_with_hips_just_outside_is_on_bed(cfg):
    o = measure(0.0, 0, person(EDGE_SIT_JUST_OUTSIDE), Scene(cfg, 1000, 1000), cfg)
    assert not o.hip_in_bed and o.edge_of_bed and leg_posture(o, cfg["posture"]) == "seated"
    assert propose(o, None, cfg).label == SITTING_ON_BED


def test_straight_legs_lying_along_the_bed_are_not_standing(cfg):
    o = make_obs(0, "standing", "bed")
    o.torso_angle, o.leg_angle, o.feet_in_bed = 30.0, 95.0, False
    assert propose(o, None, cfg).label == SITTING_ON_BED


def test_edge_distance_ignores_frame_border(cfg):
    cfg["scene"]["bed_polygon"] = [[0.2, 0.5], [0.8, 0.5], [0.8, 1.0], [0.2, 1.0]]
    scene = Scene(cfg, 1000, 1000)
    assert scene.edge_dist((500, 990)) > scene.edge_margin
    assert scene.edge_dist((500, 505)) < scene.edge_margin


def test_selector_keeps_duplicate_boxes_of_the_resident(cfg):
    scene, frame = Scene(cfg, 1000, 1000), colour_frame()
    sel = TargetSelector(cfg, scene)
    main, duplicate = person(LYING_ON_BED, track_id=1), person(LYING_ON_BED, track_id=2)
    sel.select(0.0, frame, [main, duplicate])
    assert 2 not in sel.others
    assert sel.select(0.4, frame, [duplicate])[2] == "reassociated"


def test_annotations_must_be_contiguous(tmp_path):
    path = tmp_path / "gap.json"
    path.write_text(json.dumps({"activity": [{"start": 0, "end": 10, "label": "LYING_IN_BED"},
                                             {"start": 12, "end": 20, "label": "WALKING"}]}))
    with pytest.raises(ValueError):
        load_annotations(path)


def test_event_tolerance_is_float_safe():
    assert match_events([parse_time("00:02.4")], [4.4], tol=2.0)[0] == 1


def test_stale_observation_cache_is_rejected_before_parsing(tmp_path):
    path = tmp_path / "obs.jsonl"
    write_observations(path, {"video_hash": "old"}, [])
    with open(path, "a", encoding="utf-8") as f:
        print(json.dumps({"t": 0.0, "retired_field": 1}), file=f)
    assert read_observations(path, {"video_hash": "new"}) == (None, None)


def test_hips_over_bed_without_posture_evidence_is_unclear(cfg):
    o = make_obs(0, "sitting", "bed")
    o.torso_angle = o.knee_angle = o.thigh_ratio = None
    o.bbox = [400, 200, 600, 700]
    assert propose(o, None, cfg).label == UNKNOWN
