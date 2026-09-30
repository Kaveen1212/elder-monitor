"""Regression tests for the failure cases in examples/FAILURE_CASES.md."""
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from elder_monitor.config import load_config
from elder_monitor.evaluation import load_annotations, match_events, to_grid
from elder_monitor.features import Scene, add_box_speed, add_speed, measure
from elder_monitor.pipeline import run_analysis, view_quality
from elder_monitor.schemas import (
    ACTIVITY_STATES, BED_EXIT, RETURN_TO_BED, SITTING_ON_BED, STANDING, UNKNOWN, WALKING, Observation, Proposal,
)
from elder_monitor.temporal import build_timeline, propose_all
from elder_monitor.vision import TargetSelector

from conftest import make_obs, script
from test_units import LYING_ON_BED, STANDING_BESIDE, person

ROOT = Path(__file__).resolve().parents[1]
LIE, SIT = {"pose": "lying"}, {"pose": "sitting"}
STAND_ON_BED, GONE = {"pose": "standing", "where": "bed"}, {"visible": False}
STAND_NEAR = {"pose": "standing", "where": "near"}
WALK_AWAY, WALK_NEAR = {"pose": "walking", "where": "away"}, {"pose": "walking", "where": "near"}


class FixedVLM:
    def __init__(self, posture="sitting", support="bed"):
        self.answer, self.calls = {"person_visible": True, "posture": posture, "support": support}, 0

    def ask(self, image):
        self.calls += 1
        return dict(self.answer), ""


def frames(t):
    return np.zeros((8, 8, 3), np.uint8)


def upright_reviews(r):
    return [t for t in r["trace"] if t["trigger"] == "UPRIGHT_ON_BED_REGION"]


def labels(r):
    return [s.label for s in r["activity"]]


def events(r, kind):
    return [e for e in r["events"] if e.event == kind]


# Case 1: edge-sitting facing the camera read as standing

EDGE_SIT_MISREAD = ((3, SIT), (2.4, STAND_ON_BED), (3, SIT))


def test_vlm_confirming_sitting_removes_the_false_standing(cfg):
    obs, T = script(*EDGE_SIT_MISREAD)
    vlm = FixedVLM()
    r = run_analysis(obs, T, cfg, frames=frames, vlm_factory=lambda: vlm)
    assert labels(r) == [SITTING_ON_BED] and not r["fsm"]["rejected"] and not r["fsm"]["pending_spans"]
    assert vlm.calls == cfg["agent"]["vlm_frames"]
    assert [t["outcome"] for t in upright_reviews(r)] == [SITTING_ON_BED]


@pytest.mark.parametrize("vlm", [FixedVLM("standing", "floor"), None])
def test_standing_is_kept_unless_the_vlm_confirms_sitting(cfg, vlm):
    obs, T = script(*EDGE_SIT_MISREAD)
    r = run_analysis(obs, T, cfg, frames=frames, vlm_factory=(lambda: vlm) if vlm else None)
    assert labels(r) == labels(run_analysis(obs, T, cfg, use_agent=False)) and STANDING in labels(r)
    assert bool(upright_reviews(r)) == (vlm is not None)


@pytest.mark.parametrize("parts", [
    ((3, SIT), (1.2, STAND_ON_BED), (3, WALK_AWAY)),
    ((3, SIT), (1.2, STAND_ON_BED), (0.6, SIT), (1.2, STAND_NEAR), (3, WALK_AWAY)),
])
def test_real_stand_up_is_not_questioned(cfg, parts):
    obs, T = script(*parts)
    r = run_analysis(obs, T, cfg, frames=frames, vlm_factory=FixedVLM)
    assert not upright_reviews(r) and [e.start_sec for e in events(r, BED_EXIT)] == [3.0]


def test_standing_beside_the_bed_or_moving_is_not_sent_to_the_vlm(cfg):
    obs, T = script((10, LIE), (5, SIT), (3, STAND_NEAR), (5, SIT), (10, LIE))
    r = run_analysis(obs, T, cfg, frames=frames, vlm_factory=FixedVLM)
    assert STANDING in labels(r) and r["vlm_calls"] == 0
    obs, T = script((10, LIE), (5, SIT), (3, STAND_ON_BED), (5, SIT), (10, LIE))
    for k, o in enumerate(obs):
        if 15.0 <= o.t < 18.0 and k % 4 == 0:
            o.speed = 0.6
    assert run_analysis(obs, T, cfg, frames=frames, vlm_factory=FixedVLM)["vlm_calls"] == 0


def test_vlm_cannot_relabel_a_stand_long_enough_to_confirm_an_exit(cfg):
    obs, T = script((10, LIE), (3, SIT), (40, STAND_ON_BED), (3, SIT), (10, LIE))
    r = run_analysis(obs, T, cfg, frames=frames, vlm_factory=FixedVLM)
    assert [e.start_sec for e in events(r, BED_EXIT)] == [13.0]


# Case 2: return to bed detected late after a brief stand

@pytest.mark.parametrize("use_agent", [True, False])
def test_brief_stand_during_return_keeps_the_first_sit_down(cfg, use_agent):
    obs, T = script((10, LIE), (3, SIT), (2, STAND_NEAR), (20, WALK_AWAY), (3, WALK_NEAR), (3, SIT),
                    (2.6, STAND_NEAR), (3, SIT), (10, LIE))
    r = run_analysis(obs, T, cfg, use_agent=use_agent)
    returns = events(r, RETURN_TO_BED)
    assert len(events(r, BED_EXIT)) == 1 and len(returns) == 1
    assert returns[0].start_sec == pytest.approx(38.0, abs=0.5)
    assert returns[0].evidence[-1]["action"] == "keep_return_start"


@pytest.mark.parametrize("use_agent", [True, False])
def test_walking_away_during_return_restarts_it(cfg, use_agent):
    obs, T = script((10, LIE), (3, SIT), (2, STAND_NEAR), (20, WALK_AWAY), (3, WALK_NEAR), (3, SIT),
                    (5, WALK_AWAY), (3, WALK_NEAR), (3, SIT), (10, LIE))
    returns = events(run_analysis(obs, T, cfg, use_agent=use_agent), RETURN_TO_BED)
    assert len(returns) == 1 and returns[0].start_sec == pytest.approx(49.0, abs=0.5) and not returns[0].evidence


def test_sitting_rule_counts_from_the_latest_sit_down(cfg):
    cfg["events"]["return_rule"] = "sitting"
    obs, T = script((10, LIE), (3, SIT), (2, STAND_NEAR), (20, WALK_AWAY), (3, WALK_NEAR), (1.4, SIT),
                    (3, STAND_NEAR), (1.4, SIT))
    r = run_analysis(obs, T, cfg)
    assert not events(r, RETURN_TO_BED)
    assert r["fsm"]["unresolved"][-1]["start_sec"] == pytest.approx(38.0, abs=0.5)


def test_long_unseen_absence_does_not_keep_the_first_sit_down(cfg):
    obs, T = script((10, LIE), (3, SIT), (2, STAND_NEAR), (20, WALK_AWAY), (3, WALK_NEAR), (3, SIT),
                    (2, STAND_NEAR), (600, GONE), (3, SIT), (10, LIE))
    returns = events(run_analysis(obs, T, cfg, use_agent=False), RETURN_TO_BED)
    assert len(returns) == 1 and returns[0].start_sec == pytest.approx(643.0) and not returns[0].evidence


# Case 3: walking out past the camera becomes UNKNOWN

def dropout(t, x, y=300.0, w=200.0, h=400.0):
    return Observation(t=t, visible=False, identity_ok=True, reason="low_keypoints", track_id=1, det_conf=0.8,
                       bbox=[x, y, x + w, y + h])


@pytest.mark.parametrize(("pose", "expected"), [("walking", [UNKNOWN, WALKING, WALKING]), ("lying", [UNKNOWN] * 3)])
def test_box_motion_keeps_walking_only_after_upright_postures(cfg, pose, expected):
    where = "away" if pose == "walking" else "bed"
    obs = [make_obs(round(0.2 * k, 1), pose, where) for k in range(5)]
    obs += [dropout(round(1.0 + 0.2 * k, 1), 400.0 + 60.0 * k) for k in range(3)]
    add_box_speed(obs, cfg["posture"]["box_speed_window_sec"], cfg["posture"]["speed_window_sec"])
    assert [p.label for p in propose_all(obs, cfg)[5:]] == expected


def test_walking_then_unseen_commits_after_left_view(cfg):
    walk_out = [STANDING] * 10 + [WALKING] * 3 + [UNKNOWN] * 8
    props = [Proposal(round(0.2 * k, 1), lab, 0.9) for k, lab in enumerate(walk_out)]
    segs = build_timeline(props, 4.2, cfg)
    assert [(s.label, round(s.start, 1)) for s in segs] == [(STANDING, 0.0), (WALKING, 2.0), (UNKNOWN, 2.6)]


def test_short_occlusion_after_walking_does_not_restart_the_out_of_bed_timer(cfg):
    obs, T = script((10, LIE), (3, SIT), (2, STAND_NEAR), (185, WALK_AWAY), (1.2, GONE), (180, WALK_AWAY))
    alerts = [a for a in run_analysis(obs, T, cfg, use_agent=False)["alerts"] if a["rule"] == "PROLONGED_OUT_OF_BED"]
    assert alerts and alerts[0]["start_sec"] == pytest.approx(313.0)


def test_first_visible_sample_borrows_the_box_speed_and_equal_times_are_skipped(cfg):
    a = dropout(0.0, 0.0, 0.0, 100.0, 300.0)
    b = Observation(t=0.2, visible=True, identity_ok=True, track_id=1, bbox=[60, 0, 160, 300])
    add_box_speed([a, b], 0.4, 1.0)
    assert b.speed == b.box_speed > 0.5
    same_time = Observation(t=0.2, visible=True, identity_ok=True, track_id=1, bbox=[90, 0, 190, 300])
    add_box_speed([b, same_time], 0.4, 1.0)
    assert same_time.box_speed == 0.0


@pytest.mark.parametrize("use_agent", [True, False])
def test_box_jump_while_still_in_view_is_not_a_departure(cfg, use_agent):
    obs, t = script((10, LIE), (3, SIT), (3, STAND_NEAR))
    obs += [dropout(round(t + 0.2 * k, 4), 400.0 if k < 3 else 470.0) for k in range(10)]
    tail, t2 = script((2, STAND_NEAR), (60, SIT), (2, STAND_NEAR), (20, WALK_AWAY))
    for o in tail:
        o.t = round(o.t + t + 2.0, 4)
    r = run_analysis(obs + tail, round(t + 2.0 + t2, 4), cfg, use_agent=use_agent)
    assert [e.start_sec for e in events(r, BED_EXIT)] == [80.0]


@pytest.mark.parametrize("use_agent", [True, False])
def test_lying_back_down_while_the_joints_drop_out_is_not_an_exit(cfg, use_agent):
    obs, t = script((10, LIE), (2, STAND_NEAR))
    upright, lying = [400.0, 100.0, 600.0, 700.0], [350.0, 400.0, 750.0, 600.0]
    for o in obs[50:]:
        o.bbox = list(upright)
    for k in range(1, 7):
        x1, y1, x2, y2 = (u + min(k / 3, 1.0) * (v - u) for u, v in zip(upright, lying))
        obs.append(dropout(round(t + 0.2 * (k - 1), 4), x1, y1, x2 - x1, y2 - y1))
    tail, t2 = script((10, LIE))
    for o in tail:
        o.t, o.bbox = round(o.t + t + 1.2, 4), list(lying)
    r = run_analysis(obs + tail, round(t + 1.2 + t2, 4), cfg, use_agent=use_agent)
    assert not events(r, BED_EXIT) and len(r["fsm"]["rejected"]) == 1


def test_fall_with_lost_keypoints_is_not_walking_out_of_view(cfg):
    obs, t = script((10, LIE), (3, SIT), (3, STAND_NEAR))
    obs += [dropout(round(t + 0.2 * k, 4), 400.0, 300.0 + 60 * k) for k in range(4)]
    obs += [dropout(round(t + 0.8 + 0.2 * k, 4), 300.0, 600.0, 400.0, 200.0) for k in range(150)]
    r = run_analysis(obs, round(t + 30.8, 4), cfg, use_agent=False)
    assert not events(r, BED_EXIT) and r["fsm"]["unresolved"]


def test_low_confidence_track_continuation_onto_the_bed_is_rejected(cfg):
    sel, frame = TargetSelector(cfg, Scene(cfg, 1000, 1000)), np.zeros((1000, 1000, 3), np.uint8)
    assert sel.select(0.0, frame, [person(STANDING_BESIDE)])[2] == "selected"
    phantom = person(LYING_ON_BED)
    phantom.conf = 0.36
    assert sel.select(0.2, frame, [phantom])[2] == "identity_uncertain"
    confident = person(LYING_ON_BED)
    assert sel.select(0.4, frame, [confident])[2] == "tracked"


# Case 4: handheld close-up from behind, mirror in shot

def test_tracked_close_up_at_the_frame_border_has_not_left_the_view(cfg):
    before, t0 = script((6, {"pose": "sitting", "where": "away", "truncated": True}))
    gap = [Observation(t=round(t0 + 0.2 * k, 4), visible=False, reason="low_keypoints", track_id=1, det_conf=0.9,
                       bbox=[0, 0, 600, 1000], truncated=True) for k in range(30)]
    after, _ = script((6, {"pose": "sitting", "where": "away", "truncated": True}))
    for o in after:
        o.t = round(o.t + t0 + 6.0, 4)
    r = run_analysis(before + gap + after, round(t0 + 12.0, 4), cfg, frames=frames, vlm_factory=FixedVLM)
    trace = next(t for t in r["trace"] if t["trigger"] == "AMBIGUOUS_GAP")
    assert trace["outcome"] == SITTING_ON_BED and r["vlm_calls"] == 3


def test_ignore_polygon_drops_a_lone_reflection_but_keeps_an_overlapping_body(cfg):
    cfg["scene"]["ignore_polygons"] = [[[0.0, 0.0], [0.3, 0.0], [0.3, 1.0], [0.0, 1.0]]]
    frame = np.zeros((1000, 1000, 3), np.uint8)
    assert TargetSelector(cfg, Scene(cfg, 1000, 1000)).select(0.0, frame, [person(STANDING_BESIDE)])[0] is None
    assert TargetSelector(cfg, Scene(cfg, 1000, 1000)).select(0.0, frame, [person(LYING_ON_BED)])[0] is not None


def standing(x, y=300):
    return {5: (x - 20, y), 6: (x + 20, y), 11: (x - 15, y + 150), 12: (x + 15, y + 150),
            13: (x - 15, y + 280), 14: (x + 15, y + 280), 15: (x - 15, y + 400), 16: (x + 15, y + 400)}


LYING_FLOOR = {5: (690, 870), 6: (690, 890), 11: (840, 870), 12: (840, 890),
               13: (920, 870), 14: (920, 890), 15: (1000, 870), 16: (1000, 890)}


def test_ignore_polygon_never_drops_the_tracked_resident(cfg):
    cfg["scene"]["ignore_polygons"] = [[[0.72, 0.05], [0.99, 0.05], [0.99, 0.95], [0.72, 0.95]]]
    scene, frame = Scene(cfg, 1000, 1000), np.zeros((1000, 1000, 3), np.uint8)
    sel, obs = TargetSelector(cfg, scene), []
    seq = [[person(LYING_ON_BED)]] * 50 + [[person(standing(760 + 20 * min(k, 5)))] for k in range(40)]
    seq += [[person(LYING_FLOOR)]] * 100
    for k, persons in enumerate(seq):
        p, ok, reason = sel.select(k / 5, frame, persons)
        o = measure(k / 5, k, p, scene, cfg) if p else Observation(k / 5, k, reason=reason)
        o.identity_ok = ok
        obs.append(o)
    add_speed(obs, cfg["posture"]["speed_window_sec"])
    r = run_analysis(obs, len(seq) / 5, cfg)
    assert any(a["rule"] == "LYING_ON_FLOOR" for a in r["alerts"])


def test_view_quality_flags_an_unmeasurable_view_but_not_one_dropout(cfg):
    good, _ = script((20, SIT))
    bad = [Observation(t=0.2 * k, reason="low_keypoints" if k % 2 else "") for k in range(100)]
    assert not view_quality(good, cfg)["degraded"] and view_quality(bad, cfg)["degraded"]
    night, T = script((300, LIE), (3, GONE), (297, LIE))
    for o in night:
        if not o.visible:
            o.reason, o.bbox = "low_keypoints", [400, 300, 600, 700]
    assert not view_quality(night, cfg)["degraded"]
    assert "view_quality" in run_analysis(night, T, cfg, use_agent=False)["summary"]


# Real footage: the four labelled example clips (cached keypoints, no video needed)

CLIPS = ["pexels_4049556", "pexels_4052924", "pexels_8090735", "roundtrip_4049556"]


def load_clip(name):
    with gzip.open(ROOT / "tests" / "data" / f"{name}.jsonl.gz", "rt", encoding="utf-8") as f:
        meta = json.loads(f.readline())["meta"]
        return meta, [Observation(**json.loads(line)) for line in f]


@pytest.mark.parametrize(("vlm", "min_accuracy"), [(None, 0.82), (FixedVLM, 0.89)])
def test_labelled_clips_keep_every_exit_and_return(vlm, min_accuracy):
    correct = total = 0
    for name in CLIPS:
        cfg = load_config(ROOT / "examples" / "configs" / f"pexels_{name.split('_')[1]}.yaml")
        meta, obs = load_clip(name)
        r = run_analysis(obs, meta["duration"], cfg, frames=frames, vlm_factory=vlm)
        gt = load_annotations(ROOT / "examples" / "annotations" / f"{name}.json")
        for kind in (BED_EXIT, RETURN_TO_BED):
            truth = [e["time"] for e in gt["events"] if e["event"] == kind]
            tp, fp, fn, _ = match_events(truth, [e.start_sec for e in events(r, kind)], 2.0)
            assert (tp, fp, fn) == (len(truth), [], []), (name, kind)
        T = min(gt["duration"], meta["duration"])
        g, p = (to_grid(x, T, 0.1, ACTIVITY_STATES) for x in (gt["activity"], r["activity"]))
        correct, total = correct + int((g == p).sum()), total + len(g)
    assert correct / total >= min_accuracy
