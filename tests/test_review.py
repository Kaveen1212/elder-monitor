"""Regression tests for the review of commit e1c709a (examples/FAILURE_CASES.md, "Review of commit e1c709a")."""
import json
import math

import numpy as np
import pytest

from elder_monitor.config import load_config, validate_config
from elder_monitor.evaluation import evaluate
from elder_monitor.events import departure
from elder_monitor.features import Scene
from elder_monitor.pipeline import analyze, run_analysis
from elder_monitor.schemas import BED_EXIT, RETURN_TO_BED, SITTING_ON_BED, UNKNOWN, Observation
from elder_monitor.vision import TargetSelector

from conftest import make_obs, script
from test_failure_cases import FixedVLM, dropout, events, frames, labels
from test_units import LYING_ON_BED, STANDING_BESIDE, person, write_prediction

LIE, SIT, GONE = {"pose": "lying"}, {"pose": "sitting"}, {"visible": False}
STAND_NEAR, WALK_NEAR, WALK_AWAY = ({"pose": p, "where": w} for p, w in
                                    (("standing", "near"), ("walking", "near"), ("walking", "away")))
CAREGIVER = {"identity_ok": False}


def shifted(points, dx):
    return {k: (x + dx, y) for k, (x, y) in points.items()}


# 1. Another person inheriting the resident's track ID

def two_colour_frame():
    frame = np.zeros((1000, 1000, 3), np.uint8)
    frame[:, :700], frame[:, 700:] = (200, 40, 40), (40, 40, 200)
    return frame


SITTING_UP_ON_BED = {5: (430, 330), 6: (470, 330), 11: (435, 480), 12: (465, 480),
                     13: (540, 485), 14: (540, 500), 15: (545, 600), 16: (545, 615)}


def test_another_person_given_the_residents_track_id_is_not_adopted(cfg):
    sel, frame = TargetSelector(cfg, Scene(cfg, 1000, 1000)), two_colour_frame()
    assert sel.select(0.0, frame, [person(LYING_ON_BED)])[2] == "selected"
    too_far = person(shifted(STANDING_BESIDE, -150))
    looks_different = person(shifted(STANDING_BESIDE, 560))
    for t, other in ((0.2, too_far), (0.4, looks_different), (0.6, looks_different)):
        assert sel.select(t, frame, [other]) == (None, False, "identity_uncertain")
    assert sel.track_id == 1 and sel.last_t == 0.0
    assert sel.select(0.8, frame, [person(SITTING_UP_ON_BED)])[2] == "tracked"
    assert sel.select(3.0, frame, [person(LYING_ON_BED)])[2] == "tracked"


# 2. Return confirmation across UNKNOWN

@pytest.mark.parametrize("use_agent", [True, False])
def test_return_needs_continuous_lying_after_an_unknown_gap(cfg, use_agent):
    obs, T = script((3, WALK_AWAY), (3, SIT), (1.6, LIE), (31, GONE), (3, LIE))
    returns = events(run_analysis(obs, T, cfg, use_agent=use_agent), RETURN_TO_BED)
    assert len(returns) == 1 and returns[0].start_sec == pytest.approx(3.0)
    assert returns[0].confirmed_sec == pytest.approx(38.6 + 2.0 - 0.2)


def test_sitting_rule_also_restarts_after_unknown(cfg):
    cfg["events"]["return_rule"] = "sitting"
    obs, T = script((3, WALK_AWAY), (1.4, SIT), (5, GONE), (1.4, SIT))
    assert not events(run_analysis(obs, T, cfg, use_agent=False), RETURN_TO_BED)


# 3. VLM answers only about the identified resident

class RecordingVLM(FixedVLM):
    def __init__(self, posture="sitting", support="bed"):
        super().__init__(posture, support)
        self.images = []

    def ask(self, image):
        self.images.append(image)
        return super().ask(image)


def test_vlm_is_not_asked_about_an_unselected_target(cfg):
    obs = [Observation(t=round(0.2 * k, 4), reason="target_not_selected", n_persons=2) for k in range(30)]
    tail, _ = script((10, SIT))
    for o in tail:
        o.t = round(o.t + 6.0, 4)
    vlm = RecordingVLM()
    r = run_analysis(obs + tail, 16.0, cfg, frames=frames, scene=Scene(cfg, 8, 8), vlm_factory=lambda: vlm)
    assert vlm.calls == 0 and labels(r)[0] == UNKNOWN and r["activity"][0].end == pytest.approx(6.0)


def test_vlm_cannot_relabel_a_caregiver_occlusion(cfg):
    obs, T = script((10, LIE), (40, CAREGIVER), (10, LIE))
    vlm = RecordingVLM("lying", "bed")
    r = run_analysis(obs, T, cfg, frames=frames, scene=Scene(cfg, 8, 8), vlm_factory=lambda: vlm)
    assert vlm.calls == 0 and UNKNOWN in labels(r)


def test_vlm_helps_a_weak_keypoint_resident_marked_in_the_crop(cfg):
    before, t0 = script((10, SIT))
    gap = [dropout(round(t0 + 0.2 * k, 4), 400.0) for k in range(25)]
    gap += [Observation(t=round(t0 + 5.0 + 0.2 * k, 4), identity_ok=False, reason="identity_uncertain", n_persons=2)
            for k in range(25)]
    after, _ = script((10, SIT))
    for o in after:
        o.t = round(o.t + t0 + 10.0, 4)
    vlm = RecordingVLM()
    r = run_analysis(before + gap + after, round(t0 + 20.0, 4), cfg, scene=Scene(cfg, 1000, 1000),
                     frames=lambda t: np.zeros((1000, 1000, 3), np.uint8), vlm_factory=lambda: vlm)
    assert vlm.calls == 3 and all((img == (0, 255, 0)).all(axis=2).any() for img in vlm.images)
    relabelled = [p.label for p in r["proposals"][50:100]]
    assert relabelled[:25] == [SITTING_ON_BED] * 25 and relabelled[25:] == [UNKNOWN] * 25


# 4. Evaluation coverage

def annotation(tmp_path, duration=100):
    path = tmp_path / "gt.json"
    path.write_text(json.dumps({"duration": duration,
                                "activity": [{"start": 0, "end": 80, "label": "SITTING_ON_BED"},
                                             {"start": 80, "end": duration, "label": "WALKING"}],
                                "events": [{"event": "bed_exit", "time": 80}]}))
    return str(path)


def test_truncated_prediction_is_rejected_not_scored(tmp_path):
    write_prediction(tmp_path / "short", [(0, 50, SITTING_ON_BED)], [], 50)
    with pytest.raises(ValueError, match="covers 50.00s"):
        evaluate([str(tmp_path / "short")], [annotation(tmp_path)], str(tmp_path / "eval"))


def test_rounding_sized_coverage_difference_is_scored_with_every_event(tmp_path):
    write_prediction(tmp_path / "pred", [(0, 99.97, SITTING_ON_BED)], [], 99.97)
    m = evaluate([str(tmp_path / "pred")], [annotation(tmp_path)], str(tmp_path / "eval"))
    assert m["events"]["bed_exit"]["fn"] == 1 and m["protocol"]["analysed_sec"] == 100
    assert m["activity"]["accuracy"] == pytest.approx(0.8)


@pytest.mark.parametrize("segments", [
    [(0, 60, SITTING_ON_BED), (60, 100, "FLYING")],
    [(0, 60, SITTING_ON_BED), (55, 100, "WALKING")],
    [(0, 60, SITTING_ON_BED), (65, 100, "WALKING")],
    [(0, 60, SITTING_ON_BED), (60, math.nan, "WALKING")],
    [(60, 100, "WALKING"), (0, 60, SITTING_ON_BED)],
])
def test_malformed_predictions_are_rejected(tmp_path, segments):
    write_prediction(tmp_path / "bad", segments, [], 100)
    with pytest.raises(ValueError):
        evaluate([str(tmp_path / "bad")], [annotation(tmp_path)], str(tmp_path / "eval"))


# 5. Identity conflict is not leaving the camera view

def step_then(gap):
    obs, t = script((10, LIE), (3, SIT), (1, STAND_NEAR), (0.6, WALK_NEAR), (2, gap))
    tail, t2 = script((5, SIT))
    for o in tail:
        o.t = round(o.t + t, 4)
    return obs + tail, round(t + t2, 4)


@pytest.mark.parametrize("use_agent", [True, False])
def test_caregiver_stepping_in_front_after_a_step_is_not_an_exit(cfg, use_agent):
    obs, T = step_then(CAREGIVER)
    r = run_analysis(obs, T, cfg, use_agent=use_agent)
    assert not events(r, BED_EXIT) and len(r["fsm"]["rejected"]) == 1


@pytest.mark.parametrize("reason", ["not_detected", "low_confidence"])
def test_walking_out_of_view_is_still_an_exit(cfg, reason):
    obs, t = script((10, LIE), (3, SIT), (1, STAND_NEAR), (0.6, WALK_NEAR))
    obs += [Observation(t=round(t + 0.2 * k, 4), identity_ok=reason == "not_detected", reason=reason)
            for k in range(50)]
    exits = events(run_analysis(obs, round(t + 10.0, 4), cfg, use_agent=False), BED_EXIT)
    assert len(exits) == 1 and exits[0].confirmed_sec == pytest.approx(t + 0.8)


# 6. Evidence is observed time, not a sample count

@pytest.mark.parametrize(("step", "copies", "expected"), [(0.2, 1, 1.8), (0.1, 1, 1.7), (0.2, 2, 1.8), (0.4, 1, 3.6)])
def test_departure_needs_two_observed_seconds_at_any_arrival_rate(cfg, step, copies, expected):
    walk = [make_obs(round(step * k, 4), "walking", "away") for k in range(40) for _ in range(copies)]
    assert departure(walk, 0.0, 16.0, cfg)[0] == pytest.approx(expected)


# 7. Configuration validation

def valid():
    cfg = load_config()
    cfg["scene"]["bed_polygon"] = [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]]
    return cfg


@pytest.mark.parametrize(("key", "value", "message"), [
    ("scene.bed_polygon", [[0.5, 0.5]] * 3, "zero area"),
    ("scene.bed_polygon", [[0.2, 0.2], [0.8, 0.6], [0.8, 0.2], [0.2, 0.8]], "crosses itself"),
    ("scene.bed_polygon", [[0.3, 0.3], [1.7, 0.3], [0.7, 0.7]], "normalised"),
    ("scene.bed_polygon", None, "required"),
    ("scene.chair_polygons", [[[0.1, 0.1], [0.2, 0.1]]], "chair_polygons"),
    ("sampling.fps", 0, "sampling.fps"),
    ("sampling.fps", -5, "sampling.fps"),
    ("sampling.fps", math.inf, "sampling.fps"),
    ("sampling.fps", "5", "sampling.fps"),
    ("target.point", [1.5, 0.2], "target.point"),
    ("target.time_sec", -1, "target.time_sec"),
    ("vision.det_conf", 1.5, "vision.det_conf"),
    ("identity.min_similarity", math.nan, "identity.min_similarity"),
    ("events.exit_persist_sec", 0, "events.exit_persist_sec"),
    ("temporal.smooth_samples", 0, "temporal.smooth_samples"),
    ("vlm.max_calls", -1, "vlm.max_calls"),
    ("events.return_rule", "standing", "return_rule"),
])
def test_invalid_config_is_rejected_with_a_readable_error(key, value, message):
    cfg = valid()
    *path, last = key.split(".")
    node = cfg
    for k in path:
        node = node[k]
    node[last] = value
    with pytest.raises(ValueError, match=message):
        validate_config(cfg)


def test_config_is_validated_before_the_video_is_opened(tmp_path):
    cfg = valid()
    cfg["scene"]["bed_polygon"] = [[0.5, 0.5]] * 3
    with pytest.raises(ValueError, match="zero area"):
        analyze(tmp_path / "missing.mp4", cfg, tmp_path / "out")
    validate_config(valid())
