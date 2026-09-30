"""Event timers count observed evidence, never time the timeline only inferred (review of commit 4a1a280)."""
import numpy as np
import pytest

from elder_monitor.features import Scene
from elder_monitor.pipeline import run_analysis
from elder_monitor.schemas import BED_EXIT, LYING_IN_BED, RETURN_TO_BED, UNKNOWN, Observation

from conftest import script

LIE, SIT = {"pose": "lying"}, {"pose": "sitting"}
WALK_AWAY, STAND_NEAR = {"pose": "walking", "where": "away"}, {"pose": "standing", "where": "near"}


def missing(seconds):
    return ("missing", seconds)


def session(*parts, reason="no_frame"):
    """Scripted observations; ("missing", s) inserts s seconds of samples with nothing observed."""
    obs, t = [], 0.0
    for part in parts:
        if part[0] == "missing":
            obs += [Observation(t=round(t + 0.2 * k, 4), reason=reason) for k in range(round(part[1] * 5))]
            t = round(t + part[1], 4)
            continue
        chunk, d = script(part)
        for o in chunk:
            o.t = round(o.t + t, 4)
        obs, t = obs + chunk, round(t + d, 4)
    return obs, t


def events(r, kind):
    return [e for e in r["events"] if e.event == kind]


@pytest.mark.parametrize("use_agent", [True, False])
def test_bridged_missing_frames_do_not_confirm_a_return(cfg, use_agent):
    obs, T = session((3, WALK_AWAY), (3, SIT), (1.6, LIE), missing(10), (3, LIE))
    r = run_analysis(obs, T, cfg, use_agent=use_agent)
    returns = events(r, RETURN_TO_BED)
    assert len(returns) == 1 and returns[0].start_sec == pytest.approx(3.0)
    assert returns[0].confirmed_sec == pytest.approx(17.6 + 2.0 - 0.2)
    if use_agent:
        assert [s.label for s in r["activity"]][-1] == LYING_IN_BED and len(r["activity"]) == 3
        assert all(e == UNKNOWN for o, e in zip(r["observations"], r["evidence"]) if o.reason == "no_frame")


@pytest.mark.parametrize("use_agent", [True, False])
def test_exit_dwell_counts_observed_seconds_only(cfg, use_agent):
    obs, T = session((10, LIE), *[(5, STAND_NEAR), missing(1)] * 6)
    exits = events(run_analysis(obs, T, cfg, use_agent=use_agent), BED_EXIT)
    assert len(exits) == 1 and exits[0].start_sec == pytest.approx(10.0)
    assert exits[0].confirmed_sec == pytest.approx(10.0 + 30.0 + 5 * 1.0 - 0.2)
    assert exits[0].evidence[-1]["action"] == "dwell"


class LyingVLM:
    def ask(self, image):
        return {"person_visible": True, "posture": "lying", "support": "bed"}, ""


def test_vlm_answers_support_only_the_frames_they_inspect(cfg):
    obs, T = session((3, WALK_AWAY), (3, SIT), (1.6, LIE), missing(40), (3, LIE), reason="not_detected")
    r = run_analysis(obs, T, cfg, frames=lambda t: np.zeros((80, 80, 3), np.uint8), scene=Scene(cfg, 80, 80),
                     vlm_factory=LyingVLM)
    assert r["vlm_calls"] == 3 and [s.label for s in r["activity"]][-1] == LYING_IN_BED
    gap = [e for o, e in zip(r["observations"], r["evidence"]) if o.reason == "not_detected"]
    assert gap.count(LYING_IN_BED) == 3 and gap.count(UNKNOWN) == len(gap) - 3
    assert events(r, RETURN_TO_BED)[0].confirmed_sec == pytest.approx(47.6 + 2.0 - 0.2)


def test_short_dropout_restarts_the_return_count(cfg):
    obs, T = session((3, WALK_AWAY), (3, SIT), (1.6, LIE), missing(0.4), (3, LIE))
    returns = events(run_analysis(obs, T, cfg), RETURN_TO_BED)
    assert [s.label for s in run_analysis(obs, T, cfg)["activity"]][-1] == LYING_IN_BED
    assert returns[0].confirmed_sec == pytest.approx(8.0 + 2.0 - 0.2)
