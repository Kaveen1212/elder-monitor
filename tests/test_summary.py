"""Absence metrics: continuous out-of-bed time versus the whole away episode (review of commit 4a1a280)."""
import pytest

from elder_monitor.pipeline import run_analysis

from conftest import script

LIE, SIT, GONE = {"pose": "lying"}, {"pose": "sitting"}, {"visible": False}
STAND_NEAR, WALK_AWAY, WALK_NEAR = ({"pose": p, "where": w} for p, w in
                                    (("standing", "near"), ("walking", "away"), ("walking", "near")))


def test_away_episode_spans_the_visibility_gap(cfg):
    obs, T = script((10, LIE), (3, SIT), (2, STAND_NEAR), (20, WALK_AWAY), (30, GONE), (3, WALK_NEAR), (3, SIT),
                    (10, LIE))
    s = run_analysis(obs, T, cfg)["summary"]
    assert s["longest_out_of_bed_period_sec"] == pytest.approx(22.0)
    assert s["longest_away_episode_sec"] == pytest.approx(55.0)
    (episode,) = s["away_episodes"]
    assert episode["start_sec"] == pytest.approx(13.0) and episode["end_sec"] == pytest.approx(68.0)
    assert episode["observed_out_of_bed_sec"] == pytest.approx(25.0) and episode["unknown_sec"] == pytest.approx(30.0)
    assert episode["ended_by"] == "return_to_bed"


def test_open_away_episode_runs_to_the_end_of_the_recording(cfg):
    obs, T = script((10, LIE), (3, SIT), (2, STAND_NEAR), (20, WALK_AWAY), (15, GONE))
    (episode,) = run_analysis(obs, T, cfg)["summary"]["away_episodes"]
    assert episode["ended_by"] == "end_of_recording" and episode["end_sec"] == pytest.approx(T)
