import numpy as np
import pytest

from elder_monitor.pipeline import run_analysis
from elder_monitor.schemas import (
    ACTIVITY_STATES, ALERT, BED_EXIT, BED_STATES, LYING_IN_BED, LYING_ON_FLOOR, MONITOR, RETURN_TO_BED,
    SITTING_ON_BED, STANDING, UNKNOWN,
)
from elder_monitor.temporal import durations

from conftest import script

LIE = {"pose": "lying"}
SIT = {"pose": "sitting"}
EDGE = {"pose": "sitting", "edge": True}
STAND_NEAR = {"pose": "standing", "where": "near"}
WALK_AWAY = {"pose": "walking", "where": "away"}
WALK_NEAR = {"pose": "walking", "where": "near"}
FLOOR_NEAR = {"pose": "lying", "where": "near"}
GONE = {"visible": False}


def events(result, kind):
    return [e for e in result["events"] if e.event == kind]


def labels(result):
    return [s.label for s in result["activity"]]


def rules(result, rule):
    return [a for a in result["alerts"] if a["rule"] == rule]


@pytest.mark.parametrize("use_agent", [True, False])
def test_durations_sum_to_video_length(cfg, use_agent):
    obs, T = script((10, LIE), (3, SIT), (2, STAND_NEAR), (15, WALK_AWAY), (5, GONE), (3, WALK_NEAR),
                    (3, SIT), (10, LIE))
    r = run_analysis(obs, T, cfg, use_agent=use_agent)
    assert sum(durations(r["activity"], ACTIVITY_STATES).values()) == pytest.approx(T)
    assert sum(durations(r["bed"], BED_STATES).values()) == pytest.approx(T)
    assert sum(s.duration for s in r["decisions"]) == pytest.approx(T)
    assert r["activity"][0].start == 0.0 and r["activity"][-1].end == pytest.approx(T)
    for a, b in zip(r["activity"], r["activity"][1:]):
        assert a.end == pytest.approx(b.start) and a.label != b.label


def test_sitting_up_is_not_an_exit(cfg):
    obs, T = script((10, LIE), (10, SIT), (10, LIE), (5, EDGE), (10, LIE))
    r = run_analysis(obs, T, cfg)
    assert labels(r) == [LYING_IN_BED, SITTING_ON_BED, LYING_IN_BED, SITTING_ON_BED, LYING_IN_BED]
    assert not r["events"]


def test_brief_stand_then_sit_is_rejected(cfg):
    obs, T = script((10, LIE), (3, SIT), (3, STAND_NEAR), (3, SIT), (10, LIE))
    r = run_analysis(obs, T, cfg)
    assert not events(r, BED_EXIT)
    assert len(r["fsm"]["rejected"]) == 1
    assert STANDING in labels(r)


@pytest.mark.parametrize("use_agent", [True, False])
def test_exit_and_return_are_counted_once(cfg, use_agent):
    obs, T = script((10, LIE), (3, SIT), (2, STAND_NEAR), (20, WALK_AWAY), (3, WALK_NEAR), (3, SIT), (10, LIE))
    r = run_analysis(obs, T, cfg, use_agent=use_agent)
    exits, returns = events(r, BED_EXIT), events(r, RETURN_TO_BED)
    assert len(exits) == 1 and len(returns) == 1
    assert exits[0].start_sec == pytest.approx(13.0, abs=0.5)
    assert exits[0].confirmed_sec > exits[0].start_sec
    assert exits[0].previous_state == SITTING_ON_BED
    assert exits[0].decision == MONITOR
    assert returns[0].start_sec == pytest.approx(38.0, abs=0.5)
    assert r["summary"]["bed_exit_count"] == 1 and r["summary"]["bed_return_count"] == 1
    assert r["summary"]["final_state"] == "lying_in_bed"


@pytest.mark.parametrize("use_agent", [True, False])
def test_long_stand_then_walk_away_is_one_exit(cfg, use_agent):
    obs, T = script((10, LIE), (3, SIT), (18, STAND_NEAR), (5, WALK_AWAY), (1, WALK_NEAR), (3, SIT), (10, LIE))
    r = run_analysis(obs, T, cfg, use_agent=use_agent)
    exits = events(r, BED_EXIT)
    assert len(exits) == 1 and exits[0].start_sec == pytest.approx(13.0, abs=0.5)
    assert len(events(r, RETURN_TO_BED)) == 1


def test_bedside_chair_counts_as_exit_after_dwell(cfg):
    chair = {"pose": "sitting", "where": "near"}
    obs, T = script((10, LIE), (3, SIT), (1, STAND_NEAR), (60, chair), (1, STAND_NEAR), (3, SIT), (10, LIE))
    r = run_analysis(obs, T, cfg)
    exits = events(r, BED_EXIT)
    assert len(exits) == 1 and exits[0].confirmed_sec == pytest.approx(13 + 30, abs=0.5)
    assert exits[0].evidence[-1]["action"] == "dwell"
    assert len(events(r, RETURN_TO_BED)) == 1


def test_video_starting_outside_bed_has_no_exit(cfg):
    obs, T = script((10, WALK_AWAY), (3, WALK_NEAR), (3, SIT), (10, LIE))
    r = run_analysis(obs, T, cfg)
    assert not events(r, BED_EXIT)
    assert len(events(r, RETURN_TO_BED)) == 1


def test_pending_exit_at_end_of_video_is_discarded(cfg):
    obs, T = script((10, LIE), (3, SIT), (4, STAND_NEAR))
    r = run_analysis(obs, T, cfg)
    assert not r["events"]
    assert r["fsm"]["unresolved"] and r["fsm"]["unresolved"][0]["event"] == BED_EXIT
    assert r["trace"][-1]["outcome"] == "video ended before departure was confirmed"


def test_departure_pauses_on_dropouts(cfg):
    blip = [(0.8, WALK_AWAY), (0.2, GONE)] * 4
    obs, T = script((10, LIE), (3, SIT), (2, STAND_NEAR), *blip, (60, GONE))
    r = run_analysis(obs, T, cfg)
    assert len(events(r, BED_EXIT)) == 1


def test_long_visibility_loss_becomes_unknown(cfg):
    obs, T = script((10, WALK_AWAY), (10, GONE), (10, WALK_AWAY))
    r = run_analysis(obs, T, cfg)
    unknown = [s for s in r["activity"] if s.label == UNKNOWN]
    assert len(unknown) == 1 and unknown[0].duration == pytest.approx(10.0, abs=0.5)
    assert rules(r, "UNCERTAIN")


def test_short_dropout_is_absorbed(cfg):
    obs, T = script((10, LIE), (1, GONE), (10, LIE))
    r = run_analysis(obs, T, cfg, use_agent=False)
    assert labels(r) == [LYING_IN_BED]


def test_flickering_labels_still_commit(cfg):
    flicker = [(0.6, {"pose": "standing", "where": "away"}), (0.6, WALK_AWAY)] * 20
    obs, T = script((10, LIE), *flicker)
    r = run_analysis(obs, T, cfg)
    assert labels(r)[0] == LYING_IN_BED and len(labels(r)) > 1
    assert events(r, BED_EXIT)


def test_caregiver_identity_gap_does_not_create_exit(cfg):
    obs, T = script((10, LIE), (6, {"identity_ok": False}), (10, LIE))
    r = run_analysis(obs, T, cfg)
    assert not r["events"]
    assert labels(r) == [LYING_IN_BED]


def test_agent_bridges_blanket_gap_but_baseline_does_not(cfg):
    obs, T = script((10, LIE), (15, GONE), (10, LIE))
    with_agent = run_analysis(obs, T, cfg)
    baseline = run_analysis(obs, T, cfg, use_agent=False)
    assert labels(with_agent) == [LYING_IN_BED]
    assert UNKNOWN in labels(baseline)
    assert with_agent["trace"][0]["trigger"] == "AMBIGUOUS_GAP"


def test_agent_does_not_bridge_when_resident_left_view(cfg):
    obs, T = script((10, SIT), (1, {"pose": "walking", "where": "away", "truncated": True}), (15, GONE), (10, SIT))
    r = run_analysis(obs, T, cfg)
    assert UNKNOWN in labels(r)
    gap = next(t for t in r["trace"] if t["trigger"] == "AMBIGUOUS_GAP")
    assert gap["reason"] == "resident left the camera view"
    assert len(events(r, BED_EXIT)) == 1


def test_agent_does_not_bridge_sitting_gap_without_evidence(cfg):
    obs, T = script((10, SIT), (15, GONE), (10, SIT))
    assert UNKNOWN in labels(run_analysis(obs, T, cfg))


def test_agent_does_not_bridge_out_of_view_gap(cfg):
    obs, T = script((10, WALK_AWAY), (15, GONE), (10, WALK_AWAY))
    r = run_analysis(obs, T, cfg)
    assert UNKNOWN in labels(r)
    assert r["trace"][0]["steps"][-1]["action"] == "abstain"


def test_agent_does_not_bridge_long_gap_without_vlm(cfg):
    obs, T = script((10, LIE), (40, GONE), (10, LIE))
    assert UNKNOWN in labels(run_analysis(obs, T, cfg))


def test_lying_at_bed_outline_resolved_to_bed(cfg):
    obs, T = script((10, LIE), (10, {"pose": "lying", "where": "edge_outside"}), (10, LIE))
    with_agent = run_analysis(obs, T, cfg)
    baseline = run_analysis(obs, T, cfg, use_agent=False)
    assert labels(with_agent) == [LYING_IN_BED]
    assert LYING_ON_FLOOR in labels(baseline)
    assert any(t["trigger"] == "LYING_OUTSIDE_BED" and t["outcome"] == LYING_IN_BED for t in with_agent["trace"])


@pytest.mark.parametrize("parts", [((10, LIE), (60, FLOOR_NEAR)), ((60, FLOOR_NEAR),)])
def test_fall_beside_bed_is_never_hidden(cfg, parts):
    obs, T = script(*parts)
    r = run_analysis(obs, T, cfg)
    assert labels(r)[-1] == LYING_ON_FLOOR
    assert r["summary"]["final_decision"] == ALERT


def test_fall_after_walking_raises_alert(cfg):
    obs, T = script((10, LIE), (3, SIT), (2, STAND_NEAR), (10, WALK_AWAY), (10, {"pose": "lying", "where": "away"}))
    r = run_analysis(obs, T, cfg)
    assert labels(r)[-1] == LYING_ON_FLOOR
    assert r["summary"]["final_decision"] == ALERT
    assert any(a["decision"] == ALERT for a in rules(r, "LYING_ON_FLOOR"))


def test_long_edge_sitting_triggers_monitor(cfg):
    obs, T = script((10, LIE), (70, EDGE), (10, LIE))
    edge = rules(run_analysis(obs, T, cfg), "EDGE_SIT_LONG")
    assert len(edge) == 1 and edge[0]["decision"] == MONITOR
    assert edge[0]["start_sec"] == pytest.approx(10 + 60, abs=1.0)


def test_edge_monitor_ends_when_resident_moves_to_the_middle(cfg):
    obs, T = script((10, LIE), (70, EDGE), (60, SIT), (10, LIE))
    edge = rules(run_analysis(obs, T, cfg), "EDGE_SIT_LONG")
    assert len(edge) == 1 and edge[0]["end_sec"] <= 10 + 70 + 31


def test_prolonged_out_of_bed_alert_fires_once(cfg):
    obs, T = script((10, LIE), (3, SIT), (2, STAND_NEAR), (320, WALK_AWAY))
    r = run_analysis(obs, T, cfg)
    alerts = rules(r, "PROLONGED_OUT_OF_BED")
    assert len(alerts) == 1
    assert alerts[0]["start_sec"] == pytest.approx(13 + 300, abs=1.0)
    assert r["summary"]["final_decision"] == ALERT


def test_leaving_view_after_getting_up_eventually_alerts(cfg):
    obs, T = script((10, LIE), (3, SIT), (2, STAND_NEAR), (1000, GONE))
    r = run_analysis(obs, T, cfg)
    absent = rules(r, "ABSENT_FROM_BED_LONG")
    assert len(absent) == 1 and absent[0]["start_sec"] == pytest.approx(13 + 900, abs=1.0)


def test_no_absence_alert_while_sitting_on_bed(cfg):
    obs, T = script((10, LIE), (3, SIT), (2, STAND_NEAR), (20, WALK_AWAY), (3, WALK_NEAR), (1000, SIT))
    assert not rules(run_analysis(obs, T, cfg), "ABSENT_FROM_BED_LONG")


def test_agent_exit_trace_looks_back_then_ahead(cfg):
    obs, T = script((10, LIE), (3, SIT), (20, STAND_NEAR), (20, WALK_AWAY))
    r = run_analysis(obs, T, cfg)
    exit_trace = [t for t in r["trace"] if t["trigger"] == "EXIT_CANDIDATE"]
    assert len(exit_trace) == 1 and exit_trace[0]["outcome"] == "BED_EXIT confirmed"
    actions = [s["action"] for s in exit_trace[0]["steps"]]
    assert actions[0] == "look_back" and "look_ahead" in actions
    assert events(r, BED_EXIT)[0].evidence[0]["action"] == "look_back"


def test_return_rule_sitting(cfg):
    cfg["events"]["return_rule"] = "sitting"
    obs, T = script((10, LIE), (3, SIT), (2, STAND_NEAR), (20, WALK_AWAY), (3, WALK_NEAR), (6, SIT))
    assert len(events(run_analysis(obs, T, cfg), RETURN_TO_BED)) == 1


EDGE_FLOOR = {"pose": "lying", "where": "edge_outside"}


def test_fall_right_after_standing_up_is_not_relabelled_bed(cfg):
    obs, T = script((10, LIE), (2, SIT), (2, STAND_NEAR), (60, EDGE_FLOOR))
    r = run_analysis(obs, T, cfg)
    assert labels(r)[-1] == LYING_ON_FLOOR and r["summary"]["final_decision"] == ALERT
    assert not events(r, RETURN_TO_BED)


def test_gap_after_walking_is_not_bridged_as_bed(cfg):
    obs, T = script((10, LIE), (2, SIT), (1.5, STAND_NEAR), (1.5, WALK_NEAR), (15, GONE), (10, LIE))
    r = run_analysis(obs, T, cfg)
    assert UNKNOWN in labels(r)
    assert len(events(r, BED_EXIT)) == 1 and len(events(r, RETURN_TO_BED)) == 1


def test_edge_sitting_with_caregiver_occlusions_stays_sitting(cfg):
    obs, T = script((10, LIE), *[(4, EDGE), (6, {"identity_ok": False})] * 9, (10, LIE))
    r = run_analysis(obs, T, cfg)
    assert labels(r) == [LYING_IN_BED, SITTING_ON_BED, LYING_IN_BED]
    assert rules(r, "EDGE_SIT_LONG")


def test_flickering_lying_at_bed_outline_is_reviewed(cfg):
    obs, T = script((10, LIE), *[(1.8, GONE), (0.8, EDGE_FLOOR)] * 6, (10, LIE))
    r = run_analysis(obs, T, cfg)
    assert LYING_ON_FLOOR not in labels(r) and not rules(r, "LYING_ON_FLOOR")


def test_dwell_exit_needs_observed_time_out_of_bed(cfg):
    obs, T = script((10, LIE), (3, SIT), (2, STAND_NEAR), (27, GONE), (3, STAND_NEAR), (3, SIT), (10, LIE))
    assert not events(run_analysis(obs, T, cfg), BED_EXIT)


def test_every_exit_candidate_has_one_consistent_trace(cfg):
    chair = {"pose": "sitting", "where": "near"}
    obs, T = script((10, LIE), (3, SIT), (1, STAND_NEAR), (60, chair), (3, SIT), (10, LIE))
    r = run_analysis(obs, T, cfg)
    exits = [t for t in r["trace"] if t["trigger"] == "EXIT_CANDIDATE"]
    assert len(exits) == 1 and exits[0]["outcome"] == "BED_EXIT confirmed"
    assert exits[0]["steps"][0]["action"] == "look_back"


def test_exit_into_view_loss_keeps_last_known_state(cfg):
    obs, T = script((10, LIE), (3, SIT), (2, STAND_NEAR), (1.2, WALK_NEAR), (10, GONE))
    exits = events(run_analysis(obs, T, cfg), BED_EXIT)
    assert len(exits) == 1 and exits[0].current_state != UNKNOWN


def test_floor_part_of_a_fall_is_kept_after_lying_at_the_outline(cfg):
    obs, T = script((10, LIE), (60, EDGE_FLOOR), (12, FLOOR_NEAR))
    r = run_analysis(obs, T, cfg)
    assert labels(r)[-1] == LYING_ON_FLOOR and r["summary"]["final_decision"] == ALERT


def test_fall_from_edge_sitting_is_not_relabelled_bed(cfg):
    obs, T = script((10, EDGE), (60, EDGE_FLOOR))
    r = run_analysis(obs, T, cfg)
    assert labels(r)[-1] == LYING_ON_FLOOR and r["summary"]["final_decision"] == ALERT


class StubVLM:
    def ask(self, image):
        return {"person_visible": True, "posture": "lying", "support": "bed"}, ""


def test_vlm_resolves_long_in_bed_gap(cfg):
    obs, T = script((10, LIE), (40, GONE), (10, LIE))
    r = run_analysis(obs, T, cfg, frames=lambda t: np.zeros((8, 8, 3), np.uint8), vlm_factory=StubVLM)
    assert labels(r) == [LYING_IN_BED] and r["vlm_calls"] == 3
    assert UNKNOWN in labels(run_analysis(obs, T, cfg, use_agent=False))
