import numpy as np
import pytest

from elder_monitor.live import LiveSession, chat_messages
from elder_monitor.pipeline import run_analysis
from elder_monitor.schemas import BED_EXIT, UNKNOWN

from conftest import script
from test_units import LYING_ON_BED, STANDING_BESIDE, person

LIE = {"pose": "lying"}
SIT = {"pose": "sitting"}
STAND_NEAR = {"pose": "standing", "where": "near"}
WALK_AWAY = {"pose": "walking", "where": "away"}
WALK_NEAR = {"pose": "walking", "where": "near"}
EXIT_AND_RETURN = ((10, LIE), (3, SIT), (2, STAND_NEAR), (20, WALK_AWAY), (3, WALK_NEAR), (3, SIT), (10, LIE))


def stream(cfg, obs):
    session, sent = LiveSession(cfg, tracker=object()), []
    for o in obs:
        session.obs.append(o)
        sent += session.update()["messages"]
    return session, sent


def test_live_matches_offline_and_announces_each_event_once(cfg):
    obs, T = script(*EXIT_AND_RETURN)
    session, sent = stream(cfg, obs)
    offline = run_analysis(obs, T, cfg)
    assert session.result["summary"] == offline["summary"]
    assert [m["key"] for m in sent if m["kind"] == "event"] == ["event:bed_exit", "event:return_to_bed"]
    states = [m["label"] for m in sent if m["kind"] == "state"]
    assert states == [s.label for s in offline["activity"] if s.label != UNKNOWN]


def test_answers_come_from_the_live_result(cfg):
    assert LiveSession(cfg, tracker=object()).answer()["text"] == "No frames analysed yet."
    obs, _ = script(*EXIT_AND_RETURN[:4])
    session, _ = stream(cfg, obs)
    assert "bed exit" in session.answer("any bed exits?")["text"]
    assert session.answer("status")["text"].startswith("Walking")


def test_replay_messages_are_time_ordered(cfg):
    obs, T = script(*EXIT_AND_RETURN)
    msgs = chat_messages(run_analysis(obs, T, cfg), cfg)
    assert [m["t"] for m in msgs] == sorted(m["t"] for m in msgs)
    assert {"state", "event", "alert"} <= {m["kind"] for m in msgs}


def test_server_rejects_bad_input(cfg, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from elder_monitor.server import create_app

    client = TestClient(create_app(cfg, tmp_path))
    assert client.get("/api/health").json()["bed_polygon"] == cfg["scene"]["bed_polygon"]
    r = client.post("/api/jobs", files={"file": ("a.mp4", b"x")}, data={"bed_polygon": "[[2, 3]]"})
    assert r.status_code == 400 and client.get("/api/jobs").json() == []
    assert client.get("/api/jobs/missing").status_code == 404
    with client.websocket_connect("/api/live") as ws:
        ws.send_json({"type": "ask", "text": "status"})
        assert ws.receive_json()["message"]["text"] == "Start the camera first."
        ws.send_json({"type": "start", "bed_polygon": [[5, 5]]})
        assert ws.receive_json()["type"] == "error"


def test_live_evidence_is_observed_time_at_any_frame_rate(cfg):
    """10 s in bed, then standing away from it, with frames at 1, 2, 4 and 0.5 times the 5 fps config.
    Too slow a stream leaves empty slots: nothing is confirmed from them and the view is flagged."""
    lying = person({k: (x + 2100, y) for k, (x, y) in LYING_ON_BED.items()})
    standing = person({k: (x + 2700, y) for k, (x, y) in STANDING_BESIDE.items()})
    frame, delays = np.zeros((1000, 4000, 3), np.uint8), {}
    for rate in (5, 10, 20, 2.5):
        clock = {}
        session = LiveSession(cfg, tracker=lambda f: [lying if clock["t"] < 10 else standing])
        for k in range(int(16 * rate)):
            clock["t"] = k / rate
            session.push(frame, k / rate)
        times = [o.t for o in session.obs]
        assert times == sorted(set(times)) and all(abs(t / 0.2 - round(t / 0.2)) < 1e-6 for t in times)
        delays[rate] = [e.confirmed_sec - e.start_sec for e in session.result["events"] if e.event == BED_EXIT]
    assert delays[5] == delays[10] == delays[20] == [pytest.approx(1.8)]
    assert not delays[2.5] and session.result["view"]["missing_frame_share"] == pytest.approx(0.5, abs=0.02)
    assert session.result["view"]["degraded"] and session.push(frame, 15.6)["skipped"]
