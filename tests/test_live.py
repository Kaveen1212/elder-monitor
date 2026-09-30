import pytest

from elder_monitor.live import LiveSession, chat_messages
from elder_monitor.pipeline import run_analysis
from elder_monitor.schemas import UNKNOWN

from conftest import script

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
