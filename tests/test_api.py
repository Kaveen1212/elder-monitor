"""API controls: allowed origins, API key, upload limit, message errors, retention (review of commit 4a1a280)."""
import json
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from elder_monitor.server import cleanup_runs, create_app  # noqa: E402


def client(cfg, tmp_path, **kw):
    return TestClient(create_app(cfg, tmp_path / "runs", **kw))


def test_only_configured_origins_get_cors_headers(cfg, tmp_path):
    c = client(cfg, tmp_path, origins=["http://ward-pc:8000"])
    allowed = c.get("/api/health", headers={"Origin": "http://ward-pc:8000"})
    assert allowed.headers.get("access-control-allow-origin") == "http://ward-pc:8000"
    assert "access-control-allow-origin" not in c.get("/api/health", headers={"Origin": "http://evil.example"}).headers


def test_api_key_is_required_when_set(cfg, tmp_path):
    c = client(cfg, tmp_path, api_key="s3cret")
    assert c.get("/api/health").status_code == 401
    assert c.get("/api/health", headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.get("/api/health", headers={"X-API-Key": "s3cret"}).status_code == 200
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect("/api/live") as ws:
            ws.receive_json()
    with c.websocket_connect("/api/live?api_key=s3cret") as ws:
        ws.send_json({"type": "ask", "text": "status"})
        assert ws.receive_json()["type"] == "answer"


def test_oversized_uploads_are_refused_and_not_kept(cfg, tmp_path):
    c = client(cfg, tmp_path, max_upload_mb=0.001)
    r = c.post("/api/jobs", files={"file": ("big.mp4", b"x" * 3 * 1024 * 1024)})
    assert r.status_code == 413 and "larger than" in r.json()["detail"]
    assert c.get("/api/jobs").json() == [] and not any((tmp_path / "runs").iterdir())


def test_malformed_websocket_messages_get_readable_errors(cfg, tmp_path):
    with client(cfg, tmp_path).websocket_connect("/api/live") as ws:
        for bad in ("not json", "[1, 2]", '{"type": "dance"}'):
            ws.send_text(bad)
            reply = ws.receive_json()
            assert reply["type"] == "error" and reply["error"]
        ws.send_bytes(b"\xff\xd8 not an image")
        assert "start" in ws.receive_json()["error"]
        ws.send_json({"type": "start", "bed_polygon": "nonsense"})
        assert ws.receive_json()["type"] == "error"


def make_job(runs, name, age_days, **extra):
    folder = runs / name
    folder.mkdir(parents=True)
    (folder / "job.json").write_text(json.dumps({"id": name, "created": time.time() - age_days * 86400, **extra}))
    return folder


def test_cleanup_deletes_only_old_finished_jobs_it_created(tmp_path):
    runs = tmp_path / "runs"
    old = make_job(runs, "20260101-010101-aaaaaa", 30)
    recent = make_job(runs, "20260901-010101-bbbbbb", 1)
    running = make_job(runs, "20260102-010101-cccccc", 30)
    (runs / "my-notes").mkdir()
    (runs / "my-notes" / "job.json").write_text(json.dumps({"created": 0}))
    assert cleanup_runs(runs, 7, active={running.name}, dry_run=True) == [old.name] and old.exists()
    assert cleanup_runs(runs, 7, active={running.name}) == [old.name]
    assert not old.exists() and recent.exists() and running.exists() and (runs / "my-notes").exists()


def test_retention_runs_at_start_up_only_when_configured(cfg, tmp_path):
    old = make_job(tmp_path / "runs", "20260101-010101-aaaaaa", 30)
    client(cfg, tmp_path)
    assert old.exists()
    client(cfg, tmp_path, retention_days=7)
    assert not old.exists()
