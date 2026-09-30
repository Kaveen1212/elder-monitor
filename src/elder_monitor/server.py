import asyncio
import hmac
import json
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import deep_merge, validate_config
from .live import LiveSession, chat_messages, message, pose
from .pipeline import analyze
from .reporting import event_record

RUN_KEYS = ("duration_sec", "samples", "sampling_fps", "perception_sec", "analysis_sec", "vlm_load_sec",
            "vlm_inference_sec", "total_runtime_sec", "pose_model", "vlm_model", "vlm_revision", "vlm_calls",
            "agent_enabled")
LOCAL_ORIGINS = ("http://127.0.0.1:8000", "http://localhost:8000")
CHUNK = 1 << 20


def scene_config(cfg, bed_polygon):
    out = deep_merge(cfg, {"scene": {"bed_polygon": bed_polygon}}) if bed_polygon else cfg
    validate_config(out)
    return out


def result_payload(result, cfg, manifest):
    def seg(s, **extra):
        return {"start": round(s.start, 2), "end": round(s.end, 2), "label": s.label, **extra}

    return {
        "summary": result["summary"],
        "scene": {k: cfg["scene"][k] for k in ("bed_polygon", "chair_polygons")},
        "activity": [seg(s, confidence=s.confidence) for s in result["activity"]],
        "bed": [seg(s) for s in result["bed"]],
        "decisions": [seg(s, reasons=s.reasons) for s in result["decisions"]],
        "events": [event_record(e) for e in result["events"]],
        "alerts": [{**a, "start_sec": round(a["start_sec"], 2), "end_sec": round(a["end_sec"], 2)}
                   for a in result["alerts"]],
        "messages": chat_messages(result, cfg),
        "frames": [{"t": round(o.t, 2), **pose(o)} for o in result["observations"]],
        "run": {k: manifest.get(k) for k in RUN_KEYS},
    }


def is_job_id(name):
    return len(name) == 22 and name[8] == "-" and name[15] == "-" and name.replace("-", "").isalnum()


def cleanup_runs(runs_dir, older_than_days, active=(), dry_run=False):
    """Delete finished upload jobs created more than `older_than_days` ago; returns their folder names.
    Only folders this server made (a job id name with a job.json) are touched, never a queued or running job."""
    cutoff, removed = time.time() - older_than_days * 86400, []
    for folder in sorted(Path(runs_dir).iterdir()) if Path(runs_dir).is_dir() else []:
        info = folder / "job.json"
        if not folder.is_dir() or not is_job_id(folder.name) or not info.is_file() or folder.name in active:
            continue
        try:
            created = float(json.loads(info.read_text(encoding="utf-8"))["created"])
        except (ValueError, KeyError, TypeError):
            continue
        if created < cutoff:
            removed.append(folder.name)
            if not dry_run:
                shutil.rmtree(folder)
    return removed


def create_app(cfg, runs_dir="runs", frontend=None, origins=LOCAL_ORIGINS, max_upload_mb=500, api_key=None,
               retention_days=None):
    """HTTP + WebSocket API. Browsers may call it from `origins` only; with `api_key` every /api request needs it
    (header X-API-Key, or ?api_key= where headers cannot be set, as for the WebSocket). Uploads above
    `max_upload_mb` are refused. With `retention_days`, finished jobs older than that are deleted at start-up and
    whenever a new job is created."""
    app = FastAPI(title="elder-monitor")
    runs, limit = Path(runs_dir), int(max_upload_mb * CHUNK)
    runs.mkdir(parents=True, exist_ok=True)
    jobs, worker = {}, ThreadPoolExecutor(max_workers=1)

    def authorised(conn):
        given = conn.headers.get("x-api-key") or conn.query_params.get("api_key") or ""
        return not api_key or hmac.compare_digest(given.encode(), api_key.encode())

    def prune():
        if retention_days is not None:
            active = {k for k, v in jobs.items() if v["status"] in ("queued", "running")}
            cleanup_runs(runs, retention_days, active)

    @app.middleware("http")
    async def guard(request, call_next):
        if request.url.path.startswith("/api/") and request.method != "OPTIONS":
            if not authorised(request):
                return JSONResponse({"detail": "missing or wrong API key"}, status_code=401)
            size = request.headers.get("content-length")
            if request.method == "POST" and size and size.isdigit() and int(size) > limit + CHUNK:
                return JSONResponse({"detail": f"upload larger than {max_upload_mb} MB"}, status_code=413)
        return await call_next(request)

    app.add_middleware(CORSMiddleware, allow_origins=list(origins), allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type", "X-API-Key"])

    def job_info(job_id):
        folder = runs / job_id
        if not is_job_id(job_id) or not (folder / "job.json").is_file():
            return None
        info = json.loads((folder / "job.json").read_text(encoding="utf-8"))
        default = {"status": "done" if (folder / "result.json").exists() else "interrupted", "progress": 1.0}
        return {**info, **jobs.get(job_id, default)}

    def run_job(job_id, video, job_cfg, use_vlm):
        state = jobs[job_id]
        state["status"] = "running"
        try:
            out = runs / job_id
            result = analyze(video, job_cfg, out, use_vlm=use_vlm,
                             progress=lambda p: state.update(progress=round(p, 3)))
            manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
            payload = result_payload(result, job_cfg, manifest)
            (out / "result.json").write_text(json.dumps(payload), encoding="utf-8")
            state.update(status="done", progress=1.0)
        except Exception as exc:
            state.update(status="error", error=str(exc))

    @app.get("/api/health")
    def health():
        return {"ok": True, "fps": cfg["sampling"]["fps"], "bed_polygon": cfg["scene"]["bed_polygon"],
                "vlm": cfg["vlm"]["enabled"]}

    @app.post("/api/jobs")
    def create_job(file: UploadFile = File(...), bed_polygon: str = Form(""), use_vlm: bool = Form(False)):
        try:
            job_cfg = scene_config(cfg, json.loads(bed_polygon) if bed_polygon else None)
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, f"bad bed_polygon: {exc}")
        prune()
        job_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        folder = runs / job_id
        folder.mkdir()
        video = folder / f"input{Path(file.filename or '').suffix.lower() or '.mp4'}"
        size = 0
        with open(video, "wb") as f:
            while chunk := file.file.read(CHUNK):
                size += len(chunk)
                if size > limit:
                    break
                f.write(chunk)
        if size > limit:
            shutil.rmtree(folder)
            raise HTTPException(413, f"upload larger than {max_upload_mb} MB")
        meta = {"id": job_id, "filename": file.filename, "video": video.name, "created": time.time(), "use_vlm": use_vlm}
        (folder / "job.json").write_text(json.dumps(meta), encoding="utf-8")
        jobs[job_id] = {"status": "queued", "progress": 0.0, "error": None}
        worker.submit(run_job, job_id, video, job_cfg, use_vlm)
        return job_info(job_id)

    @app.get("/api/jobs")
    def list_jobs():
        infos = [job_info(p.name) for p in runs.iterdir() if p.is_dir()]
        return sorted([i for i in infos if i], key=lambda i: i["created"], reverse=True)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        info = job_info(job_id)
        if info is None:
            raise HTTPException(404, "job not found")
        result = runs / job_id / "result.json"
        if info["status"] == "done" and result.exists():
            info["result"] = json.loads(result.read_text(encoding="utf-8"))
        return info

    @app.get("/api/jobs/{job_id}/video")
    def job_video(job_id: str):
        info = job_info(job_id)
        if info is None:
            raise HTTPException(404, "job not found")
        return FileResponse(runs / job_id / info["video"])

    @app.websocket("/api/live")
    async def live(ws: WebSocket):
        if not authorised(ws):
            await ws.close(code=1008)
            return
        await ws.accept()
        session, started = None, 0.0

        async def error(text):
            await ws.send_json({"type": "error", "error": text})

        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    if session is None:
                        await error('send {"type": "start"} before frames')
                        continue
                    frame = cv2.imdecode(np.frombuffer(msg["bytes"], np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        await ws.send_json({"type": "frame", "skipped": True, "error": "not a decodable image"})
                    else:
                        await ws.send_json(await asyncio.to_thread(session.push, frame, time.monotonic() - started))
                    continue
                try:
                    data = json.loads(msg.get("text") or "")
                except json.JSONDecodeError:
                    await error("messages must be JSON objects")
                    continue
                kind = data.get("type") if isinstance(data, dict) else None
                if kind == "start":
                    try:
                        live_cfg = scene_config(cfg, data.get("bed_polygon"))
                    except (ValueError, TypeError) as exc:
                        await error(f"bad bed_polygon: {exc}")
                        continue
                    session = await asyncio.to_thread(LiveSession, live_cfg)
                    started = time.monotonic()
                    await ws.send_json({"type": "ready", "fps": live_cfg["sampling"]["fps"],
                                        "bed_polygon": live_cfg["scene"]["bed_polygon"]})
                elif kind == "ask":
                    reply = session.answer(str(data.get("text", ""))) if session else \
                        message(0.0, "answer", "Agent", "Start the camera first.")
                    await ws.send_json({"type": "answer", "message": reply})
                else:
                    await error('unknown message; expected {"type": "start"} or {"type": "ask"}')
        except WebSocketDisconnect:
            pass

    prune()
    if frontend:
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app
