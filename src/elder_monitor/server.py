import asyncio
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
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import deep_merge, validate_config
from .live import LiveSession, chat_messages, message, pose
from .pipeline import analyze
from .reporting import event_record

RUN_KEYS = ("duration_sec", "samples", "sampling_fps", "perception_sec", "analysis_sec", "vlm_load_sec",
            "vlm_inference_sec", "total_runtime_sec", "pose_model", "vlm_model", "vlm_revision", "vlm_calls",
            "agent_enabled")


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


def create_app(cfg, runs_dir="runs", frontend=None):
    app = FastAPI(title="elder-monitor")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    runs = Path(runs_dir)
    runs.mkdir(parents=True, exist_ok=True)
    jobs, worker = {}, ThreadPoolExecutor(max_workers=1)

    def job_info(job_id):
        folder = runs / job_id
        if not job_id.replace("-", "").isalnum() or not (folder / "job.json").is_file():
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
            raise HTTPException(400, str(exc))
        job_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        folder = runs / job_id
        folder.mkdir()
        video = folder / f"input{Path(file.filename or '').suffix.lower() or '.mp4'}"
        with open(video, "wb") as f:
            shutil.copyfileobj(file.file, f)
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
        await ws.accept()
        session, started = None, 0.0
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    frame = cv2.imdecode(np.frombuffer(msg["bytes"], np.uint8), cv2.IMREAD_COLOR) if session else None
                    if frame is None:
                        await ws.send_json({"type": "frame", "skipped": True})
                    else:
                        await ws.send_json(await asyncio.to_thread(session.push, frame, time.monotonic() - started))
                    continue
                data = json.loads(msg.get("text") or "{}")
                if data.get("type") == "start":
                    try:
                        live_cfg = scene_config(cfg, data.get("bed_polygon"))
                    except (ValueError, TypeError) as exc:
                        await ws.send_json({"type": "error", "error": str(exc)})
                        continue
                    session = await asyncio.to_thread(LiveSession, live_cfg)
                    started = time.monotonic()
                    await ws.send_json({"type": "ready", "fps": live_cfg["sampling"]["fps"],
                                        "bed_polygon": live_cfg["scene"]["bed_polygon"]})
                elif data.get("type") == "ask":
                    reply = session.answer(data.get("text", "")) if session else \
                        message(0.0, "answer", "Agent", "Start the camera first.")
                    await ws.send_json({"type": "answer", "message": reply})
        except WebSocketDisconnect:
            pass

    if frontend:
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app
