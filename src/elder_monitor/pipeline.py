import platform
import time
from itertools import accumulate
from pathlib import Path

from . import __version__
from .agent import ContextAgent
from .config import config_hash, validate_scene
from .events import detect_bed_events, guard_verifier
from .features import Scene, add_box_speed, add_speed, measure
from .policy import AlertPolicy
from .reporting import SCHEMA_VERSION, read_observations, summarize, write_observations, write_outputs, write_overlay
from .schemas import Observation
from .temporal import bed_timeline, build_timeline, labels_at, propose_all
from .video import VideoReader, file_hash
from .vision import PoseTracker, TargetSelector
from .vlm import QwenVLM

PERCEPTION_KEYS = ("scene", "target", "sampling", "vision", "identity")


def perceive(reader, scene, cfg, progress=None):
    tracker, selector = PoseTracker(cfg), TargetSelector(cfg, scene)
    observations, started = [], time.time()
    for t, idx, frame in reader.sample(cfg["sampling"]["fps"]):
        if frame is None:
            observations.append(Observation(t, idx, reason="no_frame"))
            continue
        persons = tracker(frame)
        person, identity_ok, reason = selector.select(t, frame, persons)
        o = measure(t, idx, person, scene, cfg) if person else Observation(t, idx, reason=reason)
        o.identity_ok, o.n_persons = identity_ok, len(persons)
        observations.append(o)
        if progress and reader.duration:
            progress(min(t / reader.duration, 1.0))
        if len(observations) % 500 == 0:
            print(f"  perceived {t:7.1f}s of video ({len(observations) / (time.time() - started):.1f} samples/s)")
    add_speed(observations, cfg["posture"]["speed_window_sec"])
    return observations, tracker.revision


def view_quality(observations, cfg):
    """Camera-placement check: share of detected samples with too few keypoints (and the worst 10 s window)."""
    n = max(round(10 * cfg["sampling"]["fps"]), 1)
    low = [o.reason == "low_keypoints" for o in observations]
    c = [0, *accumulate(low)]
    worst = max(c[min(i + n, len(low))] - c[i] for i in range(max(len(low) - n, 0) + 1)) / n
    share = sum(low) / max(sum(o.bbox is not None or o.reason == "low_keypoints" for o in observations), 1)
    return {"worst_low_keypoint_share_10s": round(worst, 2), "low_keypoint_share": round(share, 2),
            "degraded": share >= cfg["policy"]["degraded_view_share"]}


def run_analysis(observations, duration, cfg, use_agent=True, frames=None, scene=None, vlm_factory=None):
    add_box_speed(observations, cfg["posture"]["box_speed_window_sec"], cfg["posture"]["speed_window_sec"])
    proposals = propose_all(observations, cfg)
    agent = None
    if use_agent and cfg["agent"]["enabled"]:
        agent = ContextAgent(cfg, observations, proposals, duration, frames, scene, vlm_factory)
        proposals = agent.review()

    activity = build_timeline(proposals, duration, cfg)
    bed = bed_timeline(activity)
    times = [p.t for p in proposals]
    act_labels, bed_labels = labels_at(activity, times), labels_at(bed, times)

    verify = agent.verify_exit if agent else guard_verifier(observations, cfg)
    fsm = detect_bed_events(times, act_labels, bed_labels, [p.confidence for p in proposals],
                            bed, duration, cfg, verify)
    if agent:
        agent.log_exit_candidates(fsm)
    policy = AlertPolicy(cfg)
    decisions, alerts = policy.run(times, act_labels, bed_labels, observations, fsm, duration)
    for e in fsm["events"]:
        e.decision = policy.event_decision(e, decisions)

    result = {
        "duration": duration, "observations": observations, "proposals": proposals,
        "activity": activity, "bed": bed, "events": fsm["events"], "fsm": fsm,
        "decisions": decisions, "alerts": alerts,
        "trace": agent.trace if agent else [], "vlm_calls": agent.vlm_calls if agent else 0,
        "view": view_quality(observations, cfg),
    }
    result["summary"] = summarize(result)
    return result


def analyze(video, cfg, out_dir, use_agent=True, use_vlm=True, reuse=False, overlay=False, observations_path=None,
            progress=None):
    validate_scene(cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reader = VideoReader(video)
    cfg = {**cfg, "sampling": {**cfg["sampling"], "fps": min(cfg["sampling"]["fps"], reader.fps)}}
    scene = Scene(cfg, reader.width, reader.height)
    started = time.time()
    video_hash = file_hash(video)
    perception = {k: cfg[k] for k in PERCEPTION_KEYS}
    tracker = Path(cfg["vision"]["tracker"])
    perception["vision"] = {**cfg["vision"], "tracker": tracker.read_text(encoding="utf-8") if tracker.exists()
                            else tracker.name}
    p_hash = config_hash(perception | {"speed": cfg["posture"]["speed_window_sec"], "schema": SCHEMA_VERSION})
    cache = Path(observations_path) if observations_path else out / "observations.jsonl"
    if observations_path and not cache.exists():
        raise FileNotFoundError(f"--observations: {cache} not found")
    meta = None
    if (reuse or observations_path) and cache.exists():
        meta, observations = read_observations(cache, {"video_hash": video_hash, "perception_hash": p_hash})
        if meta is None:
            print(f"perception: {cache} does not match this video / perception config; re-running")
    reused = meta is not None
    if meta is None:
        print(f"perception: {video} ({reader.width}x{reader.height}, {reader.fps:.1f} fps)")
        observations, revision = perceive(reader, scene, cfg, progress)
        meta = {"video_hash": video_hash, "perception_hash": p_hash, "duration": reader.duration,
                "width": reader.width, "height": reader.height, "pose_model": revision,
                "perception_sec": round(time.time() - started, 1)}
        cache = out / "observations.jsonl"
        write_observations(cache, meta, observations)
    else:
        print(f"perception: reusing {cache}")

    use_agent = use_agent and cfg["agent"]["enabled"]
    use_vlm = use_agent and use_vlm and cfg["vlm"]["enabled"]
    result = run_analysis(observations, meta["duration"], cfg, use_agent, reader.frame_at, scene,
                          (lambda: QwenVLM(cfg)) if use_vlm else None)
    if overlay:
        write_overlay(reader, scene, result, cfg, out / "overlay.mp4")
    portable = {**cfg, "vision": {**cfg["vision"], "tracker": tracker.name}}
    manifest = {
        "schema_version": SCHEMA_VERSION, "code_version": __version__, "mode": "offline",
        "video": str(video), "video_hash": video_hash, "duration_sec": round(meta["duration"], 2),
        "config_hash": config_hash(portable), "pose_model": meta["pose_model"],
        "vlm_model": cfg["vlm"]["model"] if use_vlm else None,
        "agent_enabled": use_agent, "vlm_calls": result["vlm_calls"],
        "sampling_fps": cfg["sampling"]["fps"], "samples": len(observations),
        "perception_sec": meta.get("perception_sec"), "reused_observations": str(cache) if reused else None,
        "total_runtime_sec": round(time.time() - started, 1), "platform": platform.platform(), "config": portable,
    }
    write_outputs(out, result, manifest)
    return result
