import hashlib
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from itertools import accumulate
from pathlib import Path

from . import __version__
from .agent import ContextAgent
from .config import config_hash, validate_config
from .events import detect_bed_events, guard_verifier
from .features import Scene, add_box_speed, add_speed, measure
from .policy import AlertPolicy
from .reporting import SCHEMA_VERSION, read_observations, summarize, write_observations, write_outputs, write_overlay
from .schemas import UNKNOWN, Observation
from .temporal import bed_timeline, build_timeline, labels_at, propose_all
from .video import VideoReader, file_hash
from .vision import PoseTracker, TargetSelector
from .vlm import QwenVLM

PERCEPTION_KEYS = ("scene", "target", "sampling", "vision", "identity")
PERCEPTION_SOURCES = ("video.py", "vision.py", "features.py")
PERCEPTION_PACKAGES = ("ultralytics", "torch", "opencv-python", "lap", "numpy")
PACKAGES = ("ultralytics", "torch", "transformers", "opencv-python", "numpy", "scipy", "lap")
ROOT = Path(__file__).resolve().parents[2]


def versions():
    out = {"python": platform.python_version()}
    for pkg in PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return out


def source_state():
    """Git commit of this checkout, and whether tracked files (all, and under src/) differ from it.
    None when the package is not run from a git checkout."""
    def git(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=20,
                              check=True).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
                "src_dirty": bool(git("status", "--porcelain", "--untracked-files=no", "--", "src"))}
    except (OSError, subprocess.SubprocessError):
        return None


def perception_key(cfg):
    """Hash of what cached observations depend on: the perception settings, the contents of the pose weights and
    the tracker config, the perception source files and the libraries that compute keypoints and tracks. Analysis
    settings are not part of it. None while the weights are not on disk, so nothing unverifiable is reused."""
    weights, tracker = Path(cfg["vision"]["model"]), Path(cfg["vision"]["tracker"])
    if not weights.is_file():
        return None
    key = {k: cfg[k] for k in PERCEPTION_KEYS}
    key["vision"] = {**cfg["vision"], "model": file_hash(weights),
                     "tracker": tracker.read_text(encoding="utf-8") if tracker.exists() else tracker.name}
    code = "".join((Path(__file__).parent / f).read_text(encoding="utf-8") for f in PERCEPTION_SOURCES)
    key |= {"speed": cfg["posture"]["speed_window_sec"], "schema": SCHEMA_VERSION,
            "code": hashlib.sha256(code.encode()).hexdigest(),
            "packages": {k: v for k, v in versions().items() if k in PERCEPTION_PACKAGES}}
    return config_hash(key)


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


def view_counts(observations, cfg):
    n = max(round(10 * cfg["sampling"]["fps"]), 1)
    low = [o.reason == "low_keypoints" for o in observations]
    c = [0, *accumulate(low)]
    worst = max(c[min(i + n, len(low))] - c[i] for i in range(max(len(low) - n, 0) + 1)) / n
    return {"low": sum(low), "detected": sum(o.bbox is not None or o.reason == "low_keypoints" for o in observations),
            "missing": sum(o.reason == "no_frame" for o in observations), "samples": len(observations), "worst": worst}


def view_quality(observations, cfg, prior=None):
    """Camera-placement check: share of detected samples with too few keypoints (and the worst 10 s window),
    and share of sampling slots with no frame at all (a live stream slower than sampling.fps). `prior` adds the
    counts of samples a live session has already archived."""
    v = view_counts(observations, cfg)
    if prior:
        v = {k: max(v[k], prior[k]) if k == "worst" else v[k] + prior[k] for k in v}
    share, missing = v["low"] / max(v["detected"], 1), v["missing"] / max(v["samples"], 1)
    limit = cfg["policy"]["degraded_view_share"]
    return {"worst_low_keypoint_share_10s": round(v["worst"], 2), "low_keypoint_share": round(share, 2),
            "missing_frame_share": round(missing, 2), "degraded": share >= limit or missing >= limit}


def run_analysis(observations, duration, cfg, use_agent=True, frames=None, scene=None, vlm_factory=None, start=0.0,
                 init=None, snapshot_at=None):
    """Proposals, agent review, committed timelines, bed events and decisions for one set of observations.

    `start` and `init` resume from a quiet state (live windows); `snapshot_at` reports the state machine and
    policy timers at that time so a later window can resume from it."""
    init = init or {}
    add_box_speed(observations, cfg["posture"]["box_speed_window_sec"], cfg["posture"]["speed_window_sec"])
    rules = propose_all(observations, cfg, init.get("prev"))
    proposals, evidence = rules, [p.label for p in rules]
    agent = None
    if use_agent and cfg["agent"]["enabled"]:
        agent = ContextAgent(cfg, observations, rules, duration, frames, scene, vlm_factory)
        proposals, evidence = agent.review(), agent.evidence

    activity = build_timeline(proposals, duration, cfg, start, init.get("state", UNKNOWN))
    bed = bed_timeline(activity)
    times = [p.t for p in proposals]
    act_labels, bed_labels = labels_at(activity, times), labels_at(bed, times)

    verify = agent.verify_exit if agent else guard_verifier(observations, cfg)
    fsm = detect_bed_events(times, act_labels, bed_labels, evidence, [p.confidence for p in proposals],
                            bed, duration, cfg, verify, init.get("fsm"), snapshot_at)
    if agent:
        agent.log_exit_candidates(fsm)
    policy = AlertPolicy(cfg)
    decisions, alerts, timers = policy.run(times, act_labels, bed_labels, observations, fsm, duration, start,
                                           init.get("policy"), snapshot_at)
    for e in fsm["events"]:
        e.decision = policy.event_decision(e, decisions)

    result = {
        "duration": duration, "observations": observations, "rules": rules, "proposals": proposals,
        "evidence": evidence,
        "activity": activity, "bed": bed, "events": fsm["events"], "fsm": fsm,
        "decisions": decisions, "alerts": alerts,
        "trace": agent.trace if agent else [], "vlm_calls": agent.vlm_calls if agent else 0,
        "vlm_revision": getattr(agent.vlm, "revision", None) if agent else None,
        "vlm_load_sec": round(agent.vlm_load_sec, 1) if agent else 0.0,
        "vlm_inference_sec": round(agent.vlm_inference_sec, 1) if agent else 0.0,
        "view": view_quality(observations, cfg),
    }
    if snapshot_at is not None:
        result["snapshot"] = {"fsm": fsm["snapshot"], "policy": timers}
    result["summary"] = summarize(result)
    return result


def analyze(video, cfg, out_dir, use_agent=True, use_vlm=True, reuse=False, overlay=False, observations_path=None,
            progress=None):
    validate_config(cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reader = VideoReader(video)
    cfg = {**cfg, "sampling": {**cfg["sampling"], "fps": min(cfg["sampling"]["fps"], reader.fps)}}
    scene = Scene(cfg, reader.width, reader.height)
    started = time.time()
    video_hash = file_hash(video)
    tracker = Path(cfg["vision"]["tracker"])
    p_hash = perception_key(cfg)
    cache = Path(observations_path) if observations_path else out / "observations.jsonl"
    if observations_path and not cache.exists():
        raise FileNotFoundError(f"--observations: {cache} not found")
    meta = None
    if (reuse or observations_path) and cache.exists():
        if p_hash:
            meta, observations = read_observations(cache, {"video_hash": video_hash, "perception_hash": p_hash})
        if meta is None:
            print(f"perception: {cache} does not match this video, perception config, weights or code; re-running")
    reused = meta is not None
    if meta is None:
        print(f"perception: {video} ({reader.width}x{reader.height}, {reader.fps:.1f} fps)")
        observations, revision = perceive(reader, scene, cfg, progress)
        meta = {"video_hash": video_hash, "perception_hash": perception_key(cfg), "duration": reader.duration,
                "width": reader.width, "height": reader.height, "pose_model": revision,
                "perception_sec": round(time.time() - started, 1),
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": source_state(),
                "versions": versions()}
        cache = out / "observations.jsonl"
        write_observations(cache, meta, observations)
    else:
        print(f"perception: reusing {cache}")

    use_agent = use_agent and cfg["agent"]["enabled"]
    use_vlm = use_agent and use_vlm and cfg["vlm"]["enabled"]
    analysis_started = time.time()
    result = run_analysis(observations, meta["duration"], cfg, use_agent, reader.frame_at, scene,
                          (lambda: QwenVLM(cfg)) if use_vlm else None)
    analysis_sec = time.time() - analysis_started
    if overlay:
        write_overlay(reader, scene, result, cfg, out / "overlay.mp4")
    portable = {**cfg, "vision": {**cfg["vision"], "tracker": tracker.name}}
    manifest = {
        "schema_version": SCHEMA_VERSION, "code_version": __version__, "mode": "offline",
        "video": str(video), "video_hash": video_hash, "duration_sec": round(meta["duration"], 2),
        "config_hash": config_hash(portable), "pose_model": meta["pose_model"],
        "vlm_model": cfg["vlm"]["model"] if use_vlm else None, "vlm_revision": result["vlm_revision"],
        "agent_enabled": use_agent, "vlm_calls": result["vlm_calls"],
        "sampling_fps": cfg["sampling"]["fps"], "samples": len(observations),
        "perception_sec": meta.get("perception_sec"), "reused_observations": str(cache) if reused else None,
        "perception": {k: meta.get(k) for k in ("created", "source", "versions", "pose_model", "perception_sec",
                                                "perception_hash")},
        "source": source_state(),
        "analysis_sec": round(analysis_sec, 1), "vlm_load_sec": result["vlm_load_sec"],
        "vlm_inference_sec": result["vlm_inference_sec"], "total_runtime_sec": round(time.time() - started, 1),
        "platform": platform.platform(), "versions": versions(), "config": portable,
    }
    write_outputs(out, result, manifest)
    return result
