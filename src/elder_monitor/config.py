import copy
import hashlib
import json
import math
from pathlib import Path

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_DIR / "default.yaml"


def deep_merge(base, override):
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path=None, overrides=None):
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if path:
        cfg = deep_merge(cfg, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    cfg = deep_merge(cfg, overrides)
    tracker = Path(cfg["vision"]["tracker"])
    if not tracker.exists() and (PACKAGE_DIR / tracker).exists():
        cfg["vision"]["tracker"] = str(PACKAGE_DIR / tracker)
    return cfg


FRACTIONS = ["scene.ignore_share", "scene.near_bed_margin", "scene.edge_margin", "vision.det_conf", "vision.kp_conf",
             "identity.min_det_conf", "identity.min_similarity", "identity.margin", "identity.track_min_similarity",
             "posture.in_bed_fraction", "posture.box_motion_conf", "agent.upright_hips_share", "policy.edge_fraction",
             "policy.degraded_view_share"]
POSITIVE = ["sampling.fps", "identity.max_jump", "identity.jump_per_sec", "events.exit_persist_sec",
            "events.left_view_sec", "events.exit_dwell_sec", "events.lookahead_sec", "events.return_persist_sec",
            "agent.lookback_sec", "agent.lookahead_sec", "policy.edge_sit_sec", "policy.max_out_of_bed_sec",
            "policy.max_absence_sec"]
NON_NEGATIVE = ["target.time_sec", "agent.min_gap_sec", "agent.max_bridge_sec"]
COUNTS = {"temporal.smooth_samples": 1, "vision.min_body_keypoints": 1, "agent.max_rounds": 1, "agent.vlm_frames": 1,
          "vlm.max_calls": 0, "vlm.max_side": 28, "vlm.max_new_tokens": 1}


def _get(cfg, key):
    for k in key.split("."):
        cfg = cfg[k]
    return cfg


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _point(p):
    return isinstance(p, (list, tuple)) and len(p) == 2 and all(_number(v) and 0 <= v <= 1 for v in p)


def _crosses(a, b, c, d):
    def side(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return side(a, b, c) * side(a, b, d) < 0 and side(c, d, a) * side(c, d, b) < 0


def _polygon(pts, name):
    if not isinstance(pts, (list, tuple)) or len(pts) < 3 or not all(_point(p) for p in pts):
        raise ValueError(f"{name} needs at least 3 [x, y] points with normalised coordinates in [0, 1]")
    n = len(pts)
    area = sum(pts[k][0] * pts[(k + 1) % n][1] - pts[(k + 1) % n][0] * pts[k][1] for k in range(n)) / 2
    if abs(area) < 1e-4:
        raise ValueError(f"{name} has (almost) zero area; mark the corners of the region")
    edges = [(pts[k], pts[(k + 1) % n]) for k in range(n)]
    if any(_crosses(*edges[a], *edges[b]) for a in range(n) for b in range(a + 2, n) if (a, b) != (0, n - 1)):
        raise ValueError(f"{name} crosses itself; mark the corners in order around the region")


def validate_config(cfg):
    """Readable errors for settings that would make the analysis meaningless, raised before any video is read."""
    s = cfg["scene"]
    if not s.get("bed_polygon"):
        raise ValueError("scene.bed_polygon is required; run `calibrate` first")
    _polygon(s["bed_polygon"], "scene.bed_polygon")
    for key in ("chair_polygons", "ignore_polygons"):
        for k, poly in enumerate(s.get(key) or []):
            _polygon(poly, f"scene.{key}[{k}]")
    point = cfg["target"].get("point")
    if point is not None and not _point(point):
        raise ValueError("target.point must be null or a normalised [x, y] in [0, 1]")
    for keys, ok, what in ((FRACTIONS, lambda v: 0 <= v <= 1, "between 0 and 1"),
                           (POSITIVE, lambda v: v > 0, "a positive number"),
                           (NON_NEGATIVE, lambda v: v >= 0, "zero or more")):
        for key in keys:
            v = _get(cfg, key)
            if not (_number(v) and ok(v)):
                raise ValueError(f"{key} must be {what}, got {v!r}")
    for key, low in COUNTS.items():
        v = _get(cfg, key)
        if not (isinstance(v, int) and not isinstance(v, bool) and v >= low):
            raise ValueError(f"{key} must be a whole number >= {low}, got {v!r}")
    dwell = cfg["temporal"]["dwell_sec"]
    if "default" not in dwell or not all(_number(v) and v >= 0 for v in dwell.values()):
        raise ValueError("temporal.dwell_sec needs a default and durations of zero or more seconds")
    p = cfg["posture"]
    if not all(_number(v) and v > 0 for v in p.values()):
        raise ValueError("posture thresholds must be positive numbers")
    if not p["lying_exit_deg"] <= p["lying_enter_deg"] <= 90:
        raise ValueError("posture needs lying_exit_deg <= lying_enter_deg <= 90")
    if not isinstance(cfg["vlm"].get("revision") or "", str):
        raise ValueError("vlm.revision must be null or a Hugging Face commit / tag string")
    if cfg["events"]["return_rule"] not in ("lying", "sitting"):
        raise ValueError("events.return_rule must be 'lying' or 'sitting'")
    if not set(cfg["policy"]["event_decision"].values()) <= {"NORMAL", "MONITOR", "ALERT"}:
        raise ValueError("policy.event_decision values must be NORMAL, MONITOR or ALERT")


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
