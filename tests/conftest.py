import pytest

from elder_monitor.config import load_config
from elder_monitor.schemas import Observation

FPS = 5
POSES = {
    "lying": (85.0, None),
    "sitting": (10.0, 90.0),
    "standing": (5.0, 175.0),
    "walking": (5.0, 175.0),
}


@pytest.fixture
def cfg():
    c = load_config()
    c["scene"]["bed_polygon"] = [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]]
    c["vlm"]["enabled"] = False
    return c


def make_obs(t, pose="lying", where="bed", visible=True, identity_ok=True, edge=False, truncated=False):
    """where: bed | near (beside the bed) | away | edge_outside (just past the outline, body partly on it)"""
    if not visible or not identity_ok:
        return Observation(t=t, visible=False, identity_ok=identity_ok,
                           reason="not_detected" if identity_ok else "identity_uncertain")
    angle, knee = POSES[pose]
    in_bed = where == "bed"
    return Observation(
        t=t, visible=True, identity_ok=True, track_id=1, det_conf=0.9, kp_conf=0.9, bbox=[400, 300, 600, 700],
        truncated=truncated, torso_angle=angle, torso_len=100.0, knee_angle=knee,
        thigh_ratio=None if knee is None else 1.0, leg_angle=None if knee is None else 3.0, anchor=[500.0, 500.0],
        body_in_bed=0.9 if in_bed else (0.3 if where == "edge_outside" else 0.0), hip_in_bed=in_bed,
        feet_in_bed=in_bed and pose not in ("standing", "walking"), bed_dist=50.0 if in_bed else -200.0,
        near_bed=where in ("bed", "near", "edge_outside"), edge_of_bed=edge,
        speed=1.0 if pose == "walking" else 0.0,
    )


def script(*parts):
    """parts: (seconds, kwargs for make_obs). Returns (observations, duration)."""
    obs, t = [], 0.0
    for seconds, kw in parts:
        for _ in range(round(seconds * FPS)):
            obs.append(make_obs(round(t, 4), **kw))
            t += 1.0 / FPS
    return obs, round(t, 4)
