import math
from collections import deque

import cv2
import numpy as np

from .schemas import Observation

L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 5, 6, 11, 12, 13, 14, 15, 16
BODY = list(range(5, 17))
LEGS = ((L_HIP, L_KNEE, L_ANK), (R_HIP, R_KNEE, R_ANK))


class Scene:
    def __init__(self, cfg, width, height):
        s = cfg["scene"]
        self.width, self.height = width, height
        self.bed = self._poly(s["bed_polygon"])
        self.chairs = [self._poly(p) for p in s.get("chair_polygons") or []]
        self.ignore_share, self.ignore_mask = s.get("ignore_share", 0.8), None
        if s.get("ignore_polygons"):
            self.ignore_mask = np.zeros((height, width), np.uint8)
            cv2.fillPoly(self.ignore_mask, [self._poly(p).astype(np.int32) for p in s["ignore_polygons"]], 1)
        self.near_margin = s["near_bed_margin"] * height
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [self.bed.astype(np.int32)], 1)
        self.inside = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        self.edge_margin = float(min(s["edge_margin"] * height, 0.4 * self.inside.max()))

    def _poly(self, pts):
        return np.array([[x * self.width, y * self.height] for x, y in pts], np.float32)

    def bed_dist(self, pt):
        return cv2.pointPolygonTest(self.bed, (float(pt[0]), float(pt[1])), True)

    def in_bed(self, pt):
        return self.bed_dist(pt) >= 0

    def edge_dist(self, pt):
        """Distance to the bed outline, ignoring outline sides that lie on the frame border."""
        d = self.bed_dist(pt)
        if d < 0:
            return -d
        x, y = min(max(int(pt[0]), 0), self.width - 1), min(max(int(pt[1]), 0), self.height - 1)
        return float(self.inside[y, x])

    def in_chair(self, pt):
        return any(cv2.pointPolygonTest(c, (float(pt[0]), float(pt[1])), False) >= 0 for c in self.chairs)

    def ignored(self, bbox):
        """A detection lying mostly inside an ignore polygon (mirror, TV screen)."""
        if self.ignore_mask is None:
            return False
        x1, y1 = max(int(bbox[0]), 0), max(int(bbox[1]), 0)
        x2, y2 = min(int(bbox[2]), self.width), min(int(bbox[3]), self.height)
        return x2 > x1 and y2 > y1 and self.ignore_mask[y1:y2, x1:x2].mean() >= self.ignore_share

    def bed_box(self):
        x, y, w, h = cv2.boundingRect(self.bed)
        return [x, y, x + w, y + h]


def _mid(kp, ok, a, b):
    pts = [kp[i, :2] for i in (a, b) if ok[i]]
    return np.mean(pts, axis=0) if pts else None


def _angle_from_vertical(top, bottom):
    dx, dy = top[0] - bottom[0], top[1] - bottom[1]
    return math.degrees(math.atan2(abs(dx), -dy))


def _mean(values):
    return float(np.mean(values)) if values else None


def measure(t, frame_idx, person, scene, cfg):
    """Raw posture and bed-geometry evidence for one detected person; posture thresholds are applied in propose()."""
    v = cfg["vision"]
    kp = np.asarray(person.keypoints, dtype=float).reshape(17, 3)
    ok = kp[:, 2] >= v["kp_conf"]
    x1, y1, x2, y2 = (float(c) for c in person.bbox)
    obs = Observation(t=t, frame_idx=frame_idx, track_id=person.track_id, det_conf=round(person.conf, 3),
                      bbox=[x1, y1, x2, y2], keypoints=kp.round(2).tolist(), kp_conf=float(kp[BODY, 2].mean()),
                      truncated=x1 <= 2 or y1 <= 2 or x2 >= scene.width - 2 or y2 >= scene.height - 2)
    if int(ok[BODY].sum()) < v["min_body_keypoints"]:
        obs.reason = "low_keypoints"
        return obs
    obs.visible = True

    sh, hip = _mid(kp, ok, L_SH, R_SH), _mid(kp, ok, L_HIP, R_HIP)
    anchor = hip if hip is not None else np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    obs.anchor = [float(anchor[0]), float(anchor[1])]
    if sh is not None and hip is not None:
        obs.torso_len = float(math.dist(sh, hip))
        obs.torso_angle = _angle_from_vertical(sh, hip)
    if ok[L_SH] and ok[R_SH]:
        obs.shoulder_width = float(math.dist(kp[L_SH, :2], kp[R_SH, :2]))

    knee_angles, thigh_ratios, drops, leg_axes, thigh_axes = [], [], [], [], []
    for h, k, a in LEGS:
        if not (ok[h] and ok[k]):
            continue
        thigh_axes.append(_angle_from_vertical(kp[h, :2], kp[k, :2]))
        if obs.torso_len:
            drops.append((kp[k, 1] - kp[h, 1]) / obs.torso_len)
        if ok[a]:
            thigh, shin = kp[k, :2] - kp[h, :2], kp[a, :2] - kp[k, :2]
            cos = np.dot(-thigh, shin) / max(np.linalg.norm(thigh) * np.linalg.norm(shin), 1e-6)
            knee_angles.append(math.degrees(math.acos(np.clip(cos, -1.0, 1.0))))
            thigh_ratios.append(np.linalg.norm(thigh) / max(np.linalg.norm(shin), 1e-6))
            leg_axes.append(_angle_from_vertical(kp[h, :2], kp[a, :2]))
    obs.knee_angle, obs.thigh_ratio = _mean(knee_angles), _mean(thigh_ratios)
    obs.knee_drop, obs.leg_angle, obs.thigh_angle = _mean(drops), _mean(leg_axes), _mean(thigh_axes)

    pts = kp[ok][:, :2]
    obs.body_in_bed = float(np.mean([scene.in_bed(q) for q in pts]))
    obs.bed_dist = float(scene.bed_dist(anchor))
    obs.hip_in_bed = obs.bed_dist >= 0
    obs.near_bed = obs.bed_dist >= -scene.near_margin
    feet = [kp[i, :2] for i in (L_ANK, R_ANK) if ok[i]]
    obs.feet_in_bed = bool(feet) and all(scene.in_bed(f) for f in feet)
    lower = [kp[i, :2] for i in (L_KNEE, R_KNEE, L_ANK, R_ANK) if ok[i]]
    obs.edge_of_bed = scene.edge_dist(anchor) < scene.edge_margin or (
        obs.hip_in_bed and any(not scene.in_bed(q) for q in lower))
    obs.in_chair = scene.in_chair(anchor)
    return obs


def add_speed(observations, window):
    """Anchor displacement over ~window seconds, in torso lengths per second."""
    j = 0
    for i, o in enumerate(observations):
        if not o.visible:
            continue
        while observations[j].t < o.t - window:
            j += 1
        ref = next((q for q in observations[j:i] if q.visible and q.track_id == o.track_id), None)
        if ref is None or o.t <= ref.t:
            continue
        scale = max(o.torso_len or 0.0, ref.torso_len or 0.0) or 0.3 * (o.bbox[3] - o.bbox[1])
        o.speed = float(math.dist(o.anchor, ref.anchor) / (o.t - ref.t) / max(scale, 1.0))


def add_box_speed(observations, window, speed_window):
    """Box-centre speed of the tracked resident in 0.3 x box-height units per second, also for samples whose keypoints
    dropped out. A visible sample with no earlier same-track sample for add_speed borrows it."""
    boxes, last_seen = deque(), {}
    for o in observations:
        o.box_speed = 0.0
        while boxes and boxes[0].t < o.t - window - 1e-9:
            boxes.popleft()
        if o.bbox is not None and o.identity_ok and o.track_id is not None:
            ref = next((q for q in boxes if q.track_id == o.track_id), None)
            if ref is not None and o.t > ref.t and _same_shape(o.bbox, ref.bbox):
                (ax, ay), (bx, by) = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in (o.bbox, ref.bbox)]
                scale = 0.3 * max(o.bbox[3] - o.bbox[1], ref.bbox[3] - ref.bbox[1], 1.0)
                o.box_speed = float(math.dist((ax, ay), (bx, by)) / (o.t - ref.t) / scale)
                if o.visible and last_seen.get(o.track_id, -math.inf) < o.t - speed_window - 1e-9:
                    o.speed = max(o.speed, o.box_speed)
            boxes.append(o)
        if o.visible:
            last_seen[o.track_id] = o.t


def _same_shape(a, b):
    """A box that changed shape (lying or sitting down) has not moved across the room."""
    return all(min(x, y) >= 0.8 * max(x, y) for x, y in ((a[2] - a[0], b[2] - b[0]), (a[3] - a[1], b[3] - b[1])))
