import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .video import file_hash


@dataclass
class Person:
    track_id: int | None
    bbox: list
    keypoints: list
    conf: float

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class PoseTracker:
    def __init__(self, cfg):
        from ultralytics import YOLO

        v = cfg["vision"]
        self.model = YOLO(v["model"])
        self.kwargs = {"persist": True, "tracker": v["tracker"], "conf": v["det_conf"], "classes": [0],
                       "verbose": False, "device": v.get("device")}

    @property
    def revision(self):
        path = getattr(self.model, "ckpt_path", None)
        if path and Path(path).is_file():
            return f"{Path(path).name} sha256:{file_hash(path)[:12]}"
        return str(path or self.model.model_name)

    def __call__(self, frame):
        r = self.model.track(frame, **self.kwargs)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        ids = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else [None] * len(boxes)
        kps = r.keypoints.data.cpu().numpy() if r.keypoints is not None else np.zeros((len(boxes), 17, 3))
        return [Person(i, b.tolist(), k.tolist(), float(c)) for i, b, k, c in zip(ids, boxes, kps, confs)]


def appearance(frame, bbox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (round(v) for v in bbox)
    crop = frame[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(hist, hist)


def same_body(a, b, kp_conf):
    ka, kb = np.asarray(a.keypoints), np.asarray(b.keypoints)
    ok = (ka[:, 2] >= kp_conf) & (kb[:, 2] >= kp_conf)
    scale = max(a.bbox[3] - a.bbox[1], b.bbox[3] - b.bbox[1], 1.0)
    return ok.sum() >= 4 and np.median(np.linalg.norm(ka[ok, :2] - kb[ok, :2], axis=1)) < 0.15 * scale


class TargetSelector:
    """Keeps the monitored resident separate from temporary tracker IDs.

    Other confident people tracked alongside the resident are remembered with their appearance, and their IDs are not
    re-attached as the resident. A tracker can later hand such an ID to the resident (it re-activates a lost track on
    them), so the ID is usable again only for a box that looks clearly more like the resident than like the person last
    seen with it. Duplicate boxes on the resident's own body are not remembered. A box that keeps the resident's track
    ID must still be within reach of the last position and look like the resident, so an ID switch becomes identity
    uncertainty. Colour histograms are weak evidence: similar-looking people who swap IDs can still be confused.
    """

    def __init__(self, cfg, scene):
        t, i = cfg["target"], cfg["identity"]
        self.scene = scene
        self.point = None if t.get("point") is None else (t["point"][0] * scene.width, t["point"][1] * scene.height)
        self.start = t.get("time_sec", 0.0) if self.point is not None else 0.0
        self.min_conf, self.min_sim, self.margin = i["min_det_conf"], i["min_similarity"], i["margin"]
        self.track_min_sim = i["track_min_similarity"]
        self.max_jump, self.jump_per_sec = i["max_jump"] * scene.height, i["jump_per_sec"] * scene.height
        self.kp_conf = cfg["vision"]["kp_conf"]
        self.selected, self.track_id, self.hist, self.last_pos, self.last_t = False, None, None, None, None
        self.others = {}
        self.last_in_bed = None

    def select(self, t, frame, persons):
        persons = [p for p in persons if (self.selected and p.track_id is not None and p.track_id == self.track_id)
                   or not self.scene.ignored(p.bbox)]
        if not self.selected:
            p = self._initial(persons) if t >= self.start else None
            if p is None:
                return None, True, "target_not_selected"
            self._adopt(t, frame, p, persons)
            return p, True, "selected"
        for p in persons:
            if p.track_id is not None and p.track_id == self.track_id:
                phantom = p.conf < self.min_conf and self.last_in_bed is False and self.scene.in_bed(self._anchor(p))
                if phantom or not self._consistent(t, frame, p):
                    break
                self._adopt(t, frame, p, persons)
                return p, True, "tracked"
        candidates = [p for p in persons if self._usable(frame, p)]
        if not candidates:
            if not persons:
                return None, True, "not_detected"
            weak = all(p.conf < self.min_conf for p in persons)
            return None, False, "low_confidence" if weak else "identity_uncertain"
        return self._reassociate(t, frame, persons, candidates)

    def _initial(self, persons):
        persons = [p for p in persons if p.conf >= self.min_conf]
        if not persons:
            return None
        if self.point is not None:
            px, py = self.point
            inside = [p for p in persons if p.bbox[0] <= px <= p.bbox[2] and p.bbox[1] <= py <= p.bbox[3]]
            return min(inside, key=lambda p: math.dist(p.center, self.point)) if inside else None
        in_bed = [p for p in persons if self.scene.in_bed(p.center)]
        if in_bed:
            return max(in_bed, key=lambda p: p.conf)
        return persons[0] if len(persons) == 1 else None

    def _reachable(self, t, p):
        return math.dist(p.center, self.last_pos) <= self.max_jump + self.jump_per_sec * (t - self.last_t)

    def _similarity(self, frame, p):
        h = appearance(frame, p.bbox)
        return None if h is None or self.hist is None else cv2.compareHist(self.hist, h, cv2.HISTCMP_CORREL)

    def _consistent(self, t, frame, p):
        """Same track ID: the loose floor lets posture changes and partial occlusion through, not another person."""
        sim = self._similarity(frame, p)
        return self._reachable(t, p) and (sim is None or sim >= self.track_min_sim)

    def _usable(self, frame, p):
        """A confident box whose ID was not last seen on another person, or now clearly looks like the resident
        rather than that person."""
        if p.conf < self.min_conf:
            return False
        if p.track_id not in self.others:
            return True
        h, other = appearance(frame, p.bbox), self.others[p.track_id]
        if h is None or self.hist is None or other is None:
            return False
        resident = cv2.compareHist(self.hist, h, cv2.HISTCMP_CORREL)
        return resident - cv2.compareHist(other, h, cv2.HISTCMP_CORREL) >= self.margin

    def _reassociate(self, t, frame, persons, candidates):
        scored = []
        for p in candidates:
            if not self._reachable(t, p):
                continue
            sim = self._similarity(frame, p)
            scored.append((-1.0 if sim is None else sim, p))
        scored.sort(key=lambda s: s[0], reverse=True)
        if scored and scored[0][0] >= self.min_sim:
            runner_up = scored[1][0] if len(scored) > 1 else -1.0
            if scored[0][0] - runner_up >= self.margin:
                self._adopt(t, frame, scored[0][1], persons)
                return scored[0][1], True, "reassociated"
        return None, False, "identity_uncertain"

    def _anchor(self, p):
        kp = np.asarray(p.keypoints)
        hips = [kp[i, :2] for i in (11, 12) if kp[i, 2] >= self.kp_conf]
        return tuple(np.mean(hips, axis=0)) if hips else p.center

    def _adopt(self, t, frame, p, persons):
        h = appearance(frame, p.bbox)
        if h is not None:
            self.hist = h if self.hist is None else 0.9 * self.hist + 0.1 * h
        if p.track_id is not None:
            self.track_id = p.track_id
        for q in persons:
            if q is not p and q.track_id is not None and q.conf >= self.min_conf and not same_body(q, p, self.kp_conf):
                self.others[q.track_id] = appearance(frame, q.bbox)
        self.others.pop(self.track_id, None)
        self.selected, self.last_pos, self.last_t = True, p.center, t
        self.last_in_bed = bool(self.scene.in_bed(self._anchor(p)))
