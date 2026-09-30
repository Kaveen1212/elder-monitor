from bisect import bisect_right
from collections import Counter, deque

from .schemas import (
    LYING_IN_BED, LYING_ON_FLOOR, LYING_STATES, OUT_OF_BED, SITTING_ON_BED, SITTING_OUTSIDE_BED,
    STANDING, UNKNOWN, WALKING, Proposal, Segment, bed_status,
)


def leg_posture(o, c):
    if o.knee_angle is not None:
        seated = o.knee_angle < c["sit_knee_angle"] or o.thigh_ratio < c["sit_thigh_ratio"]
        return "seated" if seated else "extended"
    if o.knee_drop is not None:
        return "seated" if o.knee_drop < c["sit_knee_ratio"] else "extended"
    if o.thigh_angle is not None:
        return "seated" if o.thigh_angle >= c["sit_thigh_angle"] else "extended"
    return None


def body_angle(o, legs, c):
    if o.torso_angle is not None:
        along_camera_axis = o.shoulder_width and o.torso_len < c["foreshortened_ratio"] * o.shoulder_width
        return 90.0 if legs is None and o.hip_in_bed and along_camera_axis else o.torso_angle
    if legs == "extended":
        return o.leg_angle if o.leg_angle is not None else o.thigh_angle
    return None


def propose(o, prev, cfg):
    """Frame-level state proposal from posture + bed geometry, with hysteresis on the previous label."""
    c = cfg["posture"]
    if not o.visible or not o.identity_ok:
        return Proposal(o.t, UNKNOWN, 0.0, o.reason or "not_visible")

    legs = leg_posture(o, c)
    angle = body_angle(o, legs, c)
    if angle is not None:
        lying = angle >= (c["lying_exit_deg"] if prev in LYING_STATES else c["lying_enter_deg"])
    elif o.truncated:
        return Proposal(o.t, UNKNOWN, 0.2, "posture_unclear")
    else:
        x1, y1, x2, y2 = o.bbox
        aspect = (x2 - x1) / max(y2 - y1, 1.0)
        if c["upright_aspect"] < aspect < c["lying_aspect"]:
            return Proposal(o.t, UNKNOWN, 0.2, "posture_unclear")
        lying = aspect >= c["lying_aspect"]

    conf = round(o.kp_conf, 3)
    if lying:
        if o.hip_in_bed or o.body_in_bed >= c["in_bed_fraction"]:
            return Proposal(o.t, LYING_IN_BED, conf, "lying_on_bed")
        return Proposal(o.t, LYING_ON_FLOOR, round(conf * 0.8, 3), "lying_outside_bed")

    walk = o.speed >= (c["walk_exit_speed"] if prev == WALKING else c["walk_enter_speed"])
    leg_axis = o.leg_angle if o.leg_angle is not None else o.thigh_angle
    upright_legs = legs == "extended" and (leg_axis is None or leg_axis < c["lying_exit_deg"])
    if upright_legs and not (o.hip_in_bed and o.feet_in_bed):
        return Proposal(o.t, WALKING if walk else STANDING, conf, "legs_extended")
    if o.hip_in_bed:
        if angle is None and legs is None:
            return Proposal(o.t, UNKNOWN, 0.2, "posture_unclear")
        return Proposal(o.t, SITTING_ON_BED, conf, "upright_on_bed")
    if legs == "seated" and o.edge_of_bed and not o.in_chair:
        return Proposal(o.t, SITTING_ON_BED, conf, "seated_on_bed_edge")
    if legs == "seated" or o.in_chair:
        return Proposal(o.t, SITTING_OUTSIDE_BED, conf, "seated_outside_bed")
    if walk:
        return Proposal(o.t, WALKING, round(conf * 0.8, 3), "moving_outside_bed")
    return Proposal(o.t, OUT_OF_BED, round(conf * 0.7, 3), "outside_bed_posture_unresolved")


def box_motion(o, p, prev, recent, c):
    """Keypoints dropped out but the tracked box keeps moving right after upright postures: still WALKING."""
    if p.label != UNKNOWN or p.reason not in ("low_keypoints", "posture_unclear") or not o.identity_ok or not o.bbox:
        return p
    context = [lab for t, lab in recent if t >= o.t - c["box_context_sec"] - 1e-9]
    if 2 * sum(lab in (STANDING, WALKING) for lab in context) <= len(context):
        return p
    if o.box_speed >= (c["walk_exit_speed"] if prev == WALKING else c["walk_enter_speed"]):
        return Proposal(o.t, WALKING, c["box_motion_conf"], "box_motion")
    return p


def propose_all(observations, cfg, prev=None):
    out, recent = [], deque()
    for o in observations:
        while recent and recent[0][0] < o.t - cfg["posture"]["box_context_sec"] - 1e-9:
            recent.popleft()
        p = box_motion(o, propose(o, prev, cfg), prev, recent, cfg["posture"])
        out.append(p)
        if p.label != UNKNOWN:
            prev = p.label
            if p.reason != "box_motion":
                recent.append((o.t, p.label))
    return out


def smooth(labels, k):
    if k <= 1:
        return list(labels)
    h, out = k // 2, []
    for i, lab in enumerate(labels):
        counts = Counter(labels[max(0, i - h): i + h + 1])
        top, n = counts.most_common(1)[0]
        out.append(lab if counts[lab] == n else top)
    return out


def merge(segments):
    out = []
    for s in segments:
        if s.end <= s.start:
            continue
        if out and out[-1].label == s.label and abs(out[-1].end - s.start) < 1e-9:
            out[-1].end = s.end
        else:
            out.append(Segment(s.start, s.end, s.label))
    return out


def build_timeline(proposals, duration, cfg, start=0.0, state=UNKNOWN):
    """Commit a state once it persists for its dwell time and backdate the boundary to where it began.

    If known labels keep alternating so that none can settle, the latest one is committed after
    twice the longest dwell, so a flickering person is never left in the previous state. UNKNOWN straight after
    WALKING (walked out of view) commits after left_view_sec, together with that walk. `start` and `state` resume
    a window that begins inside a committed state.
    """
    tc = cfg["temporal"]
    dt = 1.0 / cfg["sampling"]["fps"]
    dwell = tc["dwell_sec"]
    patience = 2 * max(dwell.values())
    labels = smooth([p.label for p in proposals], tc["smooth_samples"])
    segments, cand, cand_start, away = [], None, start, None
    prev_lab = walk_from = walk_out = None
    tail = len(labels)
    while tail and labels[tail - 1] == UNKNOWN:
        tail -= 1
    for i, (p, lab) in enumerate(zip(proposals, labels)):
        if lab == WALKING and prev_lab != WALKING:
            walk_from = p.t
        if lab == UNKNOWN and prev_lab == WALKING:
            walk_out = (walk_from, p.t)
        elif lab != UNKNOWN:
            walk_out = None
        prev_lab = lab
        if lab == state:
            cand = away = None
            continue
        if lab != cand:
            cand, cand_start = lab, p.t
        away = p.t if away is None else away
        walked_off = lab == UNKNOWN and walk_out is not None and cand_start == walk_out[1]
        need = dwell.get(lab, dwell["default"])
        if walked_off and i >= tail:
            need = min(need, cfg["events"]["left_view_sec"])
        if p.t + dt - cand_start >= need - 1e-9:
            if walked_off and state != WALKING and walk_out[0] > start:
                segments.append(Segment(start, walk_out[0], state))
                state, start = WALKING, walk_out[0]
            segments.append(Segment(start, cand_start, state))
            state, start, cand, away = lab, cand_start, None, None
        elif lab != UNKNOWN and p.t + dt - away >= patience - 1e-9:
            segments.append(Segment(start, away, state))
            state, start, cand, away = lab, away, None, None
    segments.append(Segment(start, max(duration, start), state))
    segments = merge(segments)
    _attach_confidence(segments, proposals)
    return segments


def _attach_confidence(segments, proposals):
    times = [p.t for p in proposals]
    for s in segments:
        i, j = bisect_right(times, s.start - 1e-9), bisect_right(times, s.end - 1e-9)
        confs = [p.confidence for p in proposals[i:j] if p.label == s.label]
        s.confidence = round(sum(confs) / len(confs), 3) if confs else 0.0


def bed_timeline(activity):
    return merge([Segment(s.start, s.end, bed_status(s.label)) for s in activity])


def labels_at(segments, times):
    starts = [s.start for s in segments]
    return [segments[max(bisect_right(starts, t) - 1, 0)].label for t in times]


def durations(segments, labels):
    out = dict.fromkeys(labels, 0.0)
    for s in segments:
        out[s.label] = out.get(s.label, 0.0) + s.duration
    return out
