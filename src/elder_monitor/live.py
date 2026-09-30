import math

import cv2

from .config import validate_config
from .features import Scene, add_speed, measure
from .pipeline import run_analysis
from .reporting import fmt_clock, fmt_human
from .schemas import BED_EXIT, NORMAL, UNKNOWN, Observation
from .vision import PoseTracker, TargetSelector

STATE_TEXT = {
    "LYING_IN_BED": "Lying in bed",
    "SITTING_ON_BED": "Sitting on the bed",
    "SITTING_OUTSIDE_BED": "Sitting outside the bed",
    "STANDING": "Standing",
    "WALKING": "Walking",
    "OUT_OF_BED": "Out of bed",
    "LYING_ON_FLOOR": "Lying outside the bed",
    "UNKNOWN": "Not clearly visible",
}

REVIEWED = {"AMBIGUOUS_GAP": "an unclear gap", "LYING_OUTSIDE_BED": "lying outside the bed",
            "UPRIGHT_ON_BED_REGION": "standing over the bed"}


def rule_text(rule, cfg):
    p = cfg["policy"]
    return {
        "LYING_ON_FLOOR": "lying outside the bed",
        "PROLONGED_OUT_OF_BED": f"out of bed for more than {fmt_human(p['max_out_of_bed_sec'])}",
        "ABSENT_FROM_BED_LONG": f"away from the bed for more than {fmt_human(p['max_absence_sec'])}",
        "EDGE_SIT_LONG": f"sitting on the bed edge for more than {fmt_human(p['edge_sit_sec'])}",
        "EVENT_PENDING": "possible bed exit, verifying",
        "UNCERTAIN": "resident not clearly observed",
    }.get(rule, rule)


def step_text(step):
    finding = step["finding"]
    if isinstance(finding, dict):
        finding = ", ".join(f"{k}={v}" for k, v in finding.items())
    a, b = step["window"]
    return f"{step['action']} {fmt_clock(a)}–{fmt_clock(b)}: {finding}"


def message(t, kind, author, text, level=NORMAL, key=None, label=None, detail=()):
    return {"t": round(t, 2), "time": fmt_clock(t), "kind": kind, "author": author, "text": text,
            "level": level, "label": label, "detail": list(detail), "key": key or kind}


def pose(o):
    if not o.bbox:
        return {}
    return {"bbox": [round(v) for v in o.bbox], "ok": o.identity_ok,
            "kp": [[round(x), round(y), round(c, 2)] for x, y, c in o.keypoints or []]}


def chat_messages(result, cfg, closed_before=None):
    """State changes, bed events, policy episodes and agent reviews in time order.
    Agent reviews whose window is still open at closed_before are held back (live mode)."""
    min_unclear = cfg["temporal"]["dwell_sec"][UNKNOWN]
    out = []
    for s in result["activity"]:
        if s.label != UNKNOWN:
            out.append(message(s.start, "state", "Vision", STATE_TEXT[s.label], key=f"state:{s.label}", label=s.label))
    for e in result["events"]:
        verb = "Bed exit confirmed" if e.event == BED_EXIT else "Returned to bed"
        text = (f"{verb}: {STATE_TEXT.get(e.previous_state, e.previous_state)} → "
                f"{STATE_TEXT.get(e.current_state, e.current_state)} (started {fmt_clock(e.start_sec)})")
        detail = [step_text(s) for s in e.evidence] + [f"confidence {e.confidence}"]
        out.append(message(e.confirmed_sec, "event", "Agent", text, e.decision, f"event:{e.event}", e.event, detail))
    for a in result["alerts"]:
        if a["rule"] == "UNCERTAIN" and a["end_sec"] - a["start_sec"] < min_unclear - 1e-9:
            continue
        text = rule_text(a["rule"], cfg)
        out.append(message(a["start_sec"], "alert", "Policy", text[0].upper() + text[1:], a["decision"],
                           f"alert:{a['rule']}", a["rule"]))
    decisions = result["decisions"]
    for prev, s in zip(decisions, decisions[1:]):
        if s.label == NORMAL and prev.label != NORMAL and min(prev.duration, s.duration) >= min_unclear:
            out.append(message(s.start, "decision", "Policy", "Back to NORMAL", key="decision:NORMAL", label=NORMAL))
    for rec in result["trace"]:
        t0, t1 = rec["window"]
        if closed_before is not None and t1 > closed_before:
            continue
        if rec["trigger"] == "EXIT_CANDIDATE":
            if rec["outcome"] == "BED_EXIT confirmed":
                continue
            text = f"Exit candidate {fmt_clock(t0)}–{fmt_clock(t1)}: {rec['outcome']}"
        else:
            what = REVIEWED.get(rec["trigger"], rec["trigger"])
            verdict = "left as unknown" if rec["outcome"] == UNKNOWN else STATE_TEXT.get(rec["outcome"], rec["outcome"])
            text = f"Reviewed {what} {fmt_clock(t0)}–{fmt_clock(t1)} → {verdict} ({rec['reason']})"
        out.append(message(t1, "agent", "Agent", text, key=f"agent:{rec['trigger']}",
                           detail=[step_text(s) for s in rec["steps"]]))
    return sorted(out, key=lambda m: m["t"])


class LiveSession:
    """Re-runs the offline analysis over the whole session on every frame, so live answers use the same
    temporal engine, agent and policy as recorded video; only messages not sent before are returned.

    Arriving frames are resampled onto the sampling.fps grid, as recorded video is: a frame for a slot that is
    already filled is dropped before the tracker, and slots with no frame become no_frame samples, so fast,
    duplicate or late frames never add observed time. The analysis is recomputed over the whole session per frame,
    so its cost grows with session length."""

    def __init__(self, cfg, tracker=None):
        validate_config(cfg)
        self.cfg, self.step = cfg, 1.0 / cfg["sampling"]["fps"]
        self.tracker = tracker or PoseTracker(cfg)
        self.scene = self.selector = self.result = self.slot = None
        self.obs, self.sent = [], {}

    def push(self, frame, t):
        slot = math.floor(t / self.step + 0.5 + 1e-9)
        if self.slot is not None and slot <= self.slot:
            return {"type": "frame", "t": round(t, 2), "skipped": True}
        if self.slot is not None:
            self.obs += [Observation(round(s * self.step, 4), reason="no_frame") for s in range(self.slot + 1, slot)]
        self.slot, t = slot, round(slot * self.step, 4)
        if self.scene is None:
            h, w = frame.shape[:2]
            self.scene = Scene(self.cfg, w, h)
            self.selector = TargetSelector(self.cfg, self.scene)
        if frame.shape[:2] != (self.scene.height, self.scene.width):
            frame = cv2.resize(frame, (self.scene.width, self.scene.height))
        persons = self.tracker(frame)
        person, identity_ok, reason = self.selector.select(t, frame, persons)
        o = measure(t, -1, person, self.scene, self.cfg) if person else Observation(t, reason=reason)
        o.identity_ok, o.n_persons = identity_ok, len(persons)
        self.obs.append(o)
        add_speed(self.obs, self.cfg["posture"]["speed_window_sec"])
        return self.update()

    def update(self):
        o = self.obs[-1]
        t = o.t
        r = self.result = run_analysis(self.obs, t + self.step, self.cfg, scene=self.scene)
        act, dec = r["activity"][-1], r["decisions"][-1]
        return {
            "type": "frame", "t": round(t, 2), "time": fmt_clock(t), "person": pose(o) or None,
            "n_persons": o.n_persons, "activity": act.label, "since": fmt_clock(act.start),
            "bed": r["bed"][-1].label, "decision": dec.label, "rules": dec.reasons,
            "summary": {k: v for k, v in r["summary"].items() if k != "human"},
            "messages": [m for m in chat_messages(r, self.cfg, t - 1.0) if self._fresh(m)],
        }

    def answer(self, question=""):
        r = self.result
        if r is None:
            return message(0.0, "answer", "Agent", "No frames analysed yet.")
        q, s = question.lower(), r["summary"]
        if any(w in q for w in ("exit", "return", "event", "left", "leave")):
            lines = [f"{fmt_clock(e.confirmed_sec)} {e.event.replace('_', ' ')} [{e.decision}]" for e in r["events"]]
            lines = lines or ["No bed exits or returns yet."]
        elif any(w in q for w in ("alert", "monitor", "warn")):
            lines = [f"{fmt_clock(a['start_sec'])}–{fmt_clock(a['end_sec'])} {a['decision']}: "
                     f"{rule_text(a['rule'], self.cfg)}" for a in r["alerts"]] or ["No alert or monitor episodes."]
        elif any(w in q for w in ("long", "time", "duration", "much")):
            lines = [f"{STATE_TEXT[k.upper()]}: {fmt_human(v)}" for k, v in s["activity_duration_sec"].items() if v]
        else:
            act, dec = r["activity"][-1], r["decisions"][-1]
            why = f" ({', '.join(rule_text(x, self.cfg) for x in dec.reasons)})" if dec.reasons else ""
            lines = [f"{STATE_TEXT[act.label]} since {fmt_clock(act.start)}. Decision {dec.label}{why}.",
                     f"In bed {fmt_human(s['total_in_bed_sec'])}, out of bed {fmt_human(s['total_out_of_bed_sec'])}, "
                     f"unclear {fmt_human(s['total_unknown_bed_sec'])} of {fmt_human(r['duration'])}.",
                     f"{s['bed_exit_count']} bed exit(s), {s['bed_return_count']} return(s), {s['alert_count']} alert(s)."]
        return message(r["duration"], "answer", "Agent", "\n".join(lines))

    def _fresh(self, m):
        seen = self.sent.setdefault(m["key"], [])
        if any(abs(m["t"] - t) <= 2.0 for t in seen):
            return False
        seen.append(m["t"])
        return True
