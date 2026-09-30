import math
from bisect import bisect_left

import cv2

from .config import validate_config
from .features import Scene, add_speed, measure
from .pipeline import run_analysis, view_counts, view_quality
from .reporting import fmt_clock, fmt_human, summarize
from .schemas import BED_EXIT, LYING_ON_FLOOR, NORMAL, UNKNOWN, Observation, Segment
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


def _join(a, b):
    """Two consecutive segment lists as one, merging the pair that meets at the boundary."""
    if a and b and a[-1].label == b[0].label and a[-1].reasons == b[0].reasons and abs(a[-1].end - b[0].start) < 1e-6:
        return [*a[:-1], Segment(a[-1].start, b[0].end, a[-1].label, a[-1].confidence, a[-1].reasons), *b[1:]]
    return [*a, *b]


def _join_alerts(a, b, t0):
    """Rule episodes of an archive and of the next window; an episode running across the boundary stays one."""
    out, open_ = list(a), {x["rule"]: k for k, x in enumerate(a) if abs(x["end_sec"] - t0) < 1e-6}
    for x in b:
        k = open_.get(x["rule"]) if abs(x["start_sec"] - t0) < 1e-6 else None
        if k is None:
            out.append(x)
        else:
            out[k] = {**out[k], "end_sec": x["end_sec"]}
    return sorted(out, key=lambda e: (e["start_sec"], e["rule"]))


def _cut(segments, t):
    return [Segment(s.start, min(s.end, t), s.label, s.confidence, list(s.reasons)) for s in segments if s.start < t - 1e-9]


class LiveSession:
    """Live analysis with the same temporal engine, agent and policy as recorded video; only messages not sent
    before are returned. Differences from offline: frames get the server's receive time, the VLM is not used, and
    a candidate can only be confirmed once the frames that confirm it have arrived.

    Arriving frames are resampled onto the sampling.fps grid, as recorded video is: a frame for a slot that is
    already filled is dropped before the tracker, and slots with no frame become no_frame samples, so fast,
    duplicate or late frames never add observed time.

    Each kept frame re-runs the analysis over a window that starts at the last checkpoint, so the cost stays bounded.
    A checkpoint is taken once the window is longer than `window_sec`, at a point `lag` seconds back, when the
    committed state has lasted since `lag` before that point (an UNKNOWN gap: long enough that it can no longer be
    bridged), no exit or return candidate is open, the resident is not lying outside the bed and the proposals around
    the point agree. Everything before it is archived: timelines, bed events, rule episodes, the agent trace and
    view counts. The state machine, policy timers and posture hysteresis at that point seed the next window, so
    totals, episode numbers and timers carry on. Without a quiet stretch the window keeps growing."""

    def __init__(self, cfg, tracker=None):
        validate_config(cfg)
        self.cfg, self.step = cfg, 1.0 / cfg["sampling"]["fps"]
        a = cfg["agent"]
        self.lag = max(a["lookback_sec"], a["lookahead_sec"], cfg["events"]["lookahead_sec"]) * max(a["max_rounds"], 1)
        self.lag += 2.0
        self.settled = a["max_bridge_sec"] + self.lag
        self.window_sec = 10 * self.lag
        self.tracker = tracker or PoseTracker(cfg)
        self.scene = self.selector = self.result = self.slot = None
        self.obs, self.sent, self.t0, self.init, self.checkpoints = [], {}, 0.0, {}, 0
        self.archive = {"activity": [], "bed": [], "decisions": [], "events": [], "alerts": [], "trace": [],
                        "rejected": [], "view": None}

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
        add_speed(self.obs, self.cfg["posture"]["speed_window_sec"], start=len(self.obs) - 1)
        return self.update()

    def update(self):
        o = self.obs[-1]
        t = o.t
        w = run_analysis(self.obs, t + self.step, self.cfg, scene=self.scene, start=self.t0, init=self.init,
                         snapshot_at=t - self.lag)
        r = self.result = self._combine(w)
        a, act, dec = self.archive, r["activity"][-1], r["decisions"][-1]
        recent = {**r, "activity": r["activity"][max(len(a["activity"]) - 1, 0):],
                  "decisions": r["decisions"][max(len(a["decisions"]) - 1, 0):], "events": w["events"],
                  "alerts": [x for x in r["alerts"] if x["end_sec"] >= self.t0 - 1e-6], "trace": w["trace"]}
        messages = [m for m in chat_messages(recent, self.cfg, t - 1.0) if m["t"] >= self.t0 - 1e-6 and self._fresh(m)]
        self._checkpoint(w)
        return {
            "type": "frame", "t": round(t, 2), "time": fmt_clock(t), "person": pose(o) or None,
            "n_persons": o.n_persons, "activity": act.label, "since": fmt_clock(act.start),
            "bed": r["bed"][-1].label, "decision": dec.label, "rules": dec.reasons,
            "summary": {k: v for k, v in r["summary"].items() if k != "human"},
            "messages": messages,
        }

    def _combine(self, w):
        """The whole session so far: the archive followed by the current window."""
        a = self.archive
        r = {**w, "activity": _join(a["activity"], w["activity"]), "bed": _join(a["bed"], w["bed"]),
             "decisions": _join(a["decisions"], w["decisions"]), "events": a["events"] + w["events"],
             "alerts": _join_alerts(a["alerts"], w["alerts"], self.t0), "trace": a["trace"] + w["trace"],
             "view": view_quality(self.obs, self.cfg, a["view"])}
        r["fsm"] = {**w["fsm"], "rejected": a["rejected"] + w["fsm"]["rejected"]}
        r["summary"] = summarize(r)
        return r

    def _checkpoint(self, w):
        now, snap = w["duration"], w.get("snapshot") or {}
        fsm, timers = snap.get("fsm"), snap.get("policy")
        if now - self.t0 < self.window_sec or not fsm or not fsm["quiet"] or timers is None:
            return
        times = [o.t for o in self.obs]
        k = bisect_left(times, now - self.step - self.lag - 1e-9)
        if k >= len(times):
            return
        f, last = times[k], w["activity"][-1]
        if last.label == LYING_ON_FLOOR or last.start > f - (self.settled if last.label == UNKNOWN else self.lag):
            return
        near = [p.label for p in w["rules"] + w["proposals"] if abs(p.t - f) <= 1.0 + 1e-9]
        if any(lab != last.label for lab in near):
            return
        a = self.archive
        a["activity"], a["bed"] = _join(a["activity"], _cut(w["activity"], f)), _join(a["bed"], _cut(w["bed"], f))
        a["decisions"] = _join(a["decisions"], _cut(w["decisions"], f))
        a["events"] += [e for e in w["events"] if e.confirmed_sec < f]
        a["alerts"] = _join_alerts(a["alerts"], [{**x, "end_sec": min(x["end_sec"], f)} for x in w["alerts"]
                                                 if x["start_sec"] < f - 1e-9], self.t0)
        a["trace"] += [rec for rec in w["trace"] if rec["window"][1] <= f + 1e-9]
        a["rejected"] += [x for x in w["fsm"]["rejected"] if x["end_sec"] < f]
        done = view_counts(self.obs[:k], self.cfg)
        a["view"] = done if a["view"] is None else {
            n: max(v, a["view"][n]) if n == "worst" else v + a["view"][n] for n, v in done.items()}
        prev = next((p.label for p in reversed(w["rules"]) if p.t < f and p.label != UNKNOWN), self.init.get("prev"))
        self.init = {"state": last.label, "prev": prev, "policy": timers,
                     "fsm": {n: fsm[n] for n in ("mode", "episode", "last_in", "last_out")}}
        self.t0, self.obs, self.checkpoints = f, self.obs[k:], self.checkpoints + 1
        self.sent = {key: [t for t in ts if t >= f - 5.0] for key, ts in self.sent.items()}

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
