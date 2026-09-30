from collections import deque

from .schemas import (
    ALERT, DECISION_RANK, IN_BED, LYING_ON_FLOOR, MONITOR, NORMAL, OUT_OF_BED, SITTING_ON_BED, UNKNOWN, Segment,
)

RULES = {
    "LYING_ON_FLOOR": ALERT,
    "PROLONGED_OUT_OF_BED": ALERT,
    "ABSENT_FROM_BED_LONG": ALERT,
    "EDGE_SIT_LONG": MONITOR,
    "EVENT_PENDING": MONITOR,
    "UNCERTAIN": MONITOR,
}


class AlertPolicy:
    """NORMAL / MONITOR / ALERT per sample from explicit rules; ALERT outranks MONITOR outranks NORMAL."""

    def __init__(self, cfg):
        self.c = cfg["policy"]
        self.dt = 1.0 / cfg["sampling"]["fps"]

    def run(self, times, activity, bed, observations, fsm, duration, start=0.0, init=None, snapshot_at=None):
        """Decision segments, rule episodes, and the timers at `snapshot_at` (None if not asked).
        `init` resumes the timers of an earlier window."""
        c, dt, init = self.c, self.dt, init or {}
        out_since, left_at, sit_since = init.get("out_since"), init.get("left_at"), init.get("sit_since")
        recent, per_sample, snapshot = deque(init.get("recent", ())), [], None
        for t, act, st, o in zip(times, activity, bed, observations):
            if snapshot_at is not None and snapshot is None and t >= snapshot_at - 1e-9:
                snapshot = {"out_since": out_since, "left_at": left_at, "sit_since": sit_since, "recent": list(recent)}
            out_since = (t if out_since is None else out_since) if st == OUT_OF_BED else None
            if st == IN_BED:
                left_at = None
            elif st == OUT_OF_BED and left_at is None:
                left_at = t
            if act == SITTING_ON_BED:
                sit_since = t if sit_since is None else sit_since
                if o.visible:
                    recent.append((t, o.edge_of_bed))
                while recent and recent[0][0] <= t - c["edge_sit_sec"]:
                    recent.popleft()
            else:
                sit_since, recent = None, deque()

            rules = set()
            if act == LYING_ON_FLOOR:
                rules.add("LYING_ON_FLOOR")
            if out_since is not None and t + dt - out_since > c["max_out_of_bed_sec"]:
                rules.add("PROLONGED_OUT_OF_BED")
            if left_at is not None and t + dt - left_at > c["max_absence_sec"]:
                rules.add("ABSENT_FROM_BED_LONG")
            if (sit_since is not None and t + dt - sit_since > c["edge_sit_sec"] and recent
                    and sum(e for _, e in recent) / len(recent) >= c["edge_fraction"]):
                rules.add("EDGE_SIT_LONG")
            if any(a <= t < b for a, b in fsm["pending_spans"]):
                rules.add("EVENT_PENDING")
            if act == UNKNOWN:
                rules.add("UNCERTAIN")
            decision = max((RULES[r] for r in rules), key=DECISION_RANK.get, default=NORMAL)
            per_sample.append((t, decision, sorted(rules)))
        ends = [t for t, _, _ in per_sample[1:]] + [duration]
        return self._segments(per_sample, ends, start), self._episodes(per_sample, ends), snapshot

    @staticmethod
    def _segments(per_sample, ends, start):
        out = []
        for k, ((t, decision, rules), end) in enumerate(zip(per_sample, ends)):
            if out and out[-1].label == decision and out[-1].reasons == rules:
                out[-1].end = end
            else:
                out.append(Segment(start if k == 0 else t, end, decision, reasons=rules))
        return out

    @staticmethod
    def _episodes(per_sample, ends):
        """One record per contiguous run of a rule, so each alert is raised once per episode."""
        done, active = [], {}
        for (t, _, rules), end in zip(per_sample, ends):
            for r in [r for r in active if r not in rules]:
                done.append(active.pop(r))
            for r in rules:
                active.setdefault(r, {"rule": r, "decision": RULES[r], "start_sec": t})["end_sec"] = end
        return sorted(done + list(active.values()), key=lambda e: (e["start_sec"], e["rule"]))

    def event_decision(self, event, decisions):
        default = self.c["event_decision"].get(event.event, NORMAL)
        at = next((s.label for s in decisions if s.start <= event.confirmed_sec < s.end), NORMAL)
        return max(default, at, key=DECISION_RANK.get)
