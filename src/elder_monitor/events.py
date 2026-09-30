import math
from bisect import bisect_left, bisect_right

from .schemas import (
    BED_EXIT, IN_BED, LYING_IN_BED, OUT_OF_BED, RETURN_TO_BED, UNKNOWN, BedEvent, Verdict, bed_status,
)


def spans(times, dt):
    """Seconds of evidence per sample: one sampling step for a sample in a new slot of the sampling grid, none for
    another sample in a slot already counted (duplicate or faster frames). Empty slots count nothing, and rounding
    to the nearest slot absorbs the timestamp jitter of the video sampler."""
    out, last = [], -math.inf
    for t in times:
        slot = math.floor(t / dt + 0.5 + 1e-9)
        out.append(dt if slot > last else 0.0)
        last = max(last, slot)
    return out


def departure(observations, t0, t1, cfg):
    """First time in [t0, t1) by which the resident has moved away from the bed: observed walking or beyond
    the near-bed band for exit_persist_sec, or walking and then out of view for left_view_sec.
    Standing still beside the bed resets the count; other unseen samples only pause it, and a tracked box that
    keeps moving while its keypoints are lost still counts as walking. Only time with nobody confidently detected
    counts as out of view; a confident unconfirmed person (a caregiver in front), lost keypoints and missing frames
    do not."""
    ec, walk_speed = cfg["events"], cfg["posture"]["walk_enter_speed"]
    times = [o.t for o in observations]
    i0, i1 = bisect_left(times, t0), bisect_left(times, t1)
    away, hidden, confs, walked, boxed = 0.0, 0.0, [], False, False
    stats = {"visible": 0, "near_bed": 0, "unseen": 0, "left_view": False}
    for o, span in zip(observations[i0:i1], spans(times, 1.0 / cfg["sampling"]["fps"])[i0:i1]):
        if not o.visible or not o.identity_ok:
            stats["unseen"] += 1
            if o.bbox is not None and o.identity_ok:
                boxed = o.box_speed >= walk_speed
                continue
            if o.reason in ("not_detected", "low_confidence"):
                hidden += span
            if (walked or boxed) and hidden >= ec["left_view_sec"] - 1e-9:
                stats["left_view"] = True
                conf = sum(confs) / len(confs) if confs else cfg["posture"]["box_motion_conf"]
                return o.t, 0.7 * conf, stats
            continue
        stats["visible"] += 1
        hidden, walked, boxed = 0.0, o.speed >= walk_speed, False
        if o.near_bed and not walked:
            stats["near_bed"] += 1
            away, confs = 0.0, []
            continue
        away += span
        confs.append(o.kp_conf)
        if away >= ec["exit_persist_sec"] - 1e-9:
            return o.t, sum(confs) / len(confs), stats
    return None, 0.0, stats


def guard_verifier(observations, cfg):
    """Baseline exit check without the agent: the deterministic departure guard only."""
    def verify(t_pending, t_now, t_limit):
        end = min(t_now + cfg["events"]["lookahead_sec"], t_limit)
        t, conf, _ = departure(observations, t_pending, end, cfg)
        return Verdict(t, end, conf)
    return verify


def detect_bed_events(times, activity, bed, evidence, confidence, bed_segments, duration, cfg, verify, init=None,
                      snapshot_at=None):
    """UNINITIALIZED -> IN_BED_BASELINE -> EXIT_PENDING -> AWAY_EPISODE -> RETURN_PENDING -> IN_BED_BASELINE.

    Modes follow the committed timelines (`activity`, `bed`). Timers follow `evidence`: the state each sample itself
    supports, UNKNOWN where nothing was observed. So the exit dwell counts observed out-of-bed samples, and a return
    needs return_persist_sec of lying (or, with return_rule sitting, any in-bed state) seen without a break. A bridged
    gap, a smoothed label or a missing frame never adds time, and each restarts the return count; the return keeps
    its start time. A pending exit falls back to the baseline if the resident is back in bed before departing; a
    pending return falls back to AWAY_EPISODE if they leave before lying down. If they get back into bed without
    having departed in between, the return keeps its first re-occupation as its start.

    `init` resumes from a quiet state (mode, episode, last_in, last_out) and `snapshot_at` reports that state at the
    first sample at or after a given time; the live session uses both to analyse a bounded window."""
    ec, init = cfg["events"], init or {}
    in_bed_starts = [s.start for s in bed_segments if s.label == IN_BED]
    mode, episode = init.get("mode", "UNINITIALIZED"), init.get("episode", 0)
    last_in, last_out = init.get("last_in"), init.get("last_out")
    events, rejected, unresolved, pending_spans, snapshot = [], [], [], [], None
    pending = recheck = ret_start = ret_from = held = None
    seen_out, checked, held_out, ret_evidence, support = 0.0, [], 0.0, [], 0.0

    def next_in_bed(t):
        i = bisect_right(in_bed_starts, t)
        return in_bed_starts[i] if i < len(in_bed_starts) else duration

    def supports_return(act, st, ev):
        if st != IN_BED or bed_status(ev) != IN_BED:
            return False
        return ec["return_rule"] == "sitting" or act == ev == LYING_IN_BED

    steps = spans(times, 1.0 / cfg["sampling"]["fps"])
    for i, (t, act, st, ev, dt) in enumerate(zip(times, activity, bed, evidence, steps)):
        if snapshot_at is not None and snapshot is None and t >= snapshot_at - 1e-9:
            snapshot = {"mode": mode, "episode": episode, "last_in": last_in, "last_out": last_out,
                        "quiet": mode in ("UNINITIALIZED", "IN_BED_BASELINE", "AWAY_EPISODE") and held is None}
        seen = bed_status(ev)
        if st == IN_BED:
            last_in = act
        elif st == OUT_OF_BED:
            last_out = act

        if mode == "UNINITIALIZED":
            if st == IN_BED:
                mode = "IN_BED_BASELINE"
            elif st == OUT_OF_BED:
                mode, episode = "AWAY_EPISODE", episode + 1

        elif mode == "IN_BED_BASELINE" and st == OUT_OF_BED:
            mode, pending, recheck, seen_out, checked = "EXIT_PENDING", t, t, 0.0, []

        if mode == "EXIT_PENDING":
            if st == IN_BED:
                rejected.append({"start_sec": pending, "end_sec": t, "reason": "back_in_bed_before_departure",
                                 "evidence": checked})
                pending_spans.append((pending, t))
                mode = "IN_BED_BASELINE"
                continue
            seen_out += dt if seen == OUT_OF_BED else 0.0
            v = None
            if t >= recheck:
                v = verify(pending, t, next_in_bed(t))
                recheck, checked = max(v.until, t + 1e-6), v.evidence
            if (v is None or v.time is None) and seen == OUT_OF_BED and seen_out >= ec["exit_dwell_sec"] - 1e-9:
                dwell = {"action": "dwell", "window": [round(pending, 2), round(t, 2)],
                         "finding": f"observed out of bed for {ec['exit_dwell_sec']}s without returning"}
                v = Verdict(t, t, confidence[i] * 0.8, [*checked, dwell])
            if v is None or v.time is None:
                continue
            episode += 1
            k = min(bisect_left(times, v.time), len(times) - 1)
            current = next((a for a in reversed(activity[:k + 1]) if a != UNKNOWN), activity[k])
            events.append(BedEvent(BED_EXIT, episode, pending, v.time, last_in, current,
                                   round(v.confidence, 2), evidence=v.evidence))
            pending_spans.append((pending, v.time))
            mode = "AWAY_EPISODE"

        elif mode == "AWAY_EPISODE":
            held_out += dt if held and st != IN_BED else 0.0
            if st == IN_BED:
                mode, ret_start, ret_from, ret_evidence = "RETURN_PENDING", t, last_out, []
                support = dt if supports_return(act, st, ev) else 0.0
                if held:
                    v = verify(held[2], t, t)
                    if v.time is None and held_out < ec["exit_dwell_sec"] - 1e-9:
                        ret_start, ret_from = held[0], held[1]
                        ret_evidence = [*v.evidence, {
                            "action": "keep_return_start", "window": [round(held[2], 2), round(t, 2)],
                            "finding": f"out of bed {held_out:.1f}s without departing; start kept at {held[0]:.1f}s"}]
                    held = None

        elif mode == "RETURN_PENDING":
            if st == OUT_OF_BED:
                mode, held, held_out = "AWAY_EPISODE", (ret_start, ret_from, t), dt
            else:
                support = support + dt if supports_return(act, st, ev) else 0.0
                if support >= ec["return_persist_sec"] - 1e-9:
                    j = bisect_left(times, ret_start)
                    conf = sum(confidence[j:i + 1]) / (i + 1 - j)
                    events.append(BedEvent(RETURN_TO_BED, episode, ret_start, t, ret_from, act, round(conf, 2),
                                           evidence=ret_evidence))
                    mode = "IN_BED_BASELINE"

    if mode == "EXIT_PENDING":
        unresolved.append({"event": BED_EXIT, "start_sec": pending, "reason": "no_departure_before_end",
                           "evidence": checked})
        pending_spans.append((pending, duration))
    if mode == "RETURN_PENDING":
        unresolved.append({"event": RETURN_TO_BED, "start_sec": ret_start, "reason": "not_sustained_before_end"})
    return {"events": events, "rejected": rejected, "unresolved": unresolved, "pending_spans": pending_spans,
            "snapshot": snapshot}
