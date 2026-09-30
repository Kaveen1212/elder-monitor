from bisect import bisect_left, bisect_right

from .schemas import BED_EXIT, IN_BED, LYING_IN_BED, OUT_OF_BED, RETURN_TO_BED, UNKNOWN, BedEvent, Verdict


def departure(observations, t0, t1, cfg):
    """First time in [t0, t1) by which the resident has moved away from the bed: observed walking or beyond
    the near-bed band for exit_persist_sec, or walking and then out of view for left_view_sec.
    Standing still beside the bed resets the count; unseen samples only pause it, and a tracked box that keeps
    moving while its keypoints are lost still counts as walking."""
    ec, walk_speed = cfg["events"], cfg["posture"]["walk_enter_speed"]
    dt = 1.0 / cfg["sampling"]["fps"]
    times = [o.t for o in observations]
    away, confs, gone, walked, boxed = 0.0, [], None, False, False
    stats = {"visible": 0, "near_bed": 0, "unseen": 0, "left_view": False}
    for o in observations[bisect_left(times, t0):bisect_left(times, t1)]:
        if not o.visible or not o.identity_ok:
            stats["unseen"] += 1
            gone = o.t if gone is None else gone
            if o.bbox is not None:
                boxed = o.identity_ok and o.box_speed >= walk_speed
            if (walked or boxed and o.bbox is None) and o.t + dt - gone >= ec["left_view_sec"] - 1e-9:
                stats["left_view"] = True
                conf = sum(confs) / len(confs) if confs else cfg["posture"]["box_motion_conf"]
                return o.t, 0.7 * conf, stats
            continue
        stats["visible"] += 1
        gone, walked, boxed = None, o.speed >= walk_speed, False
        if o.near_bed and not walked:
            stats["near_bed"] += 1
            away, confs = 0.0, []
            continue
        away += dt
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


def detect_bed_events(times, activity, bed, confidence, bed_segments, duration, cfg, verify):
    """UNINITIALIZED -> IN_BED_BASELINE -> EXIT_PENDING -> AWAY_EPISODE -> RETURN_PENDING -> IN_BED_BASELINE.

    A pending exit falls back to the baseline if the resident is back in bed before departing;
    a pending return falls back to AWAY_EPISODE if they leave before lying down. If they get back into bed
    without having departed in between, the return keeps its first re-occupation as its start."""
    ec, dt = cfg["events"], 1.0 / cfg["sampling"]["fps"]
    in_bed_starts = [s.start for s in bed_segments if s.label == IN_BED]
    mode, episode = "UNINITIALIZED", 0
    events, rejected, unresolved, pending_spans = [], [], [], []
    last_in = last_out = pending = recheck = ret_start = ret_from = lying_since = occupied = held = None
    unseen, checked, held_out, ret_evidence = 0.0, [], 0.0, []

    def next_in_bed(t):
        i = bisect_right(in_bed_starts, t)
        return in_bed_starts[i] if i < len(in_bed_starts) else duration

    for i, (t, act, st) in enumerate(zip(times, activity, bed)):
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
            mode, pending, recheck, unseen, checked = "EXIT_PENDING", t, t, 0.0, []

        if mode == "EXIT_PENDING":
            if st == IN_BED:
                rejected.append({"start_sec": pending, "end_sec": t, "reason": "back_in_bed_before_departure",
                                 "evidence": checked})
                pending_spans.append((pending, t))
                mode = "IN_BED_BASELINE"
                continue
            unseen += dt if st == UNKNOWN else 0.0
            v = None
            if t >= recheck:
                v = verify(pending, t, next_in_bed(t))
                recheck, checked = max(v.until, t + 1e-6), v.evidence
            observed_out = t - pending - unseen if st == OUT_OF_BED else 0.0
            if (v is None or v.time is None) and observed_out >= ec["exit_dwell_sec"] - 1e-9:
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
                mode, ret_start, ret_from, occupied, ret_evidence = "RETURN_PENDING", t, last_out, t, []
                if held:
                    v = verify(held[2], t, t)
                    if v.time is None and held_out < ec["exit_dwell_sec"] - 1e-9:
                        ret_start, ret_from = held[0], held[1]
                        ret_evidence = [*v.evidence, {
                            "action": "keep_return_start", "window": [round(held[2], 2), round(t, 2)],
                            "finding": f"out of bed {held_out:.1f}s without departing; start kept at {held[0]:.1f}s"}]
                    held = None
                lying_since = t if act == LYING_IN_BED else None

        elif mode == "RETURN_PENDING":
            if st == OUT_OF_BED:
                mode, held, held_out = "AWAY_EPISODE", (ret_start, ret_from, t), dt
            elif st == IN_BED:
                if ec["return_rule"] == "sitting":
                    since = occupied
                else:
                    lying_since = (t if lying_since is None else lying_since) if act == LYING_IN_BED else None
                    since = lying_since
                if since is not None and t - since >= ec["return_persist_sec"] - 1e-9:
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
    return {"events": events, "rejected": rejected, "unresolved": unresolved, "pending_spans": pending_spans}
