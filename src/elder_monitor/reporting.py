import csv
import json
from dataclasses import asdict
from pathlib import Path

import cv2

from .schemas import (
    ACTIVITY_STATES, ALERT, BED_EXIT, BED_STATES, IN_BED, MONITOR, NORMAL, OUT_OF_BED, RETURN_TO_BED, UNKNOWN,
    Observation,
)
from .temporal import durations, labels_at

SCHEMA_VERSION = "1.5"


def _hms(sec):
    sec = round(sec)
    return sec // 3600, sec % 3600 // 60, sec % 60


def fmt_clock(sec):
    h, m, s = _hms(sec)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def fmt_hms(sec):
    return "{:02d}:{:02d}:{:02d}".format(*_hms(sec))


def fmt_human(sec):
    h, m, s = _hms(sec)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s" if m else f"{s}s"


def timeline_text(segments):
    return "\n".join(f"{fmt_clock(s.start)} – {fmt_clock(s.end)}  {s.label}" for s in segments)


def summarize(result):
    T = result["duration"]
    act = durations(result["activity"], ACTIVITY_STATES)
    bed = durations(result["bed"], BED_STATES)
    exits = sum(e.event == BED_EXIT for e in result["events"])
    out_periods = [s.duration for s in result["bed"] if s.label == OUT_OF_BED]
    return {
        "observation_duration_sec": round(T, 1),
        "activity_duration_sec": {k.lower(): round(v, 1) for k, v in act.items()},
        "bed_exit_count": exits,
        "bed_return_count": sum(e.event == RETURN_TO_BED for e in result["events"]),
        "total_in_bed_sec": round(bed[IN_BED], 1),
        "total_out_of_bed_sec": round(bed[OUT_OF_BED], 1),
        "total_unknown_bed_sec": round(bed[UNKNOWN], 1),
        "longest_out_of_bed_period_sec": round(max(out_periods, default=0.0), 1),
        "final_state": result["activity"][-1].label.lower() if result["activity"] else UNKNOWN.lower(),
        "final_bed_status": result["bed"][-1].label if result["bed"] else UNKNOWN,
        "final_decision": result["decisions"][-1].label if result["decisions"] else NORMAL,
        "alert_count": sum(a["decision"] == ALERT for a in result["alerts"]),
        "monitor_count": sum(a["decision"] == MONITOR for a in result["alerts"]),
        "view_quality": result["view"],
        "human": {
            "total_observation_time": fmt_human(T),
            "activity_summary": {k.lower(): fmt_human(v) for k, v in act.items()},
            "bed_summary": {
                "time_in_bed": fmt_human(bed[IN_BED]),
                "time_out_of_bed": fmt_human(bed[OUT_OF_BED]),
                "time_unknown": fmt_human(bed[UNKNOWN]),
                "bed_exit_count": exits,
            },
        },
    }


def event_record(e):
    return {
        "event": e.event,
        "episode_id": e.episode_id,
        "start_time": fmt_hms(e.start_sec),
        "confirmed_time": fmt_hms(e.confirmed_sec),
        "start_sec": round(e.start_sec, 2),
        "confirmed_sec": round(e.confirmed_sec, 2),
        "previous_state": e.previous_state.lower(),
        "current_state": e.current_state.lower(),
        "confidence": e.confidence,
        "decision": e.decision,
        "evidence": e.evidence,
    }


def _write_segments(path, segments, extra=None):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "start_sec", "end_sec", "duration_sec", "label"] + ([extra] if extra else []))
        for s in segments:
            row = [fmt_clock(s.start), fmt_clock(s.end), round(s.start, 2), round(s.end, 2),
                   round(s.duration, 2), s.label]
            if extra == "confidence":
                row.append(s.confidence)
            elif extra == "reasons":
                row.append(";".join(s.reasons))
            w.writerow(row)


def write_outputs(out_dir, result, manifest):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_segments(out / "timeline.csv", result["activity"], "confidence")
    _write_segments(out / "bed_timeline.csv", result["bed"])
    _write_segments(out / "decisions.csv", result["decisions"], "reasons")
    (out / "timeline.txt").write_text(timeline_text(result["activity"]) + "\n", encoding="utf-8")
    events = {
        "bed_events": [event_record(e) for e in result["events"]],
        "alerts": [{**a, "start_sec": round(a["start_sec"], 2), "end_sec": round(a["end_sec"], 2),
                    "start_time": fmt_hms(a["start_sec"]), "end_time": fmt_hms(a["end_sec"])}
                   for a in result["alerts"]],
        "rejected_exit_candidates": result["fsm"]["rejected"],
        "unresolved_candidates": result["fsm"]["unresolved"],
    }
    (out / "events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")
    with open(out / "agent_trace.jsonl", "w", encoding="utf-8") as f:
        for rec in result["trace"]:
            f.write(json.dumps(rec) + "\n")
    with open(out / "proposals.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "label", "confidence", "reason", "source"])
        for p in result["proposals"]:
            w.writerow([round(p.t, 2), p.label, p.confidence, p.reason, p.source])
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_observations(path, meta, observations):
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for o in observations:
            f.write(json.dumps(asdict(o)) + "\n")


def read_observations(path, expect):
    """Cached observations, or (None, None) when the cache was made for another video or config."""
    with open(path, encoding="utf-8") as f:
        meta = json.loads(f.readline())["meta"]
        if any(meta.get(k) != v for k, v in expect.items()):
            return None, None
        return meta, [Observation(**json.loads(line)) for line in f]


COLORS = {NORMAL: (80, 200, 80), MONITOR: (0, 200, 255), ALERT: (0, 0, 255)}


def write_overlay(reader, scene, result, cfg, path):
    fps, kp_conf = cfg["sampling"]["fps"], cfg["vision"]["kp_conf"]
    times = [o.t for o in result["observations"]]
    acts, decs = labels_at(result["activity"], times), labels_at(result["decisions"], times)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (reader.width, reader.height))
    for (t, _, frame), o, act, dec in zip(reader.sample(fps), result["observations"], acts, decs):
        if frame is None:
            continue
        cv2.polylines(frame, [scene.bed.astype(int)], True, (255, 200, 0), 2)
        for chair in scene.chairs:
            cv2.polylines(frame, [chair.astype(int)], True, (200, 120, 0), 1)
        if o.bbox:
            x1, y1, x2, y2 = (int(v) for v in o.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255) if o.identity_ok else (0, 0, 255), 2)
            for x, y, c in o.keypoints or []:
                if c >= kp_conf:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
        cv2.rectangle(frame, (0, 0), (reader.width, 34), (0, 0, 0), -1)
        cv2.putText(frame, f"{fmt_clock(t)}  {act}  [{dec}]", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    COLORS.get(dec, (255, 255, 255)), 2)
        writer.write(frame)
    writer.release()
