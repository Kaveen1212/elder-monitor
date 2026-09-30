import csv
import json
from collections import Counter
from itertools import groupby, pairwise
from pathlib import Path

import cv2
import numpy as np

from .reporting import fmt_clock
from .schemas import ACTIVITY_STATES, BED_STATES, EVENT_TYPES, OUT_OF_BED, UNKNOWN, Segment
from .temporal import bed_timeline, durations
from .video import VideoReader


def parse_time(v):
    if isinstance(v, (int, float)):
        return float(v)
    sec = 0.0
    for i, part in enumerate(str(v).strip().split(":")):
        if i and not 0 <= float(part) < 60:
            raise ValueError(f"bad time {v!r}")
        sec = sec * 60 + float(part)
    return sec


def _labelled(items, labels, name):
    segs = sorted((Segment(parse_time(a["start"]), parse_time(a["end"]), a["label"].upper()) for a in items),
                  key=lambda s: s.start)
    for s in segs:
        if s.label not in labels or s.end <= s.start:
            raise ValueError(f"{name}: bad interval {s.start}-{s.end} {s.label}")
    if segs and abs(segs[0].start) > 1e-6:
        raise ValueError(f"{name}: labels must start at 0 (use UNKNOWN for unlabelled time)")
    for a, b in pairwise(segs):
        if abs(b.start - a.end) > 1e-6:
            raise ValueError(f"{name}: intervals must be contiguous; check {a.end} - {b.start}")
    return segs


def load_annotations(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    activity = _labelled(d["activity"], ACTIVITY_STATES, path)
    bed = _labelled(d["bed"], BED_STATES, path) if "bed" in d else bed_timeline(activity)
    duration = parse_time(d["duration"]) if "duration" in d else activity[-1].end
    if activity[-1].end < duration - 1e-6:
        raise ValueError(f"{path}: labels end at {activity[-1].end}, before the duration {duration}")
    events = [{"event": e["event"].lower(), "time": parse_time(e["time"])} for e in d.get("events", [])]
    for e in events:
        if e["event"] not in EVENT_TYPES or not 0 <= e["time"] <= duration:
            raise ValueError(f"{path}: bad event {e}")
    return {"duration": duration, "activity": activity, "bed": bed, "events": events}


def _read_segments(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [Segment(float(r["start_sec"]), float(r["end_sec"]), r["label"]) for r in csv.DictReader(f)]


def load_predictions(pred_dir):
    d = Path(pred_dir)
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    events = json.loads((d / "events.json").read_text(encoding="utf-8"))["bed_events"]
    return {
        "duration": summary["observation_duration_sec"],
        "activity": _read_segments(d / "timeline.csv"),
        "bed": _read_segments(d / "bed_timeline.csv"),
        "events": [{"event": e["event"], "time": e["start_sec"], "confirmed": e["confirmed_sec"]} for e in events],
    }


def _cut(x, T):
    def cut(segs):
        return [Segment(s.start, min(s.end, T), s.label) for s in segs if s.start < T]
    return {**x, "activity": cut(x["activity"]), "bed": cut(x["bed"]),
            "events": [e for e in x["events"] if e["time"] <= T]}


def to_grid(segments, T, res, labels):
    idx = {lab: i for i, lab in enumerate(labels)}
    grid = np.full(round(T / res), idx[UNKNOWN])
    for s in segments:
        grid[round(s.start / res):round(min(s.end, T) / res)] = idx.get(s.label, idx[UNKNOWN])
    return grid


def class_metrics(cm, labels):
    """Per-class P/R/F1. Macro-F1 covers classes present in the ground truth, plus wrongly predicted
    known classes; UNKNOWN abstentions are reported separately rather than averaged in."""
    per_class, f1s = {}, []
    for i, lab in enumerate(labels):
        tp, gt, pred = cm[i, i], cm[i].sum(), cm[:, i].sum()
        p = tp / pred if pred else None
        r = tp / gt if gt else None
        f1 = 2 * p * r / (p + r) if p and r else (0.0 if gt or pred else None)
        per_class[lab] = {"precision": _r(p), "recall": _r(r), "f1": _r(f1), "support_sec": _r(gt)}
        if gt or (pred and lab != UNKNOWN):
            f1s.append(f1)
    return per_class, _r(float(np.mean(f1s)) if f1s else None)


def match_events(gt, pred, tol):
    """Maximum one-to-one matching within tolerance, ties broken by smallest total time error."""
    from scipy.optimize import linear_sum_assignment  # imported late: scipy before torch crashes on some Windows builds

    if not gt or not pred:
        return 0, list(pred), list(gt), []
    diff = np.abs(np.subtract.outer(np.asarray(gt, float), np.asarray(pred, float)))
    cost = np.where(diff <= tol + 1e-6, diff - 1e6, 0.0)
    pairs = [(g, p) for g, p in zip(*linear_sum_assignment(cost)) if cost[g, p] < 0]
    used_g, used_p = {g for g, _ in pairs}, {p for _, p in pairs}
    return (len(pairs), [p for i, p in enumerate(pred) if i not in used_p],
            [g for i, g in enumerate(gt) if i not in used_g], [pred[p] - gt[g] for g, p in pairs])


def error_spans(gt, pred, res, labels, min_sec):
    spans, k = [], 0
    for wrong, group in groupby(gt != pred):
        n = len(list(group))
        if wrong and n * res >= min_sec:
            (g, p), _ = Counter(zip(gt[k:k + n], pred[k:k + n])).most_common(1)[0]
            spans.append({"start_sec": round(k * res, 1), "end_sec": round((k + n) * res, 1),
                          "ground_truth": labels[g], "predicted": labels[p]})
        k += n
    return spans


def _r(v):
    return None if v is None else round(float(v), 3)


def evaluate(pred_dirs, ann_paths, out_dir, tolerance=2.0, resolution=0.1, min_error_sec=2.0, videos=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    k_act, k_bed = len(ACTIVITY_STATES), len(BED_STATES)
    cm_act, cm_bed = np.zeros((k_act, k_act)), np.zeros((k_bed, k_bed))
    counts = {e: {"tp": 0, "fp": 0, "fn": 0, "errors": [], "delays": []} for e in EVENT_TYPES}
    duration_rows, failures, clips, total_T = [], [], [], 0.0

    for n, (pred_dir, ann_path) in enumerate(zip(pred_dirs, ann_paths)):
        gt, pred = load_annotations(ann_path), load_predictions(pred_dir)
        clip, video = str(pred_dir), videos[n] if videos else None
        T = min(gt["duration"], pred["duration"])
        gt, pred = _cut(gt, T), _cut(pred, T)
        total_T += T
        g_act, p_act = (to_grid(x["activity"], T, resolution, ACTIVITY_STATES) for x in (gt, pred))
        g_bed, p_bed = (to_grid(x["bed"], T, resolution, BED_STATES) for x in (gt, pred))
        np.add.at(cm_act, (g_act, p_act), resolution)
        np.add.at(cm_bed, (g_bed, p_bed), resolution)

        for event in EVENT_TYPES:
            g_times = [e["time"] for e in gt["events"] if e["event"] == event]
            p_events = [e for e in pred["events"] if e["event"] == event]
            tp, fp, fn, errs = match_events(g_times, [e["time"] for e in p_events], tolerance)
            c = counts[event]
            c["tp"], c["fp"], c["fn"] = c["tp"] + tp, c["fp"] + len(fp), c["fn"] + len(fn)
            c["errors"] += errs
            c["delays"] += [e["confirmed"] - e["time"] for e in p_events]
            for kind, times, truth, predicted in (("false", fp, "no event", event), ("missed", fn, event, "no event")):
                failures += [{"clip": clip, "video": video, "kind": f"{kind}_{event}", "start_sec": round(t, 1),
                              "end_sec": round(t, 1), "ground_truth": truth, "predicted": predicted} for t in times]

        for layer, keys in (("activity", ACTIVITY_STATES), ("bed", BED_STATES)):
            gd, pd = durations(gt[layer], keys), durations(pred[layer], keys)
            for key in keys:
                duration_rows.append({"clip": clip, "layer": layer, "state": key, "gt_sec": round(gd[key], 1),
                                      "pred_sec": round(pd[key], 1), "signed_error_sec": round(pd[key] - gd[key], 1),
                                      "abs_error_sec": round(abs(pd[key] - gd[key]), 1)})
        longest = [max((s.duration for s in x["bed"] if s.label == OUT_OF_BED), default=0.0) for x in (gt, pred)]
        clips.append({"clip": clip, "analysed_sec": round(T, 1),
                      "activity_accuracy": _r((g_act == p_act).mean()) if len(g_act) else None,
                      "longest_out_of_bed_gt_sec": round(longest[0], 1),
                      "longest_out_of_bed_pred_sec": round(longest[1], 1)})
        failures += [{"clip": clip, "video": video, "kind": "state", **s}
                     for s in error_spans(g_act, p_act, resolution, ACTIVITY_STATES, min_error_sec)]

    act_classes, act_macro = class_metrics(cm_act, ACTIVITY_STATES)
    bed_classes, bed_macro = class_metrics(cm_bed, BED_STATES)
    act_errors = [r["abs_error_sec"] for r in duration_rows
                  if r["layer"] == "activity" and (r["gt_sec"] or r["pred_sec"])]
    total = cm_act.sum()
    metrics = {
        "protocol": {"time_resolution_sec": resolution, "event_tolerance_sec": tolerance,
                     "event_time": "occurrence start (not confirmation)", "clips": len(clips),
                     "analysed_sec": round(total_T, 1)},
        "clips": clips,
        "activity": {"accuracy": _r(np.trace(cm_act) / total) if total else None, "macro_f1": act_macro,
                     "predicted_unknown_share": _r(cm_act[:, ACTIVITY_STATES.index(UNKNOWN)].sum() / total)
                     if total else None,
                     "per_class": act_classes,
                     "confusion_sec": {"labels": list(ACTIVITY_STATES), "rows_gt_cols_pred": cm_act.round(1).tolist()}},
        "bed_status": {"accuracy": _r(np.trace(cm_bed) / cm_bed.sum()) if cm_bed.sum() else None,
                       "macro_f1": bed_macro, "per_class": bed_classes,
                       "confusion_sec": {"labels": list(BED_STATES), "rows_gt_cols_pred": cm_bed.round(1).tolist()}},
        "events": {},
        "duration": {"activity_macro_abs_error_sec": _r(np.mean(act_errors)) if act_errors else None},
    }
    hours = total_T / 3600
    for event, c in counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        metrics["events"][event] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": _r(tp / (tp + fp)) if tp + fp else None,
            "recall": _r(tp / (tp + fn)) if tp + fn else None,
            "false_per_hour": _r(fp / hours) if hours else None,
            "mean_abs_time_error_sec": _r(np.mean(np.abs(c["errors"]))) if c["errors"] else None,
            "mean_confirmation_delay_sec": _r(np.mean(c["delays"])) if c["delays"] else None,
        }

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with open(out / "duration_errors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(duration_rows[0]) if duration_rows else ["clip"])
        w.writeheader()
        w.writerows(duration_rows)
    _write_failures(out, failures)
    _plot_confusion(cm_act, ACTIVITY_STATES, out / "confusion_matrix.png")
    return metrics


def _write_failures(out, failures):
    events = [f for f in failures if f["kind"] != "state"]
    states = sorted((f for f in failures if f["kind"] == "state"), key=lambda f: f["end_sec"] - f["start_sec"],
                    reverse=True)
    cases = events + states[:10]
    with open(out / "failure_candidates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "kind", "start_sec", "end_sec", "ground_truth", "predicted"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(events + states)
    lines = ["# Failure cases", "",
             "Generated from real disagreements with the ground truth: every false / missed bed event, then the "
             "longest wrong-state spans. Fill in cause, fix and rerun result after inspecting each snapshot.", ""]
    for n, fc in enumerate(cases, 1):
        snap = _snapshot(out, n, fc)
        lines += [f"## Case {n}: {fc['kind']} in `{fc['clip']}`", "",
                  f"- **Span:** {fmt_clock(fc['start_sec'])} – {fmt_clock(fc['end_sec'])} "
                  f"({fc['end_sec'] - fc['start_sec']:.1f}s)",
                  f"- **Ground truth:** {fc['ground_truth']}",
                  f"- **Predicted:** {fc['predicted']}",
                  f"- **Evidence snapshot:** {snap or 'n/a (pass --videos to save frames)'}",
                  "- **Effect on metric:** ",
                  "- **Cause hypothesis:** ",
                  "- **Proposed fix:** ",
                  "- **Rerun result:** ", ""]
    (out / "failure_cases.md").write_text("\n".join(lines), encoding="utf-8")


def _snapshot(out, n, fc):
    if not fc.get("video"):
        return None
    frame = VideoReader(fc["video"]).frame_at((fc["start_sec"] + fc["end_sec"]) / 2)
    if frame is None:
        return None
    (out / "failures").mkdir(exist_ok=True)
    name = f"failures/case_{n:02d}.jpg"
    cv2.imwrite(str(out / name), frame)
    return name


def _plot_confusion(cm, labels, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, f"{cm[i, j]:.0f}s", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_title("Activity confusion (seconds, colour = row share)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
