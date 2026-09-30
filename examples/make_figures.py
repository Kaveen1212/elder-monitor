"""Figures for FAILURE_CASES.md: ground truth vs. the timeline before and after the fixes, plus skeletons drawn from
the cached keypoints. Run after run_examples.py (it needs outputs/<clip>/observations.jsonl). No video frames are used.
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle

from elder_monitor.config import load_config
from elder_monitor.features import add_box_speed
from elder_monitor.reporting import read_observations
from elder_monitor.temporal import propose_all

ROOT = Path(__file__).resolve().parent
COLORS = {"LYING_IN_BED": "#4c72b0", "SITTING_ON_BED": "#55a868", "SITTING_OUTSIDE_BED": "#8172b2",
          "STANDING": "#dd8452", "WALKING": "#c44e52", "OUT_OF_BED": "#937860", "LYING_ON_FLOOR": "#222222",
          "UNKNOWN": "#c8c8c8"}
BONES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
         (0, 1), (0, 2), (1, 3), (2, 4)]
LEGS = {11, 12, 13, 14, 15, 16}
KP_CONF = 0.3

CASES = {
    "case1_edge_sit": {
        "title": "Case 1 (fixed with the VLM): edge-sitting facing the camera read as standing",
        "clip": "pexels_4049556", "window": (8.4, 10.8),
        "picks": [(11.8, "GT sitting: read correctly"), (8.6, "GT sitting: legs look straight"),
                  (9.4, "GT sitting: worst reading"), (14.0, "GT standing: real stand-up")]},
    "case2_return_late": {
        "title": "Case 2 (fixed): return to bed detected 6 s late", "clip": "roundtrip_4049556", "window": (29.8, 32.4),
        "picks": [(26.2, "GT standing: walking back in"), (27.0, "first sit-down"),
                  (30.0, "GT sitting: legs look straight"), (32.4, "second sit-down")]},
    "case3_walk_unknown": {
        "title": "Case 3 (fixed): walking out past the camera became UNKNOWN", "clip": "pexels_4052924",
        "window": (25.0, 26.0),
        "picks": [(24.8, "GT walking: last full reading"), (25.2, "GT walking: 4 scattered joints"),
                  (25.4, "GT walking: no joints, box moving"), (26.0, "after he left")]},
    "case4_close_up": {
        "title": "Case 4 (partly fixed): handheld close-up from behind, mirror in shot", "clip": "pexels_3753707",
        "window": (11.4, 17.4), "visual": [(0.0, 21.44, "SITTING_ON_BED")],
        "picks": [(1.4, "close-up while the camera moves"), (6.4, "measured: seated, hips far left of bed"),
                  (11.2, "last sample before gap 2"), (14.4, "inside gap 2: still tracked")]},
}


def segments(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [(float(r["start_sec"]), float(r["end_sec"]), r["label"]) for r in csv.DictReader(f)]


def final_proposals(clip):
    with open(ROOT / "outputs" / clip / "proposals.csv", newline="", encoding="utf-8") as f:
        return {round(float(r["t"]), 1): r for r in csv.DictReader(f)}


def strip(ax, rows, window, gt_events):
    for y, (_, segs, evs) in enumerate(rows):
        for a, b, lab in segs:
            ax.barh(y, b - a, left=a, color=COLORS[lab], edgecolor="white", height=0.8, zorder=2)
        for t, name in evs:
            ax.vlines(t, y - 0.45, y + 0.45, color="#b22222", lw=2, zorder=3)
            ax.text(t, y + 0.42, f" {name} {t:.1f}s", fontsize=6.5, color="#b22222", va="bottom", zorder=4)
    for t, name in gt_events:
        ax.axvline(t, color="black", ls="--", lw=1, zorder=3)
        ax.text(t, -0.5, f"GT {name} {t:.1f}s", fontsize=7, ha="center", va="bottom")
    ax.set_yticks(range(len(rows)), [name for name, _, _ in rows], fontsize=8)
    ax.axvspan(*window, color="#f2c14e", alpha=0.35, lw=0, zorder=0)
    ax.set_ylim(len(rows) - 0.4, -0.9)
    ax.set_xlabel("seconds", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)


def skeleton(ax, obs, rule, final, meta, bed, caption):
    w, h = meta["width"], meta["height"]
    ax.add_patch(Rectangle((0, 0), w, h, fill=False, ec="#999999", lw=1))
    ax.add_patch(Polygon([(x * w, y * h) for x, y in bed], closed=True, fc="#4c72b0", alpha=0.12, ec="#4c72b0"))
    if obs.bbox:
        x1, y1, x2, y2 = obs.bbox
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="#dd8452", lw=1.2, ls="--"))
    kp = obs.keypoints or []
    for a, b in BONES:
        if kp and kp[a][2] >= KP_CONF and kp[b][2] >= KP_CONF:
            ax.plot([kp[a][0], kp[b][0]], [kp[a][1], kp[b][1]], color="#c44e52" if {a, b} <= LEGS else "#333333", lw=2)
    for i, (x, y, c) in enumerate(kp):
        if c >= KP_CONF:
            ax.plot(x, y, "o", ms=3, color="#c44e52" if i in LEGS else "#333333")

    def fmt(v, digits=0):
        return "–" if v is None else f"{v:.{digits}f}"

    joints = sum(c >= KP_CONF for _, _, c in kp[5:17]) if kp else 0
    anchor = "–" if obs.bed_dist is None else f"{obs.bed_dist:+.0f} px"
    result = rule.label if rule.label == final["label"] else f"{rule.label} → agent: {final['label']}"
    ax.set_title(f"{obs.t:.1f} s  {caption}\nrules: {result}", fontsize=8)
    ax.text(0.01, -0.02, f"knee {fmt(obs.knee_angle)}°   thigh/shin {fmt(obs.thigh_ratio, 2)}   "
                         f"torso {fmt(obs.torso_angle)}°\nbody joints {joints}   anchor vs bed {anchor}\n"
                         f"speed {obs.speed:.2f}   box speed {obs.box_speed:.2f}", transform=ax.transAxes, fontsize=7,
            va="top")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    out = ROOT / "figures"
    before = json.loads((out / "before_fix.json").read_text(encoding="utf-8"))
    for name, spec in CASES.items():
        clip = spec["clip"]
        cfg = load_config(ROOT / "configs" / f"pexels_{clip.split('_')[1]}.yaml")
        meta, obs = read_observations(ROOT / "outputs" / clip / "observations.jsonl", {})
        add_box_speed(obs, cfg["posture"]["box_speed_window_sec"], cfg["posture"]["speed_window_sec"])
        rules, finals = propose_all(obs, cfg), final_proposals(clip)
        ann_path = ROOT / "annotations" / f"{clip}.json"
        ann = json.loads(ann_path.read_text(encoding="utf-8")) if ann_path.exists() else None
        events = json.loads((ROOT / "outputs" / clip / "events.json").read_text(encoding="utf-8"))["bed_events"]
        rows = []
        if ann:
            truth = [(float(a["start"]), float(a["end"]), a["label"]) for a in ann["activity"]]
            rows.append(("ground truth", truth, []))
        if spec.get("visual"):
            rows.append(("visual check", spec["visual"], []))
        rows.append(("before the fix", [tuple(s) for s in before[clip]["activity"]],
                     [(t, e.replace("_", " ")) for t, e in before[clip]["events"]]))
        rows.append(("after the fix", segments(ROOT / "outputs" / clip / "timeline.csv"),
                     [(e["start_sec"], e["event"].replace("_", " ")) for e in events]))
        gt_events = [(float(e["time"]), e["event"].replace("_", " ")) for e in (ann or {}).get("events", [])]

        picks = spec["picks"]
        fig = plt.figure(figsize=(3.3 * len(picks), 5.8))
        grid = fig.add_gridspec(2, len(picks), height_ratios=[1.3, 2.2])
        strip(fig.add_subplot(grid[0, :]), rows, spec["window"], gt_events)
        for k, (t, caption) in enumerate(picks):
            i = min(range(len(obs)), key=lambda n: abs(obs[n].t - t))
            skeleton(fig.add_subplot(grid[1, k]), obs[i], rules[i], finals[round(obs[i].t, 1)], meta,
                     cfg["scene"]["bed_polygon"], caption)
        handles = [Patch(color=c, label=s) for s, c in COLORS.items()] + [
            Patch(fc="#4c72b0", alpha=0.25, label="bed polygon"),
            Patch(fill=False, ec="#dd8452", ls="--", label="person box"),
            Line2D([], [], color="#c44e52", lw=2, label="legs"),
            Line2D([], [], color="#333333", lw=2, label="upper body")]
        fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7, frameon=False)
        fig.suptitle(f"{spec['title']} ({clip}); highlighted: the span that failed before the fix", fontsize=10)
        fig.tight_layout(rect=(0, 0.08, 1, 0.97))
        fig.savefig(out / f"{name}.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
