import time
from bisect import bisect_left, bisect_right
from collections import Counter
from itertools import groupby

import cv2

from .events import departure
from .schemas import (
    BED_EXIT, IN_BED_STATES, LYING_IN_BED, LYING_ON_FLOOR, LYING_STATES, SITTING_ON_BED, STANDING, UNKNOWN, Proposal,
    Verdict,
)
from .temporal import smooth
from .vlm import answer_state

UNCONFIRMED = ("identity_uncertain", "low_confidence")


def _step(action, window, finding):
    return {"action": action, "window": [round(max(window[0], 0.0), 2), round(window[1], 2)], "finding": finding}


class ContextAgent:
    """Decides when a proposal needs more temporal or visual context, picks bounded tools to get it,
    and hands the evidence back. The temporal engine and the bed-event FSM still commit.

    `props` is what the activity timeline is built from, bridged gaps included. `evidence` is what each sample
    itself shows: its own posture, an explicit re-read of an observed posture, or a VLM answer about that very
    frame. Bridging a gap changes `props` only, so event timers never count inferred time."""

    def __init__(self, cfg, observations, proposals, duration, frames=None, scene=None, vlm_factory=None):
        self.cfg, self.a = cfg, cfg["agent"]
        self.obs, self.props = observations, list(proposals)
        self.rules = [p.label for p in proposals]
        self.evidence = list(self.rules)
        self.inspected = []
        self.times = [p.t for p in proposals]
        self.dt = 1.0 / cfg["sampling"]["fps"]
        self.duration = duration
        self.frames, self.scene = frames, scene
        self.vlm_factory, self.vlm, self.vlm_calls, self.use_vlm = vlm_factory, None, 0, vlm_factory is not None
        self.max_vlm_calls = cfg["vlm"]["max_calls"]
        self.vlm_load_sec = self.vlm_inference_sec = 0.0
        self.trace, self._baselines, self._vlm_cache = [], {}, {}

    def review(self):
        for i, j in self._runs(LYING_ON_FLOOR, self.cfg["temporal"]["dwell_sec"][LYING_ON_FLOOR]):
            self._resolve_lying_outside(i, j)
        for i, j in self._runs(UNKNOWN, self.a["min_gap_sec"]):
            self._resolve_gap(i, j)
        if self.frames is not None and self.use_vlm:
            for i, j in self._runs(STANDING, self.cfg["temporal"]["dwell_sec"]["default"]):
                self._resolve_upright_on_bed(i, j)
        return self.props

    def look_back(self, t, seconds, nearest=True):
        i, j = bisect_left(self.times, t - seconds), bisect_left(self.times, t)
        return self._summarise([p for p in self.props[i:j] if p.label != UNKNOWN], t, True, nearest)

    def look_ahead(self, t, seconds):
        i, j = bisect_left(self.times, t), bisect_right(self.times, t + seconds)
        return self._summarise([p for p in self.props[i:j] if p.label != UNKNOWN], t, False, True)

    def inspect_bed_relation(self, i, j):
        seen = [o for o in self.obs[i:j] if o.visible]
        if not seen:
            return {"relation": "unobserved", "body_in_bed": 0.0}
        body = sum(o.body_in_bed for o in seen) / len(seen)
        near = sum(o.near_bed for o in seen) / len(seen)
        relation = "on_bed" if body >= self.cfg["posture"]["in_bed_fraction"] else \
            "beside_bed" if near >= 0.5 else "away_from_bed"
        return {"relation": relation, "body_in_bed": round(body, 2), "near_bed": round(near, 2)}

    def inspect_frames(self, times):
        if self.frames is None or self.vlm_calls + len(times) > self.max_vlm_calls:
            return []
        vlm = self._load_vlm()
        if vlm is None:
            return []
        answers = []
        for t in times:
            key = round(t, 2)
            if key not in self._vlm_cache:
                frame = self.frames(t)
                if frame is None:
                    return []
                self.vlm_calls += 1
                started = time.perf_counter()
                try:
                    ans, raw = vlm.ask(self._crop(frame, t))
                except Exception as exc:
                    print(f"[agent] VLM call failed, continuing with geometry only: {exc}")
                    self.vlm = None
                    return []
                finally:
                    self.vlm_inference_sec += time.perf_counter() - started
                self._vlm_cache[key] = {"t": key, "answer": ans} if ans else {"t": key, "answer": None,
                                                                              "raw": raw[:200]}
            answers.append(self._vlm_cache[key])
        return answers

    def verify_exit(self, t_pending, t_now, t_limit):
        if t_pending not in self._baselines:
            steps = []
            self._baselines[t_pending] = self._look(t_pending, steps, ahead=False, nearest=False), steps
        back, base_steps = self._baselines[t_pending]
        steps, end = list(base_steps), t_now
        for _ in range(max(self.a["max_rounds"], 1)):
            end = min(end + self.cfg["events"]["lookahead_sec"], t_limit)
            t, conf, stats = departure(self.obs, t_pending, end, self.cfg)
            steps.append(_step("look_ahead", (t_now, end), self._describe_departure(t, stats)))
            if t is not None or end >= t_limit:
                break
        if t is not None and (not back or back["label"] not in IN_BED_STATES):
            conf *= 0.8
        return Verdict(t, end, conf, steps)

    def log_exit_candidates(self, fsm):
        """One trace record per exit candidate, written from the state machine's final outcome."""
        for e in fsm["events"]:
            if e.event == BED_EXIT:
                self._log("EXIT_CANDIDATE", e.start_sec, e.confirmed_sec, e.evidence, "BED_EXIT confirmed")
        for r in fsm["rejected"]:
            self._log("EXIT_CANDIDATE", r["start_sec"], r["end_sec"], r["evidence"],
                      "back in bed before departing; not an exit")
        for u in fsm["unresolved"]:
            if u["event"] == BED_EXIT:
                self._log("EXIT_CANDIDATE", u["start_sec"], self.duration, u["evidence"],
                          "video ended before departure was confirmed")

    def _resolve_lying_outside(self, i, j):
        """Lying outside the bed polygon stays LYING_ON_FLOOR (fail-safe) unless the body overlaps the
        mattress and the VLM or the preceding lying-in-bed state says it is still the bed. Samples fully
        off the mattress are never relabelled to the bed."""
        t0, t1 = self.times[i], self.times[j - 1] + self.dt
        steps = []
        geo = self.inspect_bed_relation(i, j)
        steps.append(_step("inspect_bed_relation", (t0, t1), geo))
        label, why = LYING_ON_FLOOR, "lying outside the bed"
        if geo["body_in_bed"] >= self.cfg["posture"]["in_bed_fraction"] / 2:
            state = self._vlm_consensus(i, j, steps)
            if state in (LYING_IN_BED, LYING_ON_FLOOR):
                label, why = state, "VLM support check"
            elif state is None:
                back = self._look(t0, steps, ahead=False)
                if back and back["label"] == LYING_IN_BED:
                    label, why = LYING_IN_BED, "continues in-bed lying; body overlaps the bed outline"
        seen = set(self.inspected) if why == "VLM support check" else set()
        for k, o in enumerate(self.obs[i:j], i):
            if label == LYING_ON_FLOOR or self._target_box(o) and (not o.visible or o.body_in_bed > 0):
                self._relabel(k, label, 0.5, why, k in seen or self.rules[k] in LYING_STATES)
        self._log("LYING_OUTSIDE_BED", t0, t1, steps, label, why)

    def _resolve_upright_on_bed(self, i, j):
        """STANDING that stays over the bed without moving, between two spells of sitting on it: facing the camera,
        seated legs can look straight, so the VLM must confirm the sitting before the run is relabelled."""
        t0, t1 = self.times[i], self.times[j - 1] + self.dt
        seen = [o for o in self.obs[i:j] if o.visible]
        if not seen:
            return
        hips = sum(o.hip_in_bed for o in seen) / len(seen)
        still = max(o.speed for o in seen) < self.cfg["posture"]["walk_enter_speed"]
        if hips < self.a["upright_hips_share"] or not still or t1 - t0 >= self.cfg["events"]["exit_dwell_sec"] - 1e-9:
            return
        steps = [_step("inspect_bed_relation", (t0, t1), {"hips_over_bed": round(hips, 2), "still": still})]
        back, ahead = self._look(t0, steps, ahead=False), self._look(t1, steps, ahead=True)
        if not (back and ahead and back["label"] == ahead["label"] == SITTING_ON_BED) or not self._sits_after(j):
            return
        label, why = STANDING, "kept: the VLM did not confirm sitting on the bed"
        if self._vlm_consensus(i, j, steps) == SITTING_ON_BED:
            label, why = SITTING_ON_BED, "VLM: seated on the bed edge; the legs only look straight"
            for k in range(i, j):
                if self.props[k].label in (STANDING, UNKNOWN) and self.obs[k].visible:
                    self._relabel(k, SITTING_ON_BED, 0.45, why, k in self.inspected or self.rules[k] == STANDING)
        self._log("UPRIGHT_ON_BED_REGION", t0, t1, steps, label, why)

    def _sits_after(self, j):
        """The rule proposals right after the run (skipping dropouts) sit on the bed for at least the dwell time."""
        labels = smooth(self.rules, self.cfg["temporal"]["smooth_samples"])
        k = next((k for k in range(j, len(labels)) if labels[k] != UNKNOWN), len(labels))
        m = next((m for m in range(k, len(labels)) if labels[m] != SITTING_ON_BED), len(labels))
        need = self.cfg["temporal"]["dwell_sec"]["default"]
        return m > k and self.times[m - 1] + self.dt - self.times[k] >= need - 1e-9

    def _resolve_gap(self, i, j):
        t0, t1 = self.times[i], self.times[j - 1] + self.dt
        steps = []
        cause = Counter(p.reason for p in self.props[i:j]).most_common(1)[0][0]
        back = self._look(t0, steps, ahead=False)
        ahead = self._look(t1, steps, ahead=True)
        in_bed_both = bool(back and ahead and back["label"] in IN_BED_STATES and ahead["label"] in IN_BED_STATES)
        occluded = cause in UNCONFIRMED or (in_bed_both and back["label"] == ahead["label"] == LYING_IN_BED)
        last_seen = next((o for o in reversed(self.obs[:i]) if o.visible), None)
        still_detected = 2 * sum(o.bbox is not None and o.identity_ok for o in self.obs[i:j]) >= j - i
        left_view = last_seen is not None and last_seen.truncated and not last_seen.hip_in_bed and not still_detected
        label, why, bridged, seen = UNKNOWN, "", False, set()
        if in_bed_both and occluded and not left_view and t1 - t0 <= self.a["max_bridge_sec"]:
            label, why, bridged = back["label"], "in bed before and after a short occlusion", True
        elif (in_bed_both or cause not in UNCONFIRMED) and not left_view:
            state = self._vlm_consensus(i, j, steps)
            if state and (not in_bed_both or state in IN_BED_STATES or state == LYING_ON_FLOOR):
                label, why, seen = state, "VLM check across the gap", set(self.inspected)
        if label == UNKNOWN:
            why = "resident left the camera view" if left_view else f"insufficient evidence ({cause})"
            steps.append(_step("abstain", (t0, t1), why))
        else:
            for k, o in enumerate(self.obs[i:j], i):
                if bridged or self._target_box(o):
                    self._relabel(k, label, 0.45, why, k in seen)
        self._log("AMBIGUOUS_GAP", t0, t1, steps, label, why)

    def _look(self, t, steps, ahead, nearest=True):
        tool, key = ("look_ahead", "lookahead_sec") if ahead else ("look_back", "lookback_sec")
        found = None
        for r in range(1, max(self.a["max_rounds"], 1) + 1):
            s = self.a[key] * r
            found = self.look_ahead(t, s) if ahead else self.look_back(t, s, nearest)
            steps.append(_step(tool, (t, t + s) if ahead else (t - s, t), found or "no supported state"))
            if found:
                break
        return found

    def _summarise(self, known, t, last, nearest):
        """The smoothed state nearest to t (or the window's majority state), with its share of the window."""
        if not known:
            return None
        labels = smooth([p.label for p in known], self.cfg["temporal"]["smooth_samples"])
        label = (labels[-1] if last else labels[0]) if nearest else Counter(labels).most_common(1)[0][0]
        hits = [p.t for p, lab in zip(known, labels) if lab == label]
        gap = t - hits[-1] if last else hits[0] - t
        key = "seconds_before" if last else "seconds_after"
        return {"label": label, "share": round(labels.count(label) / len(labels), 2), key: round(gap, 1)}

    def _vlm_consensus(self, i, j, steps):
        """Ask about vlm_frames samples spread over the run where the resident is identified; all must agree."""
        n, window = self.a["vlm_frames"], (self.times[i], self.times[j - 1] + self.dt)
        ks = [k for k in range(i, j) if self._target_box(self.obs[k])]
        self.inspected = []
        if len(ks) < n:
            if self.use_vlm and self.frames is not None:
                steps.append(_step("inspect_target_crop", window, "skipped: resident not identified in enough samples"))
            return None
        picks = [ks[(m + 1) * len(ks) // (n + 1)] for m in range(n)]
        answers = self.inspect_frames([self.times[k] for k in picks])
        if len(answers) < n:
            if self.use_vlm:
                steps.append(_step("inspect_target_crop", window, "skipped: VLM unavailable or budget exhausted"))
            return None
        states = [answer_state(a["answer"]) for a in answers]
        state = states[0] if states[0] and all(s == states[0] for s in states) else None
        steps.append(_step("inspect_target_crop", window, {"answers": answers, "state": state}))
        self.inspected = picks if state else []
        return state

    def _load_vlm(self):
        if self.vlm is None and self.vlm_factory is not None:
            factory, self.vlm_factory = self.vlm_factory, None
            started = time.perf_counter()
            try:
                self.vlm = factory()
            except Exception as exc:
                print(f"[agent] VLM unavailable, continuing with geometry only: {exc}")
            self.vlm_load_sec = time.perf_counter() - started
        return self.vlm

    def _target_box(self, o):
        """What the VLM is asked about: the resident's own tracked box, or the bed when nobody at all was detected.
        Samples before the resident is selected, or with an unconfirmed person, have none."""
        if o.identity_ok and o.bbox is not None:
            return o.bbox
        return self.scene.bed_box() if self.scene and o.reason == "not_detected" else None

    def _crop(self, frame, t):
        """The resident's box and the bed with some context, the target outlined in green for the prompt."""
        o = self.obs[min(bisect_left(self.times, t), len(self.obs) - 1)]
        target = self._target_box(o)
        boxes = [b for b in (target, self.scene.bed_box() if self.scene else None) if b]
        h, w = frame.shape[:2]
        x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        px, py = 0.15 * (x2 - x1), 0.15 * (y2 - y1)
        x1, y1, x2, y2 = int(max(x1 - px, 0)), int(max(y1 - py, 0)), int(min(x2 + px, w)), int(min(y2 + py, h))
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, w, h
        crop = frame[y1:y2, x1:x2].copy()
        thick = max(2, round(0.004 * max(x2 - x1, y2 - y1)))
        cv2.rectangle(crop, (int(target[0]) - x1, int(target[1]) - y1), (int(target[2]) - x1, int(target[3]) - y1),
                      (0, 255, 0), thick)
        return crop

    def _relabel(self, k, label, confidence, why, observed):
        """Change the timeline's proposal; the sample becomes evidence only when it was itself observed."""
        self.props[k] = Proposal(self.times[k], label, confidence, why, "agent")
        if observed:
            self.evidence[k] = label

    def _runs(self, label, min_sec):
        """Runs of a smoothed label; runs of a known label separated only by short dropouts are joined."""
        labels = smooth([p.label for p in self.props], self.cfg["temporal"]["smooth_samples"])
        runs, i = [], 0
        for lab, group in groupby(labels):
            j = i + len(list(group))
            if lab == label:
                prev = runs[-1][1] if runs else None
                if (prev is not None and label != UNKNOWN and set(labels[prev:i]) == {UNKNOWN}
                        and self.times[i] - self.times[prev] < self.a["min_gap_sec"] - 1e-9):
                    runs[-1] = (runs[-1][0], j)
                else:
                    runs.append((i, j))
            i = j
        return [(i, j) for i, j in runs if self.times[j - 1] + self.dt - self.times[i] >= min_sec - 1e-9]

    @staticmethod
    def _describe_departure(t, stats):
        if t is not None and stats["left_view"]:
            return f"walks away and leaves the camera view; departure confirmed at {t:.1f}s"
        if t is not None:
            return f"walks away from the bed; departure confirmed at {t:.1f}s"
        total = stats["visible"] + stats["unseen"] or 1
        if stats["unseen"] > total / 2:
            return f"target not observed in {stats['unseen']}/{total} samples"
        if stats["near_bed"] > stats["visible"] / 2:
            return f"stays beside the bed ({stats['near_bed']}/{stats['visible']} visible samples)"
        return "moved away from the bed but not observed long enough"

    def _log(self, trigger, t0, t1, steps, outcome, why=""):
        self.trace.append({"trigger": trigger, "window": [round(t0, 2), round(t1, 2)], "steps": steps,
                           "outcome": outcome, "reason": why, "vlm_calls_total": self.vlm_calls})
