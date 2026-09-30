# Data, formats and conventions

## Inputs

| Input | Format | Notes |
|---|---|---|
| Video | Any file OpenCV can decode (`.mp4`, `.avi`, …) | A fixed camera that sees the whole bed and, ideally, the resident's full body. |
| Scene config | YAML merged over [src/elder_monitor/default.yaml](../src/elder_monitor/default.yaml) | Written by `python -m elder_monitor calibrate`. Only `scene.bed_polygon` is required. |
| Labels (evaluation only) | JSON, see [annotations/TEMPLATE.json](../annotations/TEMPLATE.json) | Activity intervals, bed events, `provenance`. Only `evaluation.py` reads them. |

Scene config keys:

| Key | Meaning |
|---|---|
| `scene.bed_polygon` | Corners of the visible mattress in normalised image coordinates `[x, y]`, 0–1, in order around it. At least 3 points, non-zero area, no self-crossing. |
| `scene.chair_polygons` | Optional chairs; hips inside one count as sitting outside the bed. |
| `scene.ignore_polygons` | Optional mirrors or screens; a detection mostly inside one is dropped unless it is the tracked resident. |
| `target.point`, `target.time_sec` | Optional click on the resident at a given time, for scenes with several people. Before that time nobody is selected. Without a point, the resident is the most confident person in the bed, or the only person in view. |
| `sampling.fps` | Analysis grid, default 5 fps (lowered to the video's rate if that is slower). |

`validate_config` checks every setting before a video is opened and names the key when one is wrong.

## Time conventions

- All times are seconds from the start of the recording, taken from the decoded frame timestamps. Samples lie on
  the `sampling.fps` grid; a grid slot with no decodable frame becomes a `no_frame` sample.
- In live mode the time is the server's receive time since the `start` message, rounded to the nearest grid slot.
  A second frame for a slot is dropped; an empty slot becomes a `no_frame` sample.
- Segments are half-open intervals `[start, end)` that cover the whole recording without gaps, so durations add
  up to its length.
- An event has two times. `start_sec` is when the transition happened, the backdated boundary of the committed
  timeline; this is what evaluation matches against the labels. `confirmed_sec` is when enough evidence had been
  seen to report it.
- Evidence is counted in grid slots: each sample that fills a new slot adds one sampling step (0.2 s at 5 fps).
  Duplicate frames and empty slots add nothing.

## Data classes ([schemas.py](../src/elder_monitor/schemas.py))

### `Observation`, one per sample

| Field | Meaning |
|---|---|
| `t`, `frame_idx` | Sample time; decoded frame index (−1 in live mode or for a missing frame). |
| `visible` | The selected resident has enough body keypoints (`vision.min_body_keypoints`) to measure a posture. |
| `identity_ok` | False when a detection is present but cannot be confirmed as the resident (`identity_uncertain`, `low_confidence`). |
| `reason` | Why a sample is not `visible`: `not_detected` (nobody detected), `low_confidence` (only weak boxes), `identity_uncertain` (a confident person who cannot be confirmed as the resident), `low_keypoints` (the resident's box with too few joints), `target_not_selected` (before the resident is chosen), `no_frame` (no frame for the slot). Empty when visible. |
| `track_id`, `n_persons`, `det_conf`, `bbox`, `keypoints`, `kp_conf`, `truncated` | Tracker ID of the selected box, detections in the frame, box confidence, box, the 17 COCO keypoints `[x, y, conf]`, their mean confidence over the body joints, and whether the box touches the frame border. |
| `torso_angle`, `torso_len`, `shoulder_width`, `leg_angle`, `knee_angle`, `thigh_ratio`, `knee_drop`, `thigh_angle` | Posture measurements; angles in degrees from vertical, lengths in pixels. |
| `anchor`, `body_in_bed`, `hip_in_bed`, `feet_in_bed`, `bed_dist`, `near_bed`, `edge_of_bed`, `in_chair` | Bed geometry: hip midpoint (or box centre), share of joints inside the bed polygon, signed distance to it (positive inside), and the derived flags. |
| `speed`, `box_speed` | Hip speed in torso lengths per second over 1 s; box-centre speed, used when joints drop out. |

### `Proposal`, one per sample

`t`, `label` (an activity state), `confidence`, `reason` (the rule that fired, or the agent's reason) and `source`
(`rules` or `agent`). Next to the proposals, the analysis keeps one *evidence* label per sample: what that sample
itself shows. It is the rule proposal, an agent re-read of an observed posture, or a VLM answer about that very
frame, and `UNKNOWN` where nothing was observed. Bridged gaps change the proposal but never the evidence.

### `Segment`

`start`, `end`, `label`, `confidence` (mean proposal confidence of the samples that agree with the label) and,
for decisions, `reasons` (the rules active in that span).

### `BedEvent`

| Field | Meaning |
|---|---|
| `event` | `bed_exit` or `return_to_bed`. |
| `episode_id` | Away episode number; an exit and its return share it. |
| `start_sec`, `confirmed_sec` | Occurrence and confirmation time (see above). |
| `previous_state`, `current_state` | Activity before the transition and at confirmation. |
| `confidence` | A heuristic score from 0 to 1, **not a calibrated probability**. For an exit, the mean keypoint confidence of the samples that confirmed the departure, × 0.7 when it was confirmed by walking out of view, × 0.8 when the look-back found no in-bed state; for the 30 s dwell fallback, 0.8 × the proposal confidence. For a return, the mean proposal confidence from its start to its confirmation. |
| `decision` | NORMAL / MONITOR / ALERT when it was confirmed. |
| `evidence` | The agent's steps (look-back, look-ahead, dwell, keep-return-start) behind it. |

### `Verdict`

Returned by an exit check: `time` (departure time or `None`), `until` (the last time looked at), `confidence`,
`evidence` (steps).

## Outputs (`outputs/<run>/`)

| File | Contents |
|---|---|
| `timeline.txt`, `timeline.csv` | Activity segments: `start`, `end` (clock), `start_sec`, `end_sec`, `duration_sec`, `label`, `confidence`. |
| `bed_timeline.csv` | Bed status segments `IN_BED` / `OUT_OF_BED` / `UNKNOWN`. |
| `decisions.csv` | Decision segments with `reasons` (`;`-separated rule names). |
| `events.json` | `bed_events` (the `BedEvent` fields plus `start_time` / `confirmed_time` as `hh:mm:ss`), `alerts` (one record per rule episode: `rule`, `decision`, `start_sec`, `end_sec`), `rejected_exit_candidates`, `unresolved_candidates`. |
| `summary.json` | Seconds per activity and bed status, event and alert counts, final state and decision, `view_quality`, and the absence measures below. A `human` block repeats the totals as `11m 42s`. |
| `agent_trace.jsonl` | One record per review: `trigger`, `window`, `steps` (tool, window, finding), `outcome`, `reason`, running VLM call count. |
| `proposals.csv` | `t`, `label`, `confidence`, `reason`, `source`, `evidence` per sample. |
| `observations.jsonl` | A `meta` line (video hash, perception hash, pose weights, perception time, creation time, git commit, library versions) followed by one `Observation` per line. It is the perception cache. |
| `run_manifest.json` | Video and config hashes, `pose_model` (weights file and sha256 prefix), `vlm_model` and `vlm_revision`, VLM calls, perception / analysis / VLM load / VLM inference seconds, `reused_observations`, a `perception` block naming the run that produced the observations, `source` (git commit, `dirty`, `src_dirty` for this run), library `versions`, and the full config. |
| `overlay.mp4` | Optional: the frames with bed polygon, box, keypoints, state and decision. |

Absence measures in `summary.json`:

- `longest_out_of_bed_period_sec`: the longest continuous `OUT_OF_BED` span of the committed bed timeline. It
  breaks at `UNKNOWN`, so it measures observed time out of bed.
- `away_episodes`: one record per confirmed exit, from its start to the start of its return (or the end of the
  recording). Each has `elapsed_sec`, `observed_out_of_bed_sec`, `unknown_sec` and `ended_by`.
  `longest_away_episode_sec` is the largest `elapsed_sec`.

Evaluation writes `metrics.json` (protocol, per-clip accuracy, activity and bed-status accuracy, per-class
precision / recall / F1, confusion in seconds, event TP / FP / FN with timing errors, confirmation delays and
`missed_with_short_context`, duration errors and per-state totals), `duration_errors.csv`, `failure_candidates.csv`
and `confusion_matrix.png`.
