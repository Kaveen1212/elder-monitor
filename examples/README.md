# Example run on public footage

```bash
python examples/run_examples.py --vlm             # development clips (or without --vlm for geometry-only)
python examples/run_examples.py --heldout --vlm   # 13 held-out clips never used to tune thresholds (see below)
```

The script downloads six short [Pexels](https://www.pexels.com/license/) clips (free licence) into `examples/data/`,
builds one constructed clip, analyses each clip with the full system (`outputs/`) and with `--no-agent`
(`outputs_baseline/`, reusing the full run's observations, so both see the same perception), and evaluates the
labelled ones (`evaluation/`). Videos and failure snapshots are not committed; the script regenerates them.

| Clip | Content | Labels | Video file |
|---|---|---|---|
| `pexels_4049556` | woman lies, sits up, sits on the edge facing the camera, stands, walks out towards the camera | yes | [mp4](https://videos.pexels.com/video-files/4049556/4049556-hd_1920_1080_30fps.mp4) |
| `pexels_4052924` | man sits in bed, gets up, walks out past the camera | yes | [mp4](https://videos.pexels.com/video-files/4052924/4052924-hd_1920_1080_25fps.mp4) |
| `pexels_8090735` | woman sits on the bed edge (low camera), stands, walks out | yes | [mp4](https://videos.pexels.com/video-files/8090735/8090735-hd_1920_1080_24fps.mp4) |
| `roundtrip_4049556` | constructed: `4049556` forwards, 5 s of the empty room, then the same footage reversed, so its return is the exit played backwards | yes | built by the script |
| `pexels_9057924` | older man lying, then sitting up in bed | no | [mp4](https://videos.pexels.com/video-files/9057924/9057924-hd_1920_1080_25fps.mp4) |
| `pexels_6898173` | woman lying under a duvet, stretching | no | [mp4](https://videos.pexels.com/video-files/6898173/6898173-hd_1920_1080_25fps.mp4) |
| `pexels_3753707` | man sitting on the bed edge, seen as a close-up of his back plus a mirror reflection | no | [mp4](https://videos.pexels.com/video-files/3753707/3753707-hd_1920_1080_25fps.mp4) |

**Read these numbers with care.** The labels are AI-assisted drafts, approximate to about ±0.3 s, and no person has
reviewed them yet ([annotations/REVIEW.md](annotations/REVIEW.md)). The development clips were also used while fixing
bugs, so this is a development set, not a held-out test. The people are adult actors (one older man), the clips are
7–44 s long, and the only return in the development set is the constructed, reversed one. The run shows the pipeline
working end to end on real video; it does not establish performance on elderly residents, overnight reliability or
clinical safety. [docs/recording_protocol.md](../docs/recording_protocol.md) lists what the data covers and how to
record better data.

## Results (4 labelled clips, 97.25 s labelled and analysed, full system with VLM enabled)

These numbers come from a fresh run on 2026-09-30: new pose inference, tracking, agent and VLM calls, then evaluation.
The code was commit `4a1a280` plus this review's uncommitted changes (the manifests say `src_dirty: true`). The
development results are identical to the committed ones: the same timelines and events on all seven clips, with and
without the agent. For the four labelled clips the new observations also match the stored test fixtures exactly.

| Metric | Value |
|---|---|
| Activity accuracy (UNKNOWN included) | **0.902** (per clip 0.875 / 0.978 / 0.766 / 0.889) |
| Activity macro-F1 | 0.801 |
| Bed-status accuracy / macro-F1 | 0.932 / 0.882 |
| Predicted UNKNOWN share | 0.149 (the labels have 0.131: the resident is out of view) |
| Bed exits | TP 4, FP 0, FN 0: precision 1.0, recall 1.0; occurrence error 0.47 s; confirmation delay 2.8 s |
| Returns to bed | TP 1, FP 0, FN 0; occurrence error 0.66 s; confirmed 9.8 s after she sits down (she sits for 8 s, then 2 s of lying must be seen) |
| Activity duration error (macro mean of per-clip absolute errors) | 0.66 s |

Per state (seconds over the four clips). The total error can hide errors that cancel between clips, so the last
numeric column adds up the per-clip absolute errors:

| State | Labels | Predicted | Error of the totals | Sum of per-clip errors | Precision | Recall |
|---|---|---|---|---|---|---|
| LYING_IN_BED | 20.2 | 18.5 | −1.7 | 1.7 | 0.98 | 0.90 |
| SITTING_ON_BED | 52.5 | 50.8 | −1.7 | 2.5 | 0.98 | 0.95 |
| STANDING | 4.5 | 4.4 | −0.1 | 2.1 | 0.55 | 0.53 |
| WALKING | 7.4 | 9.1 | +1.7 | 2.5 | 0.65 | 0.80 |
| UNKNOWN | 12.7 | 14.5 | +1.8 | 3.1 | 0.80 | 0.91 |
| bed status IN_BED / OUT_OF_BED / UNKNOWN | 72.7 / 11.9 / 12.7 | 69.3 / 13.5 / 14.5 | −3.4 / +1.6 / +1.8 | | | |

No confusion is larger than 1.8 s (see [evaluation/outputs/confusion_matrix.png](evaluation/outputs/confusion_matrix.png)).
The largest are the first second of two clips, where she is lying but not yet measured (LYING_IN_BED → UNKNOWN, 1.8 s);
walking and standing at the start and end of a walk (STANDING → WALKING 1.4 s, UNKNOWN → WALKING 1.1 s,
WALKING → STANDING 0.8 s); and the real stand-up detected 0.6 s early in two clips (SITTING_ON_BED → STANDING 1.2 s).
Full metrics are in [evaluation/outputs/metrics.json](evaluation/outputs/metrics.json).

[FAILURE_CASES.md](FAILURE_CASES.md) explains the four failures found on these clips step by step, with skeleton
figures (`python examples/make_figures.py`), the fix for each, the numbers before and after, and what is still open.
Before those fixes the same clips gave activity accuracy 0.802, bed-status accuracy 0.829 and 0 of 1 returns.

**Agent vs. baseline.** `--no-agent` gives activity accuracy 0.826, bed-status accuracy 0.856 and a duration error of
1.46 s, with the same exits and return ([evaluation/outputs_baseline](evaluation/outputs_baseline/metrics.json)). The
difference is the agent's `UPRIGHT_ON_BED_REGION` review (failure case 1): on the three windows where edge-sitting
facing the camera reads as standing, it asked Qwen2.5-VL about 3 frames each, got "sitting / bed" all 9 times, and
relabelled them SITTING_ON_BED. The agent without the VLM (`--no-vlm`) gives the baseline's numbers. Every context
request is in `agent_trace.jsonl`: exit candidates confirmed or rejected with look-back / look-ahead evidence, gaps
where it abstained, and in `pexels_3753707` two gaps sent to the VLM. There the first gap got disagreeing answers and
stayed `UNKNOWN`; the second got "sitting on the bed" three times and became SITTING_ON_BED (failure case 4). Cases
where the agent changes the outcome that these short clips do not contain (bed vs. floor at the outline, a
blanket-covered gap between lying spans, a long in-bed gap for the VLM) are covered by the scenario tests in
`tests/test_pipeline.py`. An earlier apparent agent gain (0.788 vs 0.404) turned out to be the agent masking an identity
bug in the target selector, which has since been fixed (failure case 5a).

**Runtime** (RTX 4090, from `run_manifest.json`; each clip runs in its own process):

| Stage | Time |
|---|---|
| Perception (decoding, YOLO11n-pose + ByteTrack at 5 fps, model loading included) | 4.0–6.6 s per clip |
| Analysis of cached observations without the VLM (temporal engine, agent geometry, events, policy) | under 0.05 s per clip |
| Loading Qwen2.5-VL from the local cache, once per process that needs it | 17.8–18.3 s |
| VLM inference | 19.7 s for the 15 calls over the seven clips, about 1.3 s per call |

## Other clips

| Clip | Timeline | Events / decision |
|---|---|---|
| `pexels_9057924` | LYING_IN_BED → SITTING_ON_BED (sits up) | no exit, NORMAL (sitting up is not an exit) |
| `pexels_6898173` | LYING_IN_BED for the whole clip (under a duvet, stretching) | no exit, NORMAL |
| `pexels_3753707` | UNKNOWN (0–5.8 s), SITTING_OUTSIDE_BED, SITTING_ON_BED (11.4–17.4 s, VLM), SITTING_OUTSIDE_BED | no exit; MONITOR `UNCERTAIN` for 5.8 s; camera-placement warning (`view_quality.degraded`, failure case 4) |

## Held-out test: 13 new clips

The numbers above come from the clips that were used while fixing bugs. To check the system on footage it was not
tuned on, `python examples/run_examples.py --heldout --vlm` downloads 13 other Pexels clips into `examples/data/`,
analyses them with and without the agent (`outputs/`, `outputs_baseline/`) and evaluates them
(`evaluation/heldout/`, `evaluation/heldout_baseline/`).

How the test was set up:
- The bed polygons (`configs/pexels_<id>.yaml`) were drawn from the video frames.
- The labels (`annotations/pexels_<id>.json`) were drafted blind, before the system was run on these clips, by two
  independent AI annotators: one labelled, the second made its own timeline first and then checked every boundary.
  No person has reviewed them yet.
- The first held-out run used commit `e1c709a`. Two code reviews since then found correctness problems
  ([FAILURE_CASES.md](FAILURE_CASES.md)). The fixes were checked on the development clips only; no threshold was
  tuned on these clips.
- The set is very small (4.6 minutes, 6 events), and two of its three returns (`8539664`, `8539659`) are the same
  action filmed from two cameras, so they are not independent.

| Clip | What it tests | Page | Video file |
|---|---|---|---|
| `pexels_7938959` | bed exit with a caregiver helping the resident off the bed | [page](https://www.pexels.com/video/a-young-girl-getting-off-of-the-bed-7938959/) | [mp4](https://videos.pexels.com/video-files/7938959/7938959-hd_1080_1920_24fps.mp4) |
| `pexels_4052925` | long sitting up, then a bed exit right at the end of the clip | [page](https://www.pexels.com/video/man-happily-waking-up-4052925/) | [mp4](https://videos.pexels.com/video-files/4052925/4052925-hd_1920_1080_25fps.mp4) |
| `pexels_8090730` | bed exit seen only from the legs (camera at floor level) | [page](https://www.pexels.com/video/a-person-getting-out-of-bed-8090730/) | [mp4](https://videos.pexels.com/video-files/8090730/8090730-hd_1920_1080_24fps.mp4) |
| `pexels_9615483` | dim, handheld: lying, sitting up, lying back is not an exit | [page](https://www.pexels.com/video/man-sleeping-on-bed-9615483/) | [mp4](https://videos.pexels.com/video-files/9615483/9615483-hd_1080_2048_25fps.mp4) |
| `pexels_8539664` | return to bed from outside: walks in through a doorway and lies down | [page](https://www.pexels.com/video/man-lying-on-bed-8539664/) | [mp4](https://videos.pexels.com/video-files/8539664/8539664-hd_2048_1080_25fps.mp4) |
| `pexels_8539659` | the same return from a second camera | [page](https://www.pexels.com/video/a-young-man-lying-down-on-a-bed-8539659/) | [mp4](https://videos.pexels.com/video-files/8539659/8539659-hd_1080_2048_25fps.mp4) |
| `pexels_8591515` | return: stands beside the bed, drops onto it; a projected human figure on the wall | [page](https://www.pexels.com/video/person-lying-on-bed-8591515/) | [mp4](https://videos.pexels.com/video-files/8591515/8591515-hd_2048_1080_25fps.mp4) |
| `pexels_7505324` | sitting up in bed is not an exit | [page](https://www.pexels.com/video/girl-waking-up-from-bed-7505324/) | [mp4](https://videos.pexels.com/video-files/7505324/7505324-hd_1920_1080_30fps.mp4) |
| `pexels_10608105` | kneeling and crawling on the bed is not an exit | [page](https://www.pexels.com/video/a-young-girl-getting-up-from-bed-while-looking-at-the-sea-view-10608105/) | [mp4](https://videos.pexels.com/video-files/10608105/10608105-hd_2048_1080_25fps.mp4) |
| `pexels_5983705` | sitting up in bed, side view | [page](https://www.pexels.com/video/man-waking-up-on-his-bed-5983705/) | [mp4](https://videos.pexels.com/video-files/5983705/5983705-hd_1920_1080_25fps.mp4) |
| `pexels_8862331` | older woman sitting on the bed edge facing the camera, legs out of frame | [page](https://www.pexels.com/video/an-elderly-woman-sitting-on-a-bed-8862331/) | [mp4](https://videos.pexels.com/video-files/8862331/8862331-hd_2048_1080_25fps.mp4) |
| `pexels_8088284` | sick older man sitting up in bed, a caregiver sitting beside him | [page](https://www.pexels.com/video/an-elderly-man-sick-in-bed-8088284/) | [mp4](https://videos.pexels.com/video-files/8088284/8088284-hd_1080_2048_24fps.mp4) |
| `pexels_6130024` | hospital patient in bed, a nurse working beside it, moving camera | [page](https://www.pexels.com/video/healthcare-worker-taking-care-of-sick-patient-6130024/) | [mp4](https://videos.pexels.com/video-files/6130024/6130024-hd_1920_1080_30fps.mp4) |

**Results** (278.66 s labelled; every clip is covered in full, so the analysed time is the same). The first column is
the fresh run of the current code on 2026-09-30, with new pose inference, tracking, agent and VLM calls; its metrics
are identical to the previous commit's (`4a1a280`). The second column is the saved predictions of `e1c709a`, rescored
with the current evaluator, not a new run:

| Metric | Full system (VLM), current code | Full system, `e1c709a` (rescored) | `--no-agent` (both) |
|---|---|---|---|
| Activity accuracy / macro-F1 | 0.667 / 0.504 | 0.724 / 0.578 | 0.502 / 0.438 |
| Bed-status accuracy | 0.836 | 0.940 | 0.670 |
| Predicted UNKNOWN share (labels 0.020) | 0.170 | 0.066 | 0.338 |
| Bed exits | 1 of 3 found, 0 false; 1 miss labelled 0.7 s before the clip ends | 1 of 3 found, 0 false | 1 of 3 found, 0 false |
| Returns to bed | 1 of 3 found (0.70 s timing error, confirmed 3.8 s after it starts), 0 false | 2 of 3 found, 0 false | 1 of 3 found, 0 false |
| False exits or returns on the 7 clips without any | 0 | 0 | 0 |
| ALERTs | none (none expected) | none | none |
| Activity duration error | 2.84 s | 2.16 s | 5.06 s |

The drop from `e1c709a` comes from two clips, both because the VLM now answers only about an identified resident:
- In `6130024` the config selects the resident by a clicked point at 11.5 s, and a nurse is in view from the start,
  so nobody is selected before 11.5 s. Before, the VLM was asked about "any person" and relabelled those 11.5 s as
  lying in bed. It was right, but the resident had not been identified, so that time is now `UNKNOWN`.
- In `8539659` the resident lies down and is no longer detected. The VLM is asked about the outlined bed region in
  3 frames. One of the 3 answers says nobody is visible, so there is no agreement and the gap stays `UNKNOWN`.
  The return is still pending at the end.

The earlier, higher numbers partly relied on VLM answers about people the system had not identified. The current
numbers are the ones to quote.

| Clip | Expected | Full system | Activity accuracy |
|---|---|---|---|
| `7938959` | exit at 6.1 s | **missed**: she gets off at the frame border and is only weakly detected there (see below); the rest of the clip is MONITOR `UNCERTAIN` | 0.68 |
| `4052925` | exit at 20.9 s | **missed**: the clip ends 0.7 s after he stands (see below) | 0.97 |
| `8090730` | exit at 9.7 s | exit at 10.6 s, confirmed | 0.45 |
| `9615483` | no event | no event; lying and sitting flicker in the dim, handheld view | 0.37 |
| `8539664` | return at 8.4 s | **missed**: the footboard and clothes on it hide him once he lies down (see below) | 0.18 |
| `8539659` | return at 7.2 s | **missed**: the same return from a second camera; he is no longer detected once he lies down (see below) | 0.13 |
| `8591515` | return at 7.9 s | return at 8.6 s, confirmed at 12.4 s; the projected figure is masked with an ignore polygon | 0.95 |
| `7505324` | no event | no event; sits up 1.1 s early | 0.93 |
| `10608105` | no event | no event; lying at the far pillow read as sitting for 6 s | 0.70 |
| `5983705` | no event | no event | 0.99 |
| `8862331` | no event | no event; too few joints for the rules (view warning), the VLM confirmed sitting on the bed | 1.00 (baseline 0.00) |
| `8088284` | no event | no event; the resident is kept with the caregiver beside him. Nobody is selected before the calibrated target time (6 s), and that gap stays `UNKNOWN` | 0.77 |
| `6130024` | no event | no event; the nurse is not taken for the patient. The patient is selected at the calibrated target time (11.5 s), so 0–11.5 s is `UNKNOWN`; the VLM confirmed lying for the rest (view warning) | 0.14 (`e1c709a` 0.76, baseline 0.00) |

`8591515` is confirmed 2 s later than before because the resident is not detected at 8.4–8.6 s and 10.0–10.4 s
while lying down. The committed timeline smooths those samples into lying, but a return now needs 2 s of lying that
was actually seen, which starts at 10.6 s. Its start time, and so the scores, are unchanged.

### Why four held-out events were missed

Each case below was checked frame by frame, against the observations and detector confidences at the time of the
miss.

1. **Caregiver assistance (`7938959`, exit).** From 1.2 s to about 4 s the caregiver holds her hands and helps her
   across the bed. The selector stays on the girl throughout: her track ID changes twice (2 → 10 → 14), and each time
   the box that is re-attached is hers. So the resident's identity was not lost to the caregiver. The miss has other
   causes. She crawls to the far end of the bed, at the right frame border. From 4.8 s she is cut off by the border,
   and the detector scores her 0.11–0.40 (checked at 5.0–5.8 s), below the 0.5 needed, so those samples are
   `low_confidence`. At 6.2–6.8 s she is detected again (0.70–0.80), upright at the far end of the bed with the
   mattress hiding her legs. In the image her hips are inside the bed outline, which reaches the frame border, so
   the rules read sitting on the bed. At 7.0 s she leaves the frame. No out-of-bed state is ever observed, so no exit
   candidate opens.
2. **Furniture hiding the resident (`8539664`, return).** He sits on the bed at 8.4 s and lies back behind the
   footboard, with a jacket hung over it. From about 9.2 s only his head and one arm show above it. The detector
   scores him 0.10–0.51. Apart from one sample at 9.6 s, the resident is not re-attached until a confident box at
   15.0 s. That leaves 0.9 s before the clip ends, less than the 2 s of lying a return needs.
3. **Detection loss after lying down (`8539659`, return).** This is the same action filmed from a second camera. Once
   he lies down at 7.6 s the footboard and a bag hide most of his body, and for most samples nothing is detected at
   all. The VLM was asked about the outlined bed, but its three answers did not agree. The gap stays `UNKNOWN`, and
   the return is still pending when the clip ends.
4. **The clip ends before a confirmation is possible (`4052925`, exit).** From 20.9 to 21.2 s he is standing, but
   with this low camera his hips still project inside the bed outline and his legs are not measurable, so the rules
   keep "sitting on the bed". The first out-of-bed samples come at 21.4 s, two samples before the clip ends at
   21.64 s. Committing a state takes 1 s, and confirming an exit takes 1–2 s more of evidence, so no rule could
   confirm it. Evaluation also counts it in `missed_with_short_context` (strict results unchanged).

None of these can be fixed by thresholds without guessing: the resident is not visible or not identifiable in the
frames where the event happens. Better camera placement (the whole bed and the floor around it in view, no
furniture between the camera and the mattress) and a stronger person detector are the realistic remedies.

What this shows:
- Zero false exits, false returns and ALERTs were observed in this evaluation, including sitting up, kneeling on the
  bed, edge-sitting and two caregivers. With 4.6 minutes of footage this says little about the false-alarm rate.
- Its main weakness is losing the resident when the body is hidden or cut off by the frame. The selector is strict on
  purpose (a detection must reach confidence 0.5 before it becomes the resident; the duvet "phantom" in failure
  case 3 scored 0.36), so it says `UNKNOWN` / MONITOR instead of guessing.
- The agent still helps more here than on the development clips (0.667 vs 0.502 without it), mainly through VLM
  checks of close-ups where the rules have too few joints (`8862331`: 1.00 vs 0.00).
- Total seconds per state can hide large errors that cancel. With the `e1c709a` predictions the lying totals
  differed by only 0.4 s (77.4 s labelled, 76.9 s predicted), yet the per-clip lying errors added up to 33.9 s.
  With the current code the totals are 77.4 s and 47.9 s, and the per-clip errors add up to 36.7 s.
  `metrics.json` (`duration.activity_totals_sec`) reports both.
- The development-clip accuracy (0.902) was measured on the clips used for tuning. Expect performance on new cameras
  to be closer to these held-out numbers.

**Testing your own video.** Record with a fixed camera that sees the whole bed and the resident's full body. Then run
`python -m elder_monitor calibrate --video my.mp4 --output configs/my.yaml` to draw the bed and
`python -m elder_monitor analyze --video my.mp4 --config configs/my.yaml --output outputs/my --overlay`, and check
`outputs/my/overlay.mp4`. To score it, write labels in the format of `annotations/TEMPLATE.json` and run
`python -m elder_monitor evaluate` (see the main README).
