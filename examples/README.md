# Example run on public footage

```bash
python examples/run_examples.py --vlm             # development clips (or without --vlm for geometry-only)
python examples/run_examples.py --heldout --vlm   # 13 held-out clips never used while developing (see below)
```

The script downloads six short [Pexels](https://www.pexels.com/license/) clips (free licence) into `examples/data/`,
builds one constructed clip, analyses each clip with the full system (`outputs/`) and with `--no-agent`
(`outputs_baseline/`), and evaluates the labelled ones (`evaluation/`). Videos and failure snapshots are not committed;
the script regenerates them.

| Clip | Content | Labels | Video file |
|---|---|---|---|
| `pexels_4049556` | woman lies, sits up, sits on the edge facing the camera, stands, walks out towards the camera | yes | [mp4](https://videos.pexels.com/video-files/4049556/4049556-hd_1920_1080_30fps.mp4) |
| `pexels_4052924` | man sits in bed, gets up, walks out past the camera | yes | [mp4](https://videos.pexels.com/video-files/4052924/4052924-hd_1920_1080_25fps.mp4) |
| `pexels_8090735` | woman sits on the bed edge (low camera), stands, walks out | yes | [mp4](https://videos.pexels.com/video-files/8090735/8090735-hd_1920_1080_24fps.mp4) |
| `roundtrip_4049556` | constructed: `4049556` forwards, 5 s of the empty room, then reversed, so a bed exit and a return | yes | built by the script |
| `pexels_9057924` | older man lying, then sitting up in bed | no | [mp4](https://videos.pexels.com/video-files/9057924/9057924-hd_1920_1080_25fps.mp4) |
| `pexels_6898173` | woman lying under a duvet, stretching | no | [mp4](https://videos.pexels.com/video-files/6898173/6898173-hd_1920_1080_25fps.mp4) |
| `pexels_3753707` | man sitting on the bed edge, seen as a close-up of his back plus a mirror reflection | no | [mp4](https://videos.pexels.com/video-files/3753707/3753707-hd_1920_1080_25fps.mp4) |

**Read these numbers with care.** The labels of these development clips were drafted from sampled frames with AI
assistance and are approximate to about ±0.3 s. The clips were also used while fixing bugs, so this is a development set, not a
held-out test. The people are adult actors (one older man), the clips are 7–44 s long, and only one of them contains a
return to bed. The run shows the pipeline working end to end on real video; it does not establish performance on
elderly residents or on long overnight recordings.

## Results (4 labelled clips, 97.2 s, full system with VLM enabled)

| Metric | Value |
|---|---|
| Activity accuracy (UNKNOWN included) | **0.902** (per clip 0.875 / 0.978 / 0.766 / 0.889) |
| Activity macro-F1 | 0.801 |
| Bed-status accuracy / macro-F1 | 0.932 / 0.882 |
| Predicted UNKNOWN share | 0.149 (ground truth has 0.131: the resident is out of view) |
| Bed exits | TP 4, FP 0, FN 0: precision 1.0, recall 1.0; occurrence error 0.47 s; confirmation delay 2.7 s |
| Returns to bed | TP 1, FP 0, FN 0; occurrence error 0.66 s; confirmed 10 s after she sits down (she sits for 8 s, then the lying rule needs 2 s) |
| Activity duration error (macro mean of per-clip absolute errors) | 0.67 s |

Per state (seconds over the four clips):

| State | Ground truth | Predicted | Error | Precision | Recall |
|---|---|---|---|---|---|
| LYING_IN_BED | 20.2 | 18.5 | −1.7 | 0.98 | 0.90 |
| SITTING_ON_BED | 52.5 | 50.8 | −1.7 | 0.98 | 0.95 |
| STANDING | 4.5 | 4.4 | −0.1 | 0.55 | 0.53 |
| WALKING | 7.4 | 9.1 | +1.7 | 0.65 | 0.80 |
| UNKNOWN | 12.7 | 14.5 | +1.8 | 0.80 | 0.91 |
| bed status IN_BED / OUT_OF_BED / UNKNOWN | 72.7 / 11.9 / 12.7 | 69.3 / 13.5 / 14.5 | −3.4 / +1.6 / +1.8 | | |

No confusion is larger than 1.8 s any more (see
[evaluation/outputs/confusion_matrix.png](evaluation/outputs/confusion_matrix.png)). The largest are the first second
of two clips, where she is lying but not yet measured (LYING_IN_BED → UNKNOWN, 1.8 s); walking and standing at the
start and end of a walk (STANDING → WALKING 1.4 s, UNKNOWN → WALKING 1.1 s, WALKING → STANDING 0.8 s); and the real
stand-up detected 0.6 s early in two clips (SITTING_ON_BED → STANDING 1.2 s). Full metrics are in
[evaluation/outputs/metrics.json](evaluation/outputs/metrics.json).

[FAILURE_CASES.md](FAILURE_CASES.md) explains the four failures found on these clips step by step, with skeleton
figures (`python examples/make_figures.py`), the fix for each, the numbers before and after, and what is still open.
Before those fixes the same clips gave activity accuracy 0.802, bed-status accuracy 0.829 and 0 of 1 returns.

**Agent vs. baseline.** `--no-agent` gives activity accuracy 0.826, bed-status accuracy 0.856 and a duration error of
1.47 s, with the same exits and return ([evaluation/outputs_baseline](evaluation/outputs_baseline/metrics.json)). The
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

**Runtime** (RTX 4090): perception 4–6 s per clip at 5 fps; the temporal engine, events and policy take milliseconds;
loading Qwen2.5-VL takes about 25 s, then about 1.2 s per VLM call (15 calls over the seven clips).

## Other clips

| Clip | Timeline | Events / decision |
|---|---|---|
| `pexels_9057924` | LYING_IN_BED → SITTING_ON_BED (sits up) | no exit, NORMAL (sitting up is not an exit) |
| `pexels_6898173` | LYING_IN_BED for the whole clip (under a duvet, stretching) | no exit, NORMAL |
| `pexels_3753707` | UNKNOWN (0–5.8 s), SITTING_OUTSIDE_BED, SITTING_ON_BED (11.4–17.4 s, VLM), SITTING_OUTSIDE_BED | no exit; MONITOR `UNCERTAIN` for 5.8 s; camera-placement warning (`view_quality.degraded`, failure case 4) |

## Held-out test: 13 new clips

The numbers above come from the clips that were used while fixing bugs. To check that the system works on footage it
has never seen, `python examples/run_examples.py --heldout --vlm` downloads 13 other Pexels clips into
`examples/data/`, analyses them with and without the agent (`outputs/`, `outputs_baseline/`) and evaluates them
(`evaluation/heldout/`, `evaluation/heldout_baseline/`).

How the test was set up:
- The bed polygons (`configs/pexels_<id>.yaml`) were drawn from the video frames.
- The labels (`annotations/pexels_<id>.json`) were drafted blind, before the system was run on these clips, by two
  independent AI annotators: one labelled, the second made its own timeline first and then checked every boundary.
- No code or threshold was changed after seeing the results.
- The labels have not been reviewed by a person yet.

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

**Results** (278.7 s):

| Metric | Full system (VLM) | `--no-agent` |
|---|---|---|
| Activity accuracy / macro-F1 | 0.724 / 0.578 | 0.502 / 0.438 |
| Bed-status accuracy | 0.940 | 0.670 |
| Bed exits | 1 of 3 found, 0 false | 1 of 3 found, 0 false |
| Returns to bed | 2 of 3 found (0.55 s timing error), 0 false | 1 of 3 found, 0 false |
| False exits or returns on the 7 clips without any | 0 | 0 |
| ALERTs | none (none expected) | none |
| Activity duration error | 2.15 s | 5.06 s |

| Clip | Expected | Full system | Activity accuracy |
|---|---|---|---|
| `7938959` | exit at 6.1 s | **missed**: the identity is uncertain while the caregiver holds her, then she leaves behind the bed; the rest of the clip is MONITOR `UNCERTAIN` | 0.68 |
| `4052925` | exit at 20.9 s | **missed**: the clip ends 0.7 s after he stands, shorter than the 1 s dwell; from this low camera his hips also stay over the bed polygon | 0.97 |
| `8090730` | exit at 9.7 s | exit at 10.6 s, confirmed | 0.45 |
| `9615483` | no event | no event; lying and sitting flicker in the dim, handheld view | 0.37 |
| `8539664` | return at 8.4 s | **missed**: the tracker loses him when he falls back behind the footboard. In a debug re-run his detections scored 0.14–0.41, below the 0.5 needed to re-attach the resident, so 8.8–15.0 s is `UNKNOWN` and the return is still pending at the end | 0.18 |
| `8539659` | return at 7.2 s | return at 7.6 s (the agent's VLM check filled the gap; the baseline misses it) | 0.63 |
| `8591515` | return at 7.9 s | return at 8.6 s; the projected figure is masked with an ignore polygon | 0.95 |
| `7505324` | no event | no event; sits up 1.1 s early | 0.93 |
| `10608105` | no event | no event; lying at the far pillow read as sitting for 6 s | 0.70 |
| `5983705` | no event | no event | 0.99 |
| `8862331` | no event | no event; too few joints for the rules (view warning), the VLM confirmed sitting on the bed | 1.00 (baseline 0.00) |
| `8088284` | no event | no event; the resident is kept with the caregiver beside him. Nobody is selected before the calibrated target time (6 s), and the VLM check read that gap as lying | 0.77 |
| `6130024` | no event | no event; the nurse is never taken for the patient; her 3 s lean forward is missed (view warning) | 0.76 (baseline 0.00) |

What this shows:
- On new footage the system never raised a false exit, a false return or an ALERT. That includes sitting up,
  kneeling on the bed, edge-sitting and two caregivers.
- Its main weakness is keeping the resident's identity when a caregiver holds them or furniture hides them. The
  selector is strict on purpose (a detection must reach confidence 0.5 before it becomes the resident; the duvet
  "phantom" in failure case 3 scored 0.36), so it says `UNKNOWN` / MONITOR instead of guessing. The cost is two of
  the three missed events.
- The agent's VLM checks matter more here than on the development clips (0.724 vs 0.502), mainly on close-ups where
  the rules have too few joints.
- The development-clip accuracy (0.902) was measured on the clips used for tuning. Expect performance on new cameras
  to be closer to these held-out numbers.

**Testing your own video.** Record with a fixed camera that sees the whole bed and the resident's full body. Then run
`python -m elder_monitor calibrate --video my.mp4 --output configs/my.yaml` to draw the bed and
`python -m elder_monitor analyze --video my.mp4 --config configs/my.yaml --output outputs/my --overlay`, and check
`outputs/my/overlay.mp4`. To score it, write labels in the format of `annotations/TEMPLATE.json` and run
`python -m elder_monitor evaluate` (see the main README).
