# Evaluation data: what exists, and a protocol for better data

## What the reported numbers rest on

| Set | Clips | Labelled time | Bed events | Caveats |
|---|---|---|---|---|
| Development | 3 real Pexels clips + 1 constructed clip (+ 3 unlabelled) | 97.25 s | 4 exits (3 distinct actions), 1 return | Used while fixing bugs, so not a test set. The round-trip clip is `pexels_4049556` forwards, 5 s of the empty room, then the same footage reversed: its exit repeats the real one and its return is that exit played backwards. |
| Held-out | 13 Pexels clips | 278.66 s (4.6 min) | 3 exits, 3 returns | Very small. Two of the three returns (`pexels_8539664`, `pexels_8539659`) are the same action filmed from two cameras, so they are not independent. |

All clips are staged by adult actors, 7–44 s long, filmed by stock-footage cameras. None shows a chair, a fall,
a night scene, or an absence longer than a few seconds. Every label is an AI-assisted draft that no person has
reviewed yet ([REVIEW.md](../examples/annotations/REVIEW.md)). The results show the pipeline working end to end on
real video; they do not measure reliability overnight or on elderly residents, and they are not clinical evidence.

## Real footage versus scripted scenario tests

Real clips run the whole system: video, pose model, tracker, rules, agent and state machine. Scenario tests feed
scripted observations (or single synthetic frames) into the same logic, so they test the rules and timers but not
perception.

| Scenario | Real labelled clips | Scenario tests (no video) |
|---|---|---|
| Turning in bed | none | `test_lying_hysteresis` |
| Sitting up without leaving | `7505324`, `5983705`, `9615483`, `10608105` | `test_sitting_up_is_not_an_exit` |
| Edge-sitting | `8862331` and the development clips (shorter than the 60 s rule) | `test_long_edge_sitting_triggers_monitor` |
| Brief stand, then sitting back | none | `test_brief_stand_then_sit_is_rejected` |
| Bed exit | 3 development, 3 held-out | `test_exit_and_return_are_counted_once` and others |
| Return to bed | 3 held-out (2 are one action), 1 constructed | `test_return_needs_continuous_lying_after_an_unknown_gap` and others |
| Sitting on a chair | none | `test_bedside_chair_counts_as_exit_after_dwell` |
| Walking around the room | walking in and out only | `test_flickering_labels_still_commit` |
| Blankets, temporary occlusion | `6898173` (unlabelled), `8539664` | `test_agent_bridges_blanket_gap_but_baseline_does_not`, `test_bridged_missing_frames_do_not_confirm_a_return` |
| Caregiver interaction | `7938959`, `8088284`, `6130024` | `tests/test_identity.py`, `test_caregiver_stepping_in_front_after_a_step_is_not_an_exit` |
| Long absence (> 5 and > 15 min) | none | `test_prolonged_out_of_bed_alert_fires_once`, `test_leaving_view_after_getting_up_eventually_alerts` |
| Lying outside the bed | none | `test_fall_beside_bed_is_never_hidden`, `test_fall_after_walking_raises_alert` |
| Long live session | none | `test_long_session_keeps_a_bounded_window_and_matches_offline` |

## Recording protocol

The aim is continuous sessions from fixed cameras, with enough repeats that event counts mean something.

**Consent and safety.** Get written consent from everyone filmed, caregivers included, and store the footage
locally. Stage "lying outside the bed" on a padded mat with a trained actor. Nobody should actually fall.

**Camera.** Mount the camera, never handheld, about 2 m high at the foot of the bed or in a corner. It should see
the whole mattress, the floor beside the bed and the route to the door. Record at 720p or more and at least 10 fps.
Capture one frame of the empty bed for calibration. Record each session from two positions if possible, but keep
both views of a session in the same split.

**Sessions.** Record sessions of 20–60 minutes rather than short clips, with several people, clothing, bedding and
lighting levels (day, evening, night light). Each session should include, in varying order:

1. a natural bed exit and a return, walking out of the room and back;
2. sitting in a chair beside the bed for at least one minute;
3. walking around the room;
4. standing up briefly, then sitting back down on the bed;
5. blankets pulled over the body, and short occlusions (someone walking past, an object in front of the bed);
6. caregiver interaction: helping the resident up, standing between the camera and the bed, sitting on the bed;
7. an absence of more than 5 minutes, and in some sessions more than 15 minutes;
8. sitting on the bed edge for more than one minute;
9. lying outside the bed on the mat (staged).

**Splits.** Decide before development which sessions are held out, and split by person and room, not by clip.
Never put two camera views of the same action in different splits; if both are evaluated, count the action once.

## Annotation protocol

1. Two annotators label each session independently from the full-frame-rate video, using the state definitions and
   the event policy in the README: an exit starts when the resident stands up off the bed; a return starts when they
   sit back on it. Mark `UNKNOWN` when the resident cannot be seen.
2. Compare the two labels. Report agreement on the 0.1 s grid and the event-time differences, then settle each
   disagreement by watching the footage together.
3. Record who labelled, who reviewed and when in the file's `provenance` and in the review log. Do not mark a label
   as reviewed until a person has watched the footage.
4. Freeze the held-out labels before running the system on those sessions.

**What to report.** Per-scenario results; events and false events per hour over the long sessions; event counts
with their uncertainty (with tens of events, a recall of 1.0 is still consistent with much lower true recall); and
the strict results next to any separately reported cases, such as events too close to the end of a recording.

None of this has been recorded yet. The protocol needs real footage and annotators.
