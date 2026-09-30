# Architecture

```mermaid
flowchart TD
    V[Video + scene config] --> R[video.VideoReader<br/>decoded timestamps, 5 fps grid]
    R --> P[vision.PoseTracker<br/>YOLO11n-pose + ByteTrack]
    P --> S[vision.TargetSelector<br/>resident identity, re-attach]
    S --> F[features.measure<br/>body axis, legs, bed polygon, speed, box speed]
    F --> Q[temporal.propose<br/>rules + hysteresis]
    Q --> A{agent.ContextAgent<br/>ambiguous?}
    A -- LYING_OUTSIDE_BED / AMBIGUOUS_GAP /<br/>UPRIGHT_ON_BED_REGION --> T1[look_back / look_ahead<br/>inspect_bed_relation<br/>inspect_target_crop → Qwen2.5-VL]
    T1 --> A
    A --> E[temporal.build_timeline<br/>majority filter, dwell, backdated boundaries]
    E --> L[Activity timeline + bed-status timeline]
    L --> B[events.detect_bed_events<br/>bed exit / return FSM]
    B -- EXIT_CANDIDATE --> A2[agent.verify_exit<br/>look_back + bounded look_ahead]
    A2 --> B
    B --> K[policy.AlertPolicy<br/>NORMAL / MONITOR / ALERT]
    L --> W[reporting.write_outputs]
    B --> W
    K --> W
    W --> O[timeline.csv, bed_timeline.csv, decisions.csv,<br/>events.json, summary.json, agent_trace.jsonl]
    O -.-> X[evaluation.evaluate<br/>vs. human labels]
```

| Module | Responsibility |
|---|---|
| `schemas.py` | State names, bed status mapping, dataclasses (Observation, Proposal, Segment, BedEvent, Verdict) |
| `config.py` | Default + scene YAML merge, validation, hashing |
| `video.py` | Timestamp-aware sampling on a fixed grid, frame seek for context, file hash |
| `vision.py` | YOLO11-pose + ByteTrack wrapper, resident selection and re-association, ignore polygons |
| `features.py` | Scene geometry and raw per-frame posture / bed evidence, keypoint and box speed |
| `temporal.py` | Frame proposals (box motion when the joints drop out), smoothing, dwell / hysteresis, committed timelines |
| `agent.py` | Trigger handling, bounded tool selection, evidence trace, exit verification |
| `vlm.py` | Qwen2.5-VL adapter with validated JSON answers |
| `events.py` | Departure guard and bed exit / return state machine |
| `policy.py` | Explicit alert rules, decision timeline, one record per rule episode |
| `reporting.py` | CSV / JSON outputs, summary, overlay video |
| `evaluation.py` | Accuracy, confusion, event precision / recall, duration error, failure cases |
| `pipeline.py` | Wires the stages; `run_analysis` is the testable core after perception; camera-placement check |
| `calibrate.py`, `cli.py` | Bed / chair polygon tool and the command-line entry points |
| `live.py` | Webcam session that re-runs `run_analysis` on every frame, chat messages, status answers |
| `server.py` | FastAPI app: upload jobs, results, video, live WebSocket; optionally serves the web frontend |

Data flows one way. Ground-truth labels are only read by `evaluation.py`.
The VLM never writes the timeline directly: its answers become proposals that still pass the temporal engine.
