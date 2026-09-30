# UML diagrams

The assignment brief asks for one simple architecture diagram, which is [architecture.png](architecture.png) in the
README. The diagrams below were added on request, to document behaviour and structure in more detail. They are
drawn from the code as it stands; each one names the module it describes. They are Mermaid sources, so GitHub renders
them and they can be edited as text.

Mermaid has no dedicated use-case or deployment diagram, so those two are flowcharts that follow the UML notation:
actors outside the system boundary, use cases as rounded nodes, nodes and artefacts as boxes.

## Use cases

```mermaid
flowchart LR
    operator([Operator])
    staff([Care staff])
    annotator([Evaluator / annotator])
    subgraph system [Elder Monitor]
        cal([Calibrate bed, chairs and resident])
        ana([Analyse a recorded video])
        base([Run the no-agent baseline])
        up([Upload a video through the API])
        live([Watch a live webcam])
        ask([Ask about status, exits, alerts, durations])
        out([Read timelines, events, decisions and agent trace])
        lab([Write and review labels])
        ev([Evaluate predictions against labels])
        clean([Delete old upload jobs])
    end
    operator --- cal
    operator --- ana
    operator --- base
    operator --- up
    operator --- clean
    staff --- live
    staff --- ask
    staff --- out
    annotator --- lab
    annotator --- ev
    ana -. includes .-> out
    live -. includes .-> ask
```

## Activity: analysing a recorded video (`pipeline.analyze`)

```mermaid
flowchart TD
    start((start)) --> validate[Validate the merged config]
    validate --> cache{Cached observations match video,<br/>perception settings, weights and code?}
    cache -- yes --> read[Read observations.jsonl]
    cache -- no --> sample[Sample frames on the 5 fps grid]
    sample --> pose[YOLO11n-pose + ByteTrack]
    pose --> select[TargetSelector: which box is the resident]
    select --> measure[Measure posture and bed geometry]
    measure --> write[Write observations.jsonl]
    write --> propose
    read --> propose[Rule proposal and evidence per sample]
    propose --> agentq{Agent enabled?}
    agentq -- yes --> review[Review lying outside the bed, UNKNOWN gaps,<br/>standing over the bed; VLM only on identified samples]
    agentq -- no --> timeline
    review --> timeline[Temporal engine: majority filter, dwell, backdating]
    timeline --> fsm[Bed-event state machine; exit candidates verified]
    fsm --> policy[Alert policy: NORMAL / MONITOR / ALERT]
    policy --> outputs[Write timelines, events, summary, trace, manifest]
    outputs --> stop((end))
```

## Sequence: analysis and exit verification (`pipeline.run_analysis`)

```mermaid
sequenceDiagram
    participant P as pipeline.run_analysis
    participant T as temporal
    participant A as ContextAgent
    participant V as QwenVLM
    participant E as events.detect_bed_events
    participant D as events.departure
    participant Po as AlertPolicy
    P->>T: propose_all(observations)
    T-->>P: rule proposals (also the initial evidence)
    P->>A: review()
    loop each LYING_ON_FLOOR, UNKNOWN or STANDING run
        A->>A: look_back / look_ahead / inspect_bed_relation
        opt resident identified in enough samples and a VLM is loaded
            A->>V: ask(crop with the target outlined in green) x3
            V-->>A: posture, support (must agree)
        end
        A->>A: relabel proposals, mark only observed or inspected samples as evidence
    end
    A-->>P: proposals, evidence
    P->>T: build_timeline(proposals)
    T-->>P: activity and bed-status segments
    P->>E: detect_bed_events(labels, evidence)
    loop every sample
        E->>E: mode from the committed timeline, timers from evidence
        alt exit candidate open and recheck due
            E->>A: verify_exit(start, now, next in-bed time)
            A->>D: departure(observations, window)
            D-->>A: time, confidence, stats
            A-->>E: Verdict (time or None, until, evidence)
        end
    end
    E-->>P: events, rejected, unresolved, pending spans
    P->>Po: run(labels, observations, fsm)
    Po-->>P: decision segments, rule episodes
```

## State machine: bed events (`events.detect_bed_events`)

Transitions follow the committed bed status. The two timers (observed out-of-bed time for the dwell fallback, and
the continuous supported lying that confirms a return) count evidence only.

```mermaid
stateDiagram-v2
    [*] --> UNINITIALIZED
    UNINITIALIZED --> IN_BED_BASELINE: bed status IN_BED
    UNINITIALIZED --> AWAY_EPISODE: bed status OUT_OF_BED (no exit event)
    IN_BED_BASELINE --> EXIT_PENDING: bed status OUT_OF_BED (exit start)
    EXIT_PENDING --> IN_BED_BASELINE: back IN_BED before departing (rejected)
    EXIT_PENDING --> AWAY_EPISODE: departure confirmed, or 30 s observed out of bed (BED_EXIT)
    AWAY_EPISODE --> RETURN_PENDING: bed status IN_BED (return start)
    RETURN_PENDING --> AWAY_EPISODE: bed status OUT_OF_BED (start kept if no departure)
    RETURN_PENDING --> IN_BED_BASELINE: 2 s of lying seen without a break (RETURN_TO_BED)
    EXIT_PENDING --> [*]: recording ends (unresolved)
    RETURN_PENDING --> [*]: recording ends (unresolved)
```

## Classes and data

```mermaid
classDiagram
    class Observation {
        t, frame_idx, visible, identity_ok, reason
        track_id, n_persons, det_conf, bbox, keypoints, kp_conf, truncated
        torso_angle, leg_angle, knee_angle, thigh_ratio, knee_drop, thigh_angle
        anchor, body_in_bed, hip_in_bed, feet_in_bed, bed_dist, near_bed, edge_of_bed, in_chair
        speed, box_speed
    }
    class Proposal {
        t, label, confidence, reason, source
    }
    class Segment {
        start, end, label, confidence, reasons
        duration()
    }
    class BedEvent {
        event, episode_id, start_sec, confirmed_sec
        previous_state, current_state, confidence, decision, evidence
    }
    class Verdict {
        time, until, confidence, evidence
    }
    class Person {
        track_id, bbox, keypoints, conf
        center()
    }
    class VideoReader {
        sample(fps)
        frame_at(t)
    }
    class Scene {
        bed, chairs, near_margin, edge_margin
        in_bed(pt), bed_dist(pt), ignored(bbox), bed_box()
    }
    class PoseTracker {
        __call__(frame) list~Person~
        revision
    }
    class TargetSelector {
        track_id, hist, others, last_pos
        select(t, frame, persons)
    }
    class ContextAgent {
        props, evidence, trace
        review()
        verify_exit(t_pending, t_now, t_limit) Verdict
    }
    class QwenVLM {
        revision
        ask(image) answer
    }
    class AlertPolicy {
        run(...) decisions, episodes, timers
        event_decision(event, decisions)
    }
    class LiveSession {
        obs, archive, init, t0
        push(frame, t)
        update()
        answer(question)
    }
    PoseTracker ..> Person : returns
    TargetSelector ..> Person : selects
    TargetSelector --> Scene
    ContextAgent ..> Observation : reads
    ContextAgent ..> Proposal : relabels
    ContextAgent ..> Verdict : returns
    ContextAgent --> QwenVLM : optional
    AlertPolicy ..> Segment : returns
    LiveSession --> PoseTracker
    LiveSession --> TargetSelector
    LiveSession ..> BedEvent : reports
```

`pipeline.run_analysis`, `temporal.build_timeline` and `events.detect_bed_events` are functions rather than
classes; they turn `Observation`s into `Proposal`s, `Segment`s and `BedEvent`s.

## Components (`src/elder_monitor`)

```mermaid
flowchart LR
    cli[cli] --> pipeline
    cli --> evaluation
    cli --> calibrate
    cli --> server
    server[server: FastAPI] --> pipeline
    server --> live
    live --> pipeline
    live --> vision
    pipeline --> video
    pipeline --> vision
    pipeline --> features
    pipeline --> temporal
    pipeline --> agent
    pipeline --> events
    pipeline --> policy
    pipeline --> reporting
    agent --> vlm
    agent --> events
    evaluation --> reporting
    config[config] -.-> pipeline
    vision -. ultralytics .-> yolo[(YOLO11n-pose weights)]
    vlm -. transformers .-> qwen[(Qwen2.5-VL-3B)]
    schemas[schemas] -.-> pipeline
```

## Deployment

```mermaid
flowchart LR
    subgraph host [One computer: Windows 11 or Linux, Python 3.12]
        subgraph proc [Python process]
            clip[CLI: python -m elder_monitor]
            api[Optional: uvicorn + FastAPI, 127.0.0.1:8000]
        end
        gpu[(Optional NVIDIA GPU)]
        files[(Local files: videos, configs, examples/outputs, runs/)]
        weights[(yolo11n-pose.pt, Hugging Face cache)]
    end
    browser[Browser on the same machine or LAN] -- HTTP / WebSocket --> api
    proc --> gpu
    proc --> files
    proc --> weights
```

There is no database, message queue or cloud service. Upload jobs, results and live sessions stay on the host;
live sessions are held in memory only.
