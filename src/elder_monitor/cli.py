import argparse
import json

from .calibrate import calibrate
from .config import load_config
from .evaluation import evaluate
from .pipeline import analyze
from .reporting import timeline_text


def cmd_analyze(args):
    cfg = load_config(args.config, {"sampling": {"fps": args.fps}} if args.fps else None)
    result = analyze(args.video, cfg, args.output, use_agent=not args.no_agent, use_vlm=not args.no_vlm,
                     reuse=args.reuse, overlay=args.overlay, observations_path=args.observations)
    print("\nActivity timeline")
    print(timeline_text(result["activity"]))
    print("\nBed events")
    for e in result["events"]:
        print(f"  {e.event:14s} start {e.start_sec:8.1f}s  confirmed {e.confirmed_sec:8.1f}s  "
              f"{e.previous_state.lower()} -> {e.current_state.lower()}  [{e.decision}]")
    print("\nAlerts / monitor episodes")
    for a in result["alerts"]:
        print(f"  {a['decision']:7s} {a['rule']:22s} {a['start_sec']:8.1f}s – {a['end_sec']:8.1f}s")
    print("\nSummary")
    print(json.dumps({k: v for k, v in result["summary"].items() if k != "human"}, indent=2))
    if result["view"]["degraded"]:
        print("\nWARNING: too few keypoints for long stretches; check the camera placement (whole body in view).")
    print(f"\nOutputs written to {args.output}")


def cmd_evaluate(args):
    n = len(args.predictions)
    if len(args.annotations) != n or (args.videos and len(args.videos) != n):
        raise SystemExit("--predictions, --annotations and --videos need the same number of entries")
    m = evaluate(args.predictions, args.annotations, args.output, args.tolerance, args.resolution,
                 videos=args.videos)
    act, bed = m["activity"], m["bed_status"]
    print(f"activity accuracy {act['accuracy']}  macro-F1 {act['macro_f1']}  "
          f"predicted UNKNOWN share {act['predicted_unknown_share']}")
    print(f"bed-status accuracy {bed['accuracy']}  macro-F1 {bed['macro_f1']}")
    for event, e in m["events"].items():
        print(f"{event:14s} TP {e['tp']} FP {e['fp']} FN {e['fn']}  precision {e['precision']}  recall {e['recall']}")
    print(f"activity duration macro abs error {m['duration']['activity_macro_abs_error_sec']}s")
    print(f"Evaluation written to {args.output}")


def cmd_calibrate(args):
    cfg = calibrate(args.video, args.output, args.time)
    print("cancelled" if cfg is None else f"saved {args.output}")


def cmd_serve(args):
    import uvicorn

    from .server import create_app

    uvicorn.run(create_app(load_config(args.config), args.runs, args.frontend), host=args.host, port=args.port)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="elder_monitor", description="Bed-exit and activity monitoring from video")
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="analyse one recorded video")
    a.add_argument("--video", required=True)
    a.add_argument("--config", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--fps", type=float, help="override sampling fps")
    a.add_argument("--no-agent", action="store_true", help="temporal baseline without context review")
    a.add_argument("--no-vlm", action="store_true", help="agent uses geometry/temporal tools only")
    a.add_argument("--reuse", action="store_true", help="reuse <output>/observations.jsonl when it matches")
    a.add_argument("--observations", help="reuse perception from another run's observations.jsonl")
    a.add_argument("--overlay", action="store_true", help="write overlay.mp4 for inspection")
    a.set_defaults(func=cmd_analyze)

    e = sub.add_parser("evaluate", help="compare predictions with ground-truth annotations")
    e.add_argument("--predictions", nargs="+", required=True)
    e.add_argument("--annotations", nargs="+", required=True)
    e.add_argument("--output", required=True)
    e.add_argument("--tolerance", type=float, default=2.0, help="event matching tolerance (s)")
    e.add_argument("--resolution", type=float, default=0.1, help="time grid resolution (s)")
    e.add_argument("--videos", nargs="+", help="source videos, to save failure snapshots")
    e.set_defaults(func=cmd_evaluate)

    c = sub.add_parser("calibrate", help="draw the bed / chair polygons and pick the resident")
    c.add_argument("--video", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--time", type=float, default=0.0)
    c.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("serve", help="HTTP + WebSocket API for the web frontend (upload and live webcam)")
    s.add_argument("--config", help="scene config whose bed polygon is the default")
    s.add_argument("--runs", default="runs", help="folder for uploaded videos and their results")
    s.add_argument("--frontend", help="folder with the web frontend to serve at /")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    args.func(args)
