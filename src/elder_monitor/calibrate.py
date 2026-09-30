from pathlib import Path

import cv2
import numpy as np
import yaml

from .video import VideoReader

HELP = "click: add point | n: next polygon (chair) | t: click target | u: undo | s/Enter: save | q: quit"


def calibrate(video, output, t=0.0):
    frame = VideoReader(video).frame_at(t)
    if frame is None:
        raise ValueError(f"no frame at {t}s in {video}")
    h, w = frame.shape[:2]
    state = {"polys": [[]], "target": None, "mode": "polygon"}

    def on_click(event, x, y, *_):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if state["mode"] == "target":
            state["target"], state["mode"] = (x, y), "polygon"
        else:
            state["polys"][-1].append((x, y))

    def draw():
        img = frame.copy()
        for k, poly in enumerate(state["polys"]):
            color = (255, 200, 0) if k == 0 else (200, 120, 0)
            for p in poly:
                cv2.circle(img, p, 4, color, -1)
            if len(poly) > 1:
                cv2.polylines(img, [np.array(poly, np.int32)], len(poly) > 2, color, 2)
        if state["target"]:
            cv2.drawMarker(img, state["target"], (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
        label = "BED" if len(state["polys"]) == 1 else f"CHAIR {len(state['polys']) - 1}"
        cv2.putText(img, f"{label} | {HELP}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return img

    cv2.namedWindow("calibrate", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("calibrate", on_click)
    while True:
        cv2.imshow("calibrate", draw())
        key = cv2.waitKey(30) & 0xFF
        if cv2.getWindowProperty("calibrate", cv2.WND_PROP_VISIBLE) < 1:
            return None
        if key == ord("n") and len(state["polys"][-1]) >= 3:
            state["polys"].append([])
        elif key == ord("t"):
            state["mode"] = "target"
        elif key == ord("u") and state["polys"][-1]:
            state["polys"][-1].pop()
        elif key in (ord("s"), 13) and len(state["polys"][0]) >= 3:
            break
        elif key in (ord("q"), 27):
            cv2.destroyAllWindows()
            return None
    cv2.destroyAllWindows()

    polys = [p for p in state["polys"] if len(p) >= 3]

    def norm(pts):
        return [[round(x / w, 4), round(y / h, 4)] for x, y in pts]

    cfg = {"scene": {"bed_polygon": norm(polys[0]), "chair_polygons": [norm(p) for p in polys[1:]]},
           "target": {"point": norm([state["target"]])[0] if state["target"] else None,
                      "time_sec": t if state["target"] else 0.0}}
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    cv2.imwrite(str(out.with_suffix(".png")), draw())
    return cfg
