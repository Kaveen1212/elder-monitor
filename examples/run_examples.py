"""Download the public Pexels clips, analyse them with and without the agent, and evaluate the labelled ones.

    python examples/run_examples.py                  # development clips, geometry-only agent
    python examples/run_examples.py --vlm            # agent may also consult Qwen2.5-VL
    python examples/run_examples.py --heldout --vlm  # held-out clips, never used while developing
"""
import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
DEV = {"4049556": "hd_1920_1080_30fps", "4052924": "hd_1920_1080_25fps", "8090735": "hd_1920_1080_24fps",
       "9057924": "hd_1920_1080_25fps", "6898173": "hd_1920_1080_25fps", "3753707": "hd_1920_1080_25fps"}
LABELLED = ["pexels_4049556", "pexels_4052924", "pexels_8090735", "roundtrip_4049556"]
HELDOUT = {"7938959": "hd_1080_1920_24fps", "4052925": "hd_1920_1080_25fps", "8090730": "hd_1920_1080_24fps",
           "9615483": "hd_1080_2048_25fps", "8539664": "hd_2048_1080_25fps", "8539659": "hd_1080_2048_25fps",
           "8591515": "hd_2048_1080_25fps", "7505324": "hd_1920_1080_30fps", "10608105": "hd_2048_1080_25fps",
           "5983705": "hd_1920_1080_25fps", "8862331": "hd_2048_1080_25fps", "8088284": "hd_1080_2048_24fps",
           "6130024": "hd_1920_1080_30fps"}


def elder_monitor(*args):
    subprocess.run([sys.executable, "-m", "elder_monitor", *map(str, args)], cwd=ROOT, check=True)


def fetch(clip, file):
    video = ROOT / "data" / f"pexels_{clip}.mp4"
    if not video.exists():
        video.parent.mkdir(exist_ok=True)
        url = f"https://videos.pexels.com/video-files/{clip}/{clip}-{file}.mp4"
        print(f"downloading {url}", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        video.write_bytes(urllib.request.urlopen(request, timeout=120).read())
    return video


def make_roundtrip(source):
    """The real clip forwards, 5 s of the empty room, then reversed: a bed exit followed by a return."""
    video = ROOT / "data" / "roundtrip_4049556.mp4"
    if not video.exists():
        cap, frames = cv2.VideoCapture(str(source)), []
        while (frame := cap.read()[1]) is not None:
            frames.append(cv2.resize(frame, (960, 540)))
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (960, 540))
        for frame in frames + [frames[-1]] * 150 + frames[::-1] + [frames[40]] * 120:
            writer.write(frame)
        writer.release()
    return video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm", action="store_true")
    parser.add_argument("--heldout", action="store_true", help="the held-out clips instead of the development clips")
    args = parser.parse_args()
    videos = {f"pexels_{clip}": fetch(clip, file) for clip, file in (HELDOUT if args.heldout else DEV).items()}
    if args.heldout:
        labelled, evaluation = list(videos), "evaluation/heldout"
    else:
        videos["roundtrip_4049556"] = make_roundtrip(videos["pexels_4049556"])
        labelled, evaluation = LABELLED, "evaluation/outputs"
    for name, video in videos.items():
        video, cfg = video.relative_to(ROOT), f"configs/pexels_{name.split('_')[1]}.yaml"
        elder_monitor("analyze", "--video", video, "--config", cfg, "--output", f"outputs/{name}", "--reuse",
                      *([] if args.vlm else ["--no-vlm"]))
        elder_monitor("analyze", "--video", video, "--config", cfg, "--output", f"outputs_baseline/{name}",
                      "--no-agent", "--observations", f"outputs/{name}/observations.jsonl")
    for variant, out in (("outputs", evaluation), ("outputs_baseline", f"{evaluation}_baseline")):
        elder_monitor("evaluate", "--predictions", *[f"{variant}/{c}" for c in labelled],
                      "--annotations", *[f"annotations/{c}.json" for c in labelled],
                      "--videos", *[videos[c].relative_to(ROOT) for c in labelled], "--output", out)


if __name__ == "__main__":
    main()
