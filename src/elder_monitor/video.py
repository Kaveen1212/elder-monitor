import hashlib

import cv2


class VideoReader:
    def __init__(self, path):
        self.path = str(path)
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.duration = self.frame_count / self.fps
        self._seek = None

    def sample(self, fps):
        """Yield (t, frame_idx, frame) on a fixed time grid using decoded timestamps.

        Grid slots with no decodable frame are yielded as (slot_time, -1, None).
        """
        cap = cv2.VideoCapture(self.path)
        step, half = 1.0 / min(fps, self.fps), 0.5 / self.fps
        slot, last_t, idx = 0, 0.0, -1
        while cap.grab():
            idx += 1
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t <= 0 and idx > 0:
                t = idx / self.fps
            t = last_t = max(t, last_t)
            k = int((t + half) // step)
            if k < slot:
                continue
            for s in range(slot, k):
                yield s * step, -1, None
            ok, frame = cap.retrieve()
            yield t, idx, frame if ok else None
            slot = k + 1
        cap.release()
        self.duration = last_t + 1.0 / self.fps if idx >= 0 else 0.0

    def frame_at(self, t):
        """Frame whose decoded timestamp is the first at or after t (robust to variable frame rate)."""
        cap = self._seek = self._seek or cv2.VideoCapture(self.path)
        for back in (2.0, 30.0, t):
            cap.set(cv2.CAP_PROP_POS_MSEC, max(t - back, 0.0) * 1000.0)
            ok = cap.grab()
            if ok and back < t and cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 > t:
                continue
            while ok:
                if cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 >= t - 1e-3:
                    return cap.retrieve()[1]
                ok = cap.grab()
        return None


def file_hash(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()
