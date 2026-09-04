"""Every tracked frame, one JSONL line, from day one.

Raw material for the replay harness: re-feed these through AlertEngine to retune
thresholds against real footage before shipping a change. It costs almost
nothing and cannot be reconstructed later if you didn't record it -- so don't
turn it off. Files rotate daily under <root>/detlog/.

Line format and the two timestamps are documented in README.md.
"""
import json
import logging
import threading
import time
from pathlib import Path

log = logging.getLogger("events.detlog")


class DetectionLogger:
    def __init__(self, root: Path):
        self.dir = root / "detlog"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._day = None
        self._fh = None
        self._lock = threading.Lock()

    def log(self, stream_id, frame_id, wall_ts, src_ns, tracked, capture_ns=0) -> None:
        rec = {
            "ts": round(wall_ts, 3),
            "src_ns": int(src_ns),
            # Ingest's capture instant, never re-stamped by a hop. Recording it
            # alongside src_ns lets the replay harness reconstruct true frame age
            # per record -- src_ns alone only describes the final hop.
            "cap_ns": int(capture_ns),
            "stream": stream_id,
            "frame": int(frame_id),
            "dets": [
                {
                    "t": int(tid),
                    "c": int(det.gClassId),
                    "p": round(float(det.gConfidence), 3),
                    "b": [round(float(det.gX1), 1), round(float(det.gY1), 1),
                          round(float(det.gX2), 1), round(float(det.gY2), 1)],
                }
                for det, tid in tracked
            ],
        }
        line = json.dumps(rec, separators=(",", ":")) + "\n"
        with self._lock:
            self._file_for_today().write(line)

    def _file_for_today(self):
        day = time.strftime("%Y%m%d")
        if day != self._day:
            if self._fh:
                self._fh.close()
            # buffering=1: line-buffered, each record hits the OS on write
            self._fh = open(self.dir / f"detections-{day}.jsonl", "a", buffering=1)
            self._day = day
        return self._fh

    def close(self) -> None:
        with self._lock:
            if self._fh:
                self._fh.close()
                self._fh = None
                self._day = None
