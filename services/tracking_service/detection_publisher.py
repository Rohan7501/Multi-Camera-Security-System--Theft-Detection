"""Publish the latest tracked detections per stream, for the display service.

Tracking owns the detections, the display owns the frames and the drawing. This
writes one small "latest" JSON per stream to a tmpfs dir; the display joins it
against the shm frame named by frame_id. No gRPC and no shared address between
the two -- they agree on a directory path and nothing else.
"""
import json
import os
from pathlib import Path

DEFAULT_FEED_DIR = "/dev/shm/edge-display"


def _safe(stream_id: str) -> str:
    return stream_id.replace("/", "_").replace("..", "_")


class DetectionPublisher:
    def __init__(self, root: str = None):
        self.dir = Path(root or os.getenv("DISPLAY_FEED_DIR", DEFAULT_FEED_DIR))
        self.dir.mkdir(parents=True, exist_ok=True)

    def publish(self, stream_id: str, frame_id, timestamp_ns, tracked) -> None:
        rec = {
            "stream": stream_id,
            "frame_id": int(frame_id),
            "src_ns": int(timestamp_ns),        # pipeline steady_clock; NOT wall time
            "dets": [
                {
                    "x1": round(float(d.gX1), 1), "y1": round(float(d.gY1), 1),
                    "x2": round(float(d.gX2), 1), "y2": round(float(d.gY2), 1),
                    "cls": int(d.gClassId),
                    "conf": round(float(d.gConfidence), 3),
                    "track": int(tid),
                }
                for d, tid in tracked
            ],
        }
        name = _safe(stream_id)
        tmp = self.dir / f".{name}.json.tmp"
        final = self.dir / f"{name}.json"
        tmp.write_text(json.dumps(rec, separators=(",", ":")))
        os.replace(tmp, final)                  # atomic on the same filesystem (tmpfs)
