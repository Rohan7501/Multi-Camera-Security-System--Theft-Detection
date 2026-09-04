"""Join shm frames with the tracking detection feed, draw, and JPEG-encode.

This is the heart of the Option-A display service: frames come from the shm ring
(ShmReader, the Python mirror of common/shm.hpp), detections come from the
tracking DetectionPublisher's per-stream JSON in DISPLAY_FEED_DIR, and this
module joins them by frame_id, draws the boxes, and returns a JPEG.

Requires FRAME_TRANSPORT=shm upstream -- in grpc mode ingest never fills the
ring, so there are no frames to serve here (that's the transport's nature, not a
bug). When a frame isn't available we serve a placeholder so the browser still
shows the stream is known.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np

# ShmReader is the canonical Python view of common/shm.hpp; reuse it rather than
# duplicate the byte layout (same cross-service import pattern tracking uses).
sys.path.append(str(Path(__file__).resolve().parent.parent / "tracking_service"))
from shm_reader import ShmReader  # noqa: E402

FEED_DIR = Path(os.getenv("DISPLAY_FEED_DIR", "/dev/shm/edge-display"))
JPEG_QUALITY = int(os.getenv("DISPLAY_JPEG_QUALITY", "75"))
# Per-camera cell in the mosaic. The wall is cols*CELL_W x rows*CELL_H, with the
# grid sized to ceil(sqrt(n)) -- 8 cameras -> 3x3 -> 1440x810 at the default.
CELL_W = int(os.getenv("DISPLAY_CELL_W", "480"))
CELL_H = int(os.getenv("DISPLAY_CELL_H", "270"))

CLASS_NAMES = {0: "Product", 1: "Product-Picked", 2: "Regular", 3: "Shoplifting"}
CLASS_COLORS = {0: (0, 200, 0), 1: (0, 180, 220), 2: (220, 160, 0), 3: (0, 0, 255)}
DEFAULT_COLOR = (200, 200, 200)


def _encode(img) -> bytes:
    ok, buf = cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else b""


def _placeholder(stream_id: str, msg: str) -> np.ndarray:
    img = np.full((360, 640, 3), 40, np.uint8)
    cv.putText(img, f"{stream_id}: {msg}", (20, 190),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv.LINE_AA)
    return img


def draw(img, dets) -> None:
    """Draw detection dicts (from the tracking feed) onto img in place."""
    for d in dets:
        x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
        cls = int(d.get("cls", -1))
        color = CLASS_COLORS.get(cls, DEFAULT_COLOR)
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = CLASS_NAMES.get(cls, str(cls))
        conf = d.get("conf")
        if conf is not None:
            label += f" {conf:.2f}"
        tid = d.get("track", -1)
        if tid and int(tid) > 0:
            label += f" #{int(tid)}"
        (tw, th), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ytop = max(0, y1 - th - 4)
        cv.rectangle(img, (x1, ytop), (x1 + tw + 2, ytop + th + 4), color, -1)
        cv.putText(img, label, (x1 + 1, ytop + th),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)


class FrameJoiner:
    """Lazily-opened ShmReader + detection-feed reader. One per process."""

    def __init__(self):
        self._shm = None

    def _reader(self):
        if self._shm is None:
            self._shm = ShmReader()          # raises if the segment isn't up yet
        return self._shm

    def streams(self) -> list:
        """Streams currently publishing detections (feed-dir *.json)."""
        if not FEED_DIR.exists():
            return []
        return sorted(p.stem for p in FEED_DIR.glob("*.json") if not p.name.startswith("."))

    def _feed(self, stream_id: str):
        """(record, wall_age_seconds) from the stream's feed file, or (None, None).

        Age comes from the file mtime, never the record's src_ns. See README.md.
        """
        p = FEED_DIR / f"{stream_id}.json"
        try:
            age = time.time() - p.stat().st_mtime
            return json.loads(p.read_text()), age
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            return None, None

    def render_bgr(self, stream_id: str):
        """Return (bgr_image, age_seconds, had_frame) for one annotated frame.

        Separate from render() so the mosaic can tile raw images and encode once.
        """
        rec, age = self._feed(stream_id)
        if rec is None:
            return _placeholder(stream_id, "no detection feed"), None, False

        # Fetch the exact frame the detections belong to; fall back to the ring's
        # newest if that one already lapped out.
        frame = None
        try:
            reader = self._reader()
            frame = reader.get_frame(stream_id, rec["frame_id"])
            if frame is None:
                latest = reader.latest_frame_id(stream_id)
                if latest is not None:
                    frame = reader.get_frame(stream_id, latest)
        except FileNotFoundError:
            return _placeholder(stream_id, "shm not up (FRAME_TRANSPORT=shm?)"), age, False

        if frame is None:
            return _placeholder(stream_id, "frame lapped"), age, False

        draw(frame, rec.get("dets", []))
        return frame, age, True

    def render(self, stream_id: str):
        """Return (jpeg_bytes, age_seconds, had_frame) for one annotated frame."""
        img, age, had = self.render_bgr(stream_id)
        return _encode(img), age, had

    def render_wall(self, stream_ids: list):
        """Composite every stream into ONE annotated grid image.

        Returns (jpeg, [(stream_id, age, had_frame), ...]) -- see README.md.
        """
        if not stream_ids:
            return _encode(_placeholder("wall", "no streams publishing")), []

        cols = math.ceil(math.sqrt(len(stream_ids)))
        rows = math.ceil(len(stream_ids) / cols)
        wall = np.zeros((rows * CELL_H, cols * CELL_W, 3), np.uint8)

        stats = []
        for i, sid in enumerate(stream_ids):
            img, age, had = self.render_bgr(sid)
            stats.append((sid, age, had))
            cell = cv.resize(img, (CELL_W, CELL_H), interpolation=cv.INTER_AREA)
            # Caption bar: stream id + staleness, so a frozen tile is obvious
            # even though every tile shares one connection.
            caption = sid if age is None else f"{sid}  {age:.1f}s"
            cv.rectangle(cell, (0, 0), (CELL_W, 22), (0, 0, 0), -1)
            cv.putText(cell, caption, (6, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 255, 0) if had else (0, 165, 255), 1, cv.LINE_AA)
            r, c = divmod(i, cols)
            wall[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W] = cell

        return _encode(wall), stats
