"""Per-stream tracking manager (algorithm-agnostic).

Keeps one Tracker per stream_id and adapts between the proto grpcDetection
messages and the algorithm's TrackerInput. The concrete algorithm is injected as
a factory (defaults to ByteTrack), so swapping trackers touches nothing here:

    TrackerEngine(lambda stream_id: MyTracker())   # any Tracker implementation

update() fills each detection's gTrackId in place and returns
[(detection, track_id), ...]; an untracked detection gets track_id -1.
"""
from typing import Callable, Optional

import numpy as np

from tracker_base import Tracker, TrackerInput
from byte_tracker import ByteTrackTracker


def _default_factory(stream_id: str) -> Tracker:
    return ByteTrackTracker()


class TrackerEngine:
    def __init__(self, tracker_factory: Optional[Callable[[str], Tracker]] = None):
        # tracker_factory(stream_id) -> a fresh per-stream Tracker. Default: ByteTrack.
        self._make = tracker_factory or _default_factory
        self._trackers: dict[str, Tracker] = {}

    def _tracker(self, stream_id: str) -> Tracker:
        t = self._trackers.get(stream_id)
        if t is None:
            t = self._make(stream_id)
            self._trackers[stream_id] = t
        return t

    def update(self, stream_id: str, detections, frame=None,
               frame_id: int = 0, timestamp_ns: int = 0) -> list:
        """Track one frame's detections; returns [(detection, track_id), ...].

        Empty frames still step the tracker so lost tracks age out.
        """
        dets = list(detections)
        if dets:
            boxes = np.array([[d.gX1, d.gY1, d.gX2, d.gY2] for d in dets], dtype=np.float32)
            scores = np.array([d.gConfidence for d in dets], dtype=np.float32)
            classes = np.array([d.gClassId for d in dets], dtype=np.int32)
        else:
            boxes = np.zeros((0, 4), np.float32)
            scores = np.zeros((0,), np.float32)
            classes = np.zeros((0,), np.int32)

        inp = TrackerInput(boxes=boxes, scores=scores, classes=classes,
                           stream_id=stream_id, frame_id=frame_id,
                           timestamp_ns=timestamp_ns, frame=frame)
        ids = self._tracker(stream_id).update(inp)

        result = []
        for i, d in enumerate(dets):
            tid = int(ids[i]) if i < len(ids) else -1
            d.gTrackId = tid                          # fill the proto field in place
            result.append((d, tid))
        return result
