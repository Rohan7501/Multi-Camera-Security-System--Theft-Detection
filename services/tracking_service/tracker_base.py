"""Algorithm-agnostic tracking contract.

TrackerInput is the exhaustive per-frame input any tracker might want -- pixels,
detections, and the stream/frame/timestamp bookkeeping our multi-camera pipeline
needs. A Tracker returns one track id per detection. Swapping ByteTrack for
SORT, DeepSORT or OC-SORT means implementing this and nothing else.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TrackerInput:
    """One frame's worth of tracker input.

    The exhaustive set any tracker might want. Fields in README.md.
    """
    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray
    stream_id: str
    frame_id: int = 0
    timestamp_ns: int = 0
    frame: Optional[np.ndarray] = None


class Tracker(ABC):
    """A single-stream, stateful multi-object tracker.

    One instance per stream_id; TrackerEngine owns the mapping.
    """

    @abstractmethod
    def update(self, inp: TrackerInput) -> list:
        """Return a track id for each detection, aligned to inp.boxes order.

        Use -1 for a detection the algorithm chose not to track this frame.
        """
        raise NotImplementedError
