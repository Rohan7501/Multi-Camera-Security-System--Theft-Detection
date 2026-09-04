"""Per-track suspicion state: EWMA + hysteresis.

A single frame's classification flickers -- Shoplifting at 0.6 one frame, gone
the next. The EWMA integrates that over time so suspicion only rises on
sustained evidence, and hysteresis (raise above RAISE_AT, clear only below
CLEAR_AT) stops a track hovering at the threshold from flapping every frame.

Timestamps come from the caller and are boot-relative steady_clock ns; never
compare them to wall time. See README.md.
"""
from collections import deque
from dataclasses import dataclass, field

ALPHA = 0.30            # EWMA weight of the newest observation
RAISE_AT = 0.35         # raise `raised` when ewma crosses above this... .55
CLEAR_AT = 0.15        # ...clear it only when ewma falls back below this .25
HISTORY_LEN = 32        # recent observations kept per track (for alert context)
STALE_NS = 30 * 1_000_000_000   # forget a track not seen for this long

# Per-class contribution to the suspicion signal (retail labels, see
# proto/services.proto). Product-Picked could contribute a partial weight
# (e.g. 1: 0.4) once the tracker is trusted; start with the explicit class.
CLASS_WEIGHT = {3: 1.0}         # 3 = Shoplifting


@dataclass
class Observation:
    frame_id: int
    mono_ns: int
    class_id: int
    confidence: float
    box: tuple          # (x1, y1, x2, y2) in frame pixels


@dataclass
class TrackState:
    stream_id: str
    track_id: int
    ewma: float = 0.0
    raised: bool = False
    raised_since_ns: int = 0        # monotonic ns when `raised` last flipped True
    last_seen_ns: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))

    def observe(self, obs: Observation) -> None:
        signal = CLASS_WEIGHT.get(obs.class_id, 0.0) * obs.confidence
        self.ewma = ALPHA * signal + (1.0 - ALPHA) * self.ewma
        self.last_seen_ns = obs.mono_ns
        self.history.append(obs)

        if not self.raised and self.ewma >= RAISE_AT:
            self.raised = True
            self.raised_since_ns = obs.mono_ns
        elif self.raised and self.ewma <= CLEAR_AT:
            self.raised = False
            self.raised_since_ns = 0


class SuspicionTracker:
    """The per-track state buffer: (stream_id, track_id) -> TrackState.

    Not thread-safe on its own; AlertEngine serializes access.
    """

    def __init__(self):
        self._tracks: dict[tuple, TrackState] = {}

    def observe_frame(self, stream_id, tracked, frame_id, mono_ns) -> list[TrackState]:
        """Feed one frame's [(detection, track_id), ...]; return the states touched.

        Untracked detections (id < 0) are skipped -- nothing to accumulate against.
        """
        touched = []
        for det, tid in tracked:
            if tid is None or int(tid) < 0:
                continue
            key = (stream_id, int(tid))
            state = self._tracks.get(key)
            if state is None:
                state = self._tracks[key] = TrackState(stream_id, int(tid))
            state.observe(Observation(
                frame_id=int(frame_id),
                mono_ns=mono_ns,
                class_id=int(det.gClassId),
                confidence=float(det.gConfidence),
                box=(float(det.gX1), float(det.gY1), float(det.gX2), float(det.gY2)),
            ))
            touched.append(state)
        return touched

    def prune(self, mono_ns) -> int:
        """Drop tracks not seen for STALE_NS; bounds memory over long runs."""
        dead = [k for k, s in self._tracks.items() if mono_ns - s.last_seen_ns > STALE_NS]
        for k in dead:
            del self._tracks[k]
        return len(dead)

    def __len__(self):
        return len(self._tracks)
