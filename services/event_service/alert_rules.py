"""Alert rules over per-track suspicion state.

Rules read accumulated TrackState, never a sequence of events, so a dropped or
out-of-order frame can't break them. Each returns an Alert or None -- dedupe and
cooldown are the engine's job, not the rule's. Rules also stay clock-free; the
engine stamps timing onto whatever they return. See README.md.
"""
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from suspicion import TrackState


@dataclass
class Alert:
    rule: str
    stream_id: str
    track_id: int
    score: float                # EWMA suspicion at fire time
    frame_id: int               # frame the rule fired on
    class_id: int
    confidence: float
    box: tuple                  # (x1, y1, x2, y2)
    detail: dict = field(default_factory=dict)
    wall_ts: float = field(default_factory=time.time)   # human/event time
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    # Set by the engine when this fire landed inside the cooldown window of a
    # previous alert for the same (stream, track, rule): saved as evidence
    # attached to that parent alert, not delivered as a fresh alert.
    suppressed: bool = False
    parent_id: str = ""
    # How old the frame was when this alert fired (ingest RTSP read -> here),
    # stamped by the engine from gCaptureTimestampNs. Rules stay clock-free, so
    # they never set this. -1.0 = unknown (no capture stamp on the frame, e.g.
    # an older peer or a replayed record) -- distinct from 0.0, "instant".
    frame_age_s: float = -1.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "rule": self.rule,
            "wall_ts": round(self.wall_ts, 3),
            "stream_id": self.stream_id,
            "track_id": self.track_id,
            "score": round(self.score, 4),
            "frame_id": self.frame_id,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 4),
            "box": [round(v, 1) for v in self.box],
            "detail": self.detail,
            "suppressed": self.suppressed,
            "parent_id": self.parent_id,
            "frame_age_s": round(self.frame_age_s, 4),
        }


class Rule(ABC):
    name = "rule"

    @abstractmethod
    def evaluate(self, state: TrackState, mono_ns: int):
        """Return an Alert if the rule fires for this track, else None."""
        raise NotImplementedError


class SustainedSuspicionRule(Rule):
    """Fire when a track's suspicion has stayed raised for sustain_s seconds.

    Asks "has the EWMA been above the band this long?", never "in what order?".
    """
    name = "sustained_suspicion"

    def __init__(self, sustain_s: float = 0.5):#default 2
        self.sustain_ns = int(sustain_s * 1e9)

    def evaluate(self, state: TrackState, mono_ns: int):
        if not state.raised or not state.history:
            return None
        raised_for = mono_ns - state.raised_since_ns
        if raised_for < self.sustain_ns:
            return None
        last = state.history[-1]
        return Alert(
            rule=self.name,
            stream_id=state.stream_id,
            track_id=state.track_id,
            score=state.ewma,
            frame_id=last.frame_id,
            class_id=last.class_id,
            confidence=last.confidence,
            box=last.box,
            detail={"raised_for_s": round(raised_for / 1e9, 2)},
        )
