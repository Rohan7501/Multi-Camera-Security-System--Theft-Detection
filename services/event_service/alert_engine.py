"""The event service's front door, called once per tracked frame.

Tracking calls update() in-process; there's no server here yet. Each call logs
the frame, updates per-track suspicion, runs the rules, applies dedupe and
cooldown, and hands whatever survives to the sinks.

Timing runs on the frame's own capture instant, not on when we processed it --
that distinction is load-bearing and explained in README.md. Data lands under
EVENT_DATA_DIR (default <repo>/data).
"""
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from alert_rules import Alert, SustainedSuspicionRule
from detection_log import DetectionLogger
from event_store import build_sinks
from suspicion import SuspicionTracker

log = logging.getLogger("events.engine")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = _REPO_ROOT / "data"

PRUNE_EVERY_NS = 5 * 1_000_000_000


@dataclass
class _FireRecord:
    """Cooldown state for one (stream, track, rule) key."""
    fired_ns: int               # when the parent alert fired (cooldown anchor)
    alert_id: str               # parent alert -- suppressed evidence links here
    last_suppressed_ns: int = 0 # throttle for suppressed-evidence records


class AlertEngine:
    def __init__(self, data_dir=None, rules=None, cooldown_s: float = 60.0,
                 suppressed_every_s: float = 5.0,
                 annotate=None, clock=time.monotonic_ns):
        root = Path(data_dir or os.getenv("EVENT_DATA_DIR") or DEFAULT_DATA_DIR)
        root.mkdir(parents=True, exist_ok=True)

        self.rules = rules if rules is not None else [SustainedSuspicionRule()]
        self.sinks = build_sinks(root)
        self.detlog = DetectionLogger(root)
        self.suspicion = SuspicionTracker()
        self.cooldown_ns = int(cooldown_s * 1e9)
        # During cooldown the rule keeps firing every frame; saving all of that
        # is noise. One evidence record per this interval bounds the volume
        # (cooldown_s / suppressed_every_s records per suppressed window).
        self.suppressed_every_ns = int(suppressed_every_s * 1e9)

        self._annotate = annotate       # optional: annotate(frame, tracked) in place
        self._clock = clock
        self._fired: dict[tuple, _FireRecord] = {}   # (stream, track, rule) -> cooldown state
        self._lock = threading.Lock()
        self._last_prune = 0
        log.info("AlertEngine: data=%s rules=%s cooldown=%.0fs suppressed_every=%.0fs",
                 root, [r.name for r in self.rules], cooldown_s, suppressed_every_s)

    def update(self, stream_id, tracked, frame_id=0, timestamp_ns=0, frame=None,
               capture_timestamp_ns=0) -> list[Alert]:
        """Process one tracked frame; returns the alerts fired (usually []).

        capture_timestamp_ns is the timeline everything runs on. See README.md.
        """
        wall = self._clock()
        self.detlog.log(stream_id, frame_id, time.time(), timestamp_ns, tracked,
                        capture_ns=capture_timestamp_ns)

        # -1 = unknown age (no capture stamp). Clamped at 0 so clock skew can't
        # fake that sentinel.
        age_s = (max(0.0, (wall - capture_timestamp_ns) / 1e9)
                 if capture_timestamp_ns > 0 else -1.0)

        # Time everything by the frame's own capture instant, not by when we got
        # round to it. A backlogged stream arrives in bursts, and on the
        # processing clock a burst looks like zero elapsed time -- sustain would
        # never accumulate and the track would never alert. See README.md.
        now = capture_timestamp_ns if capture_timestamp_ns > 0 else wall

        fired: list[Alert] = []
        evidence: list[Alert] = []        # suppressed fires saved during cooldown
        with self._lock:
            for state in self.suspicion.observe_frame(stream_id, tracked, frame_id, now):
                for rule in self.rules:
                    # Dedupe key is per (stream, track, rule): each track alerts
                    # independently -- one stream with three shoplifting tracks
                    # produces three alerts, each with its own cooldown.
                    key = (state.stream_id, state.track_id, rule.name)
                    alert = rule.evaluate(state, now)
                    if alert is None:
                        continue
                    # Rules are clock-free by design, so the engine stamps age.
                    alert.frame_age_s = age_s
                    rec = self._fired.get(key)
                    if rec is not None and now - rec.fired_ns < self.cooldown_ns:
                        # In cooldown: don't re-alert, but SAVE the still-firing
                        # evidence (throttled), linked to the parent alert.
                        if now - rec.last_suppressed_ns >= self.suppressed_every_ns:
                            rec.last_suppressed_ns = now
                            alert.suppressed = True
                            alert.parent_id = rec.alert_id
                            evidence.append(alert)
                        continue
                    self._fired[key] = _FireRecord(fired_ns=now, alert_id=alert.id,
                                                   last_suppressed_ns=now)
                    fired.append(alert)

            if now - self._last_prune > PRUNE_EVERY_NS:
                self.suspicion.prune(now)
                self._prune_fired(now)
                self._last_prune = now

        for alert in fired + evidence:    # sink I/O outside the state lock
            self._emit(alert, frame, tracked)
        return fired

    def _emit(self, alert: Alert, frame, tracked) -> None:
        annotated = None
        if frame is not None:
            annotated = frame.copy()      # never draw on the caller's frame
            if self._annotate is not None:
                try:
                    self._annotate(annotated, tracked)
                except Exception:
                    log.exception("annotate failed; dumping plain frame")
        for sink in self.sinks:
            try:
                sink.emit(alert, frame=annotated)
            except Exception:
                log.exception("sink %s failed for alert %s",
                              type(sink).__name__, alert.id)
        if alert.suppressed:
            log.info("suppressed evidence %s (parent %s) stream=%s track=%d score=%.2f frame=%d",
                     alert.id, alert.parent_id, alert.stream_id, alert.track_id,
                     alert.score, alert.frame_id)
        else:
            age = (f"{alert.frame_age_s * 1000:.0f}ms"
                   if alert.frame_age_s >= 0 else "unknown")
            log.warning("ALERT %s rule=%s stream=%s track=%d score=%.2f frame=%d age=%s",
                        alert.id, alert.rule, alert.stream_id, alert.track_id,
                        alert.score, alert.frame_id, age)

    def _prune_fired(self, now) -> None:
        # Entries past cooldown no longer suppress anything; drop them.
        dead = [k for k, rec in self._fired.items()
                if now - rec.fired_ns >= self.cooldown_ns]
        for k in dead:
            del self._fired[k]

    def close(self) -> None:
        self.detlog.close()
        for sink in self.sinks:
            sink.close()
