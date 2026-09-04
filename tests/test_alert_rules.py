"""Tests for the suspicion/rule/engine chain in services/event_service.

These run without a GPU, a camera or a network, because the whole alerting path
is pure functions over an injectable clock. The one test that matters most is
test_burst_drain_still_alerts -- it pins the bug where a backlogged stream
silently stopped alerting.
"""
import pytest

from alert_engine import AlertEngine
from alert_rules import Alert, SustainedSuspicionRule
from suspicion import CLEAR_AT, RAISE_AT, SuspicionTracker, TrackState

SHOPLIFTING = 3
REGULAR = 2
SEC = 1_000_000_000        # ns


class FakeDetection:
    """Duck-types the protobuf detection that observe_frame() reads."""

    def __init__(self, class_id=SHOPLIFTING, confidence=0.9, box=(0, 0, 10, 10)):
        self.gClassId = class_id
        self.gConfidence = confidence
        self.gX1, self.gY1, self.gX2, self.gY2 = box


def tracked(class_id=SHOPLIFTING, confidence=0.9, track_id=1):
    """One frame's worth of [(detection, track_id)]."""
    return [(FakeDetection(class_id, confidence), track_id)]


# ---------------------------------------------------------------- suspicion

def test_ewma_rises_on_sustained_shoplifting():
    t = SuspicionTracker()
    for i in range(20):
        t.observe_frame("cam1", tracked(), frame_id=i, mono_ns=i * SEC)
    state = t._tracks[("cam1", 1)]
    assert state.ewma > RAISE_AT
    assert state.raised


def test_single_frame_flicker_does_not_raise():
    """One shoplifting frame among regular ones must not trip the threshold."""
    t = SuspicionTracker()
    frames = [REGULAR] * 5 + [SHOPLIFTING] + [REGULAR] * 5
    for i, cls in enumerate(frames):
        t.observe_frame("cam1", tracked(class_id=cls), frame_id=i, mono_ns=i * SEC)
    assert not t._tracks[("cam1", 1)].raised


def test_hysteresis_holds_between_the_thresholds():
    """Once raised, the track stays raised until ewma falls below CLEAR_AT --
    not merely below RAISE_AT. That gap is what stops per-frame flapping."""
    state = TrackState("cam1", 1)
    state.ewma = RAISE_AT + 0.1
    state.raised = True
    # Decay into the band between CLEAR_AT and RAISE_AT.
    mid = (RAISE_AT + CLEAR_AT) / 2
    state.ewma = mid
    state.observe(_obs(class_id=REGULAR, confidence=0.0, mono_ns=SEC))
    assert CLEAR_AT < state.ewma < RAISE_AT
    assert state.raised, "must not clear inside the hysteresis band"


def test_untracked_detections_are_skipped():
    t = SuspicionTracker()
    t.observe_frame("cam1", [(FakeDetection(), -1)], frame_id=0, mono_ns=0)
    assert len(t) == 0


def test_prune_drops_stale_tracks():
    t = SuspicionTracker()
    t.observe_frame("cam1", tracked(), frame_id=0, mono_ns=0)
    assert len(t) == 1
    assert t.prune(60 * SEC) == 1
    assert len(t) == 0


def _obs(class_id, confidence, mono_ns, frame_id=0):
    from suspicion import Observation
    return Observation(frame_id=frame_id, mono_ns=mono_ns, class_id=class_id,
                       confidence=confidence, box=(0, 0, 10, 10))


# ------------------------------------------------------------------- rules

def test_rule_silent_before_sustain_elapses():
    rule = SustainedSuspicionRule(sustain_s=2.0)
    state = TrackState("cam1", 1)
    state.observe(_obs(SHOPLIFTING, 0.9, mono_ns=0))
    state.raised, state.raised_since_ns = True, 0
    assert rule.evaluate(state, mono_ns=1 * SEC) is None


def test_rule_fires_once_sustain_elapses():
    rule = SustainedSuspicionRule(sustain_s=2.0)
    state = TrackState("cam1", 1)
    state.observe(_obs(SHOPLIFTING, 0.9, mono_ns=0, frame_id=42))
    state.raised, state.raised_since_ns = True, 0
    alert = rule.evaluate(state, mono_ns=3 * SEC)
    assert isinstance(alert, Alert)
    assert alert.stream_id == "cam1"
    assert alert.track_id == 1
    assert alert.frame_id == 42
    assert alert.detail["raised_for_s"] == pytest.approx(3.0)


def test_rule_silent_when_not_raised():
    rule = SustainedSuspicionRule(sustain_s=0.1)
    state = TrackState("cam1", 1)
    state.observe(_obs(SHOPLIFTING, 0.9, mono_ns=0))
    state.raised = False
    assert rule.evaluate(state, mono_ns=10 * SEC) is None


def test_rules_are_order_tolerant():
    """Rules read accumulated state, so a dropped or reordered frame can't break
    them -- feeding the same observations in reverse must reach the same verdict."""
    rule = SustainedSuspicionRule(sustain_s=1.0)
    forward, backward = TrackState("cam1", 1), TrackState("cam1", 1)
    obs = [_obs(SHOPLIFTING, 0.9, mono_ns=i * SEC, frame_id=i) for i in range(10)]
    for o in obs:
        forward.observe(o)
    for o in reversed(obs):
        backward.observe(o)
    assert forward.raised == backward.raised
    assert (rule.evaluate(forward, 20 * SEC) is None) == \
           (rule.evaluate(backward, 20 * SEC) is None)


# ------------------------------------------------------------------ engine

@pytest.fixture
def engine(tmp_path):
    """An engine writing to a temp dir, on a clock we control."""
    def make(clock, **kw):
        kw.setdefault("rules", [SustainedSuspicionRule(sustain_s=1.0)])
        kw.setdefault("cooldown_s", 60.0)
        e = AlertEngine(data_dir=tmp_path, clock=clock, **kw)
        return e
    return make


def _feed(e, n, start_ns, step_ns, stream="cam1"):
    """Push n frames whose CAPTURE stamps advance by step_ns."""
    out = []
    for i in range(n):
        out += e.update(stream, tracked(), frame_id=i, timestamp_ns=start_ns + i * step_ns,
                        capture_timestamp_ns=start_ns + i * step_ns)
    return out


def test_engine_fires_after_sustained_suspicion(engine):
    e = engine(clock=lambda: 10 * SEC)
    fired = _feed(e, n=30, start_ns=SEC, step_ns=SEC // 10)
    assert len(fired) == 1
    assert fired[0].rule == "sustained_suspicion"
    e.close()


def test_cooldown_suppresses_repeat_alerts(engine):
    e = engine(clock=lambda: 10 * SEC, cooldown_s=60.0)
    first = _feed(e, n=30, start_ns=SEC, step_ns=SEC // 10)
    more = _feed(e, n=30, start_ns=5 * SEC, step_ns=SEC // 10)
    assert len(first) == 1
    assert more == [], "second burst is inside the cooldown window"
    e.close()


def test_cooldown_expiry_allows_a_new_alert(engine):
    e = engine(clock=lambda: 10 * SEC, cooldown_s=5.0)
    first = _feed(e, n=30, start_ns=SEC, step_ns=SEC // 10)
    later = _feed(e, n=30, start_ns=100 * SEC, step_ns=SEC // 10)
    assert len(first) == 1
    assert len(later) == 1, "cooldown has expired; the track may alert again"
    e.close()


def test_dedupe_is_per_track_not_per_stream(engine):
    """Two suspicious tracks on one camera are two alerts, not one."""
    e = engine(clock=lambda: 10 * SEC)
    fired = []
    for i in range(30):
        cap = SEC + i * (SEC // 10)
        fired += e.update("cam1",
                          [(FakeDetection(), 1), (FakeDetection(), 2)],
                          frame_id=i, timestamp_ns=cap, capture_timestamp_ns=cap)
    assert {a.track_id for a in fired} == {1, 2}
    e.close()


def test_frame_age_is_unknown_without_a_capture_stamp(engine):
    """proto3 omits a 0 default, so a missing stamp must read as -1 (unknown),
    never as time-since-boot. Without a capture stamp the engine falls back to
    its own clock, so that clock has to advance for sustain to accumulate."""
    ticks = iter(range(0, 10_000))
    e = engine(clock=lambda: next(ticks) * (SEC // 10))
    fired = []
    for i in range(30):
        fired += e.update("cam1", tracked(), frame_id=i,
                          timestamp_ns=0, capture_timestamp_ns=0)
    assert fired, "should still alert on the fallback clock"
    assert all(a.frame_age_s == -1.0 for a in fired)
    e.close()


def test_frame_age_measures_capture_to_now(engine):
    """Age is measured from the frame that actually fired -- which is the frame
    where sustain first elapsed, not the last one fed in."""
    now = 10 * SEC
    e = engine(clock=lambda: now)
    fired = _feed(e, n=30, start_ns=SEC, step_ns=SEC // 10)
    assert len(fired) == 1
    captured_at = (SEC + fired[0].frame_id * (SEC // 10)) / 1e9
    assert fired[0].frame_age_s == pytest.approx(10.0 - captured_at, abs=0.01)
    e.close()


def test_burst_drain_still_alerts(engine):
    """Regression: alerting must run on FOOTAGE time, not processing time.

    Every camera is multiplexed onto one inference->tracking stream handled by
    one thread, so a stream that falls behind drains its backlog in a burst --
    dozens of frames processed microseconds apart. Timed on the processing
    clock, 4s of footage looks like ~10ms, `raised_for` never reaches sustain_s,
    and a genuinely suspicious track never alerts.

    Here the wall clock is frozen (the burst takes no processing time at all)
    while capture stamps advance normally. It must still fire.
    """
    e = engine(clock=lambda: 10 * SEC)          # frozen: zero processing elapsed
    fired = _feed(e, n=40, start_ns=SEC, step_ns=SEC // 10)   # 4s of footage
    assert len(fired) == 1, "a backlogged stream must alert like a healthy one"
    e.close()


def test_healthy_and_lagging_streams_agree(engine):
    """The same footage must produce the same verdict whether it arrives live or
    as a backlog burst. This is the pair that originally disagreed (1 vs 0)."""
    healthy = engine(clock=lambda: 10 * SEC)
    lagging = engine(clock=lambda: 10 * SEC)
    a = _feed(healthy, n=40, start_ns=SEC, step_ns=SEC // 10, stream="live")
    b = _feed(lagging, n=40, start_ns=SEC, step_ns=SEC // 10, stream="backlog")
    assert len(a) == len(b) == 1
    healthy.close()
    lagging.close()


def test_alert_serialises_for_the_sinks(engine):
    e = engine(clock=lambda: 10 * SEC)
    fired = _feed(e, n=30, start_ns=SEC, step_ns=SEC // 10)
    d = fired[0].to_dict()
    for key in ("id", "rule", "stream_id", "track_id", "score", "frame_id",
                "class_id", "confidence", "box", "suppressed", "parent_id",
                "frame_age_s"):
        assert key in d, f"{key} missing from the sink payload"
    assert len(d["box"]) == 4
    e.close()


def test_alerts_reach_the_sqlite_sink(tmp_path):
    """End-to-end through the real sinks: an alert must land as a queryable row."""
    import sqlite3
    e = AlertEngine(data_dir=tmp_path, clock=lambda: 10 * SEC,
                    rules=[SustainedSuspicionRule(sustain_s=1.0)])
    fired = _feed(e, n=30, start_ns=SEC, step_ns=SEC // 10)
    e.close()

    db = sqlite3.connect(str(tmp_path / "alerts.db"))
    rows = db.execute("SELECT id, stream_id, track_id, frame_age_s "
                      "FROM alerts WHERE suppressed=0").fetchall()
    db.close()
    assert len(rows) == len(fired) == 1
    assert rows[0][0] == fired[0].id
    assert rows[0][1] == "cam1"
    assert rows[0][3] >= 0
