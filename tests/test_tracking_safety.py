"""Tests for the tracking service's two safety settings in main.py.

Both exist because of a failure measured on the Jetson, where end-to-end latency
climbed past 30 s:

* the age-based drop (TRACKING_MAX_AGE_MS): a frame that's already past its
  capture-age limit when it arrives is skipped and counted, so a backlog drains
  instead of growing;
* OPENBLAS_NUM_THREADS defaulting to 1: numpy's OpenBLAS helper threads busy-wait,
  and ByteTrack's per-frame 4x4 solves kept ~5 cores spinning.

No GPU, camera or network: the servicer is driven with hand-built messages.
"""
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from prometheus_client import REGISTRY

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACKING_MAIN = REPO_ROOT / "services" / "tracking_service" / "main.py"
MS = 1_000_000            # ns


@pytest.fixture(scope="module")
def tracking_main():
    """tracking_service/main.py, loaded by path.

    `import main` would find event_service/main.py first (see conftest.py).
    Loading it also applies its OPENBLAS_NUM_THREADS default to os.environ; put
    that back so it can't leak into other tests' subprocesses.
    """
    saved = os.environ.get("OPENBLAS_NUM_THREADS")
    spec = importlib.util.spec_from_file_location("tracking_main", TRACKING_MAIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if saved is None:
        os.environ.pop("OPENBLAS_NUM_THREADS", None)
    else:
        os.environ["OPENBLAS_NUM_THREADS"] = saved
    return module


@pytest.fixture
def make_servicer(tracking_main, monkeypatch, tmp_path):
    """Build a servicer with a given TRACKING_MAX_AGE_MS (None = unset), writing
    its event data under tmp_path and fetching no pixels."""
    def make(max_age_ms=None):
        if max_age_ms is None:
            monkeypatch.delenv("TRACKING_MAX_AGE_MS", raising=False)
        else:
            monkeypatch.setenv("TRACKING_MAX_AGE_MS", str(max_age_ms))
        monkeypatch.setenv("EVENT_DATA_DIR", str(tmp_path))
        for var in ("TRACKING_PIXELS", "DISPLAY_DETECTIONS", "DISPLAY_PUBLISH"):
            monkeypatch.delenv(var, raising=False)
        return tracking_main.TrackingServicer()
    return make


class FakeContext:
    def peer(self):
        return "test"


def message(pb2, stream, frame_id, capture_ns):
    return pb2.grpcDetectionResponse(gStreamId=stream, gFrameId=frame_id,
                                     gTimestampNs=capture_ns,
                                     gCaptureTimestampNs=capture_ns)


def counter(name, stream):
    # Prometheus metrics are process-global, so each test uses its own stream id.
    return REGISTRY.get_sample_value(name, {"stream": stream}) or 0.0


# ---------------------------------------------------------------- age drop

@pytest.mark.parametrize("raw, expected_ms", [
    (None, 80), ("", 80), ("120", 120), ("0", 0), ("junk", 80), ("-5", 80)])
def test_max_age_parsing(tracking_main, monkeypatch, raw, expected_ms):
    """Same rules as inference's INFERENCE_MAX_AGE_MS: 0 disables, junk falls back."""
    if raw is None:
        monkeypatch.delenv("TRACKING_MAX_AGE_MS", raising=False)
    else:
        monkeypatch.setenv("TRACKING_MAX_AGE_MS", raw)
    assert tracking_main.max_age_ns() == expected_ms * MS


def test_stale_frames_are_skipped_and_counted(tracking_main, make_servicer):
    servicer = make_servicer(max_age_ms=80)
    pb2 = tracking_main.services_pb2
    now = time.monotonic_ns()
    stream = [message(pb2, "drop-on", 1, now),              # fresh: tracked
              message(pb2, "drop-on", 2, now - 1000 * MS),  # 1 s old: skipped
              message(pb2, "drop-on", 3, 0)]                # no stamp: never dropped

    servicer.grpcStreamDetections(iter(stream), FakeContext())

    assert counter("tracking_frame_stale_total", "drop-on") == 1
    assert counter("tracking_frames_total", "drop-on") == 2


def test_zero_disables_the_drop(tracking_main, make_servicer):
    servicer = make_servicer(max_age_ms=0)
    pb2 = tracking_main.services_pb2
    old = time.monotonic_ns() - 10_000 * MS

    servicer.grpcStreamDetections(iter([message(pb2, "drop-off", 1, old)]), FakeContext())

    assert counter("tracking_frame_stale_total", "drop-off") == 0
    assert counter("tracking_frames_total", "drop-off") == 1


# ---------------------------------------------------------------- OpenBLAS

# Loads main.py in a fresh interpreter (the pool is sized once, when numpy first
# loads, so this can't be checked in-process) and reports the resolved setting
# plus the thread count after numpy and scipy have loaded.
PROBE = """
import importlib.util, os, sys
sys.path.insert(0, {tracking_dir!r})
spec = importlib.util.spec_from_file_location("tracking_main", {main!r})
spec.loader.exec_module(importlib.util.module_from_spec(spec))
print(os.environ["OPENBLAS_NUM_THREADS"], len(os.listdir("/proc/self/task")))
"""


def probe(blas_env):
    env = {k: v for k, v in os.environ.items() if k != "OPENBLAS_NUM_THREADS"}
    if blas_env is not None:
        env["OPENBLAS_NUM_THREADS"] = blas_env
    code = PROBE.format(tracking_dir=str(TRACKING_MAIN.parent), main=str(TRACKING_MAIN))
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                         text=True, check=True, timeout=120).stdout.split()
    return out[-2], int(out[-1])


@pytest.mark.skipif(not Path("/proc/self/task").is_dir(), reason="needs Linux /proc")
def test_openblas_defaults_to_one_thread_before_numpy_loads():
    """Unset, junk or non-positive -> "1", applied before numpy sizes its pool: the
    process has exactly as many threads as with an explicit 1, so no helpers are
    spinning. An explicit positive value is left alone."""
    _, baseline = probe("1")
    for raw in (None, "junk", "0"):
        value, threads = probe(raw)
        assert value == "1", raw
        assert threads == baseline, raw
    assert probe("3")[0] == "3"
