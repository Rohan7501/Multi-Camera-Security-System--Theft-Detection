"""Prometheus metrics for the display service.

These make the display service a Prometheus *target* (scrape /metrics). They
describe the display's own health -- how many frames it serves, how fresh they
are, how often the shm join misses -- NOT the pipeline's detection numbers,
which each upstream service exposes on its own /metrics.

Labels are keyed by `stream` only (a small, bounded set of cameras). Never label
by track_id: ByteTrack ids are unbounded and would explode Prometheus cardinality.
"""
from prometheus_client import Counter, Gauge

FRAMES_SERVED = Counter(
    "display_frames_served_total", "MJPEG frames sent to viewers", ["stream"])
SHM_MISS = Counter(
    "display_shm_miss_total", "frame unavailable in the shm ring (lapped/empty)", ["stream"])
VIEWERS = Gauge(
    "display_viewers", "active MJPEG viewers", ["stream"])
FRAME_AGE = Gauge(
    "display_frame_age_seconds", "wall-clock age of the latest served frame", ["stream"])
