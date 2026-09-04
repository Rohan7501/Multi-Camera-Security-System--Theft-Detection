"""Prometheus metrics for the tracking service.

`start_metrics()` starts a small HTTP server on its own port that answers
`GET /metrics` for a Prometheus scrape. It is SEPARATE from the gRPC server on
:50052 -- gRPC speaks HTTP/2 + protobuf and cannot serve a scrape -- but it runs
in the same process, so the counters below are plain in-memory objects.

Address from METRICS_ADDR (default 0.0.0.0:9103); set METRICS_ADDR=off to run
without the endpoint. When disabled the metric objects still exist, so callers
never need a feature check.

Cardinality: label by `stream` (bounded camera set) and class_id (4 retail
classes). NEVER by track_id -- ByteTrack ids are unbounded and monotonic.
"""
import logging
import os

from prometheus_client import Counter, Gauge, Histogram, start_http_server

log = logging.getLogger("tracking.metrics")

DEFAULT_ADDR = "0.0.0.0:9103"

FRAMES = Counter(
    "tracking_frames_total", "detection messages processed", ["stream"])
DETECTIONS = Counter(
    "tracking_detections_received_total",
    "detections received, by retail class id "
    "(0=Product 1=Product-Picked 2=Regular 3=Shoplifting)",
    ["stream", "class_id"])
ACTIVE_TRACKS = Gauge(
    "tracking_active_tracks", "tracks with an id assigned in the last frame", ["stream"])
LATENCY = Histogram(
    "tracking_process_seconds", "tracker + alert engine time per frame", ["stream"],
    buckets=(0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0))
# Time a frame spent inside the inference module: queue wait + infer() + the
# gRPC hop. Buckets run higher than tracking's own process time because a
# backed-up queue can hold a frame for seconds. See README.md.
INFERENCE_MODULE_LATENCY = Histogram(
    "inference_module_latency_seconds",
    "seconds from inference receiving a frame to tracking receiving its detections "
    "(FrameQueue wait + infer() + gRPC hop)", ["stream"],
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1,
             0.25, 0.5, 1.0, 2.5, 5.0, 10.0))
# End-to-end frame age: RTSP read -> handled here. The number that answers "how
# stale is the footage we're acting on?", and it should equal the sum of the
# per-module latencies. See README.md.
PIPELINE_FRAME_AGE = Histogram(
    "pipeline_frame_age_seconds",
    "seconds from the ingest RTSP read to the frame being handled by tracking "
    "(end-to-end; ingest + inference + transport)", ["stream"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25,
             0.5, 1.0, 2.5, 5.0, 10.0, 30.0))
ALERTS = Counter(
    "tracking_alerts_total", "alerts emitted by the event engine",
    ["stream", "rule", "suppressed"])
STREAMS = Gauge(
    "tracking_streams_open", "inference->tracking gRPC streams currently open")


def start_metrics() -> bool:
    """Start the scrape endpoint. Returns False when disabled or the port is busy
    (a metrics failure must never take the tracking service down)."""
    addr = os.getenv("METRICS_ADDR", DEFAULT_ADDR)
    if not addr or addr in ("off", "0"):
        log.info("metrics disabled (METRICS_ADDR=%s)", addr)
        return False
    host, _, port = addr.rpartition(":")
    try:
        start_http_server(int(port), addr=host or "0.0.0.0")
        log.info("metrics scrape endpoint on http://%s/metrics", addr)
        return True
    except OSError as e:
        log.warning("metrics bind %s failed: %s (continuing without metrics)", addr, e)
        return False