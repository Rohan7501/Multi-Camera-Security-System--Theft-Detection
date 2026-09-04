"""TrackingService gRPC server -- the end of the forward flow.

Inference pushes a detection stream here; we assign track IDs with ByteTrack,
run the alert engine in-process, optionally publish a feed for the display
service, and return a grpcAck when the client half-closes.

Run from THIS directory (the generated stubs use flat imports):
    python3 main.py [--addr 0.0.0.0:50052]

Regenerating stubs after a proto change, and the rest, is in README.md.
"""
import argparse
import logging
import os
import sys
import time
from concurrent import futures
from pathlib import Path

import grpc

import services_pb2
import services_pb2_grpc
import metrics
from tracker_engine import TrackerEngine

# The alert engine lives in the sibling event_service (flat modules, like this
# service). Appended (not prepended) so this directory's modules keep priority.
sys.path.append(str(Path(__file__).resolve().parent.parent / "event_service"))
from alert_engine import AlertEngine

log = logging.getLogger("tracking")


class TrackingServicer(services_pb2_grpc.TrackingServiceServicer):
    def __init__(self):
        self.engine = TrackerEngine()

        # Opt-in debug overlay: draw detections on the frame and imshow it.
        # Purely presentational -- the frame is fetched here and passed in.
        self._display = None
        if os.getenv("DISPLAY_DETECTIONS") == "1":
            from display import displayFrameWithDetections
            self._display = displayFrameWithDetections
            log.info("DISPLAY_DETECTIONS=1: drawing detections")

        # Pixels, only if something actually wants them: a pixel-consuming
        # tracker or the debug overlay. ByteTrack doesn't, so by default we skip
        # the fetch entirely. See README.md.
        self._frames = None
        if os.getenv("TRACKING_PIXELS") == "1" or self._display is not None:
            from frame_reader import make_frame_reader
            self._frames = make_frame_reader()
            log.info("resolving frames via %s", type(self._frames).__name__)

        # Alerting (event_service, in-process): EWMA suspicion + rules + sinks.
        # Annotated frame dumps only happen when a frame is available above.
        try:
            from display import draw_detections
        except Exception:
            draw_detections = None
        self.alerts = AlertEngine(annotate=draw_detections)

        # Detection feed for the standalone display_service (Option A): publish
        # latest boxes per stream; the display joins them with the shm frame.
        self._publisher = None
        if os.getenv("DISPLAY_PUBLISH") == "1":
            from detection_publisher import DetectionPublisher
            self._publisher = DetectionPublisher()
            log.info("DISPLAY_PUBLISH=1: publishing detections to %s", self._publisher.dir)

    def grpcStreamDetections(self, request_iterator, context):
        # Client-streaming: consume detection responses until the client closes.
        # Inference reconnects on failure, so this handler runs once per stream
        # incarnation -- the gauge tracks how many are open right now.
        frames = 0
        metrics.STREAMS.inc()
        try:
            for resp in request_iterator:
                frames += 1
                t0 = time.perf_counter()

                # How long the frame spent in the inference stage. monotonic_ns()
                # is the same clock C++ steady_clock uses, so the epochs line up
                # on this box. Guard on > 0: proto3 omits a 0 default and an
                # unset stamp would read as time-since-boot. See README.md.
                now_ns = time.monotonic_ns()
                if resp.gTimestampNs > 0:
                    delta_ns = now_ns - resp.gTimestampNs
                    metrics.INFERENCE_MODULE_LATENCY.labels(resp.gStreamId).observe(
                        delta_ns / 1e9 if delta_ns > 0 else 0.0)

                frame = self._frames.get(resp) if self._frames else None
                tracked = self.engine.update(resp.gStreamId, resp.gDetections,
                                             frame=frame,
                                             frame_id=resp.gFrameId,
                                             timestamp_ns=resp.gTimestampNs)

                log.info("stream=%s frame=%s detections=%d",
                         resp.gStreamId, resp.gFrameId, len(tracked))
                # Alerting before display: display draws on `frame` in place, while
                # the alert path annotates its own copy of the clean frame.
                alerts = self.alerts.update(resp.gStreamId, tracked,
                                            frame_id=resp.gFrameId,
                                            timestamp_ns=resp.gTimestampNs,
                                            frame=frame,
                                            capture_timestamp_ns=resp.gCaptureTimestampNs)

                now_ns = time.monotonic_ns()
                # End-to-end frame age: gCaptureTimestampNs is set once by ingest
                # and never re-stamped, so it spans the whole pipeline.
                if resp.gCaptureTimestampNs > 0:
                    age_ns = now_ns - resp.gCaptureTimestampNs
                    metrics.PIPELINE_FRAME_AGE.labels(resp.gStreamId).observe(
                        age_ns / 1e9 if age_ns > 0 else 0.0)                
                
                if self._publisher is not None:
                    self._publisher.publish(resp.gStreamId, resp.gFrameId,
                                            resp.gTimestampNs, tracked)
                # if self._display is not None:
                    # self._display(frame, tracked, window=f"tracking: {resp.gStreamId}")

                metrics.FRAMES.labels(resp.gStreamId).inc()
                metrics.LATENCY.labels(resp.gStreamId).observe(time.perf_counter() - t0)
                # -1 = not yet confirmed by ByteTrack; don't count as a live track.
                metrics.ACTIVE_TRACKS.labels(resp.gStreamId).set(
                    sum(1 for _, tid in tracked if tid >= 0))
                for det, track_id in tracked:
                    metrics.DETECTIONS.labels(resp.gStreamId, str(det.gClassId)).inc()
                    log.debug("  cls=%d conf=%.2f box=(%d,%d,%d,%d) track=%d",
                              det.gClassId, det.gConfidence,
                              int(det.gX1), int(det.gY1), int(det.gX2), int(det.gY2), track_id)
                for a in alerts or ():
                    metrics.ALERTS.labels(resp.gStreamId, a.rule,
                                          str(bool(a.suppressed)).lower()).inc()
        finally:
            metrics.STREAMS.dec()
        log.info("stream from %s ended after %d frames", context.peer(), frames)
        return services_pb2.grpcAck(ok=True)


def serve(addr: str) -> None:
    # Scrape endpoint first: its own HTTP port, independent of the gRPC server.
    metrics.start_metrics()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_TrackingServiceServicer_to_server(TrackingServicer(), server)
    server.add_insecure_port(addr)
    server.start()
    log.info("TrackingService listening on %s", addr)
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="TrackingService gRPC server")
    parser.add_argument("--addr", default="0.0.0.0:50052",
                        help="listen address  [%(default)s]")
    serve(parser.parse_args().addr)
