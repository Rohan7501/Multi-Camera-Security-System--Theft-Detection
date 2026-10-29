"""Opt-in per-frame latency trace: one CSV line per tracked frame.

Off unless LATENCY_TRACE names a file. tests/benchmark_pipeline.py turns it on to
get exact tail percentiles and a per-hop split; the Prometheus histograms only
place each frame in a bucket. Format and cost are in README.md.
"""
import os

HEADER = "stream,frame,capture_ns,inference_rx_ns,tracking_rx_ns,done_ns\n"


class LatencyTrace:
    def __init__(self, path: str):
        self.path = path
        # One os.write per line on an O_APPEND fd: each line reaches the kernel
        # whole, so nothing is lost when the service is killed, and two streams
        # (an inference reconnect overlapping the old one) never interleave
        # mid-line. There is no Python-side buffer to flush.
        self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        if os.fstat(self._fd).st_size == 0:
            os.write(self._fd, HEADER.encode())

    def record(self, resp, tracking_rx_ns: int, done_ns: int) -> None:
        """All four stamps share CLOCK_MONOTONIC on this box: capture (ingest),
        inference receipt (resp.gTimestampNs), tracking receipt, and done."""
        os.write(self._fd, (f"{resp.gStreamId},{resp.gFrameId},{resp.gCaptureTimestampNs},"
                            f"{resp.gTimestampNs},{tracking_rx_ns},{done_ns}\n").encode())
