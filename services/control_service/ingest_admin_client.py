"""Client for the ingest runtime control plane (IngestAdmin RPC).

Live camera operations on a *running* ingest, no restart -- as distinct from
lifecycle.py, which starts and stops the process itself. Talks to
INGEST_ADMIN_ADDR (default 127.0.0.1:50053).
"""
import logging
import os
from dataclasses import dataclass

import grpc

import services_pb2 as pb
import services_pb2_grpc as rpc

log = logging.getLogger("control.ingest_admin")


@dataclass
class CameraStatus:
    stream_id: str
    url: str
    running: bool
    fps: float
    frames_total: int
    reconnects: int
    last_frame_ts_ms: int


class IngestAdminError(RuntimeError):
    pass


class IngestAdminClient:
    def __init__(self, addr: str = None):
        self.addr = addr or os.getenv("INGEST_ADMIN_ADDR", "127.0.0.1:50053")
        self._chan = grpc.insecure_channel(self.addr)
        self._stub = rpc.IngestAdminStub(self._chan)

    def _ok(self, reply, action: str):
        # Domain failures come back as ok=False + message (transport still OK).
        if not reply.ok:
            raise IngestAdminError(f"{action}: {reply.message}")
        return reply

    def add_camera(self, stream_id: str, url: str):
        return self._ok(self._stub.AddCamera(pb.ingestCamera(gStreamId=stream_id, gUrl=url)),
                        f"add_camera {stream_id}")

    def start_stream(self, stream_id: str):
        return self._ok(self._stub.StartStream(pb.ingestStreamId(gStreamId=stream_id)),
                        f"start_stream {stream_id}")

    def stop_stream(self, stream_id: str):
        return self._ok(self._stub.StopStream(pb.ingestStreamId(gStreamId=stream_id)),
                        f"stop_stream {stream_id}")

    def remove_camera(self, stream_id: str):
        return self._ok(self._stub.RemoveCamera(pb.ingestStreamId(gStreamId=stream_id)),
                        f"remove_camera {stream_id}")

    def list_cameras(self) -> list:
        reply = self._stub.ListCameras(pb.ingestEmpty())
        return [
            CameraStatus(c.gStreamId, c.gUrl, c.gRunning, c.gFps,
                         c.gFramesTotal, c.gReconnects, c.gLastFrameTsMs)
            for c in reply.gCameras
        ]

    def close(self):
        self._chan.close()
