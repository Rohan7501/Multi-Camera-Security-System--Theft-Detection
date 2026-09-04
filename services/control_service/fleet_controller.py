"""One operator-facing API over the two things you can do to the fleet.

Restarting a service and adding a camera are genuinely different operations --
one needs the process manager, the other talks to a process that's already
running -- so they go to different collaborators: a LifecycleBackend
(systemd/compose) and an IngestAdminClient. This class is where they meet.

No FastAPI in here on purpose: it keeps the thing unit-testable and makes the
HTTP layer a thin shell. Swapping systemd for compose changes construction only.
"""
import logging

from service_config import SHARED_INVARIANTS, spec, SERVICES
from lifecycle import LifecycleBackend, ServiceStatus

log = logging.getLogger("control.fleet")


class FleetController:
    def __init__(self, backend: LifecycleBackend, ingest_admin=None):
        self.backend = backend
        # Lazily built so the controller is usable for lifecycle-only ops when
        # ingest isn't up yet (the admin RPC would refuse a connection).
        self._ingest_admin = ingest_admin

    # ---- lifecycle: start/stop/restart + launch-time config ----------------

    def start(self, service: str) -> None:
        self.backend.start(spec(service))

    def stop(self, service: str) -> None:
        self.backend.stop(spec(service))

    def restart(self, service: str) -> None:
        self.backend.restart(spec(service))

    def configure(self, service: str, env: dict, restart: bool = False) -> None:
        """Stage launch-time config; optionally restart to apply it now."""
        s = spec(service)
        self.backend.apply_config(s, env)
        if restart:
            self.backend.restart(s)

    def set_frame_transport(self, transport: str) -> None:
        """FRAME_TRANSPORT is a fleet-wide invariant (checkTransport aborts a
        peer on mismatch): render it to EVERY service and restart them together."""
        if transport not in ("shm", "grpc"):
            raise ValueError("transport must be 'shm' or 'grpc'")
        for name, s in SERVICES.items():
            if "FRAME_TRANSPORT" in s.params:
                self.backend.apply_config(s, {"FRAME_TRANSPORT": transport})
        for name in SERVICES:  # restart order is the caller's concern; keep simple
            self.backend.restart(spec(name))
        log.info("FRAME_TRANSPORT=%s applied fleet-wide (%s)", transport, SHARED_INVARIANTS)

    def status(self) -> list:
        return [self.backend.status(spec(n)) for n in SERVICES]

    # ---- runtime: cameras on a running ingest, no restart ------------------

    @property
    def ingest(self):
        if self._ingest_admin is None:
            from ingest_admin_client import IngestAdminClient
            self._ingest_admin = IngestAdminClient()
        return self._ingest_admin

    def add_camera(self, stream_id: str, url: str):
        return self.ingest.add_camera(stream_id, url)

    def start_stream(self, stream_id: str):
        return self.ingest.start_stream(stream_id)

    def stop_stream(self, stream_id: str):
        return self.ingest.stop_stream(stream_id)

    def remove_camera(self, stream_id: str):
        return self.ingest.remove_camera(stream_id)

    def cameras(self) -> list:
        return self.ingest.list_cameras()
