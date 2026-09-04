"""What the control service is allowed to configure, per service.

A ServiceSpec names one service and allowlists the env keys we may set for it.
These are read once at process init, so changing one means re-render the env
file and restart -- that's the LifecycleBackend's job. Camera add/remove on a
running ingest is a different thing entirely and lives on the IngestAdmin RPC.

See README.md for which keys the services actually read today, and for what
METRICS_ADDR does.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ServiceSpec:
    name: str            # logical name used by the control API
    unit: str            # systemd unit
    compose_service: str # docker-compose service name
    env_file: str        # EnvironmentFile= path (systemd) / env_file: (compose)
    params: tuple        # allowlist of launch-time env keys we may set


INGEST = ServiceSpec(
    name="ingest", unit="edge-ingest.service", compose_service="ingest",
    env_file="/etc/edge-ai/ingest.env",
    params=("INFERENCE_ADDR", "INGEST_ADMIN_ADDR", "FRAME_TRANSPORT", "METRICS_ADDR"),
)

INFERENCE = ServiceSpec(
    name="inference", unit="edge-inference.service", compose_service="inference",
    env_file="/etc/edge-ai/inference.env",
    params=("INFERENCE_BIND", "EXECUTION_PROVIDER", "WORKER_COUNT", "BATCH_SIZE",
            "TRACKING_ADDR", "FRAME_TRANSPORT", "TRACKING_PIXELS", "METRICS_ADDR"),
)

TRACKING = ServiceSpec(
    name="tracking", unit="edge-tracking.service", compose_service="tracking",
    env_file="/etc/edge-ai/tracking.env",
    params=("TRACKING_BIND", "TRACKING_ALGO", "FRAME_TRANSPORT", "TRACKING_PIXELS",
            "METRICS_ADDR"),
)

SERVICES = {s.name: s for s in (INGEST, INFERENCE, TRACKING)}

# FRAME_TRANSPORT is a deployment-wide invariant enforced by the C++ checkTransport
# handshake -- a mismatch aborts a peer at startup. The control service must render
# the SAME value to every service and restart them together when it changes.
SHARED_INVARIANTS = ("FRAME_TRANSPORT",)


def spec(name: str) -> ServiceSpec:
    try:
        return SERVICES[name]
    except KeyError:
        raise ValueError(f"unknown service '{name}'; known: {sorted(SERVICES)}")
