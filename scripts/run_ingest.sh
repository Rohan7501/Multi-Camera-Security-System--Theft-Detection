#!/usr/bin/env bash
# Ingest service only: reads cameras from config.yaml and pushes frames onward.
#
# It does NOT start an RTSP server or publish any simulated streams -- bring the
# cameras up yourself first (see "Simulating RTSP streams" in README.md), or
# point config.yaml at real ones. Requires inference_service already listening
# on :50051, since ingest is the gRPC client.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Transport for the frame data path: "shm" (default) or "grpc". MUST match the
# inference launcher — the binary reads this from the environment, so it has to
# be exported. Override at the call site: FRAME_TRANSPORT=grpc ./scripts/run_ingest.sh
export FRAME_TRANSPORT="${FRAME_TRANSPORT:-shm}"

cd "$REPO"
exec ./build/services/ingest_service/ingest_service "$@"
