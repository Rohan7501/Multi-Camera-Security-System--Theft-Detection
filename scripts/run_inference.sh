#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

export LD_LIBRARY_PATH="$(gpu_ld_path):${LD_LIBRARY_PATH:-}"

# Transport for the frame data path: "shm" (default) or "grpc". MUST match the
# ingest launcher — the binary reads this from the environment, so it has to be
# exported, not just set. Override at the call site: FRAME_TRANSPORT=grpc ./scripts/run_inference.sh
export FRAME_TRANSPORT="${FRAME_TRANSPORT:-shm}"
# Does the tracking algorithm consume pixels (appearance/re-ID)? 1 => in grpc
# mode inference inlines the frame on the detection hop (gInline). Set the same
# value on the tracker (run_tracker.sh). ByteTrack doesn't need pixels: default 0.
export TRACKING_PIXELS="${TRACKING_PIXELS:-1}"

cd "$REPO"
exec ./build/services/inference_service/inference_service "$@"