#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
VIDEO="${SIM_VIDEO:-$HOME/Downloads/videoplayback.mp4}"
# Transport for the frame data path: "shm" (default) or "grpc". MUST match the
# inference launcher — the binary reads this from the environment, so it has to
# be exported. Override at the call site: FRAME_TRANSPORT=grpc ./scripts/run_ingest.sh
export FRAME_TRANSPORT="${FRAME_TRANSPORT:-shm}"
cd "$REPO"

pids=()
cleanup() { [ ${#pids[@]} -gt 0 ] && kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# 1. RTSP server first
cd "$REPO/rtsp_server" && ./mediamtx &
pids+=($!)
until bash -c '</dev/tcp/127.0.0.1/8554' 2>/dev/null; do sleep 0.2; done   # wait til :8554 is up

# 2. Publish the two camera feeds (cam1=live, cam2=live1)
ffmpeg -re -stream_loop -1 -i "$VIDEO" -c copy -f rtsp rtsp://localhost:8554/live  &
pids+=($!)
ffmpeg -re -stream_loop -1 -i "$VIDEO" -c copy -f rtsp rtsp://localhost:8554/live1 &
pids+=($!)
sleep 2                                                                   # let streams register before ingest

# 3. Ingest (foreground — NOT exec, so cleanup still fires on exit).
#    Requires inference_service (run_local.sh) already running on :50051.
./build/services/ingest_service/ingest_service "$@"
