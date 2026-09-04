#!/usr/bin/env bash
# Standalone display service (FastAPI + MJPEG from shm + /metrics).
# Requires: pipeline running with FRAME_TRANSPORT=shm, and tracking started with
# DISPLAY_PUBLISH=1 (run_tracker.sh) so the detection feed exists.
#   FRAME_TRANSPORT  shm     -- must match the pipeline (frames come from the ring)
#   DISPLAY_FEED_DIR         -- tracking<->display detection feed (default /dev/shm/edge-display)
#   DISPLAY_FPS      15      -- MJPEG cap per viewer
#   EVENT_DATA_DIR           -- event_service data root (for /alerts); default <repo>/data
export FRAME_TRANSPORT="${FRAME_TRANSPORT:-shm}"
PORT="${DISPLAY_PORT:-8088}"
cd "$SEC_SYS_ROOT_DIR/services/display_service" \
  && exec uvicorn main:app --host 0.0.0.0 --port "$PORT"