#!/usr/bin/env bash
# Flags MUST match run_inference.sh / run_ingest.sh:
#  FRAME_TRANSPORT     shm|grpc -- where frames live (shm ring vs inline gRPC)
#  TRACKING_PIXELS     0|1      -- tracker consumes pixels (fetch via FrameReader)
#  DISPLAY_DETECTIONS  0|1      -- in-process debug overlay (cv2 window)
#  DISPLAY_PUBLISH     0|1      -- publish detections to the display_service feed
# NOTE: must be exported -- `VAR=1 && cmd` does NOT put VAR in cmd's environment.
export FRAME_TRANSPORT="${FRAME_TRANSPORT:-shm}"
export TRACKING_PIXELS="${TRACKING_PIXELS:-1}"
export DISPLAY_DETECTIONS="${DISPLAY_DETECTIONS:-1}"
export DISPLAY_PUBLISH="${DISPLAY_PUBLISH:-1}"
cd "$SEC_SYS_ROOT_DIR/services/tracking_service" && exec python3 main.py "$@"
