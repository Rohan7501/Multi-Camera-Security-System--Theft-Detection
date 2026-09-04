"""Draw detections on a frame and show it (debug view).

Purely presentational: the caller fetches the frame -- via the FrameReader
strategy (frame_reader.py), which resolves pixels from the shm ring or the
inline gInline payload depending on FRAME_TRANSPORT -- and passes it in here
with the detections to draw. This module never touches a transport.

Standalone live view straight from the shm ring (no detections):
    python3 display.py --stream cam1
"""
import logging

import cv2 as cv

log = logging.getLogger("tracking.display")

# Retail class IDs (see proto/services.proto): 0=Product .. 3=Shoplifting.
CLASS_NAMES = {0: "Product", 1: "Product-Picked", 2: "Regular", 3: "Shoplifting"}
CLASS_COLORS = {                 # BGR
    0: (0, 200, 0),              # green
    1: (0, 180, 220),            # amber
    2: (220, 160, 0),            # blue
    3: (0, 0, 255),              # red  -- Shoplifting
}
DEFAULT_COLOR = (200, 200, 200)


def _unpack(item):
    """Accept either a proto grpcDetection or an (detection, track_id) tuple."""
    if isinstance(item, tuple):
        det, track_id = item
        return det, track_id
    return item, (getattr(item, "gTrackId", 0) or None)


def draw_detections(image, detections):
    """Draw each detection (box + class/conf/track label) onto image in place."""
    for item in detections:
        det, track_id = _unpack(item)
        x1, y1, x2, y2 = int(det.gX1), int(det.gY1), int(det.gX2), int(det.gY2)
        cls = int(det.gClassId)
        color = CLASS_COLORS.get(cls, DEFAULT_COLOR)
        cv.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = CLASS_NAMES.get(cls, str(cls))
        conf = getattr(det, "gConfidence", None)
        if conf is not None:
            label += f" {conf:.2f}"
        if track_id and int(track_id) > 0:      # -1 == untracked this frame
            label += f" #{int(track_id)}"

        (tw, th), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ytop = max(0, y1 - th - 4)
        cv.rectangle(image, (x1, ytop), (x1 + tw + 2, ytop + th + 4), color, -1)
        cv.putText(image, label, (x1 + 1, ytop + th),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
    return image


def displayFrameWithDetections(frame, detections, window="tracking"):
    """Draw detections on `frame` (BGR, in place) and show it.

    False means there was no frame to draw on -- see README.md.
    """
    if frame is None:
        log.debug("no frame to display for window '%s'", window)
        return False

    draw_detections(frame, detections)
    cv.imshow(window, frame)
    cv.waitKey(1)
    return True


if __name__ == "__main__":
    import argparse
    import time

    from shm_reader import ShmReader

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Live-view a stream's frames from shm (no detections).")
    ap.add_argument("--stream", required=True, help="stream id to view (e.g. cam1)")
    args = ap.parse_args()

    reader = ShmReader()
    log.info("viewing stream '%s' from shm (press q to quit)", args.stream)
    try:
        while True:
            fid = reader.latest_frame_id(args.stream)
            if fid is None:
                log.info("stream '%s' not registered yet...", args.stream)
                time.sleep(0.2)
                continue
            frame = reader.get_frame(args.stream, fid)
            if frame is not None:
                cv.imshow(f"shm: {args.stream}", frame)
            if (cv.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cv.destroyAllWindows()
        reader.close()
