"""Consumer-side frame transport: given a detection message, get me the pixels.

Python mirror of the C++ FrameReader strategy. GrpcFrameReader decodes the
inline gInline payload; ShmFrameReader fetches from the shm ring by
(gStreamId, gFrameId). The choice is made once in make_frame_reader() from
FRAME_TRANSPORT, so nothing downstream branches on it.
"""
import logging
import os
from abc import ABC, abstractmethod

import numpy as np

log = logging.getLogger("tracking.frame_reader")


class FrameReader(ABC):
    @abstractmethod
    def get(self, resp):
        """grpcDetectionResponse -> BGR ndarray (h, w, 3) uint8, or None."""
        raise NotImplementedError


class GrpcFrameReader(FrameReader):
    """Pixels arrived inline on the detection message (gInline present)."""

    def get(self, resp):
        if not resp.HasField("gInline"):
            return None
        f = resp.gInline
        n = f.gWidth * f.gHeight * f.gChannels
        if f.gChannels != 3 or n == 0 or len(f.gData) != n:
            log.warning("malformed gInline: %dx%dx%d with %dB payload",
                        f.gWidth, f.gHeight, f.gChannels, len(f.gData))
            return None
        # copy(): writable, and independent of the protobuf message's lifetime
        return np.frombuffer(f.gData, np.uint8).reshape(f.gHeight, f.gWidth, 3).copy()


class ShmFrameReader(FrameReader):
    """The message is a reference; fetch pixels from the shm ring.

    Opened lazily so tracking can start before the segment exists. See README.md.
    """

    def __init__(self):
        self._reader = None
        self._warned = False

    def get(self, resp):
        if self._reader is None:
            try:
                from shm_reader import ShmReader
                self._reader = ShmReader()
                self._warned = False
            except Exception as e:
                if not self._warned:
                    log.warning("shm unavailable (%s); is the pipeline running "
                                "with FRAME_TRANSPORT=shm?", e)
                    self._warned = True
                return None
        return self._reader.get_frame(resp.gStreamId, resp.gFrameId)


def make_frame_reader() -> FrameReader:
    """Selected once at startup from FRAME_TRANSPORT (default grpc) -- must be
    set identically across ingest, inference, and tracking."""
    if os.getenv("FRAME_TRANSPORT") == "shm":
        return ShmFrameReader()
    return GrpcFrameReader()
