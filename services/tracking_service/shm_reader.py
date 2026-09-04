"""Read frames out of the C++ shared-memory ring from Python.

Mirrors the FrameStore / StreamRing / Slot layout byte-for-byte, x86-64 padding
included. If you change common/shm.hpp you MUST update the offsets here and bump
SHM_VERSION on both sides, or reads come back as garbage -- __init__ validates
the magic, version and segment size to make that failure loud instead of silent.

Layout constants and the seqlock read protocol are in README.md.
"""
import mmap
import os
import struct

import numpy as np

SHM_PATH = "/dev/shm/sec-sys-shm"

# ---- constants (must match common/shm.hpp) ---------------------------------
MAX_STREAMS = 16
RING_DEPTH = 240
SLOT_BYTES = 640 * 640 * 3            # 1_228_800

SHM_MAGIC = 0x53484D31               # "SHM1"; identifies an edge-ai segment
SHM_VERSION = 1                      # layout epoch; bump with common/shm.hpp

# ---- Slot { atomic<u32> seq; u64 frame_id; i32 w,h,c; i64 ts_ns; u8 data[]; }
SLOT_SEQ_OFF = 0
SLOT_FRAMEID_OFF = 8                  # u64 -> 8-aligned (4B pad after seq)
SLOT_W_OFF = 16
SLOT_H_OFF = 20
SLOT_C_OFF = 24
SLOT_TS_OFF = 32                      # i64 -> 8-aligned (4B pad after channels)
SLOT_DATA_OFF = 40
SLOT_SIZE = SLOT_DATA_OFF + SLOT_BYTES              # 1_228_840

# ---- StreamRing { atomic<u32> state; char id[32]; atomic<u64> latest; Slot[]; }
RING_STATE_OFF = 0
RING_ID_OFF = 4
RING_ID_LEN = 32
RING_LATEST_OFF = 40                 # u64 -> 8-aligned (4B pad after id ends at 36)
RING_SLOTS_OFF = 48
RING_SIZE = RING_SLOTS_OFF + RING_DEPTH * SLOT_SIZE

# ---- FrameStore { atomic<u32> magic; atomic<u32> version; atomic<u32> transport_mode;
#                   StreamRing streams[MAX_STREAMS]; }
FS_MAGIC_OFF = 0
FS_VERSION_OFF = 4
FS_TRANSPORT_OFF = 8
FS_STREAMS_OFF = 16                   # StreamRing -> 8-aligned (4B pad after transport_mode)
FS_SIZE = FS_STREAMS_OFF + MAX_STREAMS * RING_SIZE

STATE_READY = 2                      # StreamRing.state: registered + published
SEQLOCK_RETRIES = 6                  # bounded retries when racing a writer


class ShmReader:
    """Opens the shm segment read-only and fetches frames by (stream_id, frame_id)."""

    def __init__(self, path: str = SHM_PATH):
        self._fd = os.open(path, os.O_RDONLY)          # raises FileNotFoundError if absent
        size = os.fstat(self._fd).st_size
        if size < FS_SIZE:
            os.close(self._fd)
            raise RuntimeError(
                f"shm segment {path} is {size}B but the FrameStore layout expects "
                f"{FS_SIZE}B -- shm_reader.py is out of sync with common/shm.hpp")
        self._mm = mmap.mmap(self._fd, FS_SIZE, mmap.MAP_SHARED, mmap.PROT_READ)

        # Header check: refuse a not-yet-initialized (magic 0 / mid-init) or
        # incompatible (wrong magic/version) segment instead of reading garbage.
        magic = struct.unpack_from("<I", self._mm, FS_MAGIC_OFF)[0]
        version = struct.unpack_from("<I", self._mm, FS_VERSION_OFF)[0]
        if magic != SHM_MAGIC:
            self._mm.close(); os.close(self._fd)
            raise RuntimeError(
                f"shm {path}: bad/uninitialized magic 0x{magic:08x} "
                f"(want 0x{SHM_MAGIC:08x}); producer not up yet or incompatible build")
        if version != SHM_VERSION:
            self._mm.close(); os.close(self._fd)
            raise RuntimeError(
                f"shm {path}: version {version} != expected {SHM_VERSION}; "
                f"rebuild/clear the segment (rm {path})")

    def close(self) -> None:
        self._mm.close()
        os.close(self._fd)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- internals -----------------------------------------------------------

    def _find_ring(self, stream_id):
        """Return the byte offset of the StreamRing for stream_id, or None."""
        want = stream_id.encode() if isinstance(stream_id, str) else stream_id
        for i in range(MAX_STREAMS):
            base = FS_STREAMS_OFF + i * RING_SIZE
            state = struct.unpack_from("<I", self._mm, base + RING_STATE_OFF)[0]
            if state != STATE_READY:
                continue
            raw = self._mm[base + RING_ID_OFF: base + RING_ID_OFF + RING_ID_LEN]
            if raw.split(b"\x00", 1)[0] == want:
                return base
        return None

    # -- public --------------------------------------------------------------

    def latest_frame_id(self, stream_id):
        """Newest frame_id written for stream_id, or None if the stream is unknown."""
        base = self._find_ring(stream_id)
        if base is None:
            return None
        return struct.unpack_from("<Q", self._mm, base + RING_LATEST_OFF)[0]

    def get_frame(self, stream_id, frame_id):
        """Return the BGR frame for (stream_id, frame_id), or None.

        None = unknown stream, lapped out of the ring, or lost the seqlock race.
        """
        base = self._find_ring(stream_id)
        if base is None:
            return None

        slot = base + RING_SLOTS_OFF + (frame_id % RING_DEPTH) * SLOT_SIZE
        for _ in range(SEQLOCK_RETRIES):
            # seqlock: even = stable, odd = writer mid-write. Snapshot, then re-check.
            s1 = struct.unpack_from("<I", self._mm, slot + SLOT_SEQ_OFF)[0]
            if s1 & 1:
                continue

            fid = struct.unpack_from("<Q", self._mm, slot + SLOT_FRAMEID_OFF)[0]
            w = struct.unpack_from("<i", self._mm, slot + SLOT_W_OFF)[0]
            h = struct.unpack_from("<i", self._mm, slot + SLOT_H_OFF)[0]
            c = struct.unpack_from("<i", self._mm, slot + SLOT_C_OFF)[0]
            # Torn reads produce nonsense here; retry rather than trust it.
            if not (0 < w and 0 < h and c == 3):
                continue
            nbytes = w * h * c
            if nbytes > SLOT_BYTES:
                continue

            raw = self._mm[slot + SLOT_DATA_OFF: slot + SLOT_DATA_OFF + nbytes]  # copy

            if struct.unpack_from("<I", self._mm, slot + SLOT_SEQ_OFF)[0] != s1:
                continue                         # writer touched the slot -> torn, retry
            if fid != frame_id:
                return None                      # lapped: too old for the ring

            return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, c).copy()

        return None
