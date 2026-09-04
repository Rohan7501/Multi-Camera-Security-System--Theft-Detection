#pragma once
#include <atomic>
#include <cstdint>
#include <string>
#include <opencv2/opencv.hpp>

// Shared-memory frame ring: per-stream rings addressed by (stream_id, frame_id).
// Producer writes with writeFrame(); any process reads with getFrame(). Slots use
// a seqlock so a reader never sees a half-written frame.

constexpr int    MAX_STREAMS = 16;
constexpr int    RING_DEPTH  = 240;
constexpr size_t SLOT_BYTES  = 640 * 640 * 3;   // max bytes of a stored frame

// Segment header identity. `magic` marks this as an edge-ai shm segment; `version`
// is the layout epoch. BUMP SHM_VERSION on ANY change to FrameStore/StreamRing/Slot
// (field order, sizes, MAX_STREAMS, RING_DEPTH, SLOT_BYTES) so a peer built against a
// different layout fails fast instead of reading garbage. Mirrored in shm_reader.py.
constexpr uint32_t SHM_MAGIC   = 0x53484D31;    // "SHM1"
constexpr uint32_t SHM_VERSION = 1;

// Which data path carries frames. Selected once at startup from FRAME_TRANSPORT
// and shared through the segment so producer and consumer must agree (see
// checkTransport). Values are the on-wire encoding of FrameStore::transport_mode
// (0 is reserved for "unset" / fresh segment).
enum class Transport : uint32_t { Grpc = 1, Shm = 2 };

struct Slot {
    std::atomic<uint32_t> seq;      // seqlock: even = stable, odd = being written
    uint64_t frame_id;              // which frame this slot currently holds
    int32_t  width, height, channels;
    int64_t  timestamp_ns;
    uint8_t  data[SLOT_BYTES];
};

struct StreamRing {
    std::atomic<uint32_t> state;    // 0 = free, 1 = reserved (registering), 2 = ready
    char     stream_id[32];
    std::atomic<uint64_t> latest;   // newest frame_id written (for "give me the latest")
    Slot     slots[RING_DEPTH];
};

struct FrameStore {
    std::atomic<uint32_t> magic;            // SHM_MAGIC once initialized (0 = fresh)
    std::atomic<uint32_t> version;          // SHM_VERSION; layout epoch
    std::atomic<uint32_t> transport_mode;   // 0 = unset, else a Transport value (handshake)
    StreamRing streams[MAX_STREAMS];
};   // the whole /dev/shm segment

// Producer: create (or re-open) the segment read-write. Returns mapped pointer or nullptr.
FrameStore* createShm(const char* name = "/sec-sys-shm");
// Consumer: open an existing segment read-only. Returns mapped pointer or nullptr.
FrameStore* openShm(const char* name = "/sec-sys-shm");
// Unmap the mapping (both roles). Does not remove the segment.
void closeShm(FrameStore* fs);
// Remove the segment from /dev/shm (producer only, on shutdown).
void unlinkShm(const char* name = "/sec-sys-shm");

// Producer: store `frame` (CV_8UC3, continuous, <= 640x640) for (sid, fid).
bool writeFrame(FrameStore* fs, const std::string& sid, uint64_t fid,
                int64_t ts_ns, const cv::Mat& frame);
// Consumer: fetch the frame for (sid, fid); false if unknown stream or lapped.
bool getFrame(FrameStore* fs, const std::string& sid, uint64_t fid, cv::Mat& out);

// Read FRAME_TRANSPORT from the environment (default Grpc). Single source of
// truth for the spelling so producer, consumer, and both factories agree.
Transport frameTransport();

// Fail-fast handshake: publish `want` into the shared segment; if a peer already
// published a DIFFERENT transport, print a diagnostic and abort. Guarantees ingest
// and inference agree on gRPC vs shm before any frame flows. No-op if fs is null.
// NOTE: transport_mode persists in the segment, so switching FRAME_TRANSPORT
// requires unlinkShm() (or `rm /dev/shm/sec-sys-shm`) first — restart both peers.
void checkTransport(FrameStore* fs, Transport want);
