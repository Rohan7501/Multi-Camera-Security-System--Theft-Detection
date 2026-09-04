#include "shm.hpp"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

// Transient value in `magic` while the creator initializes the header (see
// initHeader). Never a valid resting state; a fresh segment is 0, a ready one
// is SHM_MAGIC.
static constexpr uint32_t SHM_MAGIC_CLAIMING = 0xFFFFFFFF;

// ---- mapping helpers --------------------------------------------------------

static FrameStore* mapShm(const char* name, int oflag, int prot)
{
    // 0600: owner-only. Everything here runs as one user; a shared box must not
    // let other users read the video frames. Use a shared group (0640) instead
    // if producer and consumer run as different service accounts.
    int fd = shm_open(name, oflag, 0600);
    if (fd < 0) { perror("shm_open"); return nullptr; }

    if (oflag & O_CREAT) {
        // Guard against attaching to a pre-existing segment from a different
        // build whose layout changed size (belt-and-suspenders with SHM_VERSION).
        struct stat st;
        if (fstat(fd, &st) == 0 && st.st_size != 0 &&
            static_cast<size_t>(st.st_size) != sizeof(FrameStore)) {
            std::fprintf(stderr,
                "FATAL: existing shm '%s' is %lld bytes but this build expects %zu "
                "(incompatible layout). Remove it: rm /dev/shm%s\n",
                name, static_cast<long long>(st.st_size), sizeof(FrameStore), name);
            close(fd);
            std::abort();
        }
        if (ftruncate(fd, sizeof(FrameStore)) != 0) {
            perror("ftruncate");
            close(fd);
            return nullptr;
        }
    }

    void* p = mmap(nullptr, sizeof(FrameStore), prot, MAP_SHARED, fd, 0);
    close(fd);                                   // mapping stays valid after close
    if (p == MAP_FAILED) { perror("mmap"); return nullptr; }
    return reinterpret_cast<FrameStore*>(p);
}

// Stamp magic+version exactly once across racing creators. The winner of the CAS
// (fresh 0 -> CLAIMING) writes version then publishes magic with a release store;
// losers spin past CLAIMING so they observe a fully-initialized header. Validation
// happens in checkTransport / ShmReader.
static void initHeader(FrameStore* fs)
{
    uint32_t expected = 0;
    if (fs->magic.compare_exchange_strong(expected, SHM_MAGIC_CLAIMING,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        fs->version.store(SHM_VERSION, std::memory_order_relaxed);
        fs->magic.store(SHM_MAGIC, std::memory_order_release);   // publish
    } else {
        while (fs->magic.load(std::memory_order_acquire) == SHM_MAGIC_CLAIMING)
            std::this_thread::yield();
    }
}

FrameStore* createShm(const char* name)
{
    // A freshly ftruncate'd tmpfs segment is zero-filled, and a zeroed std::atomic
    // is a valid initialized value here (seq/state/latest = 0 => stable / free /
    // empty). So we deliberately do NOT value-init -- that would memset the whole
    // (~1.3 GB) struct and commit every page instead of paging in lazily.
    FrameStore* fs = mapShm(name, O_CREAT | O_RDWR, PROT_READ | PROT_WRITE);
    if (fs) initHeader(fs);
    return fs;
}

FrameStore* openShm(const char* name)
{
    return mapShm(name, O_RDONLY, PROT_READ);
}

void closeShm(FrameStore* fs)
{
    if (fs) munmap(fs, sizeof(FrameStore));
}

void unlinkShm(const char* name)
{
    shm_unlink(name);
}

// ---- transport selection / handshake ---------------------------------------

static const char* transportName(Transport t)
{
    switch (t) {
        case Transport::Grpc: return "grpc";
        case Transport::Shm:  return "shm";
        default:              return "unset";
    }
}

Transport frameTransport()
{
    const char* t = std::getenv("FRAME_TRANSPORT");
    return (t && std::strcmp(t, "shm") == 0) ? Transport::Shm : Transport::Grpc;
}

void checkTransport(FrameStore* fs, Transport want)
{
    if (!fs) return;                                   // no segment => nothing to reconcile

    // Header compatibility: a stale/incompatible segment (old layout left in
    // /dev/shm, or a peer built at a different SHM_VERSION) must fail loudly, not
    // be read as garbage. createShm has already stamped the header on our side.
    const uint32_t m = fs->magic.load(std::memory_order_acquire);
    const uint32_t v = fs->version.load(std::memory_order_acquire);
    if (m != SHM_MAGIC || v != SHM_VERSION) {
        std::fprintf(stderr,
            "FATAL: shm header mismatch (magic=0x%08x want 0x%08x, version=%u want %u). "
            "Stale/incompatible segment -- rm /dev/shm/sec-sys-shm and restart all peers.\n",
            m, SHM_MAGIC, v, SHM_VERSION);
        std::abort();
    }

    const uint32_t desired = static_cast<uint32_t>(want);
    uint32_t expected = 0;                             // first writer wins from the fresh (zeroed) state
    if (!fs->transport_mode.compare_exchange_strong(
            expected, desired,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        // Someone published first; `expected` now holds their value.
        if (expected != desired) {
            std::fprintf(stderr,
                "FATAL: FRAME_TRANSPORT mismatch -- this process selected '%s' but the "
                "peer already selected '%s'. Set FRAME_TRANSPORT identically on ingest and "
                "inference (and `rm /dev/shm/sec-sys-shm` if you changed it).\n",
                transportName(want),
                transportName(static_cast<Transport>(expected)));
            std::abort();
        }
    }
}

// ---- stream registration (lock-free, thread-safe) --------------------------

// Returns the row for `sid`, claiming a free one if needed. Safe for concurrent
// writers: rows are reserved with a CAS on `state` (0 -> 1), so two threads can
// never grab the same free row; the id is published with a release store (state
// -> 2) that readers acquire before reading `stream_id`. NOTE: each stream should
// still have a single writer (per-stream slot writes are not mutually synced).
static StreamRing* find_or_claim(FrameStore* fs, const std::string& sid)
{
    // 1. already-registered row?
    for (auto& s : fs->streams)
        if (s.state.load(std::memory_order_acquire) == 2 && sid == s.stream_id)
            return &s;

    // 2. atomically reserve a free row and register the id
    for (auto& s : fs->streams) {
        uint32_t expected = 0;
        if (s.state.compare_exchange_strong(expected, 1,
                std::memory_order_acq_rel, std::memory_order_relaxed)) {
            std::strncpy(s.stream_id, sid.c_str(), sizeof(s.stream_id) - 1);
            s.stream_id[sizeof(s.stream_id) - 1] = '\0';
            s.state.store(2, std::memory_order_release);   // publish stream_id
            return &s;
        }
    }
    return nullptr;   // table full
}

// ---- producer / consumer ----------------------------------------------------

bool writeFrame(FrameStore* fs, const std::string& sid, uint64_t fid,
                int64_t ts_ns, const cv::Mat& frame)
{
    if (frame.type() != CV_8UC3 || !frame.isContinuous())
        return false;
    const size_t bytes = frame.total() * frame.elemSize();
    if (bytes > SLOT_BYTES) return false;                  // too big for a 640x640 slot

    StreamRing* r = find_or_claim(fs, sid);
    if (!r) return false;                                  // table full

    // direct-index the slot; seqlock write
    Slot& s = r->slots[fid % RING_DEPTH];

    // Force the seqlock ODD at write-start via `| 1` (not `+1`): if a previous
    // writer crashed mid-write it left this slot odd, and `+1` would flip it even
    // during the write (torn reads) and leave it odd when done (stuck "writing"
    // forever). OR-ing to odd heals that slot instead.
    uint32_t writing = s.seq.load(std::memory_order_relaxed) | 1u;
    s.seq.store(writing, std::memory_order_release);       // -> odd (writing)

    s.frame_id     = fid;
    s.width        = frame.cols;
    s.height       = frame.rows;
    s.channels     = frame.channels();
    s.timestamp_ns = ts_ns;
    std::memcpy(s.data, frame.data, bytes);

    s.seq.store(writing + 1, std::memory_order_release);   // -> even (stable): publishes the writes
    r->latest.store(fid, std::memory_order_release);
    return true;
}

bool getFrame(FrameStore* fs, const std::string& sid, uint64_t fid, cv::Mat& out)
{
    // 1. find the registered stream row
    StreamRing* r = nullptr;
    for (auto& s : fs->streams)
        if (s.state.load(std::memory_order_acquire) == 2 && sid == s.stream_id) { r = &s; break; }
    if (!r) return false;                                  // unknown stream

    // 2. direct index
    Slot& s = r->slots[fid % RING_DEPTH];

    // 3. seqlock read
    for (int t = 0; t < 4; ++t) {
        uint32_t s1 = s.seq.load(std::memory_order_acquire);
        if (s1 & 1) continue;                              // writer mid-write
        uint64_t got = s.frame_id;
        int w = s.width, h = s.height;
        cv::Mat snap(h, w, CV_8UC3, s.data);
        cv::Mat copy = snap.clone();                       // snapshot before re-check
        if (s.seq.load(std::memory_order_acquire) != s1) continue;   // torn -> retry

        // 4. verify it's still YOUR frame
        if (got != fid) return false;                      // lapped: too old for the ring
        out = std::move(copy);
        return true;                                       // hit
    }
    return false;
}
