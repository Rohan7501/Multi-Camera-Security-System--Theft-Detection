// Unit tests for InferenceWorker's age-based drop -- the safety net that skips a
// frame already older than INFERENCE_MAX_AGE_MS when a worker takes it, before
// the shm fetch and infer() spend anything on it.
//
// A fake detector, frame reader and writer stand in for ONNX Runtime, the frame
// transport and the TrackingService, so this runs anywhere: no model, no GPU, no
// gRPC peer, no shm segment.

#include "inference_worker.hpp"

#include <prometheus/registry.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

static int failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

namespace {

constexpr int64_t kMs = 1'000'000;            // ns
constexpr int64_t kLimit = 1000 * kMs;        // generous, so "fresh" can't flake
constexpr int64_t kOld = 5000 * kMs;          // well past kLimit

int64_t nowNs()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct FakeDetector : Detector {
    std::atomic<int> calls{0};
    bool load_model(const std::string&) override { return true; }
    std::vector<Detection> infer(const cv::Mat&) override { ++calls; return {}; }
};

struct FakeReader : FrameReader {
    std::atomic<int> calls{0};
    bool get(const FramePacket&, cv::Mat& out) override
    {
        ++calls;
        out = cv::Mat(2, 2, CV_8UC3, cv::Scalar(0, 0, 0));
        return true;
    }
};

struct FakeWriter : DetectionWriter {
    std::mutex mu;
    std::vector<uint64_t> frame_ids;          // published, in order
    void publish(const DetectionPacket& pkt) override
    {
        std::lock_guard<std::mutex> lock(mu);
        frame_ids.push_back(pkt.frame_id);
    }
    size_t count()
    {
        std::lock_guard<std::mutex> lock(mu);
        return frame_ids.size();
    }
};

FramePacket packet(const std::string& stream, uint64_t frame_id, int64_t capture_ns)
{
    FramePacket p;
    p.stream_id = stream;
    p.frame_id = frame_id;
    p.timestamp_ns = capture_ns;
    p.capture_timestamp_ns = capture_ns;
    return p;
}

// One worker with its own metrics registry and fakes.
struct Rig {
    prometheus::Registry registry;
    inference_metrics::Metrics metrics{registry};
    FakeDetector detector;
    FakeReader reader;
    FakeWriter writer;

    double stale(const std::string& s) { return metrics.forStream(s).stale->Value(); }
    double frames(const std::string& s) { return metrics.forStream(s).frames->Value(); }

    // Queues every packet first, so the worker takes them in order, then runs it
    // until each one has been published or counted stale.
    void run(int64_t max_age_ns, std::vector<FramePacket> packets)
    {
        FrameQueue<FramePacket> queue(64);
        const size_t n = packets.size();
        for (auto& p : packets) queue.push(std::move(p));

        InferenceWorker worker(queue, detector, writer, reader, metrics, max_age_ns);
        std::thread t(&InferenceWorker::run, &worker);

        // pop() returns false on shutdown even with items still queued, so wait
        // for the worker to account for all of them before releasing it.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (writer.count() + stale("cam1") + stale("cam2") < n &&
               std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));

        queue.shutdown();
        t.join();
    }
};

}  // namespace

static void test_fresh_frames_run_normally()
{
    // The steady state: inside the limit, every frame is fetched, inferred and
    // published, and the stale counter stays flat.
    Rig r;
    const int64_t now = nowNs();
    r.run(kLimit, {packet("cam1", 1, now), packet("cam1", 2, now)});

    CHECK(r.writer.count() == 2);
    CHECK(r.detector.calls == 2);
    CHECK(r.frames("cam1") == 2);
    CHECK(r.stale("cam1") == 0);
}

static void test_stale_frame_skips_fetch_and_infer()
{
    // The point of checking at pickup: a stale frame costs neither the shm fetch
    // nor infer(), and isn't counted as a frame run through the detector.
    Rig r;
    r.run(kLimit, {packet("cam1", 1, nowNs() - kOld)});

    CHECK(r.stale("cam1") == 1);
    CHECK(r.reader.calls == 0);
    CHECK(r.detector.calls == 0);
    CHECK(r.writer.count() == 0);
    CHECK(r.frames("cam1") == 0);
}

static void test_only_stale_frames_are_dropped()
{
    // A backlog mixes old and new frames. The fresh ones still run, in order.
    Rig r;
    const int64_t now = nowNs(), old = now - kOld;
    r.run(kLimit, {packet("cam1", 1, old), packet("cam1", 2, now),
                   packet("cam1", 3, old), packet("cam1", 4, now)});

    CHECK(r.stale("cam1") == 2);
    CHECK((r.writer.frame_ids == std::vector<uint64_t>{2, 4}));
}

static void test_stale_counter_is_per_stream()
{
    Rig r;
    const int64_t now = nowNs();
    r.run(kLimit, {packet("cam1", 1, now), packet("cam2", 1, now - kOld)});

    CHECK(r.stale("cam1") == 0);
    CHECK(r.stale("cam2") == 1);
}

static void test_unset_capture_stamp_is_never_dropped()
{
    // proto3 omits a 0 default. Treating 0 as a stamp would make every frame look
    // as old as the machine's uptime and drop the whole stream.
    Rig r;
    r.run(kLimit, {packet("cam1", 1, 0)});

    CHECK(r.stale("cam1") == 0);
    CHECK(r.writer.count() == 1);
}

static void test_zero_limit_disables_the_drop()
{
    // INFERENCE_MAX_AGE_MS=0: every frame runs, however old.
    Rig r;
    r.run(0, {packet("cam1", 1, nowNs() - kOld)});

    CHECK(r.stale("cam1") == 0);
    CHECK(r.writer.count() == 1);
}

int main()
{
    test_fresh_frames_run_normally();
    test_stale_frame_skips_fetch_and_infer();
    test_only_stale_frames_are_dropped();
    test_stale_counter_is_per_stream();
    test_unset_capture_stamp_is_never_dropped();
    test_zero_limit_disables_the_drop();

    if (failures == 0) {
        std::printf("test_inference_worker: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_inference_worker: %d check(s) failed\n", failures);
    return 1;
}
