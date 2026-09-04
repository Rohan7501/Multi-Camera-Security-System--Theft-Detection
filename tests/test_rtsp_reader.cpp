// Unit tests for RtspReader -- ingest's per-camera capture thread.
//
// These deliberately test the FAILURE path, because it is the one that runs in
// production: cameras go offline, DNS fails, a URL is typo'd in config.yaml. An
// unreachable camera must never hang start(), crash ingest, or wedge stop() --
// the other cameras have to keep running. A happy-path test needs a live RTSP
// server and belongs in an integration suite, not here.
//
// Uses its own shm segment name so it can never disturb a running pipeline.

#include "rtsp_reader.hpp"
#include "shm.hpp"

#include <grpcpp/grpcpp.h>

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

static int failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

namespace {

constexpr const char* kTestShm = "/sec-sys-shm-test-rtsp";

// A channel to a port nothing listens on. RtspReader owns an InferenceClient,
// but a dead camera never produces a frame to send, so this is never dialed.
std::shared_ptr<grpc::Channel> deadChannel()
{
    return grpc::CreateChannel("127.0.0.1:1", grpc::InsecureChannelCredentials());
}

struct ShmFixture {
    FrameStore* fs = nullptr;
    ShmFixture()  { unlinkShm(kTestShm); fs = createShm(kTestShm); }
    ~ShmFixture() { if (fs) closeShm(fs); unlinkShm(kTestShm); }
};

}  // namespace

// An unreachable URL must not block start(); capture retries on its own thread.
static void test_start_does_not_block_on_a_dead_url(ShmFixture& shm)
{
    RtspReader reader("cam-dead", "rtsp://127.0.0.1:1/nope", deadChannel(), shm.fs);

    const auto t0 = std::chrono::steady_clock::now();
    reader.start();
    const auto elapsed = std::chrono::steady_clock::now() - t0;

    CHECK(elapsed < std::chrono::seconds(5));
    reader.stop();
}

// stop() must join the worker even while it is mid-reconnect.
static void test_stop_is_clean_while_reconnecting(ShmFixture& shm)
{
    RtspReader reader("cam-dead", "rtsp://127.0.0.1:1/nope", deadChannel(), shm.fs);
    reader.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));   // let it fail + retry

    const auto t0 = std::chrono::steady_clock::now();
    reader.stop();
    const auto elapsed = std::chrono::steady_clock::now() - t0;

    CHECK(elapsed < std::chrono::seconds(10));
}

// No frames from a dead camera, and no phantom frames either.
static void test_dead_camera_produces_no_frames(ShmFixture& shm)
{
    RtspReader reader("cam-dead", "rtsp://127.0.0.1:1/nope", deadChannel(), shm.fs);
    reader.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    CHECK(reader.stats().frames_total == 0);
    CHECK(!reader.latest_frame().has_value());

    reader.stop();
}

// A malformed URL is a config typo, not a crash.
static void test_garbage_url_is_survivable(ShmFixture& shm)
{
    RtspReader reader("cam-garbage", "not-a-url-at-all", deadChannel(), shm.fs);
    reader.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    CHECK(reader.stats().frames_total == 0);
    reader.stop();
}

// stop() without start(), and a double stop(), must both be no-ops. The admin
// RPC can remove a camera that was never started.
static void test_stop_without_start_is_a_noop(ShmFixture& shm)
{
    RtspReader reader("cam-idle", "rtsp://127.0.0.1:1/nope", deadChannel(), shm.fs);
    reader.stop();
    reader.stop();
    CHECK(reader.stats().frames_total == 0);
}

// Destruction while running must not leak the thread -- ~RtspReader joins it.
static void test_destructor_stops_a_running_reader(ShmFixture& shm)
{
    {
        RtspReader reader("cam-dtor", "rtsp://127.0.0.1:1/nope", deadChannel(), shm.fs);
        reader.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }   // no explicit stop(): the destructor owns it
    CHECK(true);   // reaching here without a hang or abort is the assertion
}

// Several dead cameras at once: one bad URL must not starve the others.
static void test_many_dead_readers_coexist(ShmFixture& shm)
{
    std::vector<std::unique_ptr<RtspReader>> readers;
    for (int i = 0; i < 4; ++i)
        readers.push_back(std::make_unique<RtspReader>(
            "cam" + std::to_string(i), "rtsp://127.0.0.1:1/nope",
            deadChannel(), shm.fs));

    for (auto& r : readers) r->start();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    for (auto& r : readers) {
        CHECK(r->stats().frames_total == 0);
        r->stop();
    }
}

int main()
{
    ShmFixture shm;
    if (shm.fs == nullptr) {
        std::fprintf(stderr, "test_rtsp_reader: could not create %s\n", kTestShm);
        return 1;
    }

    test_start_does_not_block_on_a_dead_url(shm);
    test_stop_is_clean_while_reconnecting(shm);
    test_dead_camera_produces_no_frames(shm);
    test_garbage_url_is_survivable(shm);
    test_stop_without_start_is_a_noop(shm);
    test_destructor_stops_a_running_reader(shm);
    test_many_dead_readers_coexist(shm);

    if (failures == 0) {
        std::printf("test_rtsp_reader: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_rtsp_reader: %d check(s) failed\n", failures);
    return 1;
}
