#include "detector_onnx.hpp"
#include "frame_queue.hpp"
#include "inference_worker.hpp"
#include "inference_server.hpp"
#include "inference_metrics.hpp"
#include "tracking_client.hpp"
#include "shm.hpp"
#include "frame_reader.hpp"

#include <grpcpp/grpcpp.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <signal.h>
#include <execinfo.h>
#include <unistd.h>
#include <iostream>

#define LOG_INFO(msg)  std::cout << "[INFO] "  << __FILE__ << ":" << __LINE__ << " " << msg << std::endl;
#define LOG_ERROR(msg) std::cerr << "[ERROR] " << __FILE__ << ":" << __LINE__ << " " << msg << std::endl;

std::atomic<bool> running(true);

// Size of the worker pool, from NUM_THREADS. Anything unset, unparseable or
// < 1 falls back, so a bad value degrades to the default instead of no workers.
static int workerCount(int fallback)
{
    const char* v = std::getenv("NUM_THREADS");
    if (!v || !*v) return fallback;
    const int n = std::atoi(v);
    return n > 0 ? n : fallback;
}

void signal_handler(int) { running = false; }

void segfault_handler(int)
{
    void* array[20];
    size_t size = backtrace(array, 20);
    std::cerr << "FATAL: Segmentation fault\n";
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    _exit(1);
}

int main()
{
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGSEGV, segfault_handler);

    FrameQueue<FramePacket> frame_queue(400);

    OnnxDetector detector;
    detector.load_model("models/best.onnx");

    FrameStore* fs = createShm();
    checkTransport(fs, frameTransport());   // abort now if ingest picked a different transport
    auto reader = makeFrameReader(fs);      // gRPC or shm, chosen once from FRAME_TRANSPORT

    // Forward flow: detections are pushed to the TrackingService. Override the
    // endpoint with TRACKING_ADDR (default 127.0.0.1:50052).
    const char* env = std::getenv("TRACKING_ADDR");
    const std::string tracking_addr = env ? env : "127.0.0.1:50052";
    auto tracking_channel = grpc::CreateChannel(tracking_addr, grpc::InsecureChannelCredentials());
    TrackingClient tracking(tracking_channel);
    tracking.start();
    LOG_INFO("TrackingClient -> " + tracking_addr);

    // Reference-only, or inline pixels when TRACKING_PIXELS=1 in grpc mode.
    auto det_writer = makeDetectionWriter(tracking);

    // Prometheus scrape endpoint. This is its OWN HTTP server on its OWN port --
    // the gRPC server above speaks HTTP/2+protobuf and cannot serve /metrics.
    metrics::MetricsServer metrics_server(metrics::addrFromEnv("0.0.0.0:9102"));
    inference_metrics::Metrics metrics(metrics_server.registry());

    // Worker pool: NUM_THREADS threads all popping the one shared queue. Workers
    // hold only references, so one per thread costs nothing and keeps ownership
    // obvious. They share a single detector, so infer() itself stays serialized.
    const int num_threads = 4;
    std::vector<std::unique_ptr<InferenceWorker>> workers;
    std::vector<std::thread> worker_threads;
    workers.reserve(num_threads);
    worker_threads.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        workers.push_back(std::make_unique<InferenceWorker>(
            frame_queue, detector, *det_writer, *reader, metrics));
        worker_threads.emplace_back(&InferenceWorker::run, workers.back().get());
    }
    LOG_INFO("InferenceWorker pool started (" + std::to_string(num_threads) + " threads)");

    InferenceServer server(frame_queue, metrics);
    std::thread server_thread(&InferenceServer::start, &server);
    LOG_INFO("InferenceServer started");

    // Single-owner sampler for process-wide gauges. Doing this from the workers
    // would double-count (N threads mirroring one queue), so it lives here.
    uint64_t last_dropped = 0, last_reconnects = 0;
    while (running) {
        metrics.depth.Set(static_cast<double>(frame_queue.size()));
        const uint64_t d = frame_queue.dropped();
        metrics.dropped.Increment(static_cast<double>(d - last_dropped));
        last_dropped = d;

        const uint64_t r = tracking.reconnects();
        metrics.reconnects.Increment(static_cast<double>(r - last_reconnects));
        last_reconnects = r;
        metrics.up.Set(tracking.connected() ? 1.0 : 0.0);

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    server.shutdown();
    frame_queue.shutdown();
    for (auto& t : worker_threads) t.join();
    tracking.stop();
    server_thread.join();

    return 0;
}
