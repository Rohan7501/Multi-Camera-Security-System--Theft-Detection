#include <csignal>
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "rtsp_reader.hpp"
#include "ingest_server.hpp"
#include "ingest_metrics.hpp"
// #include "inference_client.hpp"

// #include "../../common/logging/logger.hpp"
// #include "../../common/config/config_loader.hpp"

std::atomic<bool> g_running{true};

void signal_handler(int signal)
{
    // std::cout << "Received shutdown signal: " << signal << std::endl;
    g_running = false;
}

int main()
{
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    IngestServer server;

    try
    {
        server.start();

        // Prometheus scrape endpoint: its OWN HTTP server on its OWN port. The
        // IngestAdmin gRPC server (:50053) speaks HTTP/2+protobuf and cannot
        // serve /metrics.
        metrics::MetricsServer metrics_server(metrics::addrFromEnv("0.0.0.0:9101"));
        ingest_metrics::Metrics metrics(metrics_server.registry());

        // Mirror per-stream Stats into Prometheus once a second. Sampling here
        // (rather than in run_loop) keeps the capture path untouched.
        auto next_sample = std::chrono::steady_clock::now();
        while (g_running)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));

            const auto now = std::chrono::steady_clock::now();
            if (now >= next_sample) {
                next_sample = now + std::chrono::seconds(1);
                // Same clock RtspReader stamps frames with (steady, not wall).
                const int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now.time_since_epoch()).count();
                metrics.sample(server.listCameras(), now_ms);
            }
        }

        std::cout << "Shutdown signal received.\n";

        server.stop();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Server shut down cleanly.\n";
    return EXIT_SUCCESS;
}