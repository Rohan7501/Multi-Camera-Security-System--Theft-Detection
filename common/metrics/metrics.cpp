#include "metrics.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>

namespace metrics {

std::string addrFromEnv(const std::string& fallback)
{
    const char* e = std::getenv("METRICS_ADDR");
    if (!e)              return fallback;      // unset -> service default
    std::string v(e);
    if (v.empty() || v == "off" || v == "0")
        return "";                             // explicitly disabled
    return v;
}

prometheus::Histogram::BucketBoundaries latencyBuckets()
{
    return {0.001, 0.002, 0.005, 0.01, 0.02, 0.03,
            0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0};
}

prometheus::Histogram::BucketBoundaries transportBuckets()
{
    return {0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005,
            0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0};
}

MetricsServer::MetricsServer(const std::string& addr)
    : registry_(std::make_shared<prometheus::Registry>())
{
    if (addr.empty()) {
        std::cout << "[metrics] disabled (METRICS_ADDR=off)" << std::endl;
        return;
    }
    try {
        // Own HTTP server + own threads; unrelated to the gRPC server.
        exposer_ = std::make_unique<prometheus::Exposer>(addr);
        exposer_->RegisterCollectable(registry_);
        std::cout << "[metrics] scrape endpoint on http://" << addr << "/metrics" << std::endl;
    } catch (const std::exception& e) {
        // A busy port must not take the service down -- run without metrics.
        exposer_.reset();
        std::cerr << "[metrics] failed to bind " << addr << ": " << e.what()
                  << " (continuing without metrics)" << std::endl;
    }
}

MetricsServer::~MetricsServer() = default;

}  // namespace metrics