#pragma once
// Prometheus metrics for the C++ services.
//
// This owns an HTTP server that exists ONLY to answer `GET /metrics` for a
// Prometheus scrape. It is deliberately SEPARATE from each service's gRPC
// server: a grpc::Server dispatches HTTP/2 + protobuf on a registered service
// contract and has no hook for plain HTTP/1.1, so the two cannot share a port.
// prometheus::Exposer embeds its own (civetweb) HTTP server on its own socket
// and threads; both live in the same process and share the in-memory registry.
//
//   gRPC   :50051  HTTP/2 + protobuf  <- ingest / tracking peers
//   metrics:9102   HTTP/1.1 + text    <- Prometheus
//
// Address comes from METRICS_ADDR (per-service default at the call site); set it
// to "off" or "" to run without an exposer. When disabled the registry and every
// metric object still exist and increments are simply never scraped -- so the hot
// path never needs a null check or a feature flag.
//
// Cardinality rule: label by `stream` (a small, bounded set of cameras) and by
// class_id (4 retail classes). NEVER label by track_id or frame_id -- those are
// unbounded and would blow up Prometheus' index.

#include <prometheus/counter.h>
#include <prometheus/exposer.h>
#include <prometheus/family.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>
#include <prometheus/registry.h>

#include <memory>
#include <string>

namespace metrics {

// METRICS_ADDR if set, else `fallback`. Returns "" when explicitly disabled
// (METRICS_ADDR unset is NOT disabled -- it takes the fallback).
std::string addrFromEnv(const std::string& fallback);

// Latency buckets (seconds) tuned for per-frame edge work: sub-ms to ~1s.
prometheus::Histogram::BucketBoundaries latencyBuckets();

// Buckets for a TRANSPORT hop (ingest -> inference). Starts two decades lower
// than latencyBuckets(): a same-box gRPC handoff is tens of microseconds, so a
// 1ms floor would pile every observation into the first bucket and make any
// quantile meaningless. Extends to 2s to keep a backlog visible instead of
// dumping it in +Inf (where histogram_quantile can no longer estimate).
prometheus::Histogram::BucketBoundaries transportBuckets();

class MetricsServer
{
public:
    // addr like "0.0.0.0:9102"; "" -> registry only, no HTTP listener.
    explicit MetricsServer(const std::string& addr);
    ~MetricsServer();

    prometheus::Registry& registry() { return *registry_; }
    std::shared_ptr<prometheus::Registry> registryPtr() const { return registry_; }
    bool enabled() const { return exposer_ != nullptr; }

private:
    std::shared_ptr<prometheus::Registry> registry_;
    std::unique_ptr<prometheus::Exposer> exposer_;   // null when disabled
};

}  // namespace metrics