// gRPC client for TrackingService.grpcStreamDetections -- the forward flow out
// of inference. The worker calls send() per frame; tracking fills in gTrackId.
//
// Self-healing: if the stream breaks because tracking isn't up yet or restarted,
// we tear it down and reopen once the channel is connected again, with capped
// backoff. It never blocks the worker and never latches permanently broken, so
// detections resume on their own. See README.md.

#pragma once

#include "types.hpp"
#include "services.grpc.pb.h"
#include <grpcpp/grpcpp.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>

class TrackingClient
{
public:
    explicit TrackingClient(std::shared_ptr<grpc::Channel> channel);
    ~TrackingClient();

    void start();                           // brief connect grace + open the stream
    // Forward one frame's detections. attach_frame=true additionally inlines
    // pkt.frame as gInline (grpc transport + pixel-consuming tracker; see
    // detection_writer.hpp -- callers go through a DetectionWriter, not this).
    void send(const DetectionPacket& pkt, bool attach_frame = false);
    void stop();                            // WritesDone + Finish; suppress reconnect

    // Observability (mirrored into Prometheus by main.cpp's sampler thread).
    // Atomic so the sampler can read without taking write_mtx_ and stalling sends.
    uint64_t reconnects() const { return reconnects_.load(std::memory_order_relaxed); }
    bool connected() const { return connected_.load(std::memory_order_relaxed); }

private:
    bool open_locked();                     // (re)open a stream if the channel is READY
    void close_locked();                    // Finish current stream + schedule a retry
    void schedule_retry_locked();           // bump next_retry_ / grow backoff

    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<inference::TrackingService::Stub> stub_;
    std::unique_ptr<grpc::ClientContext> context_;   // one per stream instance
    inference::grpcAck ack_;
    std::unique_ptr<grpc::ClientWriter<inference::grpcDetectionResponse>> stream_;
    std::mutex write_mtx_;
    std::chrono::steady_clock::time_point next_retry_{};   // don't reopen before this
    std::chrono::milliseconds backoff_;                    // grows on repeated failure
    bool stopping_ = false;                                // shutdown: stop reconnecting
    bool opened_once_ = false;                             // first open isn't a "reconnect"

    std::atomic<uint64_t> reconnects_{0};   // streams reopened after a break
    std::atomic<bool> connected_{false};    // stream currently open
};