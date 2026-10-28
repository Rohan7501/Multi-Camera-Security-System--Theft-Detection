#include "inference_worker.hpp"
#include <chrono>
#include <cstdint>
#include <iostream>

#define LOG_ERROR(msg) std::cerr << "[ERROR] " << __FILE__ << ":" << __LINE__ << " " << msg << std::endl;

namespace {
// Same clock ingest stamps gCaptureTimestampNs with (steady_clock ==
// CLOCK_MONOTONIC), which is what makes the subtraction valid -- but only
// because both processes run on this box. See README.md.
int64_t steadyNowNs()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
}

InferenceWorker::InferenceWorker(FrameQueue<FramePacket>& frame_queue,
                                 Detector& detector,
                                 DetectionWriter& writer,
                                 FrameReader& reader,
                                 inference_metrics::Metrics& metrics,
                                 int64_t max_age_ns)
    : frame_queue_(frame_queue),
      detector_(detector),
      writer_(writer),
      reader_(reader),
      metrics_(metrics),
      max_age_ns_(max_age_ns)
{}

void InferenceWorker::run()
{
    FramePacket packet;

    while (frame_queue_.pop(packet))
    {
        // Per-frame metrics only. Queue-level gauges (depth/dropped) are sampled
        // by the single-owner thread in main.cpp -- mirroring them from N workers
        // would double-count.
        auto m = metrics_.forStream(packet.stream_id);

        // Safety net, not the steady state: a healthy pool takes frames within a
        // few ms of capture, so this only fires once the pool has fallen behind.
        // A frame that old would miss the end-to-end budget anyway, so skip it
        // before the shm fetch and infer() spend anything on it. Measured from
        // the capture stamp, so it covers the ingest->inference hop too.
        // Guard on > 0: proto3 omits a 0 default, so an unset stamp would read
        // as time-since-boot and drop every frame. See README.md.
        if (max_age_ns_ > 0 && packet.capture_timestamp_ns > 0 &&
            steadyNowNs() - packet.capture_timestamp_ns > max_age_ns_) {
            m.stale->Increment();
            continue;
        }

        // Resolve pixels via the configured transport (gRPC inline / shm ring).
        // A miss (empty gRPC frame, or shm lapped/unavailable) -> skip this frame.
        cv::Mat frame;
        if (!reader_.get(packet, frame)) {
            m.unavailable->Increment();
            LOG_ERROR("frame unavailable: " << packet.stream_id << " #" << packet.frame_id);
            continue;
        }

        DetectionPacket result;
        result.stream_id    = packet.stream_id;
        result.frame_id     = packet.frame_id;
        result.timestamp_ns = packet.timestamp_ns;
        result.capture_timestamp_ns = packet.capture_timestamp_ns;   // never re-stamped
        result.frame        = frame;   // refcount share; serialized only by InlineDetectionWriter

        const auto t0 = std::chrono::steady_clock::now();
        result.detections   = detector_.infer(frame);
        const std::chrono::duration<double> dt = std::chrono::steady_clock::now() - t0;

        m.latency->Observe(dt.count());
        m.frames->Increment();
        for (const auto& det : result.detections)
            metrics_.forClass(packet.stream_id, det.class_id).Increment();

        writer_.publish(result);   // forward downstream to TrackingService
    }
}