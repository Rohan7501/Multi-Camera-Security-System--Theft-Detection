#include "inference_worker.hpp"
#include <chrono>
#include <iostream>

#define LOG_ERROR(msg) std::cerr << "[ERROR] " << __FILE__ << ":" << __LINE__ << " " << msg << std::endl;

InferenceWorker::InferenceWorker(FrameQueue<FramePacket>& frame_queue,
                                 Detector& detector,
                                 DetectionWriter& writer,
                                 FrameReader& reader,
                                 inference_metrics::Metrics& metrics)
    : frame_queue_(frame_queue),
      detector_(detector),
      writer_(writer),
      reader_(reader),
      metrics_(metrics)
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