#pragma once

#include "frame_queue.hpp"
#include "types.hpp"
#include "detector.hpp"
#include "detection_writer.hpp"
#include "frame_reader.hpp"
#include "inference_metrics.hpp"

// Single worker: pops frames, obtains the pixels via the FrameReader (gRPC inline
// or shm ring), runs the shared detector, and publishes detections downstream via
// the DetectionWriter (reference-only or with inline pixels; forward flow).
// Records Prometheus metrics (frames, latency, detections) as it goes.
//
// max_age_ns is the safety-net capture-age limit: a frame already older than this
// when the worker takes it is skipped and counted stale. 0 = never drop.
class InferenceWorker
{
public:
    InferenceWorker(FrameQueue<FramePacket>& frame_queue,
                    Detector& detector,
                    DetectionWriter& writer,
                    FrameReader& reader,
                    inference_metrics::Metrics& metrics,
                    int64_t max_age_ns);

    void run();

private:
    FrameQueue<FramePacket>& frame_queue_;
    Detector& detector_;
    DetectionWriter& writer_;
    FrameReader& reader_;
    inference_metrics::Metrics& metrics_;
    const int64_t max_age_ns_;
};
