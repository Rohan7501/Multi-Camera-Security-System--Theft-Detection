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
class InferenceWorker
{
public:
    InferenceWorker(FrameQueue<FramePacket>& frame_queue,
                    Detector& detector,
                    DetectionWriter& writer,
                    FrameReader& reader,
                    inference_metrics::Metrics& metrics);

    void run();

private:
    FrameQueue<FramePacket>& frame_queue_;
    Detector& detector_;
    DetectionWriter& writer_;
    FrameReader& reader_;
    inference_metrics::Metrics& metrics_;
};
