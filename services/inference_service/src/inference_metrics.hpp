#pragma once
// Prometheus metrics for the inference service.
//
// One instance is built in main.cpp and handed to the worker(s) by reference.
// Metric *families* are created once here; per-stream children are resolved with
// forStream(...) on the hot path (prometheus-cpp's Add() is a hash lookup under a
// mutex, so we keep it out of the innermost loop where it matters).
//
// Labels: `stream` only (bounded set of cameras). Never track_id/frame_id.

#include "metrics.hpp"

#include <string>

namespace inference_metrics {

struct Metrics
{
    explicit Metrics(prometheus::Registry& reg)
        : frames(prometheus::BuildCounter()
              .Name("inference_frames_total")
              .Help("frames run through the detector")
              .Register(reg)),
          detections(prometheus::BuildCounter()
              .Name("inference_detections_total")
              .Help("detections emitted, by retail class id (0=Product 1=Product-Picked 2=Regular 3=Shoplifting)")
              .Register(reg)),
          unavailable(prometheus::BuildCounter()
              .Name("inference_frame_unavailable_total")
              .Help("frames skipped: pixels could not be resolved from the transport")
              .Register(reg)),
          latency(prometheus::BuildHistogram()
              .Name("inference_latency_seconds")
              .Help("detector infer() wall time per frame")
              .Register(reg)),
          ingest_latency(prometheus::BuildHistogram()
              .Name("ingest_module_latency_seconds")
              .Help("seconds from the ingest capture stamp to the frame being read "
                    "by inference (RTSP read -> gRPC receipt; transport + queueing)")
              .Register(reg)),
          queue_depth(prometheus::BuildGauge()
              .Name("inference_queue_depth")
              .Help("frames waiting in the FrameQueue")
              .Register(reg)),
          queue_dropped(prometheus::BuildCounter()
              .Name("inference_queue_dropped_total")
              .Help("frames dropped: queue full, oldest evicted")
              .Register(reg)),
          tracking_reconnects(prometheus::BuildCounter()
              .Name("inference_tracking_reconnects_total")
              .Help("TrackingService stream reopened after a break")
              .Register(reg)),
          tracking_up(prometheus::BuildGauge()
              .Name("inference_tracking_stream_up")
              .Help("1 when the TrackingService stream is open, 0 while reconnecting")
              .Register(reg)),
          // Unlabelled singletons -- resolve once, not per frame.
          depth(queue_depth.Add({})),
          dropped(queue_dropped.Add({})),
          reconnects(tracking_reconnects.Add({})),
          up(tracking_up.Add({}))
    {}

    // Per-stream children, resolved once per frame in the worker.
    struct Stream {
        prometheus::Counter*   frames;
        prometheus::Counter*   unavailable;
        prometheus::Histogram* latency;
    };

    Stream forStream(const std::string& id)
    {
        return Stream{
            &frames.Add({{"stream", id}}),
            &unavailable.Add({{"stream", id}}),
            &latency.Add({{"stream", id}}, metrics::latencyBuckets()),
        };
    }

    prometheus::Counter& forClass(const std::string& stream, int class_id)
    {
        return detections.Add({{"stream", stream},
                               {"class_id", std::to_string(class_id)}});
    }

    // Ingest->inference transit. Separate from forStream() because the gRPC
    // handler needs only this one and shouldn't pay for the other Add() lookups.
    prometheus::Histogram& ingestLatency(const std::string& stream)
    {
        return ingest_latency.Add({{"stream", stream}}, metrics::transportBuckets());
    }

    prometheus::Family<prometheus::Counter>&   frames;
    prometheus::Family<prometheus::Counter>&   detections;
    prometheus::Family<prometheus::Counter>&   unavailable;
    prometheus::Family<prometheus::Histogram>& latency;
    prometheus::Family<prometheus::Histogram>& ingest_latency;
    prometheus::Family<prometheus::Gauge>&     queue_depth;
    prometheus::Family<prometheus::Counter>&   queue_dropped;
    prometheus::Family<prometheus::Counter>&   tracking_reconnects;
    prometheus::Family<prometheus::Gauge>&     tracking_up;

    prometheus::Gauge&   depth;
    prometheus::Counter& dropped;
    prometheus::Counter& reconnects;
    prometheus::Gauge&   up;
};

}  // namespace inference_metrics