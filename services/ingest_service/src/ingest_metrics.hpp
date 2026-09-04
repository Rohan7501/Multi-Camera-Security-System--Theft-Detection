#pragma once
// Prometheus metrics for the ingest service.
//
// Ingest already maintains per-stream `Stats` (fps, frames_total, reconnects,
// last_frame_ts_ms) inside each RtspReader, and IngestServer::listCameras()
// hands back a snapshot of all of them. So instead of instrumenting the capture
// hot path, main.cpp samples that snapshot once a second and mirrors it here --
// no extra locking in run_loop(), and one owner for every counter.
//
// Counters are mirrored by DELTA (Prometheus counters only move forward, and a
// restarted reader resets its Stats to 0; see reset handling in sample()).
//
// Labels: `stream` only (bounded set of cameras).

#include "metrics.hpp"
#include "ingest_server.hpp"

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

namespace ingest_metrics {

struct Metrics
{
    explicit Metrics(prometheus::Registry& reg)
        : frames(prometheus::BuildCounter()
              .Name("ingest_frames_captured_total")
              .Help("frames read from the camera and published downstream")
              .Register(reg)),
          reconnects(prometheus::BuildCounter()
              .Name("ingest_rtsp_reconnects_total")
              .Help("RTSP capture reopened after a failed open/read")
              .Register(reg)),
          fps(prometheus::BuildGauge()
              .Name("ingest_fps")
              .Help("frames per second measured over the last second")
              .Register(reg)),
          stream_up(prometheus::BuildGauge()
              .Name("ingest_stream_up")
              .Help("1 when a reader is active for this camera, 0 when stopped")
              .Register(reg)),
          frame_age(prometheus::BuildGauge()
              .Name("ingest_last_frame_age_seconds")
              .Help("time since the last frame was captured")
              .Register(reg)),
          cameras_known(prometheus::BuildGauge()
              .Name("ingest_cameras_known")
              .Help("cameras registered (running or stopped)")
              .Register(reg)),
          cameras_active(prometheus::BuildGauge()
              .Name("ingest_cameras_active")
              .Help("cameras with a live reader")
              .Register(reg)),
          known(cameras_known.Add({})),
          active(cameras_active.Add({}))
    {}

    // Mirror one listCameras() snapshot. `now_ms` must come from the SAME clock
    // RtspReader stamps frames with (steady_clock) -- never wall time.
    void sample(const std::vector<CameraInfo>& cams, int64_t now_ms)
    {
        int running = 0;
        for (const auto& c : cams) {
            auto& prev = last_[c.id];

            // A stopped/respawned reader restarts its Stats at 0; treat any
            // decrease as a reset and resume from the new value.
            if (c.stats.frames_total < prev.frames) prev.frames = 0;
            if (c.stats.reconnects   < prev.recon)  prev.recon  = 0;

            frames.Add({{"stream", c.id}})
                .Increment(static_cast<double>(c.stats.frames_total - prev.frames));
            reconnects.Add({{"stream", c.id}})
                .Increment(static_cast<double>(c.stats.reconnects - prev.recon));
            prev.frames = c.stats.frames_total;
            prev.recon  = c.stats.reconnects;

            fps.Add({{"stream", c.id}}).Set(c.running ? c.stats.fps : 0.0);
            stream_up.Add({{"stream", c.id}}).Set(c.running ? 1.0 : 0.0);
            if (c.stats.last_frame_ts_ms > 0)
                frame_age.Add({{"stream", c.id}})
                    .Set(static_cast<double>(now_ms - c.stats.last_frame_ts_ms) / 1000.0);
            if (c.running) running++;
        }
        known.Set(static_cast<double>(cams.size()));
        active.Set(static_cast<double>(running));
    }

    prometheus::Family<prometheus::Counter>& frames;
    prometheus::Family<prometheus::Counter>& reconnects;
    prometheus::Family<prometheus::Gauge>&   fps;
    prometheus::Family<prometheus::Gauge>&   stream_up;
    prometheus::Family<prometheus::Gauge>&   frame_age;
    prometheus::Family<prometheus::Gauge>&   cameras_known;
    prometheus::Family<prometheus::Gauge>&   cameras_active;

    prometheus::Gauge& known;
    prometheus::Gauge& active;

private:
    struct Prev { uint64_t frames = 0; uint64_t recon = 0; };
    std::unordered_map<std::string, Prev> last_;   // per-stream delta bookkeeping
};

}  // namespace ingest_metrics