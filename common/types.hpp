#pragma once
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

template<typename T>
class FrameQueue;


struct Detection {
  int class_id = 0;
  float confidence = 0.0f;
  // xyxy in pixels
  int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
};

struct Stats {
  double fps = 0.0;
  uint64_t frames_total = 0;
  uint64_t frames_dropped = 0;
  uint64_t reconnects = 0;
  int64_t last_frame_ts_ms = 0;
};

struct Frame {
  int width = 0;
  int height = 0;
  int channels = 0;
  int64_t ts_ms = 0;
  std::vector<uint8_t> bgr; // tightly packed BGR
};

// Two timestamps, both steady_clock ns, with DIFFERENT lifetimes:
//   timestamp_ns         -- RE-STAMPED at each hop with that stage's receipt
//                           time, so each service can measure the one segment it
//                           just completed (*_module_latency_seconds).
//   capture_timestamp_ns -- set ONCE by ingest at the RTSP read, never touched
//                           again, so frame age survives to the end of the pipeline.
struct DetectionPacket
{
    std::string stream_id;
    uint64_t frame_id = 0;
    int64_t  timestamp_ns = 0;          // carried forward for tracking/alerting
    int64_t  capture_timestamp_ns = 0;  // ingest capture instant (never re-stamped)
    std::vector<Detection> detections;
    cv::Mat frame;
};

struct FramePacket
{
    std::string stream_id;
    uint64_t frame_id = 0;
    int64_t  timestamp_ns = 0;
    int64_t  capture_timestamp_ns = 0;  // ingest capture instant (never re-stamped)
    cv::Mat frame;
};

