#include "rtsp_reader.hpp"

#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>
#define LOG_ERROR(msg) std::cerr << "[ERROR] " << __FILE__ << ":" << __LINE__ << " " << msg << std::endl;

// #include "../../proto/inference-ingest.grpc.pb.h"
// #include "../../inference_service/src/inference_server.hpp"

static int64_t now_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::steady_clock::now().time_since_epoch()
  ).count();
}

// The timestamp that travels downstream as gTimestampNs. MUST be nanoseconds --
// publishing now_ms() here once made the wire value 1e6x too small. Same
// steady_clock as now_ms(), so they share an epoch. See README.md.
static int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()
  ).count();
}

// RtspReader::RtspReader(std::string stream_id, std::string url)
// : stream_id_(std::move(stream_id)), url_(std::move(url)) {}
// RtspReader::RtspReader(std::string stream_id,
//                        std::string url,
//                        std::shared_ptr<grpc::Channel> channel)
//     : stream_id_(stream_id),
//       url_(url),
//       channel_(channel)
// {
// }
// RtspReader::RtspReader(std::string stream_id, std::string url,
//           std::shared_ptr<grpc::Channel> channel)
//         : client_(std::make_unique<InferenceClient>(channel)),
//         stream_id_(stream_id),
//         url_(url)
//         // channel_(channel)
// {};

RtspReader::RtspReader(std::string stream_id, std::string url,
                       std::shared_ptr<grpc::Channel> channel, FrameStore* fs)
    : stream_id_(stream_id),
      url_(url),
      client_(std::make_unique<InferenceClient>(channel)),
      writer_(makeFrameWriter(fs, *client_))   // gRPC or shm, chosen once from FRAME_TRANSPORT
{}

RtspReader::~RtspReader() {
    stop();
}

void RtspReader::start() {
    running_ = true;
    client_->StartStream(stream_id_);
    // Why do we need to reference RtspReader Object??
    worker_ = std::thread(&RtspReader::run_loop, this);

    // capture_thread_ = std::thread(client_.ReadResponses(), this);
}

// void RtspReader::stop() { running_ = false; }
void RtspReader::stop() {
    running_ = false;
    if (worker_.joinable()) {
        worker_.join();
        client_->StopStream();
    }
}

std::optional<Frame> RtspReader::latest_frame() {
  std::lock_guard<std::mutex> lk(mtx_);
  return latest_;
}

Stats RtspReader::stats() const {
  std::lock_guard<std::mutex> lk(mtx_);
  return stats_;
}

void RtspReader::run_loop() {
  cv::VideoCapture cap;

  while (running_) {
    // ---- Connection lifecycle: (re)open the capture ----
    if (!cap.isOpened()) {
      if (!cap.open(url_)) {
        {
          std::lock_guard<std::mutex> lk(mtx_);
          stats_.reconnects++;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        continue; // retry the open
      }
    }

    auto t0 = now_ms();
    uint64_t frames_since = 0;

    cv::Mat mat;

    // ---- Frame pump: breaks out only to trigger a reconnect ----
    while (running_) {
      if (!cap.read(mat) || mat.empty()) {
        cap.release(); // drop the dead capture; outer loop reconnects
        break;         // breaks the INNER loop only, not the thread
      }

      if (!mat.isContinuous()) {
        mat = mat.clone();
      }

      #ifdef ENABLE_VISUALIZATION
        cv::imshow("RTSP Stream " + stream_id_ , mat);
        cv::waitKey(1);
      #endif

      frame_counter_++;

      // Downscale to <= 640x640 and publish. Both transports carry THIS frame, so
      // inference gets identical pixels/coordinates whether it's gRPC or shm.
      const int TARGET = 640;
      const float r = std::min(static_cast<float>(TARGET) / mat.cols,
                               static_cast<float>(TARGET) / mat.rows);
      const int new_w = static_cast<int>(std::round(mat.cols * r));
      const int new_h = static_cast<int>(std::round(mat.rows * r));

      try {
        cv::Mat resized;
        cv::resize(mat, resized, cv::Size(new_w, new_h));
        // ns, not ms: this is gTimestampNs on the wire and inference subtracts
        // it from a ns clock to measure ingest->inference transit.
        writer_->publish(stream_id_, frame_counter_, now_ns(), resized);
      }
      catch (const cv::Exception& e) {
        LOG_ERROR("resize/publish failed: " << e.what());
      }

      Frame f;
      f.width = mat.cols;
      f.height = mat.rows;
      f.channels = mat.channels();
      f.ts_ms = now_ms();
      f.bgr.assign(mat.data, mat.data + mat.total() * mat.elemSize());

      {
        std::lock_guard<std::mutex> lk(mtx_);
        // latest-only: overwrite previous without queueing
        latest_ = std::move(f);
        stats_.frames_total++;
        stats_.last_frame_ts_ms = latest_->ts_ms;
      }

      frames_since++;
      auto t1 = now_ms();
      if (t1 - t0 >= 1000) {
        std::lock_guard<std::mutex> lk(mtx_);
        stats_.fps = (double)frames_since * 1000.0 / (double)(t1 - t0);
        t0 = t1;
        frames_since = 0;
      }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(250)); // backoff before reconnect
  }

  cap.release();
  #ifdef ENABLE_VISUALIZATION
    cv::destroyAllWindows();
  #endif
}