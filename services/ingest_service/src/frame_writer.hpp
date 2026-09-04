// Producer-side frame transport, named from ingest's point of view: we WRITE
// frames, so we hold a FrameWriter.
//
// publish() sends one frame for (stream_id, frame_id). Both transports carry the
// SAME already-resized frame, so inference sees identical pixels either way. The
// gRPC-vs-shm choice is made once in makeFrameWriter().

#pragma once
#include "shm.hpp"
#include "inference_client.hpp"
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <memory>
#include <string>

class FrameWriter {
public:
    virtual ~FrameWriter() = default;
    virtual void publish(const std::string& sid, uint64_t fid,
                         int64_t ts_ns, const cv::Mat& frame) = 0;
};

// gRPC transport: send the pixels inline over the frame stream.
class GrpcFrameWriter : public FrameWriter {
public:
    explicit GrpcFrameWriter(InferenceClient& client) : client_(client) {}
    void publish(const std::string&, uint64_t fid, int64_t ts_ns, const cv::Mat& frame) override {
        client_.SendFrame(frame, fid, ts_ns);
    }
private:    
    InferenceClient& client_;
};

// shm transport: write pixels to the ring, then send a metadata-only gRPC
// notification (empty frame) so inference knows a frame is ready to fetch.
class ShmFrameWriter : public FrameWriter {
public:
    ShmFrameWriter(FrameStore* fs, InferenceClient& client) : fs_(fs), client_(client) {}
    void publish(const std::string& sid, uint64_t fid, int64_t ts_ns, const cv::Mat& frame) override {
        if (fs_) writeFrame(fs_, sid, fid, ts_ns, frame);
        client_.SendFrame(cv::Mat(), fid, ts_ns);   // notification only, no pixels
    }
private:
    FrameStore* fs_;
    InferenceClient& client_;
};

// Selected once at startup from FRAME_TRANSPORT (default gRPC). checkTransport()
// must have already validated that the peer agrees.
inline std::unique_ptr<FrameWriter> makeFrameWriter(FrameStore* fs, InferenceClient& client) {
    if (frameTransport() == Transport::Shm) return std::make_unique<ShmFrameWriter>(fs, client);
    return std::make_unique<GrpcFrameWriter>(client);
}
