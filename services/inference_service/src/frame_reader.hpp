// Consumer-side frame transport, named from inference's point of view: we READ
// frames, so we hold a FrameReader.
//
// Given a FramePacket -- stream_id, frame_id, and possibly inline gRPC pixels --
// yield the cv::Mat to run inference on. The gRPC-vs-shm choice is made once in
// makeFrameReader(); nothing downstream branches on it.

#pragma once
#include "types.hpp"
#include "shm.hpp"
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <memory>
#include <string>

class FrameReader {
public:
    virtual ~FrameReader() = default;
    virtual bool get(const FramePacket& pkt, cv::Mat& out) = 0;   // false = unavailable
};

// gRPC transport: pixels already arrived inline (decoded by the gRPC server).
class GrpcFrameReader : public FrameReader {
public:
    bool get(const FramePacket& pkt, cv::Mat& out) override {
        if (pkt.frame.empty()) return false;
        out = pkt.frame;
        return true;
    }
};

// shm transport: the packet is metadata only; fetch pixels from the ring.
class ShmFrameReader : public FrameReader {
public:
    explicit ShmFrameReader(FrameStore* fs) : fs_(fs) {}
    bool get(const FramePacket& pkt, cv::Mat& out) override {
        return fs_ && getFrame(fs_, pkt.stream_id, pkt.frame_id, out);
    }
private:
    FrameStore* fs_;
};

// Selected once at startup from FRAME_TRANSPORT (default gRPC). checkTransport()
// must have already validated that the peer agrees.
inline std::unique_ptr<FrameReader> makeFrameReader(FrameStore* fs) {
    if (frameTransport() == Transport::Shm) return std::make_unique<ShmFrameReader>(fs);
    return std::make_unique<GrpcFrameReader>();
}
