// Producer-side detection transport, mirroring frame_writer / frame_reader.
//
// The worker publishes one DetectionPacket per frame. Both variants send the
// same metadata and boxes to the TrackingService and differ only in whether the
// pixels ride along as gInline. Picked once in makeDetectionWriter(), so there
// is no per-frame branching.

#pragma once
#include "types.hpp"
#include "tracking_client.hpp"
#include "shm.hpp"   // frameTransport() / Transport
#include <cstdlib>
#include <cstring>
#include <memory>

class DetectionWriter {
public:
    virtual ~DetectionWriter() = default;
    virtual void publish(const DetectionPacket& pkt) = 0;
};

// Reference-only: detections + (stream_id, frame_id). Tracking resolves pixels
// itself from the shm ring, or doesn't need them (motion-only tracker like
// ByteTrack). This is the default in BOTH transports.
class RefDetectionWriter : public DetectionWriter {
public:
    explicit RefDetectionWriter(TrackingClient& client) : client_(client) {}
    void publish(const DetectionPacket& pkt) override {
        client_.send(pkt, /*attach_frame=*/false);
    }
private:
    TrackingClient& client_;
};

// Inline: attach pkt.frame as gInline. Only for a pixel-consuming tracker when
// there is no shm ring to read from (grpc transport).
class InlineDetectionWriter : public DetectionWriter {
public:
    explicit InlineDetectionWriter(TrackingClient& client) : client_(client) {}
    void publish(const DetectionPacket& pkt) override {
        client_.send(pkt, /*attach_frame=*/true);
    }
private:
    TrackingClient& client_;
};

// Deployment-wide flag: does the tracking algorithm consume pixels (appearance /
// re-ID)? Set TRACKING_PIXELS=1 consistently on inference AND tracking.
inline bool trackingWantsPixels() {
    const char* t = std::getenv("TRACKING_PIXELS");
    return t && std::strcmp(t, "1") == 0;
}

// Pixels ride the detection hop only when the tracker consumes them AND the
// transport is grpc -- in shm mode tracking reads the ring by (stream, frame),
// so inlining would be a redundant copy.
inline std::unique_ptr<DetectionWriter> makeDetectionWriter(TrackingClient& client) {
    if (frameTransport() == Transport::Grpc && trackingWantsPixels())
        return std::make_unique<InlineDetectionWriter>(client);
    return std::make_unique<RefDetectionWriter>(client);
}
