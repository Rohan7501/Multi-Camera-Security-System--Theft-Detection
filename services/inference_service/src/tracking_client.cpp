#include "tracking_client.hpp"
#include <algorithm>
#include <iostream>

#define LOG_INFO(msg)  std::cout << "[INFO] "  << __FILE__ << ":" << __LINE__ << " " << msg << std::endl;
#define LOG_ERROR(msg) std::cerr << "[ERROR] " << __FILE__ << ":" << __LINE__ << " " << msg << std::endl;

namespace {
constexpr std::chrono::milliseconds kMinBackoff{200};    // first retry gap
constexpr std::chrono::milliseconds kMaxBackoff{5000};   // cap for a long-down peer
}

TrackingClient::TrackingClient(std::shared_ptr<grpc::Channel> channel)
    : channel_(std::move(channel)),
      stub_(inference::TrackingService::NewStub(channel_)),
      backoff_(kMinBackoff)
{}

TrackingClient::~TrackingClient() { stop(); }

void TrackingClient::start()
{
    std::lock_guard<std::mutex> lock(write_mtx_);
    // Brief grace window so a just-launched tracking (the readiness gate usually
    // makes this instant) is connected before the first detections arrive. If it
    // isn't up, this returns after the deadline and send() reconnects later.
    channel_->WaitForConnected(
        std::chrono::system_clock::now() + std::chrono::seconds(2));
    open_locked();
}

bool TrackingClient::open_locked()
{
    if (std::chrono::steady_clock::now() < next_retry_)
        return false;                       // still inside the backoff window

    // Only open a stream when the channel is actually connected -- otherwise
    // writes would pour into a doomed stream. GetState(true) kicks off a
    // (re)connect in the background and returns immediately (never blocks).
    if (channel_->GetState(/*try_to_connect=*/true) != GRPC_CHANNEL_READY) {
        schedule_retry_locked();
        return false;
    }

    context_ = std::make_unique<grpc::ClientContext>();
    stream_ = stub_->grpcStreamDetections(context_.get(), &ack_);
    if (!stream_) {
        context_.reset();
        schedule_retry_locked();
        return false;
    }
    backoff_ = kMinBackoff;                  // healthy again: reset the backoff
    if (opened_once_)                        // don't count the first open as a reconnect
        reconnects_.fetch_add(1, std::memory_order_relaxed);
    opened_once_ = true;
    connected_.store(true, std::memory_order_relaxed);
    LOG_INFO("TrackingClient stream opened");
    return true;
}

void TrackingClient::schedule_retry_locked()
{
    next_retry_ = std::chrono::steady_clock::now() + backoff_;
    backoff_ = std::min(backoff_ * 2, kMaxBackoff);
}

void TrackingClient::close_locked()
{
    if (stream_) {
        stream_->WritesDone();
        stream_->Finish();                   // reap status/resources; result ignored
        stream_.reset();
        context_.reset();
    }
    connected_.store(false, std::memory_order_relaxed);
    schedule_retry_locked();
}

void TrackingClient::send(const DetectionPacket& pkt, bool attach_frame)
{
    std::lock_guard<std::mutex> lock(write_mtx_);
    if (stopping_)
        return;
    if (!stream_ && !open_locked())
        return;                              // not connected yet; drop this frame

    inference::grpcDetectionResponse resp;
    resp.set_gstreamid(pkt.stream_id);
    resp.set_gframeid(pkt.frame_id);
    resp.set_gtimestampns(pkt.timestamp_ns);
    resp.set_gcapturetimestampns(pkt.capture_timestamp_ns);   // ingest's instant, untouched

    for (const auto& det : pkt.detections)
    {
        auto* d = resp.add_gdetections();
        d->set_gx1(static_cast<float>(det.x1));
        d->set_gy1(static_cast<float>(det.y1));
        d->set_gx2(static_cast<float>(det.x2));
        d->set_gy2(static_cast<float>(det.y2));
        d->set_gconfidence(det.confidence);
        d->set_gclassid(det.class_id);
        d->set_gtrackid(-1);   // tracking assigns the real id
    }

    // Inline the pixels only when asked (InlineDetectionWriter: grpc transport +
    // pixel-consuming tracker). Otherwise gInline stays absent and tracking
    // resolves pixels from the shm ring -- or doesn't need them at all.
    if (attach_frame && !pkt.frame.empty()) {
        cv::Mat f = pkt.frame.isContinuous() ? pkt.frame : pkt.frame.clone();
        auto* inl = resp.mutable_ginline();
        inl->set_gwidth(f.cols);
        inl->set_gheight(f.rows);
        inl->set_gchannels(f.channels());
        inl->set_gdata(f.data, f.total() * f.elemSize());
    }

    if (!stream_->Write(resp)) {
        LOG_ERROR("TrackingService stream broken; will reconnect");
        close_locked();        // tear down + schedule a retry; next send() reopens
    }
}

void TrackingClient::stop()
{
    std::lock_guard<std::mutex> lock(write_mtx_);
    stopping_ = true;          // no more reconnects
    connected_.store(false, std::memory_order_relaxed);
    if (!stream_)
        return;

    stream_->WritesDone();
    grpc::Status status = stream_->Finish();
    if (!status.ok())
        LOG_ERROR("TrackingService finished with error: " << status.error_message());
    stream_.reset();
    context_.reset();
}