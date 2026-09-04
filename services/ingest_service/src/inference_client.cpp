#include "inference_client.hpp"
#include <chrono>
#include <iostream>

InferenceClient::~InferenceClient() {
    StopStream();
}

void InferenceClient::StartStream(const std::string& stream_id) {
    stream_id_ = stream_id;
    stream_ = stub_->grpcStreamFrames(&context_, &ack_);   // client-streaming
}

void InferenceClient::SendFrame(const cv::Mat& frame, u_int64_t frame_ID, int64_t ts) {
    inference::grpcFrameRequest request;

    request.set_gstreamid(stream_id_);
    // request.set_gframeid(frame_counter_++);
    request.set_gframeid(frame_ID);
    // request.set_gtimestampns(
    //     std::chrono::duration_cast<std::chrono::nanoseconds>(
    //         std::chrono::steady_clock::now().time_since_epoch()).count());
    request.set_gtimestampns(ts);
    // Capture instant == publish instant here (ingest is the origin). Downstream
    // hops re-stamp gTimestampNs but leave this one alone, so frame age survives
    // to the end of the pipeline.
    request.set_gcapturetimestampns(ts);

    // gRPC transport inlines the pixels; shm transport passes an empty Mat and
    // sends a metadata-only notification (gInline absent -- pixels are in the ring).
    if (!frame.empty()) {
        auto* inl = request.mutable_ginline();
        inl->set_gwidth(frame.cols);
        inl->set_gheight(frame.rows);
        inl->set_gchannels(frame.channels());
        inl->set_gdata(frame.data, frame.total() * frame.elemSize());
    }

    std::lock_guard<std::mutex> lock(write_mtx_);
    if (stream_ && !stream_->Write(request)) {
        std::cerr << "Frame stream broken for " << stream_id_ << "\n";
    }
    // Time metric Out-time
}

void InferenceClient::StopStream() {
    if (!stream_)
        return;

    {
        std::lock_guard<std::mutex> lock(write_mtx_);
        stream_->WritesDone();
    }
    grpc::Status status = stream_->Finish();
    if (!status.ok()) {
        std::cerr << "Stream closed with error: " << status.error_message() << "\n";
    }
    stream_.reset();
}
