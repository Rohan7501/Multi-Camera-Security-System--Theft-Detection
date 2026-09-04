#pragma once
#include <memory>
#include <mutex>
#include <string>
#include <grpcpp/grpcpp.h>
#include <opencv2/opencv.hpp>
#include "services.grpc.pb.h"

// Client for InferenceService.grpcStreamFrames (client-streaming, forward flow):
// pushes frames to the inference server and gets back only a grpcAck on close.
// No detections are returned here -- they flow forward to the TrackingService.
class InferenceClient {
public:
    InferenceClient(std::shared_ptr<grpc::Channel> channel)
        : stub_(inference::InferenceService::NewStub(channel)) {}

    ~InferenceClient();

    void StartStream(const std::string& stream_id);
    void SendFrame(const cv::Mat& frame, u_int64_t frame_ID, int64_t ts);
    void StopStream();

private:
    std::unique_ptr<inference::InferenceService::Stub> stub_;
    std::unique_ptr<grpc::ClientWriter<inference::grpcFrameRequest>> stream_;

    grpc::ClientContext context_;
    inference::grpcAck ack_;
    std::mutex write_mtx_;

    std::string stream_id_;
    uint64_t frame_counter_{0};
};
