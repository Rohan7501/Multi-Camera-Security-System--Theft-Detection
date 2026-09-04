#pragma once
#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include "../common/types.hpp"
#include <grpcpp/grpcpp.h>
#include "services.grpc.pb.h"
#include "inference_client.hpp"
#include "shm.hpp"
#include "frame_writer.hpp"

class RtspReader {
private:
    void run_loop();

    std::string stream_id_;
    std::string url_;
    std::atomic<bool> running_{false};
    std::thread worker_;

    mutable std::mutex mtx_;
    std::optional<Frame> latest_;
    Stats stats_;

    uint64_t frame_counter_{0};

    std::unique_ptr<InferenceClient> client_;   // owned; drives the gRPC stream lifecycle
    std::unique_ptr<FrameWriter> writer_;       // publishes frames (gRPC or shm); wraps client_

public:
    RtspReader(std::string stream_id, std::string url,
               std::shared_ptr<grpc::Channel> channel, FrameStore* fs);
    ~RtspReader();

    void start();
    void stop();

    // returns latest frame (copy) if available
    std::optional<Frame> latest_frame();
    Stats stats() const;
};
