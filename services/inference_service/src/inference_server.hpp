#pragma once

#include "frame_queue.hpp"
#include "types.hpp"
#include "services.grpc.pb.h"
#include "inference_metrics.hpp"

namespace grpc {
    class Server;
}

// InferenceService server. `grpcStreamFrames` is CLIENT-STREAMING (forward flow):
// ingest pushes frames, the handler decodes them onto the shared FrameQueue, and
// returns a single grpcAck when the stream ends. Detections are NOT returned here
// -- the InferenceWorker forwards them onward to the TrackingService.
class InferenceServer final : public inference::InferenceService::Service
{
public:
    InferenceServer(FrameQueue<FramePacket>& frame_queue, inference_metrics::Metrics& metrics);
    ~InferenceServer();

    void start();
    void shutdown();

    grpc::Status grpcStreamFrames(
        grpc::ServerContext* context,
        grpc::ServerReader<inference::grpcFrameRequest>* reader,
        inference::grpcAck* response) override;

private:
    FrameQueue<FramePacket>& queue_;
    std::unique_ptr<grpc::Server> server_;
    inference_metrics::Metrics& metrics_;
};
