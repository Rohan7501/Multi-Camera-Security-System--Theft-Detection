#include "inference_server.hpp"
#include <grpcpp/grpcpp.h>
#include <chrono>
#include <iostream>

#define LOG_INFO(msg)  std::cout << "[INFO] "  << __FILE__ << ":" << __LINE__ << " " << msg << std::endl;
#define LOG_ERROR(msg) std::cerr << "[ERROR] " << __FILE__ << ":" << __LINE__ << " " << msg << std::endl;

namespace {
// Same clock ingest stamps gTimestampNs with (steady_clock == CLOCK_MONOTONIC),
// which is what makes the subtraction valid -- but only because both processes
// run on this box. See README.md.
int64_t steadyNowNs()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
}

InferenceServer::InferenceServer(FrameQueue<FramePacket>& frame_queue,inference_metrics::Metrics& metrics)
    : queue_(frame_queue),
    metrics_(metrics)
{}

InferenceServer::~InferenceServer() = default;

void InferenceServer::start()
{
    std::string server_address = "127.0.0.1:50051";

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(this);

    server_ = builder.BuildAndStart();
    std::cout << "Inference server listening on " << server_address << std::endl;

    server_->Wait();
}

void InferenceServer::shutdown()
{
    std::cout << "Shutting down server\n";
    server_->Shutdown();
    queue_.shutdown();
}

grpc::Status InferenceServer::grpcStreamFrames(
    grpc::ServerContext* context,
    grpc::ServerReader<inference::grpcFrameRequest>* reader,
    inference::grpcAck* response)
{
    LOG_INFO("Frame stream started");

    inference::grpcFrameRequest request;

    while (reader->Read(&request))
    {
        if (context->IsCancelled()) {
            LOG_ERROR("Client cancelled stream");
            return grpc::Status::CANCELLED;
        }

        FramePacket packet;
        packet.stream_id    = request.gstreamid();
        packet.frame_id     = request.gframeid();
        // Re-stamp with THIS hop's receipt time (ns -- must match the unit ingest
        // sends, or the subtraction below is meaningless). Each stage re-stamps,
        // so every service measures the segment it just completed.
        packet.timestamp_ns = steadyNowNs();
        // Pass the capture instant through UNCHANGED -- this is the one stamp no
        // hop may overwrite, so tracking can still compute end-to-end frame age.
        packet.capture_timestamp_ns = request.gcapturetimestampns();

        // ingest_module_latency_seconds: ingest's RTSP read -> received here.
        // Guard on > 0: proto3 omits a 0 default, so an unset stamp would
        // otherwise be measured as time-since-boot. Clamp negatives, which a
        // coarse clock can yield for a sub-microsecond hop.
        if (request.gtimestampns() > 0) {
            const int64_t delta_ns = packet.timestamp_ns - request.gtimestampns();
            metrics_.ingestLatency(packet.stream_id)
                    .Observe(delta_ns > 0 ? static_cast<double>(delta_ns) / 1e9 : 0.0);
        }

        // gRPC transport carries pixels inline (gInline present); shm transport
        // sends a metadata-only notification (gInline absent). Always queue the
        // packet; the worker's FrameReader resolves the pixels for its transport.
        if (request.has_ginline() && !request.ginline().gdata().empty()) {
            const auto& inl = request.ginline();
            // The cv::Mat wraps the protobuf buffer; the clone is load-bearing
            // because `request` is reused on the next Read().
            cv::Mat frame(inl.gheight(), inl.gwidth(), CV_8UC3,
                          (void*)inl.gdata().data());
            #ifdef ENABLE_VISUALIZATION
            cv::imshow("Inference", frame);
            cv::waitKey(1);
            #endif
            packet.frame = frame.clone();
        }

        queue_.push(packet);
    }

    #ifdef ENABLE_VISUALIZATION
    cv::destroyAllWindows();
    #endif

    response->set_ok(true);
    return grpc::Status::OK;
}
