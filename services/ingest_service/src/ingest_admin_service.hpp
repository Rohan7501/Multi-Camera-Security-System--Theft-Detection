#pragma once
#include "services.grpc.pb.h"

class IngestServer;

// gRPC facade for runtime camera management. Every RPC delegates to IngestServer
// and never fails the transport: domain errors come back as ingestAdminReply.ok
// = false + message, so the control_service gets a clean result either way.
class IngestAdminService final : public inference::IngestAdmin::Service {
public:
    explicit IngestAdminService(IngestServer& server) : server_(server) {}

    grpc::Status AddCamera(grpc::ServerContext*, const inference::ingestCamera*,
                           inference::ingestAdminReply*) override;
    grpc::Status StartStream(grpc::ServerContext*, const inference::ingestStreamId*,
                             inference::ingestAdminReply*) override;
    grpc::Status StopStream(grpc::ServerContext*, const inference::ingestStreamId*,
                            inference::ingestAdminReply*) override;
    grpc::Status RemoveCamera(grpc::ServerContext*, const inference::ingestStreamId*,
                              inference::ingestAdminReply*) override;
    grpc::Status ListCameras(grpc::ServerContext*, const inference::ingestEmpty*,
                             inference::ingestCameraList*) override;

private:
    IngestServer& server_;
};
