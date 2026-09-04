#include "ingest_admin_service.hpp"
#include "ingest_server.hpp"

namespace {
// Domain error string -> reply. Empty err == success.
void fill(inference::ingestAdminReply* rep, const std::string& err) {
    rep->set_ok(err.empty());
    rep->set_message(err.empty() ? "ok" : err);
}
}  // namespace

grpc::Status IngestAdminService::AddCamera(grpc::ServerContext*,
        const inference::ingestCamera* req, inference::ingestAdminReply* rep) {
    fill(rep, server_.addCamera(req->gstreamid(), req->gurl()));
    return grpc::Status::OK;
}

grpc::Status IngestAdminService::StartStream(grpc::ServerContext*,
        const inference::ingestStreamId* req, inference::ingestAdminReply* rep) {
    fill(rep, server_.startStream(req->gstreamid()));
    return grpc::Status::OK;
}

grpc::Status IngestAdminService::StopStream(grpc::ServerContext*,
        const inference::ingestStreamId* req, inference::ingestAdminReply* rep) {
    fill(rep, server_.stopStream(req->gstreamid()));
    return grpc::Status::OK;
}

grpc::Status IngestAdminService::RemoveCamera(grpc::ServerContext*,
        const inference::ingestStreamId* req, inference::ingestAdminReply* rep) {
    fill(rep, server_.removeCamera(req->gstreamid()));
    return grpc::Status::OK;
}

grpc::Status IngestAdminService::ListCameras(grpc::ServerContext*,
        const inference::ingestEmpty*, inference::ingestCameraList* rep) {
    for (const auto& c : server_.listCameras()) {
        auto* s = rep->add_gcameras();
        s->set_gstreamid(c.id);
        s->set_gurl(c.url);
        s->set_grunning(c.running);
        s->set_gfps(c.stats.fps);
        s->set_gframestotal(c.stats.frames_total);
        s->set_greconnects(c.stats.reconnects);
        s->set_glastframetsms(c.stats.last_frame_ts_ms);
    }
    return grpc::Status::OK;
}
