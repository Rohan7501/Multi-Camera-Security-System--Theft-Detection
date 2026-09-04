#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <grpcpp/grpcpp.h>
#include <yaml-cpp/yaml.h>

#include "rtsp_reader.hpp"
#include "shm.hpp"
#include "types.hpp"

namespace grpc { class Server; }
class IngestAdminService;

struct Camera {
    std::string id;
    std::string url;
};

// Snapshot of one camera for the admin ListCameras RPC. Plain struct so the
// admin service (which knows the proto types) does the translation, not this class.
struct CameraInfo {
    std::string id;
    std::string url;
    bool        running = false;   // an RtspReader is currently active for this id
    Stats       stats;
};

// Owns the RTSP readers plus an IngestAdmin gRPC server for runtime camera
// management. `known_` is every camera ever added; `active_` is the subset with
// a live reader, so a stopped stream can be respawned. See README.md.
class IngestServer {
public:
    IngestServer();
    ~IngestServer();

    void start();   // dial inference, open shm, start configured cameras + admin server
    void stop();    // stop admin server + all active readers

    std::vector<Camera> LoadCameras(const std::string& filename);

    // Runtime camera ops (called by IngestAdminService). Return "" on success,
    // else a human-readable error.
    std::string addCamera(const std::string& id, const std::string& url);
    std::string startStream(const std::string& id);
    std::string stopStream(const std::string& id);
    std::string removeCamera(const std::string& id);
    std::vector<CameraInfo> listCameras();

private:
    void spawnReader(const std::string& id, const std::string& url);   // caller holds mtx_

    std::mutex mtx_;
    std::unordered_map<std::string, std::string> known_;                       // id -> url
    std::unordered_map<std::string, std::unique_ptr<RtspReader>> active_;      // id -> reader
    std::shared_ptr<grpc::Channel> channel_;
    FrameStore* fs_ = nullptr;

    std::unique_ptr<IngestAdminService> admin_service_;
    std::unique_ptr<grpc::Server> admin_server_;
};
