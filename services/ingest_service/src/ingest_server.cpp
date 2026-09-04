#include "ingest_server.hpp"
#include "ingest_admin_service.hpp"

#include <yaml-cpp/yaml.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

IngestServer::IngestServer() = default;
IngestServer::~IngestServer() { stop(); }

std::vector<Camera> IngestServer::LoadCameras(const std::string& filename)
{
    std::vector<Camera> camerasOut;

    try {
        YAML::Node config = YAML::LoadFile(filename);

        if (!config["cameras"] || !config["cameras"].IsSequence()) {
            throw std::runtime_error("Missing or invalid 'cameras' section");
        }

        const YAML::Node& cameras = config["cameras"];
        int cameraEntry = 1;
        for (const auto& camNode : cameras) {
            if (!camNode["id"] || !camNode["url"]) {
                std::cerr << "Warning: Skipping invalid camera entry" << cameraEntry << "\n";
                cameraEntry++;
                continue;
            }

            Camera cam;
            cam.id  = camNode["id"].as<std::string>();
            cam.url = camNode["url"].as<std::string>();

            camerasOut.push_back(std::move(cam));
            cameraEntry++;
            std::cout << "Camera added: " << cam.id << cam.url << std::endl;
        }
    }
    catch (const YAML::Exception& e) {
        std::cerr << "YAML error: " << e.what() << "\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }

    return camerasOut;
}

// ---- runtime camera ops (caller-facing; return "" on success) --------------

void IngestServer::spawnReader(const std::string& id, const std::string& url)
{
    auto reader = std::make_unique<RtspReader>(id, url, channel_, fs_);
    reader->start();
    active_[id] = std::move(reader);
}

std::string IngestServer::addCamera(const std::string& id, const std::string& url)
{
    if (id.empty() || url.empty()) return "stream_id and url are required";
    std::lock_guard<std::mutex> lk(mtx_);
    if (active_.count(id)) return "stream already running: " + id;
    known_[id] = url;
    spawnReader(id, url);
    std::cout << "[admin] add camera " << id << " " << url << std::endl;
    return "";
}

std::string IngestServer::startStream(const std::string& id)
{
    std::lock_guard<std::mutex> lk(mtx_);
    if (active_.count(id)) return "stream already running: " + id;
    auto it = known_.find(id);
    if (it == known_.end()) return "unknown camera: " + id;
    spawnReader(id, it->second);
    std::cout << "[admin] start stream " << id << std::endl;
    return "";
}

std::string IngestServer::stopStream(const std::string& id)
{
    std::unique_ptr<RtspReader> reader;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = active_.find(id);
        if (it == active_.end()) return "stream not running: " + id;
        reader = std::move(it->second);
        active_.erase(it);
    }
    reader->stop();   // join OUTSIDE the lock: a frozen stream can't wedge admin
    std::cout << "[admin] stop stream " << id << std::endl;
    return "";
}

std::string IngestServer::removeCamera(const std::string& id)
{
    stopStream(id);   // idempotent: ignore "not running"
    std::lock_guard<std::mutex> lk(mtx_);
    known_.erase(id);
    std::cout << "[admin] remove camera " << id << std::endl;
    return "";
}

std::vector<CameraInfo> IngestServer::listCameras()
{
    std::lock_guard<std::mutex> lk(mtx_);
    std::vector<CameraInfo> out;
    out.reserve(known_.size());
    for (const auto& [id, url] : known_) {
        CameraInfo info;
        info.id  = id;
        info.url = url;
        auto it = active_.find(id);
        info.running = (it != active_.end());
        if (info.running) info.stats = it->second->stats();
        out.push_back(std::move(info));
    }
    return out;
}

// ---- lifecycle -------------------------------------------------------------

void IngestServer::start()
{
    grpc::ChannelArguments args;
    args.SetMaxSendMessageSize(50 * 1024 * 1024);
    args.SetMaxReceiveMessageSize(50 * 1024 * 1024);

    // Launch-time config (Bucket A): inference endpoint from the environment,
    // rendered by control_service/lifecycle.py; defaults to the local server.
    const char* inf = std::getenv("INFERENCE_ADDR");
    const std::string inference_addr = inf ? inf : "localhost:50051";
    channel_ = grpc::CreateCustomChannel(inference_addr,
                                         grpc::InsecureChannelCredentials(), args);

    fs_ = createShm();
    checkTransport(fs_, frameTransport());   // abort on gRPC/shm mismatch with the peers

    // Seed the configured cameras through the same runtime path used by the admin RPC.
    for (const auto& cam : LoadCameras("config.yaml")) {
        std::string err = addCamera(cam.id, cam.url);
        if (err.empty())
            std::cout << "Started streaming: " << cam.id << cam.url << std::endl;
        else
            std::cerr << "config camera " << cam.id << ": " << err << "\n";
    }

    // Runtime control plane for the control_service.
    const char* a = std::getenv("INGEST_ADMIN_ADDR");
    const std::string admin_addr = a ? a : "127.0.0.1:50053";
    admin_service_ = std::make_unique<IngestAdminService>(*this);
    grpc::ServerBuilder builder;
    builder.AddListeningPort(admin_addr, grpc::InsecureServerCredentials());
    builder.RegisterService(admin_service_.get());
    admin_server_ = builder.BuildAndStart();
    if (admin_server_)
        std::cout << "[admin] IngestAdmin listening on " << admin_addr << std::endl;
    else
        std::cerr << "[admin] failed to bind " << admin_addr << "\n";
}

void IngestServer::stop()
{
    if (admin_server_) {
        admin_server_->Shutdown();
        admin_server_.reset();
    }
    // Move readers out under the lock, join them outside it.
    std::vector<std::unique_ptr<RtspReader>> readers;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        for (auto& [id, r] : active_)
            readers.push_back(std::move(r));
        active_.clear();
    }
    for (auto& r : readers)
        if (r) r->stop();
}
