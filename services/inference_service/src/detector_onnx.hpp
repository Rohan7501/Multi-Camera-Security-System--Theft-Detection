// #pragma once
// #include "detector.hpp"
// #include <string>

// class OnnxDetector : public IDetector {
// public:
//   explicit OnnxDetector(const std::string& model_path);
//   std::vector<Detection> detect(const Frame& frame) override;

// private:
//   std::string model_path_;
//   // TODO: Ort::Env, Ort::Session, allocator, input/output names
// };

#pragma once
#include "detector.hpp"
#include <onnxruntime_cxx_api.h>

// Which execution provider to register; CPU is the implicit last resort and ORT
// falls back per-op down the list. Auto/TensorRT = TRT -> CUDA -> CPU (the
// production default); CUDA = CUDA -> CPU; CPU = CPU only.
enum class ExecutionProvider { Auto, CUDA, TensorRT, CPU };

class OnnxDetector : public Detector
{

public:

    explicit OnnxDetector(ExecutionProvider ep = ExecutionProvider::Auto);

    bool load_model(const std::string& path) override;

    std::vector<Detection> infer(const cv::Mat& frame) override;

private:

    ExecutionProvider ep_;
    bool initialized_ = false;

    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::SessionOptions session_options_;

    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<std::string> input_name_strings_;
    std::vector<std::string> output_name_strings_;

    int input_width_ = 640;
    int input_height_ = 640;
};