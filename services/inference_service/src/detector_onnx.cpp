// #include "detector_onnx.hpp"

// For MVP skeleton, we stub.
// You’ll replace with ONNX Runtime session + preprocess + postprocess.

// OnnxDetector::OnnxDetector(const std::string& model_path)
// : model_path_(model_path) {}

// std::vector<Detection> OnnxDetector::detect(const Frame& frame) {
//   (void)frame;
//   return {}; // TODO: implement
// }

#include "detector_onnx.hpp"
#include "preprocess.hpp"
#include "nms.hpp"
#include "postprocess.hpp"
#define LOG_INFO(msg) std::cout << "[INFO] " << __FILE__ << ":" << __LINE__ << " " << msg << std::endl;
#define LOG_ERROR(msg) std::cerr << "[ERROR] " << __FILE__ << ":" << __LINE__ << " " << msg << std::endl;



#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdint>

OnnxDetector::OnnxDetector(ExecutionProvider ep)
    // : env_(ORT_LOGGING_LEVEL_VERBOSE, "inference")
    : ep_(ep), env_(ORT_LOGGING_LEVEL_WARNING, "inference")
{

    // session_options_.SetIntraOpNumThreads(1);

    // session_options_.SetGraphOptimizationLevel(
    //     GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // OrtCUDAProviderOptions cuda_options;
    // session_options_.AppendExecutionProvider_CUDA(cuda_options);
}

// --- Execution-provider helpers ---------------------------------------------
static void append_cuda_ep(Ort::SessionOptions& opts)
{
    OrtCUDAProviderOptions cuda{};
    opts.AppendExecutionProvider_CUDA(cuda);
}

static void append_trt_ep(Ort::SessionOptions& opts)
{
    // TensorRT EP builds + caches an engine on first run (slow first inference).
    OrtTensorRTProviderOptionsV2* trt = nullptr;
    Ort::ThrowOnError(Ort::GetApi().CreateTensorRTProviderOptions(&trt));
    std::vector<const char*> k{"device_id","trt_fp16_enable","trt_engine_cache_enable","trt_engine_cache_path"};
    std::vector<const char*> v{"0","1","1","trt_cache"};
    Ort::ThrowOnError(Ort::GetApi().UpdateTensorRTProviderOptions(trt, k.data(), v.data(), k.size()));
    opts.AppendExecutionProvider_TensorRT_V2(*trt);
    Ort::GetApi().ReleaseTensorRTProviderOptions(trt);
}

bool OnnxDetector::load_model(const std::string& path)
{
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    try {
        switch (ep_) {
            case ExecutionProvider::CUDA:
                append_cuda_ep(opts);
                LOG_INFO("Provider: CUDA -> CPU");
                break;
            case ExecutionProvider::CPU:
                LOG_INFO("Provider: CPU");
                break;
            case ExecutionProvider::TensorRT:
            case ExecutionProvider::Auto:
            default:
                append_trt_ep(opts);   // highest priority
                append_cuda_ep(opts);  // fallback for ops TRT won't take
                LOG_INFO("Provider: TensorRT -> CUDA -> CPU");
                break;
        }

        session_ = Ort::Session(env_, path.c_str(), opts);

    } catch (const Ort::Exception& e) {
        // Only Auto silently degrades to CPU. An explicitly requested EP must fail
        // loudly so callers (e.g. the EP benchmark) can skip it instead of being
        // handed a CPU session mislabelled as CUDA/TensorRT.
        if (ep_ != ExecutionProvider::Auto) {
            LOG_ERROR("Provider init failed for requested EP: " << e.what());
            throw;
        }

        LOG_ERROR("Provider init failed: " << e.what() << " -- falling back to CPU");

        Ort::SessionOptions cpu_options;
        cpu_options.SetIntraOpNumThreads(1);
        cpu_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        session_ = Ort::Session(env_, path.c_str(), cpu_options);

        LOG_INFO("Using CPU execution provider");
    }

    auto type_info = session_.GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto input_dims = tensor_info.GetShape();
    // std::cout<<"Height: "<<input_dims[2]<<" Width: "<<input_dims[3]<<std::endl;
    
    initialized_ = true;
    Ort::AllocatorWithDefaultOptions allocator;

    // Collect every name string FIRST, then take .c_str(). Grabbing the pointer
    // inside the loop dangles the moment the vector reallocates -- see README.md.
    const size_t num_input_nodes = session_.GetInputCount();
    input_name_strings_.reserve(num_input_nodes);
    for (size_t i = 0; i < num_input_nodes; i++)
        input_name_strings_.push_back(session_.GetInputNameAllocated(i, allocator).get());

    const size_t num_output_nodes = session_.GetOutputCount();
    output_name_strings_.reserve(num_output_nodes);
    for (size_t i = 0; i < num_output_nodes; i++)
        output_name_strings_.push_back(session_.GetOutputNameAllocated(i, allocator).get());

    for (const auto& s : input_name_strings_)  input_names_.push_back(s.c_str());
    for (const auto& s : output_name_strings_) output_names_.push_back(s.c_str());

    std::cout << "ONNX model loaded successfully\n";

    return true;
}

static std::vector<float> mat_to_tensor(const cv::Mat& image)
{
    std::vector<cv::Mat> channels(3);
    cv::split(image, channels);

    std::vector<float> tensor;
    tensor.reserve(3 * image.rows * image.cols);

    for(int i=0;i<3;i++)
    {
        CV_Assert(channels[i].isContinuous());

        tensor.insert(
            tensor.end(),
            (float*)channels[i].datastart,
            (float*)channels[i].dataend
        );
    }

    return tensor;
}

std::vector<Detection> OnnxDetector::infer(const cv::Mat& frame)
{
    cv::Mat preprocessed = preprocess(frame);

    std::vector<float> tensor = mat_to_tensor(preprocessed);

    // std::array<int64_t,4> input_shape{1,3,input_height_,input_width_};
    // LOG_INFO("Frame: " << preprocessed.cols << "x" << preprocessed.rows);
    // LOG_INFO("Target: " << input_width_ << "x" << input_height_);
    std::array<int64_t,4> input_shape{1,3,640,640};

    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault
        );

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        tensor.data(),
        tensor.size(),
        input_shape.data(),
        input_shape.size()
    );

    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names_.data(),
        &input_tensor,
        1,
        output_names_.data(),
        output_names_.size()
    );

    // --- EfficientNMS model: 4 outputs (num_dets, det_boxes, det_scores, det_classes).
    // NMS already ran in the engine; det_boxes are xyxy in 640 letterboxed space. We
    // wrap them as Detections and hand them to postprocess() for the box conversion. ---
    if (output_tensors.size() >= 4)
    {
        auto idx = [&](const char* n) -> int {
            for (size_t i = 0; i < output_name_strings_.size(); ++i)
                if (output_name_strings_[i] == n) return static_cast<int>(i);
            return -1;
        };
        const int i_num = idx("num_dets"),   i_box = idx("det_boxes"),
                  i_scr = idx("det_scores"), i_cls = idx("det_classes");
        if (i_num < 0 || i_box < 0 || i_scr < 0 || i_cls < 0) {
            LOG_ERROR("EfficientNMS outputs not found by name");
            return {};
        }

        const int32_t* num_p  = output_tensors[i_num].GetTensorData<int32_t>();
        const float*   boxes  = output_tensors[i_box].GetTensorData<float>();   // [1, N, 4] xyxy @640
        const float*   scores = output_tensors[i_scr].GetTensorData<float>();   // [1, N]
        const int32_t* cls    = output_tensors[i_cls].GetTensorData<int32_t>(); // [1, N]

        const int cap = static_cast<int>(
            output_tensors[i_box].GetTensorTypeAndShapeInfo().GetElementCount() / 4);
        const int n = std::clamp(num_p[0], 0, cap);

        std::vector<Detection> dets;
        dets.reserve(n);
        for (int i = 0; i < n; ++i) {
            Detection d;
            d.class_id   = cls[i];
            d.confidence = scores[i];
            d.x1 = static_cast<int>(std::lround(boxes[i*4+0]));  // xyxy in 640 space
            d.y1 = static_cast<int>(std::lround(boxes[i*4+1]));
            d.x2 = static_cast<int>(std::lround(boxes[i*4+2]));
            d.y2 = static_cast<int>(std::lround(boxes[i*4+3]));
            dets.push_back(d);
        }
        return postprocess(dets, frame.cols, frame.rows);   // 640 -> original coords
    }

    // --- Raw YOLOv8 model: [1,8,8400] -> nms() (decode + greedy NMS) -> postprocess() ---
    float* raw_output = output_tensors[0].GetTensorMutableData<float>();
    size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> output_vector(raw_output, raw_output + output_size);
    return postprocess(nms(output_vector), frame.cols, frame.rows);
}