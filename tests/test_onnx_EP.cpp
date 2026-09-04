// End-to-end NMS benchmark: CPU NMS (raw model) vs GPU NMS (EfficientNMS model).
//
//   ./onnx_test [--raw p] [--nms p] [--image p] [--warmup n] [--iters n]
//   defaults:   --raw models/best.onnx   --nms models/best_nms.onnx
//               --image tests/frame_666.jpg   --warmup 20   --iters 300
//
// Times the whole infer() path (preprocess -> session.Run -> NMS) for:
//   * raw.onnx + CUDA EP -> CPU NMS   (postprocess: decode + greedy NMS on CPU)
//   * raw.onnx + TRT  EP -> CPU NMS
//   * nms.onnx + TRT  EP -> GPU NMS   (EfficientNMS runs inside the TensorRT engine)
//
// Run from the repo root (relative model/image paths) and through the runtime
// wrapper that puts the sandboxed CUDA/cuDNN/TensorRT libs on LD_LIBRARY_PATH.

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "detector_onnx.hpp"

// Benchmarks one EP. Any failure (provider init, model load, inference) is caught
// and reported; returns false so the caller can move on to the next EP.
static bool benchmark(const char* name, ExecutionProvider ep,
                      const std::string& model, const cv::Mat& frame,
                      int warmup, int iters)
{
    std::cout << "\n=== " << name << " ===\n";

    try {
        OnnxDetector det(ep);
        if (!det.load_model(model)) {
            std::cerr << "  [skip] " << name << ": load_model returned false\n";
            return false;
        }

        // Warm-up hides the TensorRT engine build + cache warm and allocator
        // settling, so the timed loop measures steady-state inference only.
        size_t ndet = 0;
        for (int i = 0; i < warmup; ++i) ndet = det.infer(frame).size();

        std::vector<double> ms;
        ms.reserve(iters);
        for (int i = 0; i < iters; ++i) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            const auto dets = det.infer(frame);
            const auto t1 = std::chrono::high_resolution_clock::now();
            ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            ndet = dets.size();
        }

        std::sort(ms.begin(), ms.end());
        const double mean = std::accumulate(ms.begin(), ms.end(), 0.0) / ms.size();
        const auto pct = [&](double p) {
            return ms[std::min(ms.size() - 1, static_cast<size_t>(p * ms.size()))];
        };

        std::cout << std::fixed << std::setprecision(2)
                  << "  detections=" << ndet
                  << "  mean=" << mean << " ms"
                  << "  p50=" << pct(0.50)
                  << "  p90=" << pct(0.90)
                  << "  p99=" << pct(0.99)
                  << "  min=" << ms.front()
                  << "  fps=" << (1000.0 / mean) << "\n";
        return true;
    } catch (const std::exception& e) {
        std::cerr << "  [skip] " << name << ": " << e.what() << "\n";
        return false;
    }
}

static void usage(const char* prog)
{
    std::cout <<
        "usage: " << prog << " [options]\n"
        "  --raw <path>     raw model, CPU NMS           [models/best.onnx]\n"
        "  --nms <path>     EfficientNMS model, GPU NMS  [models/best_nms.onnx]\n"
        "  --image <path>   test image                   [tests/frame_666.jpg]\n"
        "  --warmup <n>     warm-up iterations           [20]\n"
        "  --iters <n>      timed iterations             [300]\n"
        "  -h, --help       show this help\n";
}

int main(int argc, char* argv[])
{
    std::string raw_model = "models/best.onnx";      // [1,8,8400] -> CPU NMS
    std::string nms_model = "models/best_nms.onnx";  // EfficientNMS -> GPU NMS
    std::string image     = "tests/frame_666.jpg";
    int         warmup    = 20;
    int         iters     = 300;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) { std::cerr << flag << " needs a value\n"; std::exit(2); }
            return argv[++i];
        };
        if      (a == "--raw")    raw_model = value("--raw");
        else if (a == "--nms")    nms_model = value("--nms");
        else if (a == "--image")  image     = value("--image");
        else if (a == "--warmup") warmup    = std::stoi(value("--warmup"));
        else if (a == "--iters")  iters     = std::stoi(value("--iters"));
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { std::cerr << "unknown argument: " << a << "\n"; usage(argv[0]); return 2; }
    }

    cv::Mat frame = cv::imread(image);
    if (frame.empty()) {
        std::cerr << "Failed to load image: " << image << "\n";
        return 1;
    }

    std::cout << "raw model: " << raw_model << "  nms model: " << nms_model << "\n"
              << "image: " << image << " (" << frame.cols << "x" << frame.rows << ")"
              << "  warmup=" << warmup << "  iters=" << iters << "\n";

    bool any = false;
    // CPU NMS: raw.onnx emits [1,8,8400]; postprocess() decodes + runs greedy NMS on the CPU.
    any |= benchmark("CUDA EP + CPU-NMS (raw)", ExecutionProvider::CUDA,     raw_model, frame, warmup, iters);
    any |= benchmark("TRT  EP + CPU-NMS (raw)", ExecutionProvider::TensorRT, raw_model, frame, warmup, iters);
    // GPU NMS: nms.onnx runs EfficientNMS inside the TensorRT engine.
    any |= benchmark("TRT  EP + GPU-NMS (nms)", ExecutionProvider::TensorRT, nms_model, frame, warmup, iters);

    if (!any) {
        std::cerr << "\nAll benchmarks failed.\n";
        return 1;
    }
    return 0;
}
