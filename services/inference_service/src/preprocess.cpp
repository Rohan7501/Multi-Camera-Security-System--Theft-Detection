#include "preprocess.hpp"
#include <algorithm>
#include <cmath>
#define LOG_INFO(msg) std::cout << "[INFO] " << __FILE__ << ":" << __LINE__ << " " << msg << std::endl;
#define LOG_ERROR(msg) std::cerr << "[ERROR] " << __FILE__ << ":" << __LINE__ << " " << msg << std::endl;

// Target square size the model expects.
static constexpr int TARGET = 640;

// Padding colour for the letterbox bars. YOLOv8 is trained with gray (114),
// so we match that for best accuracy. Set to cv::Scalar(0,0,0) for true black.
static const cv::Scalar PAD_COLOR(114, 114, 114);

cv::Mat preprocess(const cv::Mat& frame)
{
    if (frame.empty()) {
        LOG_ERROR("Empty frame received in preprocess");
        return {};
    }

    const int w0 = frame.cols;
    const int h0 = frame.rows;

    // Uniform scale so the image fits inside TARGET x TARGET without distortion.
    // Clamp to 1.0: never upscale. Frames already <= TARGET on both dims keep
    // their pixels (r == 1, no resize); larger frames shrink to fit.
    // (postprocess must apply the identical clamp to invert this correctly.)
    const float r = std::min(1.0f, std::min(static_cast<float>(TARGET) / w0,
                                            static_cast<float>(TARGET) / h0));

    const int new_w = static_cast<int>(std::round(w0 * r));
    const int new_h = static_cast<int>(std::round(h0 * r));

    // Skip the resize entirely when the frame already fits (r == 1 => dims
    // unchanged): copying pixels through cv::resize would be wasted work.
    cv::Mat resized;
    if (new_w == w0 && new_h == h0) {
        resized = frame;
    } else {
        try {
            cv::resize(frame, resized, cv::Size(new_w, new_h));
        }
        catch (const cv::Exception& e) {
            LOG_ERROR("Resize failed: " << e.what());
            return {};
        }
    }

    // Pad the remaining space to reach TARGET x TARGET, centering the image.
    // (postprocess recomputes these same pad/scale values to undo the mapping.)
    // Only pad if a dimension falls short of TARGET; a frame that already fills
    // TARGET x TARGET needs no border.
    cv::Mat padded;
    if (new_w < TARGET || new_h < TARGET) {
        const int dw = TARGET - new_w;
        const int dh = TARGET - new_h;
        const int top    = dh / 2;
        const int bottom = dh - top;
        const int left   = dw / 2;
        const int right  = dw - left;

        cv::copyMakeBorder(resized, padded, top, bottom, left, right,
                           cv::BORDER_CONSTANT, PAD_COLOR);
    } else {
        padded = resized;
    }

    padded.convertTo(padded, CV_32F, 1.0 / 255);

    cv::cvtColor(padded, padded, cv::COLOR_BGR2RGB);

    return padded;
}
