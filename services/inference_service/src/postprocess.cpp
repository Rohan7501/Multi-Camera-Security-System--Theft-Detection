#include "postprocess.hpp"
#include <algorithm>
#include <cmath>

namespace {
constexpr int INPUT_SIZE = 640;  // model input is 640x640 (must match preprocess.cpp)
} // namespace

std::vector<Detection> postprocess(std::vector<Detection> dets,
                                   int orig_width,
                                   int orig_height)
{
    // Reconstruct the letterbox transform applied in preprocess so we can invert
    // it: a uniform scale r plus centered padding. Must match preprocess exactly,
    // including the clamp to 1.0 (frames <= INPUT_SIZE are never upscaled, so r
    // stays 1 and pad_x/pad_y collapse to 0 when the frame fills the input).
    const float r = std::min(1.0f, std::min(static_cast<float>(INPUT_SIZE) / orig_width,
                                            static_cast<float>(INPUT_SIZE) / orig_height));
    const int new_w = static_cast<int>(std::round(orig_width  * r));
    const int new_h = static_cast<int>(std::round(orig_height * r));
    const int pad_x = (INPUT_SIZE - new_w) / 2;
    const int pad_y = (INPUT_SIZE - new_h) / 2;

    // Boxes arrive xyxy in 640 letterboxed space; remove the centered padding,
    // divide by the uniform scale, and clamp into the original image.
    for (auto& d : dets)
    {
        d.x1 = std::clamp(static_cast<int>(std::lround((d.x1 - pad_x) / r)), 0, orig_width  - 1);
        d.y1 = std::clamp(static_cast<int>(std::lround((d.y1 - pad_y) / r)), 0, orig_height - 1);
        d.x2 = std::clamp(static_cast<int>(std::lround((d.x2 - pad_x) / r)), 0, orig_width  - 1);
        d.y2 = std::clamp(static_cast<int>(std::lround((d.y2 - pad_y) / r)), 0, orig_height - 1);
    }

    return dets;
}
