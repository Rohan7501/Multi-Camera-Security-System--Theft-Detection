// Decode the raw YOLOv8 output tensor and run greedy per-class NMS.
//
// Input is [1, 4+NUM_CLASSES, 8400], channel-major. Returned boxes are xyxy in
// 640x640 LETTERBOXED space -- call postprocess() to get original pixels. Only
// the raw-model path uses this; the EfficientNMS model runs NMS inside the
// TensorRT engine and goes straight to postprocess().

#pragma once
#include "types.hpp"
#include <vector>

std::vector<Detection> nms(const std::vector<float>& raw);
