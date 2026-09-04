// Map boxes from 640x640 letterboxed space back to original image pixels.
//
// Reconstructs the preprocess letterbox (uniform scale + centered padding),
// removes the padding, divides by the scale, clamps to bounds. Both detection
// paths converge here -- raw goes through nms() first, EfficientNMS arrives
// straight from the engine. orig_width/orig_height are the dimensions BEFORE
// the preprocess resize.

#pragma once
#include "types.hpp"
#include <vector>

std::vector<Detection> postprocess(std::vector<Detection> dets,
                                   int orig_width,
                                   int orig_height);
