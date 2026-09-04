#include "nms.hpp"
#include <algorithm>
#include <cmath>

namespace {

constexpr int   NUM_BOXES      = 8400;  // anchors in the [1, 8, 8400] output
constexpr int   NUM_CLASSES    = 4;     // 8 attributes - 4 box coords
constexpr float CONF_THRESHOLD = 0.25f; // min class score to keep a box
constexpr float IOU_THRESHOLD  = 0.45f; // overlap above which we suppress

// Intersection-over-Union of two xyxy boxes. IoU is invariant to the uniform
// scale + translation of the letterbox, so running NMS in 640 space gives the
// same result as running it in original-image space.
float iou(const Detection& a, const Detection& b)
{
    const int ix1 = std::max(a.x1, b.x1);
    const int iy1 = std::max(a.y1, b.y1);
    const int ix2 = std::min(a.x2, b.x2);
    const int iy2 = std::min(a.y2, b.y2);

    const int iw = std::max(0, ix2 - ix1);
    const int ih = std::max(0, iy2 - iy1);
    const float inter = static_cast<float>(iw) * static_cast<float>(ih);

    const float area_a = static_cast<float>(a.x2 - a.x1) * (a.y2 - a.y1);
    const float area_b = static_cast<float>(b.x2 - b.x1) * (b.y2 - b.y1);
    const float uni = area_a + area_b - inter;

    return uni > 0.0f ? inter / uni : 0.0f;
}

} // namespace

std::vector<Detection> nms(const std::vector<float>& raw)
{
    // Channel-major: attribute `a` of anchor `i` is raw[a * NUM_BOXES + i].
    // 0..3 = cx,cy,w,h (in 640 letterboxed space); 4..7 = per-class scores
    // (already sigmoid'd, no objectness).

    // --- 1. Decode + confidence filter (boxes stay in 640 space) ---
    std::vector<Detection> candidates;
    candidates.reserve(256);

    for (int i = 0; i < NUM_BOXES; ++i)
    {
        int   best_class = -1;
        float best_score = 0.0f;
        for (int c = 0; c < NUM_CLASSES; ++c)
        {
            const float score = raw[(4 + c) * NUM_BOXES + i];
            if (score > best_score)
            {
                best_score = score;
                best_class = c;
            }
        }

        if (best_score < CONF_THRESHOLD)
            continue;

        const float cx = raw[0 * NUM_BOXES + i];
        const float cy = raw[1 * NUM_BOXES + i];
        const float w  = raw[2 * NUM_BOXES + i];
        const float h  = raw[3 * NUM_BOXES + i];

        // center form -> corners, still in 640 letterboxed space. postprocess()
        // maps these back to the original image.
        Detection det;
        det.class_id   = best_class;
        det.confidence = best_score;
        det.x1 = static_cast<int>(std::lround(cx - w / 2.0f));
        det.y1 = static_cast<int>(std::lround(cy - h / 2.0f));
        det.x2 = static_cast<int>(std::lround(cx + w / 2.0f));
        det.y2 = static_cast<int>(std::lround(cy + h / 2.0f));

        candidates.push_back(det);
    }

    // --- 2. Sort by confidence, highest first ---
    std::sort(candidates.begin(), candidates.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    // --- 3. Greedy per-class NMS ---
    std::vector<Detection> result;
    std::vector<bool> suppressed(candidates.size(), false);

    for (size_t i = 0; i < candidates.size(); ++i)
    {
        if (suppressed[i])
            continue;

        result.push_back(candidates[i]);

        // suppress lower-confidence boxes of the SAME class that overlap too much
        for (size_t j = i + 1; j < candidates.size(); ++j)
        {
            if (suppressed[j])
                continue;
            if (candidates[j].class_id != candidates[i].class_id)
                continue;
            if (iou(candidates[i], candidates[j]) > IOU_THRESHOLD)
                suppressed[j] = true;
        }
    }

    return result;
}
