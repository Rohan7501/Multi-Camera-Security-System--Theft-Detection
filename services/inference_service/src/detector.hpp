// #pragma once
// #include "../common/types.hpp"
// #include <vector>

// class IDetector {
// public:
//   virtual ~IDetector() = default;
//   virtual std::vector<Detection> detect(const Frame& frame) = 0;
// };

#pragma once
#include "types.hpp"

class Detector
{
public:

    virtual ~Detector() = default;

    virtual bool load_model(const std::string& path) = 0;

    virtual std::vector<Detection> infer(const cv::Mat& frame) = 0;
};