#pragma once
#include <cstdint>
#include <memory>
namespace Scanner {
// 占位类型（算子迁入后 leftGray/rightGray 改 std::shared_ptr<cv::Mat>）
struct Frame {
    std::shared_ptr<void> leftGray;
    std::shared_ptr<void> rightGray;
    uint64_t frameId = 0;
    uint64_t timestamp = 0;
    double temperature = 25.0;
};
struct FrameResult {
    uint64_t frameId = 0;
};
}  // namespace Scanner
