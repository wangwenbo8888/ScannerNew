#include "frame_filter_cuda.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <gtest/gtest.h>

using calib::FrameFilterCUDA;
using calib::FrameFilterParams;

// 高占比 → 标记点帧通过
TEST(FrameFilterCUDA, MarkerFrameHighRatio) {
    cv::Mat mask = cv::Mat::zeros(100, 100, CV_8UC1);
    cv::rectangle(mask, cv::Rect(10, 10, 50, 50), 255, cv::FILLED);  // 2500 非零 → 0.25
    cv::cuda::GpuMat d_mask(mask);
    FrameFilterCUDA ff(FrameFilterParams{0.1});
    auto r = ff.Execute(d_mask);
    ASSERT_TRUE(r.success);
    EXPECT_TRUE(r.isMarkerFrame);
    EXPECT_NEAR(r.maskRatio, 0.25, 0.002);
}

// 低占比 → 激光线帧
TEST(FrameFilterCUDA, LaserLineFrameLowRatio) {
    cv::Mat mask = cv::Mat::zeros(100, 100, CV_8UC1);
    cv::line(mask, {0, 50}, {99, 50}, 255, 1);  // ~100 非零 → 0.01
    cv::cuda::GpuMat d_mask(mask);
    FrameFilterCUDA ff(FrameFilterParams{0.05});
    auto r = ff.Execute(d_mask);
    ASSERT_TRUE(r.success);
    EXPECT_FALSE(r.isMarkerFrame);
}

// shared_ptr 透传：Result.d_cleanedMask 与传入同一对象
TEST(FrameFilterCUDA, PassthroughShared) {
    auto d_mask = std::make_shared<cv::cuda::GpuMat>(100, 100, CV_8UC1);
    d_mask->setTo(255);
    FrameFilterCUDA ff;
    auto r = ff.Execute(d_mask);
    ASSERT_TRUE(r.success);
    EXPECT_EQ(r.d_cleanedMask.get(), d_mask.get());
}

// 默认阈值 0.0 → 全部通过(含全黑帧)
TEST(FrameFilterCUDA, DefaultThresholdPassAll) {
    cv::Mat mask = cv::Mat::zeros(100, 100, CV_8UC1);  // 全黑，占比 0
    cv::cuda::GpuMat d_mask(mask);
    FrameFilterCUDA ff;  // 默认 0.0
    auto r = ff.Execute(d_mask);
    ASSERT_TRUE(r.success);
    EXPECT_TRUE(r.isMarkerFrame);  // 0 >= 0.0
}

// 错误类型 → success=false
TEST(FrameFilterCUDA, WrongTypeFails) {
    cv::cuda::GpuMat d_mask(100, 100, CV_32FC1);
    FrameFilterCUDA ff;
    auto r = ff.Execute(d_mask);
    EXPECT_FALSE(r.success);
}
