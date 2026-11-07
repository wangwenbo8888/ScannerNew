/**
 * @file example_usage.cpp
 * @brief 激光线与标记点掩膜分离算子使用示例（细激光形态）
 */

#include "laser_markingpoint_mask_separation_cuda.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    try {
        LaserMarkingSeparationParams params;   // 默认值即细激光形态调定值
        params.threshold = 80;
        params.noiseErodeSize = 3;
        params.laserKillSize = 9;
        params.markerKillSize = 37;
        params.laserEdgeDilateSize = 9;
        params.markingEdgeDilateSize = 7;

        LaserMarkingSeparationCUDA separator(params);

        separator.Warmup(1080, 1920);
        std::cout << "GPU warmup completed" << std::endl;

        cv::Mat grayImage = cv::imread("test_image.png", cv::IMREAD_GRAYSCALE);
        if (grayImage.empty()) {
            std::cerr << "Failed to load test image, using synthetic image" << std::endl;

            grayImage = cv::Mat::zeros(480, 640, CV_8UC1);
            // 竖直激光条 w=5
            cv::rectangle(grayImage, cv::Point(150, 80), cv::Point(154, 440),
                          cv::Scalar(200), -1);
            // 标记点圆斑 φ35
            cv::circle(grayImage, cv::Point(120, 400), 17, cv::Scalar(200), -1);
            cv::circle(grayImage, cv::Point(520, 400), 17, cv::Scalar(200), -1);
        }

        LaserMarkingSeparationResult result = separator.Execute(grayImage);

        if (result.success) {
            std::cout << "Separation successful!" << std::endl;

            cv::Mat laserMask, markingMask, combinedMask;
            result.d_laserMask->download(laserMask);
            result.d_markingPointMask->download(markingMask);
            result.d_combinedMask->download(combinedMask);

            cv::imwrite("laser_mask_output.png", laserMask);
            cv::imwrite("marking_point_mask_output.png", markingMask);
            cv::imwrite("combined_mask_output.png", combinedMask);
            std::cout << "Results saved" << std::endl;
        } else {
            std::cerr << "Separation failed: " << result.message << std::endl;
            return -1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
