/**
 * @file example_usage.cpp
 * @brief 激光线与标记点掩膜分离算子使用示例
 */

#include "laser_markingpoint_mask_separation_cuda.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    try {
        LaserMarkingSeparationParams params;
        params.gaussianSize = 5;
        params.threshold = 80;
        params.step2_erodeSize = 3;
        params.step2_dilateSize = 3;
        params.step3_erodeSize = 5;
        params.step3_dilateSize = 7;
        params.step4_erodeSize = 5;
        params.step4_dilateSize = 9;
        params.step6_dilateSize = 5;

        LaserMarkingSeparationCUDA separator(params);

        separator.Warmup(1080, 1920);
        std::cout << "GPU warmup completed" << std::endl;

        cv::Mat grayImage = cv::imread("test_image.png", cv::IMREAD_GRAYSCALE);
        if (grayImage.empty()) {
            std::cerr << "Failed to load test image, using synthetic image" << std::endl;

            grayImage = cv::Mat::zeros(480, 640, CV_8UC1);
            cv::rectangle(grayImage, cv::Point(10, 230), cv::Point(630, 250),
                          cv::Scalar(200), -1);
            cv::circle(grayImage, cv::Point(100, 100), 8, cv::Scalar(200), -1);
            cv::circle(grayImage, cv::Point(540, 100), 8, cv::Scalar(200), -1);
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
