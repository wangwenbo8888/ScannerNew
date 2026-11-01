/**
 * @file test_laser_match_scan_cuda.cpp
 * @brief 婵€鍏夌嚎鍖归厤鎵弿CUDA绠楀瓙鍗曞厓娴嬭瘯
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <nlohmann/json.hpp>
#include <opencv2/core/cuda.hpp>
#include "laser_match_scan_cuda.h"

using namespace calib;

#include <cstdio>
#include <fstream>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>

using namespace calib;

class LaserMatchScanTest : public ::testing::Test {
protected:
    void SetUp() override {
        LaserMatchScanParams params;
        params.match_threshold = 1.0f;
        params.epipolar_row_step = 0.5f;
        matcher_.reset(new LaserMatchScanCuda(params));
    }

    void TearDown() override {
        matcher_.reset();
    }

    std::unique_ptr<LaserMatchScanCuda> matcher_;
    cv::cuda::Stream stream_;
};

static std::string createTempMapJson(
    const std::vector<std::array<float,4>>& mapData,
    double temperature = 25.0)
{
    char tmpPath[L_tmpnam];
    tmpnam(tmpPath);
    std::string path = std::string(tmpPath) + ".json";

    nlohmann::json j;
    j["table"] = nlohmann::json::array();
    nlohmann::json entry;
    entry["temperature"] = temperature;
    nlohmann::json mapArr = nlohmann::json::array();
    for (const auto& row : mapData) {
        mapArr.push_back({row[0], row[1], row[2], row[3]});
    }
    entry["mapData"] = mapArr;
    j["table"].push_back(entry);

    std::ofstream ofs(path);
    ofs << j.dump(2);
    ofs.close();
    return path;
}

TEST_F(LaserMatchScanTest, EmptyInput) {
    cv::cuda::GpuMat d_empty_left, d_empty_left_ids;
    cv::cuda::GpuMat d_empty_right, d_empty_right_ids;

    auto result = matcher_->Execute(
        d_empty_left, d_empty_left_ids,
        d_empty_right, d_empty_right_ids, stream_);
    stream_.waitForCompletion();

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.matchedCount, 0);
}

TEST_F(LaserMatchScanTest, PerfectMatch) {
    std::vector<std::array<float,4>> mapData = {
        {100.0f, 10.0f, 200.0f, 0.0f},
    };
    std::string jsonPath = createTempMapJson(mapData);

    ASSERT_TRUE(matcher_->LoadTempTable(jsonPath));
    matcher_->SetCurrentTemperature(25.0);

    std::vector<cv::Point2f> leftPts = {{100.0f, 10.0f}};
    std::vector<int> leftLids = {0};
    std::vector<cv::Point2f> rightPts = {{200.0f, 10.0f}};
    std::vector<int> rightLids = {0};

    cv::Mat h_lp(1, 1, CV_32FC2, leftPts.data());
    cv::Mat h_ll(1, 1, CV_32SC1, leftLids.data());
    cv::Mat h_rp(1, 1, CV_32FC2, rightPts.data());
    cv::Mat h_rl(1, 1, CV_32SC1, rightLids.data());

    cv::cuda::GpuMat d_lp, d_ll, d_rp, d_rl;
    d_lp.upload(h_lp); d_ll.upload(h_ll);
    d_rp.upload(h_rp); d_rl.upload(h_rl);

    auto result = matcher_->Execute(d_lp, d_ll, d_rp, d_rl, stream_);
    stream_.waitForCompletion();

    EXPECT_TRUE(result.success) << result.message;
    EXPECT_EQ(result.matchedCount, 1);

    std::remove(jsonPath.c_str());
}

TEST_F(LaserMatchScanTest, AmbiguousMatch_TwoCandidates) {
    std::vector<std::array<float,4>> mapData = {
        {100.0f, 10.0f, 200.0f, 0.0f},
    };
    std::string jsonPath = createTempMapJson(mapData);
    ASSERT_TRUE(matcher_->LoadTempTable(jsonPath));
    matcher_->SetCurrentTemperature(25.0);

    std::vector<cv::Point2f> leftPts = {{100.0f, 10.0f}};
    std::vector<int> leftLids = {0};
    // Two candidates: 199 and 201, both within threshold=1.0 of uR_expected=200
    std::vector<cv::Point2f> rightPts = {{199.0f, 10.0f}, {201.0f, 10.0f}};
    std::vector<int> rightLids = {0, 0};

    cv::Mat h_lp(1, 1, CV_32FC2, leftPts.data());
    cv::Mat h_ll(1, 1, CV_32SC1, leftLids.data());
    cv::Mat h_rp(1, 2, CV_32FC2, rightPts.data());
    cv::Mat h_rl(1, 2, CV_32SC1, rightLids.data());

    cv::cuda::GpuMat d_lp, d_ll, d_rp, d_rl;
    d_lp.upload(h_lp); d_ll.upload(h_ll);
    d_rp.upload(h_rp); d_rl.upload(h_rl);

    auto result = matcher_->Execute(d_lp, d_ll, d_rp, d_rl, stream_);
    stream_.waitForCompletion();

    EXPECT_TRUE(result.success) << result.message;
    EXPECT_EQ(result.matchedCount, 0);
    EXPECT_EQ(result.excludedLeftCount, 1);
    EXPECT_EQ(result.excludedRightCount, 2);

    std::remove(jsonPath.c_str());
}

TEST_F(LaserMatchScanTest, AlreadyOccupied) {
    // Two map entries: both predict uR=200 for different uL values
    std::vector<std::array<float,4>> mapData = {
        {99.0f,  10.0f, 200.0f, 0.0f},
        {101.0f, 10.0f, 200.0f, 0.0f},
    };
    std::string jsonPath = createTempMapJson(mapData);
    ASSERT_TRUE(matcher_->LoadTempTable(jsonPath));
    matcher_->SetCurrentTemperature(25.0);

    // Two left points: uL=99 and uL=101, both at v=10, both lineId=0
    // After sorting by uL: 99 comes first 鈫?matches right point (200, 10)
    // Then uL=101: uR_expected=200, but right point already occupied 鈫?excluded
    std::vector<cv::Point2f> leftPts = {{99.0f, 10.0f}, {101.0f, 10.0f}};
    std::vector<int> leftLids = {0, 0};
    std::vector<cv::Point2f> rightPts = {{200.0f, 10.0f}};
    std::vector<int> rightLids = {0};

    cv::Mat h_lp(1, 2, CV_32FC2, leftPts.data());
    cv::Mat h_ll(1, 2, CV_32SC1, leftLids.data());
    cv::Mat h_rp(1, 1, CV_32FC2, rightPts.data());
    cv::Mat h_rl(1, 1, CV_32SC1, rightLids.data());

    cv::cuda::GpuMat d_lp, d_ll, d_rp, d_rl;
    d_lp.upload(h_lp); d_ll.upload(h_ll);
    d_rp.upload(h_rp); d_rl.upload(h_rl);

    auto result = matcher_->Execute(d_lp, d_ll, d_rp, d_rl, stream_);
    stream_.waitForCompletion();

    EXPECT_TRUE(result.success) << result.message;
    EXPECT_EQ(result.matchedCount, 1);
    EXPECT_EQ(result.excludedLeftCount, 1);

    std::remove(jsonPath.c_str());
}

TEST_F(LaserMatchScanTest, NoMatch_NoRightCandidate) {
    std::vector<std::array<float,4>> mapData = {
        {100.0f, 10.0f, 200.0f, 0.0f},
    };
    std::string jsonPath = createTempMapJson(mapData);
    ASSERT_TRUE(matcher_->LoadTempTable(jsonPath));
    matcher_->SetCurrentTemperature(25.0);

    // Left point: lineId=0, uR_expected=200
    std::vector<cv::Point2f> leftPts = {{100.0f, 10.0f}};
    std::vector<int> leftLids = {0};
    // Right point exists but at lineId=1 (different line)
    std::vector<cv::Point2f> rightPts = {{200.0f, 10.0f}};
    std::vector<int> rightLids = {1};

    cv::Mat h_lp(1, 1, CV_32FC2, leftPts.data());
    cv::Mat h_ll(1, 1, CV_32SC1, leftLids.data());
    cv::Mat h_rp(1, 1, CV_32FC2, rightPts.data());
    cv::Mat h_rl(1, 1, CV_32SC1, rightLids.data());

    cv::cuda::GpuMat d_lp, d_ll, d_rp, d_rl;
    d_lp.upload(h_lp); d_ll.upload(h_ll);
    d_rp.upload(h_rp); d_rl.upload(h_rl);

    auto result = matcher_->Execute(d_lp, d_ll, d_rp, d_rl, stream_);
    stream_.waitForCompletion();

    EXPECT_TRUE(result.success) << result.message;
    EXPECT_EQ(result.matchedCount, 0);
    EXPECT_EQ(result.excludedLeftCount, 1);

    std::remove(jsonPath.c_str());
}

TEST_F(LaserMatchScanTest, InvalidInput_WrongType) {
    cv::Mat h_wrong(1, 1, CV_32FC1);
    h_wrong.at<float>(0, 0) = 100.0f;
    cv::Mat h_lids(1, 1, CV_32SC1);
    h_lids.at<int>(0, 0) = 0;

    cv::cuda::GpuMat d_wrong, d_lids;
    d_wrong.upload(h_wrong);
    d_lids.upload(h_lids);

    auto result = matcher_->Execute(d_wrong, d_lids, d_wrong, d_lids, stream_);
    stream_.waitForCompletion();

    EXPECT_FALSE(result.success);
}

TEST_F(LaserMatchScanTest, MultiEpipolarRow) {
    // 3 rows, each with one matched pair
    std::vector<std::array<float,4>> mapData = {
        {100.0f, 10.0f, 200.0f, 0.0f},
        {110.0f, 20.0f, 210.0f, 0.0f},
        {120.0f, 30.0f, 220.0f, 0.0f},
    };
    std::string jsonPath = createTempMapJson(mapData);
    ASSERT_TRUE(matcher_->LoadTempTable(jsonPath));
    matcher_->SetCurrentTemperature(25.0);

    std::vector<cv::Point2f> leftPts = {
        {100.0f, 10.0f},
        {110.0f, 20.0f},
        {120.0f, 30.0f}
    };
    std::vector<int> leftLids = {0, 0, 0};
    std::vector<cv::Point2f> rightPts = {
        {200.0f, 10.0f},
        {210.0f, 20.0f},
        {220.0f, 30.0f}
    };
    std::vector<int> rightLids = {0, 0, 0};

    cv::Mat h_lp(1, 3, CV_32FC2, leftPts.data());
    cv::Mat h_ll(1, 3, CV_32SC1, leftLids.data());
    cv::Mat h_rp(1, 3, CV_32FC2, rightPts.data());
    cv::Mat h_rl(1, 3, CV_32SC1, rightLids.data());

    cv::cuda::GpuMat d_lp, d_ll, d_rp, d_rl;
    d_lp.upload(h_lp); d_ll.upload(h_ll);
    d_rp.upload(h_rp); d_rl.upload(h_rl);

    auto result = matcher_->Execute(d_lp, d_ll, d_rp, d_rl, stream_);
    stream_.waitForCompletion();

    EXPECT_TRUE(result.success) << result.message;
    EXPECT_EQ(result.matchedCount, 3);
    EXPECT_EQ(result.excludedLeftCount, 0);

    std::remove(jsonPath.c_str());
}

TEST_F(LaserMatchScanTest, TempTableLookup) {
    // Create multi-temperature JSON
    char tmpPath[L_tmpnam];
    tmpnam(tmpPath);
    std::string jsonPath = std::string(tmpPath) + ".json";

    nlohmann::json j;
    j["table"] = nlohmann::json::array();

    // Temp 20: uR=190
    {
        nlohmann::json entry;
        entry["temperature"] = 20.0;
        entry["mapData"] = nlohmann::json::array({{100.0f, 10.0f, 190.0f, 0.0f}});
        j["table"].push_back(entry);
    }
    // Temp 25: uR=200
    {
        nlohmann::json entry;
        entry["temperature"] = 25.0;
        entry["mapData"] = nlohmann::json::array({{100.0f, 10.0f, 200.0f, 0.0f}});
        j["table"].push_back(entry);
    }
    // Temp 30: uR=210
    {
        nlohmann::json entry;
        entry["temperature"] = 30.0;
        entry["mapData"] = nlohmann::json::array({{100.0f, 10.0f, 210.0f, 0.0f}});
        j["table"].push_back(entry);
    }

    std::ofstream(jsonPath) << j.dump(2);

    ASSERT_TRUE(matcher_->LoadTempTable(jsonPath));

    // Query temp=28 鈫?nearest is 30 (diff=2) 鈫?uR=210
    matcher_->SetCurrentTemperature(28.0);

    // Left point expects uR from temp=30 entry 鈫?uR=210
    std::vector<cv::Point2f> leftPts = {{100.0f, 10.0f}};
    std::vector<int> leftLids = {0};
    // Right point at uR=210 鈫?should match
    std::vector<cv::Point2f> rightPts = {{210.0f, 10.0f}};
    std::vector<int> rightLids = {0};

    cv::Mat h_lp(1, 1, CV_32FC2, leftPts.data());
    cv::Mat h_ll(1, 1, CV_32SC1, leftLids.data());
    cv::Mat h_rp(1, 1, CV_32FC2, rightPts.data());
    cv::Mat h_rl(1, 1, CV_32SC1, rightLids.data());

    cv::cuda::GpuMat d_lp, d_ll, d_rp, d_rl;
    d_lp.upload(h_lp); d_ll.upload(h_ll);
    d_rp.upload(h_rp); d_rl.upload(h_rl);

    auto result = matcher_->Execute(d_lp, d_ll, d_rp, d_rl, stream_);
    stream_.waitForCompletion();

    EXPECT_TRUE(result.success) << result.message;
    EXPECT_EQ(result.matchedCount, 1);

    std::remove(jsonPath.c_str());
}

TEST_F(LaserMatchScanTest, LargeScale) {
    const int numLines = 256;
    const int rowsPerLine = 50;

    std::vector<std::array<float,4>> mapData;
    for (int lid = 0; lid < numLines; ++lid) {
        for (int r = 0; r < rowsPerLine; ++r) {
            float vL = 10.0f + r * 0.5f;
            float uL = 100.0f + lid * 2.0f;
            float uR = 200.0f + lid * 2.0f;
            mapData.push_back({uL, vL, uR, static_cast<float>(lid)});
        }
    }
    std::string jsonPath = createTempMapJson(mapData);
    ASSERT_TRUE(matcher_->LoadTempTable(jsonPath));
    matcher_->SetCurrentTemperature(25.0);

    int totalPoints = numLines * rowsPerLine;
    std::vector<cv::Point2f> leftPts(totalPoints), rightPts(totalPoints);
    std::vector<int> leftLids(totalPoints), rightLids(totalPoints);

    for (int lid = 0; lid < numLines; ++lid) {
        for (int r = 0; r < rowsPerLine; ++r) {
            int idx = lid * rowsPerLine + r;
            float vL = 10.0f + r * 0.5f;
            float uL = 100.0f + lid * 2.0f;
            float uR = 200.0f + lid * 2.0f;
            leftPts[idx] = cv::Point2f(uL, vL);
            leftLids[idx] = lid;
            rightPts[idx] = cv::Point2f(uR, vL);
            rightLids[idx] = lid;
        }
    }

    cv::Mat h_lp(1, totalPoints, CV_32FC2, leftPts.data());
    cv::Mat h_ll(1, totalPoints, CV_32SC1, leftLids.data());
    cv::Mat h_rp(1, totalPoints, CV_32FC2, rightPts.data());
    cv::Mat h_rl(1, totalPoints, CV_32SC1, rightLids.data());

    cv::cuda::GpuMat d_lp, d_ll, d_rp, d_rl;
    d_lp.upload(h_lp); d_ll.upload(h_ll);
    d_rp.upload(h_rp); d_rl.upload(h_rl);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = matcher_->Execute(d_lp, d_ll, d_rp, d_rl, stream_);
    stream_.waitForCompletion();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    EXPECT_TRUE(result.success) << result.message;
    EXPECT_EQ(result.matchedCount, totalPoints);
    EXPECT_EQ(result.excludedLeftCount, 0);

    printf("  LargeScale: %d points matched in %.2f ms\n", totalPoints, ms);
    fflush(stdout);

    std::remove(jsonPath.c_str());
}

TEST_F(LaserMatchScanTest, VeryLargeScale) {
    const int numLines = 800;
    const int rowsPerLine = 50;

    std::vector<std::array<float,4>> mapData;
    for (int lid = 0; lid < numLines; ++lid) {
        for (int r = 0; r < rowsPerLine; ++r) {
            float vL = 10.0f + r * 0.5f;
            float uL = 100.0f + lid * 2.0f;
            float uR = 200.0f + lid * 2.0f;
            mapData.push_back({uL, vL, uR, static_cast<float>(lid)});
        }
    }
    std::string jsonPath = createTempMapJson(mapData);
    ASSERT_TRUE(matcher_->LoadTempTable(jsonPath));
    matcher_->SetCurrentTemperature(25.0);

    int totalPoints = numLines * rowsPerLine;
    std::vector<cv::Point2f> leftPts(totalPoints), rightPts(totalPoints);
    std::vector<int> leftLids(totalPoints), rightLids(totalPoints);

    for (int lid = 0; lid < numLines; ++lid) {
        for (int r = 0; r < rowsPerLine; ++r) {
            int idx = lid * rowsPerLine + r;
            float vL = 10.0f + r * 0.5f;
            float uL = 100.0f + lid * 2.0f;
            float uR = 200.0f + lid * 2.0f;
            leftPts[idx] = cv::Point2f(uL, vL);
            leftLids[idx] = lid;
            rightPts[idx] = cv::Point2f(uR, vL);
            rightLids[idx] = lid;
        }
    }

    cv::Mat h_lp(1, totalPoints, CV_32FC2, leftPts.data());
    cv::Mat h_ll(1, totalPoints, CV_32SC1, leftLids.data());
    cv::Mat h_rp(1, totalPoints, CV_32FC2, rightPts.data());
    cv::Mat h_rl(1, totalPoints, CV_32SC1, rightLids.data());

    cv::cuda::GpuMat d_lp, d_ll, d_rp, d_rl;
    d_lp.upload(h_lp); d_ll.upload(h_ll);
    d_rp.upload(h_rp); d_rl.upload(h_rl);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = matcher_->Execute(d_lp, d_ll, d_rp, d_rl, stream_);
    stream_.waitForCompletion();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    EXPECT_TRUE(result.success) << result.message;
    EXPECT_EQ(result.matchedCount, totalPoints);
    EXPECT_EQ(result.excludedLeftCount, 0);

    printf("  VeryLargeScale: %d points matched in %.2f ms\n", totalPoints, ms);
    fflush(stdout);

    std::remove(jsonPath.c_str());
}

TEST_F(LaserMatchScanTest, RealisticScale) {
    const int numLines = 17;
    const int ptsPerLine = 1300;
    const int totalPoints = numLines * ptsPerLine;

    std::vector<std::array<float,4>> mapData;
    for (int lid = 0; lid < numLines; ++lid) {
        for (int p = 0; p < ptsPerLine; ++p) {
            float vL = p * (999.5f / (ptsPerLine - 1));
            float uL = 100.0f + lid * 10.0f;
            float uR = 200.0f + lid * 10.0f;
            mapData.push_back({uL, vL, uR, static_cast<float>(lid)});
        }
    }
    std::string jsonPath = createTempMapJson(mapData);
    ASSERT_TRUE(matcher_->LoadTempTable(jsonPath));
    matcher_->SetCurrentTemperature(25.0);

    std::vector<cv::Point2f> leftPts(totalPoints), rightPts(totalPoints);
    std::vector<int> leftLids(totalPoints), rightLids(totalPoints);

    for (int lid = 0; lid < numLines; ++lid) {
        for (int p = 0; p < ptsPerLine; ++p) {
            int idx = lid * ptsPerLine + p;
            float vL = p * (999.5f / (ptsPerLine - 1));
            float uL = 100.0f + lid * 10.0f;
            float uR = 200.0f + lid * 10.0f;
            leftPts[idx] = cv::Point2f(uL, vL);
            leftLids[idx] = lid;
            rightPts[idx] = cv::Point2f(uR, vL);
            rightLids[idx] = lid;
        }
    }

    cv::Mat h_lp(1, totalPoints, CV_32FC2, leftPts.data());
    cv::Mat h_ll(1, totalPoints, CV_32SC1, leftLids.data());
    cv::Mat h_rp(1, totalPoints, CV_32FC2, rightPts.data());
    cv::Mat h_rl(1, totalPoints, CV_32SC1, rightLids.data());

    cv::cuda::GpuMat d_lp, d_ll, d_rp, d_rl;
    d_lp.upload(h_lp); d_ll.upload(h_ll);
    d_rp.upload(h_rp); d_rl.upload(h_rl);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = matcher_->Execute(d_lp, d_ll, d_rp, d_rl, stream_);
    stream_.waitForCompletion();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    EXPECT_TRUE(result.success) << result.message;
    EXPECT_EQ(result.matchedCount, totalPoints);
    EXPECT_EQ(result.excludedLeftCount, 0);

    printf("  RealisticScale: %d lines x %d pts = %d points, matched in %.2f ms\n",
           numLines, ptsPerLine, totalPoints, ms);
    fflush(stdout);

    std::remove(jsonPath.c_str());
}

// Scanning mode: input line_id is UNKNOWN (uniformly 0, as produced by steger
// Flat mode). The operator must search the table by epipolar row to determine
// which calibration line each point falls on. The calibration line_id becomes
// an OUTPUT of the lookup, not an input.
TEST_F(LaserMatchScanTest, ScanMode_LineIdIsOutput) {
    // Two calibration lines at DIFFERENT epipolar rows.
    //   line 0: xL=100, yL=10 (row 20), uR=200
    //   line 1: xL=150, yL=20 (row 40), uR=250
    std::vector<std::array<float,4>> mapData = {
        {100.0f, 10.0f, 200.0f, 0.0f},
        {150.0f, 20.0f, 250.0f, 1.0f},
    };
    std::string jsonPath = createTempMapJson(mapData);
    ASSERT_TRUE(matcher_->LoadTempTable(jsonPath));
    matcher_->SetCurrentTemperature(25.0);

    // Left points land exactly on the two table entries, but input line_id is
    // uniformly 0 (line_id unknown in scan mode).
    std::vector<cv::Point2f> leftPts = {{100.0f, 10.0f}, {150.0f, 20.0f}};
    std::vector<int> leftLids = {0, 0};
    std::vector<cv::Point2f> rightPts = {{200.0f, 10.0f}, {250.0f, 20.0f}};
    std::vector<int> rightLids = {0, 0};

    cv::Mat h_lp(1, 2, CV_32FC2, leftPts.data());
    cv::Mat h_ll(1, 2, CV_32SC1, leftLids.data());
    cv::Mat h_rp(1, 2, CV_32FC2, rightPts.data());
    cv::Mat h_rl(1, 2, CV_32SC1, rightLids.data());

    cv::cuda::GpuMat d_lp, d_ll, d_rp, d_rl;
    d_lp.upload(h_lp); d_ll.upload(h_ll);
    d_rp.upload(h_rp); d_rl.upload(h_rl);

    auto result = matcher_->Execute(d_lp, d_ll, d_rp, d_rl, stream_);
    stream_.waitForCompletion();

    EXPECT_TRUE(result.success) << result.message;
    ASSERT_EQ(result.matchedCount, 2);
    ASSERT_TRUE(result.d_matched_line_ids != nullptr);

    cv::Mat h_out_lids;
    result.d_matched_line_ids->download(h_out_lids);
    ASSERT_EQ(h_out_lids.total(), 2u);

    std::vector<int> outLids = {
        h_out_lids.at<int>(0, 0),
        h_out_lids.at<int>(0, 1),
    };
    std::sort(outLids.begin(), outLids.end());

    // The lookup determined each point's calibration line_id from the table by
    // row-search, even though the input line_id was uniformly 0. Under the old
    // line-id-based lookup both points would have searched line 0 only, missing
    // the line-1 point entirely (matchedCount would be 1).
    EXPECT_EQ(outLids[0], 0);
    EXPECT_EQ(outLids[1], 1);

    std::remove(jsonPath.c_str());
}
