// ============================================================================
// test_scan_session.cpp — ScanSessionData 契约测试（C-T10 装表/pushFrame 入环/丢帧）
//
// 档表经仓库载荷自建（rectify/planeMap 各 3 档 25.0/25.2/25.4，矩阵首元素作档
// 身份标记）——06 不跨模块取 07 test_scan_enricher 夹具。
// 降级语义断言依据 FrameEnricher.h 契约：单表空→warning 且 success 仍 true、
// 两表全空→fail（Result 字段 success/qualityFlag 判定）。
// ============================================================================

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "CalibrationRepository.h"
#include "ScanSessionData.h"

#include <chrono>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

using namespace Scanner::data;

namespace {

std::string tmpPath(const char* name) {
    return (fs::temp_directory_path() / name).string();
}

nlohmann::json mat3(double v) {
    return nlohmann::json::array({nlohmann::json::array({v, 0, 0}),
                                  nlohmann::json::array({0, v, 0}),
                                  nlohmann::json::array({0, 0, 1})});
}

nlohmann::json mat34(double v) {
    return nlohmann::json::array({nlohmann::json::array({v, 0, 0, 0}),
                                  nlohmann::json::array({0, v, 0, 0}),
                                  nlohmann::json::array({0, 0, v, 0})});
}

nlohmann::json mat44(double v) {
    return nlohmann::json::array({nlohmann::json::array({v, 0, 0, 0}),
                                  nlohmann::json::array({0, v, 0, 0}),
                                  nlohmann::json::array({0, 0, v, 0}),
                                  nlohmann::json::array({0, 0, 0, 1})});
}

// rectify 单档（tempC＋五矩阵；矩阵首元素 mark 作档身份标记）
nlohmann::json rectifyTier(double tempC, double mark) {
    return {{"temperature", tempC}, {"R1", mat3(mark)}, {"R2", mat3(mark)},
            {"P1", mat34(mark)}, {"P2", mat34(mark)}, {"Q", mat44(mark)}};
}

// planeMap 单档（档温＋参数原文占位）
nlohmann::json pmTier(double tempC) {
    return {{"temperature", tempC}, {"deltaT", 0.0}, {"totalPairs", 10}};
}

// 载荷：stereo 最小合法＋rectify/planeMap 各 n 档（25.0 起步 0.2；mark=100+i*100）
nlohmann::json tieredPayload(int rectifyN, int pmN) {
    nlohmann::json rectifyTable = nlohmann::json::array();
    for (int i = 0; i < rectifyN; ++i)
        rectifyTable.push_back(rectifyTier(25.0 + i * 0.2, 100.0 + i * 100));
    nlohmann::json pmTable = nlohmann::json::array();
    for (int i = 0; i < pmN; ++i)
        pmTable.push_back(pmTier(25.0 + i * 0.2));
    nlohmann::json j = {
        {"stereo",
         {{"cameraMatrixL", mat3(1000)},
          {"distCoeffsL", nlohmann::json::array({nlohmann::json::array({0, 0, 0, 0, 0})})},
          {"cameraMatrixR", mat3(1001)},
          {"distCoeffsR", nlohmann::json::array({nlohmann::json::array({0, 0, 0, 0, 0})})},
          {"R", mat3(1)},
          {"T", nlohmann::json::array({nlohmann::json::array({10}),
                                       nlohmann::json::array({0}),
                                       nlohmann::json::array({0})})},
          {"R1", mat3(1)}, {"R2", mat3(1)},
          {"P1", mat34(1)}, {"P2", mat34(1)}, {"Q", mat44(1)},
          {"reprojError", 0.5}}},
        {"tempTables",
         {{"intrinsic", {{"success", true}, {"table", nlohmann::json::array()}}},
          {"extrinsic", {{"success", true}, {"table", nlohmann::json::array()}}},
          {"rectify", {{"success", true}, {"table", rectifyTable}}},
          {"laserExtrinsic", {{"success", true}, {"table", nlohmann::json::array()}}}}},
        {"pjc", {{"success", true}}},
        {"planeMap",
         {{"map", {{"success", true}}},
          {"tempTable", {{"success", true}, {"table", pmTable}}}}},
        {"quality", {{"ok", true}, {"items", nlohmann::json::array()}}}};
    return j;
}

bool writeRepo(CalibrationRepository& repo, int rectifyN, int pmN, const char* tmpName) {
    return repo.write(tieredPayload(rectifyN, pmN).dump(), cv::Size(2560, 1440),
                      tmpPath(tmpName)).success;
}

} // namespace

// T10-1：装配——rectify/planeMap 各一档 → assemble ok 且槽位 16（构造期定死）
TEST(ScanSession, AssembleLoadsTables) {
    CalibrationRepository repo;
    ASSERT_TRUE(writeRepo(repo, 1, 1, "t10_assemble.json"));
    ScanSessionData s;                                  // 默认 16 槽
    EXPECT_TRUE(s.assemble(repo).success);
    EXPECT_EQ(s.ringCapacity(), 16u);
    fs::remove(tmpPath("t10_assemble.json"));
}

// T10-2：pushFrame 出口查表入环——两帧不同温度命中档 0/档 2，
//        回读 frameId/帧温/双表档索引/立体档身份标记一致，无丢帧
TEST(ScanSession, PushFrameEnrichesIntoRing) {
    CalibrationRepository repo;
    ASSERT_TRUE(writeRepo(repo, 3, 3, "t10_push.json"));
    ScanSessionData s;
    ASSERT_TRUE(s.assemble(repo).success);

    const cv::Mat gray(2, 2, CV_8UC1, cv::Scalar(7));
    EXPECT_TRUE(s.pushFrame(gray, gray, 25.0, 0).success);
    EXPECT_TRUE(s.pushFrame(gray, gray, 25.4, 1).success);

    auto& ring = s.ring();
    ASSERT_TRUE(ring.waitFor(0, std::chrono::milliseconds(1000)));
    ASSERT_TRUE(ring.waitFor(1, std::chrono::milliseconds(1000)));
    auto f0 = ring.read(0);
    auto f1 = ring.read(1);
    ASSERT_NE(f0, nullptr);
    ASSERT_NE(f1, nullptr);
    EXPECT_EQ(f0->frameId, 0u);
    EXPECT_EQ(f0->snapshot.stereoTier, 0);
    EXPECT_EQ(f0->snapshot.laserTier, 0);
    EXPECT_DOUBLE_EQ(f0->snapshot.R1(0, 0), 100.0);   // 档 0 身份标记
    EXPECT_DOUBLE_EQ(f0->temperature, 25.0);
    EXPECT_EQ(f1->frameId, 1u);
    EXPECT_EQ(f1->snapshot.stereoTier, 2);
    EXPECT_EQ(f1->snapshot.laserTier, 2);
    EXPECT_DOUBLE_EQ(f1->snapshot.R1(0, 0), 300.0);   // 档 2 身份标记
    EXPECT_DOUBLE_EQ(f1->temperature, 25.4);
    EXPECT_EQ(s.droppedFrames(), 0u);
    fs::remove(tmpPath("t10_push.json"));
}

// T10-3：两表全空——assemble fail「标定表未就绪」；pushFrame enrich fail →
//        丢帧不写环＋droppedFrames 计数＋Result 透传非 success
TEST(ScanSession, EnrichFailDropsFrame) {
    CalibrationRepository repo;
    ASSERT_TRUE(writeRepo(repo, 0, 0, "t10_drop.json"));   // 两表键在、档空
    ScanSessionData s;
    EXPECT_FALSE(s.assemble(repo).success);

    const cv::Mat gray(2, 2, CV_8UC1, cv::Scalar(7));
    const auto r = s.pushFrame(gray, gray, 25.0, 0);
    EXPECT_FALSE(r.success);                           // enrich fail 透传
    EXPECT_TRUE(r.isFault());
    EXPECT_EQ(s.droppedFrames(), 1u);
    EXPECT_EQ(s.ring().writePtr(), 0u);                // 未写环
    EXPECT_EQ(s.ring().read(0), nullptr);
    fs::remove(tmpPath("t10_drop.json"));
}

// T10-4：单表空降级——rectify 有档/planeMap 空 → assemble ok；pushFrame 返
//        warning（success 仍 true）且帧入环（laserTier=-1 自证降级），不计数
TEST(ScanSession, SingleEmptyTableWarningFrameWritten) {
    CalibrationRepository repo;
    ASSERT_TRUE(writeRepo(repo, 1, 0, "t10_warn.json"));
    ScanSessionData s;
    ASSERT_TRUE(s.assemble(repo).success);             // 单表空装配放行
    const cv::Mat gray(2, 2, CV_8UC1, cv::Scalar(7));
    const auto r = s.pushFrame(gray, gray, 25.0, 0);
    EXPECT_TRUE(r.success);                            // warning＝success 仍 true
    EXPECT_TRUE(r.hasWarning());
    ASSERT_TRUE(s.ring().waitFor(0, std::chrono::milliseconds(1000)));
    auto f = s.ring().read(0);
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f->snapshot.stereoTier, 0);
    EXPECT_EQ(f->snapshot.laserTier, -1);              // 空表档索引自证降级
    EXPECT_EQ(s.droppedFrames(), 0u);                  // warning 不算丢帧
    fs::remove(tmpPath("t10_warn.json"));
}
