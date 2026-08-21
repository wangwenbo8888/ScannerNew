// ============================================================================
// test_calib_session.cpp — CalibSessionData 契约测试（C-T11 配置装载/姿态环）
//
// 会话档格式（§8-2 定案，JSON 三键）：initialParams（07 PostureInitialParams
// 字段原文——K1,D1,K2,D2,R1,R2,P1,P2,Q＋imageWidth/imageHeight/maskRatioThreshold，
// 06 原文透传不解析）/ targets（每条 16 数）/ boardPoints（每点 [x,y,z]）。
// ============================================================================

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "CalibSessionData.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
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

// 样例会话档（initialParams 字段照 07 PosturePipeline.h PostureInitialParams：
// 九 Mat＋imageWidth/imageHeight/maskRatioThreshold；targets 测试 2 条、板点 3 个）
nlohmann::json sampleSessionDoc() {
    nlohmann::json targets = nlohmann::json::array();
    for (int t = 0; t < 2; ++t) {
        nlohmann::json row = nlohmann::json::array();
        for (int i = 0; i < 16; ++i) row.push_back(t * 100.0 + i);
        targets.push_back(row);
    }
    nlohmann::json board = nlohmann::json::array();
    for (int p = 0; p < 3; ++p)
        board.push_back(nlohmann::json::array({p * 1.0, p * 2.0, 0.0}));
    return {
        {"initialParams",
         {{"K1", mat3(1000)},
          {"D1", nlohmann::json::array({nlohmann::json::array({0, 0, 0, 0, 0})})},
          {"K2", mat3(1001)},
          {"D2", nlohmann::json::array({nlohmann::json::array({0, 0, 0, 0, 0})})},
          {"R1", mat3(1)}, {"R2", mat3(1)},
          {"P1", mat34(1)}, {"P2", mat34(1)}, {"Q", mat44(1)},
          {"imageWidth", 2560}, {"imageHeight", 1440}, {"maskRatioThreshold", 0.05}}},
        {"targets", targets},
        {"boardPoints", board}};
}

void writeTmp(const char* name, const std::string& content) {
    std::ofstream ofs(tmpPath(name), std::ios::binary | std::ios::trunc);
    ofs << content;
}

} // namespace

// T11-1：全字段装载——targets 2 条×16 元素、boardPoints 3 点、initialParams 原文含 K1
TEST(CalibSession, LoadAllFields) {
    writeTmp("t11_session.json", sampleSessionDoc().dump());
    CalibSessionData s;
    CalibSessionConfig cfg;
    ASSERT_TRUE(s.load(tmpPath("t11_session.json"), cfg).success);
    ASSERT_EQ(cfg.targets.size(), 2u);
    EXPECT_EQ(cfg.targets[0].size(), 16u);
    EXPECT_DOUBLE_EQ(cfg.targets[0][0], 0.0);
    EXPECT_DOUBLE_EQ(cfg.targets[1][15], 115.0);
    ASSERT_EQ(cfg.boardPoints.size(), 3u);
    EXPECT_FLOAT_EQ(cfg.boardPoints[1].x, 1.0f);
    EXPECT_FLOAT_EQ(cfg.boardPoints[1].y, 2.0f);
    ASSERT_TRUE(cfg.initialParams.contains("K1"));
    EXPECT_EQ(cfg.initialParams["imageWidth"], 2560);
    EXPECT_EQ(cfg.initialParams["maskRatioThreshold"], 0.05);
    fs::remove(tmpPath("t11_session.json"));
}

// T11-2：坏档/缺键/空档——垃圾不崩、缺 targets fail、空 targets/空 boardPoints fail；
//        initialParams 缺省容空对象（01 防言语义兜底）
TEST(CalibSession, BadFileFails) {
    CalibSessionData s;
    CalibSessionConfig cfg;

    writeTmp("t11_garbage.json", "{ not json");
    EXPECT_FALSE(s.load(tmpPath("t11_garbage.json"), cfg).success);      // 坏档不崩

    nlohmann::json doc = sampleSessionDoc();
    doc.erase("targets");
    writeTmp("t11_no_targets.json", doc.dump());
    EXPECT_FALSE(s.load(tmpPath("t11_no_targets.json"), cfg).success);   // 缺 targets

    doc = sampleSessionDoc();
    doc["targets"] = nlohmann::json::array();
    writeTmp("t11_empty_targets.json", doc.dump());
    EXPECT_FALSE(s.load(tmpPath("t11_empty_targets.json"), cfg).success);

    doc = sampleSessionDoc();
    doc["boardPoints"] = nlohmann::json::array();
    writeTmp("t11_empty_board.json", doc.dump());
    EXPECT_FALSE(s.load(tmpPath("t11_empty_board.json"), cfg).success);

    doc = sampleSessionDoc();
    doc.erase("initialParams");
    writeTmp("t11_no_params.json", doc.dump());
    EXPECT_TRUE(s.load(tmpPath("t11_no_params.json"), cfg).success);     // initialParams 可缺省
    EXPECT_TRUE(cfg.initialParams.is_object());
    EXPECT_TRUE(cfg.initialParams.empty());

    for (const char* n : {"t11_garbage.json", "t11_no_targets.json",
                          "t11_empty_targets.json", "t11_empty_board.json",
                          "t11_no_params.json"})
        fs::remove(tmpPath(n));
}

// T11-3：姿态环——cycleRing() 可写读回（SlotRing<CycleUnit> Backpressure 8 槽）
TEST(CalibSession, CycleRingAttached) {
    CalibSessionData s;
    auto& ring = s.cycleRing();
    auto unit = std::make_shared<CycleUnit>();
    unit->id = 5;
    unit->temperature = 26.5;
    unit->markerL = cv::Mat(2, 2, CV_8UC1, cv::Scalar(9));
    ring.write(unit);
    ASSERT_TRUE(ring.waitFor(0, std::chrono::milliseconds(1000)));
    auto back = ring.read(0);
    ASSERT_NE(back, nullptr);
    EXPECT_EQ(back->id, 5u);
    EXPECT_DOUBLE_EQ(back->temperature, 26.5);
    ring.done();                              // Backpressure 记账（单帧亦按约定 done）
}
