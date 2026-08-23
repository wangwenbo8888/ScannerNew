// ============================================================================
// test_calib_repository.cpp — 标定仓库契约测试（B-T6 write/load/原子落盘；
// T7 typed getters；T8 readyForScan 五样门禁）
//
// 载荷键名照抄 07 CalibSerialize.cpp（stereo/tempTables/pjc/planeMap/quality 五键）；
// Mat JSON＝二维嵌套数组 CV_64F 行主序（对齐 09 json_utils matToJson 格式），
// 形状对齐目标类型：P1/P2=3×4、Q=4×4、T=3×1——T7 matxFromJson 逐元素转换的前提。
// ============================================================================

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "CalibrationRepository.h"

#include <filesystem>
#include <fstream>
#include <sstream>
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

// 最小合法载荷（真实键名照抄 CalibSerialize.cpp:23-43,113-128）
nlohmann::json samplePayloadJson() {
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
          {"rectify", {{"success", true}, {"table", nlohmann::json::array({
               {{"temperature", 25.0}, {"R1", mat3(1)}, {"R2", mat3(1)},
                {"P1", mat34(1)}, {"P2", mat34(1)}, {"Q", mat44(1)}}})}}},
          {"laserExtrinsic", {{"success", true}, {"table", nlohmann::json::array()}}}}},
        {"pjc", {{"success", true}}},
        {"planeMap",
         {{"map", {{"success", true}}},
          {"tempTable", {{"success", true}, {"table", nlohmann::json::array({
               {{"temperature", 25.0}, {"deltaT", 0.0}, {"totalPairs", 10}}})}}}}},
        {"quality", {{"ok", true}, {"items", nlohmann::json::array()}}}};
    return j;
}

std::string samplePayload() { return samplePayloadJson().dump(); }

bool loadSample(CalibrationRepository& repo, const char* tmpName) {
    return repo.write(samplePayload(), cv::Size(2560, 1440), tmpPath(tmpName)).success;
}

} // namespace

// —— T6：write/load/lastPath/原子落盘 ——
TEST(CalibRepository, WriteLoadRoundtrip) {
    CalibrationRepository repo;
    const std::string path = tmpPath("t06_repo.json");
    ASSERT_TRUE(repo.write(samplePayload(), cv::Size(2560, 1440), path).success);
    EXPECT_EQ(repo.lastPath(), path);

    CalibrationRepository repo2;
    ASSERT_TRUE(repo2.load(path).success);
    EXPECT_EQ(repo2.lastPath(), path);
    EXPECT_EQ(repo2.stereo().imageSize, cv::Size(2560, 1440));
    EXPECT_EQ(repo2.stereoTempTable().tiers.size(), 1u);
    fs::remove(path);
}

TEST(CalibRepository, BadJsonFailsNotCrash) {
    CalibrationRepository repo;
    EXPECT_FALSE(repo.write("{ not json", cv::Size(1, 1)).success);
}

TEST(CalibRepository, LoadBadFileFailsNotCrash) {
    CalibrationRepository repo;
    EXPECT_FALSE(repo.load("Z:/no/such/calib.json").success);   // 打不开
    const auto p = tmpPath("t06_bad_repo.json");
    { std::ofstream ofs(p); ofs << "{ not json"; }
    EXPECT_FALSE(repo.load(p).success);                          // 坏档不崩
    fs::remove(p);
}

TEST(CalibRepository, AtomicWriteNoHalfFile) {
    CalibrationRepository repo;
    const std::string path = tmpPath("t06_atomic.json");
    ASSERT_TRUE(repo.write(samplePayload(), cv::Size(2560, 1440), path).success);
    // 二次写同路径＝走 remove+rename 覆盖分支（Windows rename 不覆盖既有）
    ASSERT_TRUE(repo.write(samplePayload(), cv::Size(2560, 1440), path).success);

    std::ifstream ifs(path, std::ios::binary);
    ASSERT_TRUE(ifs.is_open());
    std::stringstream ss;
    ss << ifs.rdbuf();
    nlohmann::json doc;
    ASSERT_NO_THROW(doc = nlohmann::json::parse(ss.str()));  // 半文件/坏档会抛
    EXPECT_TRUE(doc.contains("meta"));
    EXPECT_EQ(doc["meta"]["imageSize"]["width"], 2560);
    EXPECT_FALSE(fs::exists(path + ".tmp"));                 // tmp 已被改名消费
    ifs.close();
    fs::remove(path);
}

// —— T7：typed getters ——
TEST(CalibRepository, StereoGettersRoundtrip) {
    CalibrationRepository repo;
    ASSERT_TRUE(loadSample(repo, "t07_stereo.json"));

    const StereoData st = repo.stereo();
    EXPECT_EQ(st.cameraMatrixL.at<double>(0, 0), 1000);
    EXPECT_EQ(st.cameraMatrixR.at<double>(0, 0), 1001);
    EXPECT_EQ(st.R.at<double>(0, 0), 1);
    EXPECT_EQ(st.T.at<double>(0, 0), 10);
    ASSERT_EQ(st.P1.rows, 3);
    ASSERT_EQ(st.P1.cols, 4);
    EXPECT_EQ(st.P1.at<double>(0, 0), 1);
    ASSERT_EQ(st.Q.rows, 4);
    ASSERT_EQ(st.Q.cols, 4);
    EXPECT_EQ(st.imageSize, cv::Size(2560, 1440));
    fs::remove(tmpPath("t07_stereo.json"));
}

TEST(CalibRepository, StereoTempTableTiers) {
    CalibrationRepository repo;
    ASSERT_TRUE(loadSample(repo, "t07_rectify.json"));

    const StereoTempTable table = repo.stereoTempTable();
    ASSERT_EQ(table.tiers.size(), 1u);
    const StereoTempTier& tier = table.tiers[0];
    EXPECT_DOUBLE_EQ(tier.tempC, 25.0);
    EXPECT_DOUBLE_EQ(tier.R1(0, 0), 1);
    EXPECT_DOUBLE_EQ(tier.R2(0, 0), 1);
    EXPECT_DOUBLE_EQ(tier.P1(0, 0), 1);
    EXPECT_DOUBLE_EQ(tier.P2(0, 0), 1);
    EXPECT_DOUBLE_EQ(tier.Q(3, 3), 1);
    fs::remove(tmpPath("t07_rectify.json"));
}

TEST(CalibRepository, PlaneMapTiersAndRawSubtrees) {
    CalibrationRepository repo;
    ASSERT_TRUE(loadSample(repo, "t07_pm.json"));

    const PlaneMapTempTableRef ref = repo.planeMapTiers();
    ASSERT_EQ(ref.tiers.size(), 1u);
    EXPECT_DOUBLE_EQ(ref.tiers[0].tempC, 25.0);

    const nlohmann::json pmRaw = repo.planeMapTempTableRaw();
    ASSERT_TRUE(pmRaw.contains("table"));
    ASSERT_TRUE(pmRaw["table"].is_array() && !pmRaw["table"].empty());
    EXPECT_TRUE(pmRaw["table"][0].contains("deltaT"));

    const nlohmann::json ttRaw = repo.tempTablesRaw();
    EXPECT_TRUE(ttRaw.contains("intrinsic"));
    EXPECT_TRUE(ttRaw.contains("extrinsic"));
    EXPECT_TRUE(ttRaw.contains("rectify"));
    EXPECT_TRUE(ttRaw.contains("laserExtrinsic"));
    fs::remove(tmpPath("t07_pm.json"));
}

// —— T8：readyForScan 五样门禁（内参/外参/立体温度表/激光档表/imageSize）——
namespace {

bool missingHas(const ReadyReport& rep, const char* sub) {
    for (const auto& m : rep.missing) {
        if (m.find(sub) != std::string::npos) return true;
    }
    return false;
}

bool writeMutatedReadyCheck(const char* tmpName, void (*mutate)(nlohmann::json&),
                            ReadyReport& rep) {
    nlohmann::json j = samplePayloadJson();
    mutate(j);
    CalibrationRepository repo;
    if (!repo.write(j.dump(), cv::Size(2560, 1440), tmpPath(tmpName)).success) return false;
    repo.readyForScan(rep);
    fs::remove(tmpPath(tmpName));
    return true;
}

} // namespace

TEST(CalibRepository, ReadyForScanMissingIntrinsicL) {
    ReadyReport rep;
    ASSERT_TRUE(writeMutatedReadyCheck("t08_no_kl.json", [](nlohmann::json& j) {
        j["stereo"].erase("cameraMatrixL");
    }, rep));
    EXPECT_FALSE(rep.ready);
    EXPECT_TRUE(missingHas(rep, "内参 L"));
}

TEST(CalibRepository, ReadyForScanMissingIntrinsicR) {
    ReadyReport rep;
    ASSERT_TRUE(writeMutatedReadyCheck("t08_no_kr.json", [](nlohmann::json& j) {
        j["stereo"].erase("cameraMatrixR");
    }, rep));
    EXPECT_FALSE(rep.ready);
    EXPECT_TRUE(missingHas(rep, "内参 R"));
}

TEST(CalibRepository, ReadyForScanMissingExtrinsic) {
    ReadyReport rep;
    ASSERT_TRUE(writeMutatedReadyCheck("t08_no_rt.json", [](nlohmann::json& j) {
        j["stereo"].erase("R");
        j["stereo"].erase("T");
    }, rep));
    EXPECT_FALSE(rep.ready);
    EXPECT_TRUE(missingHas(rep, "外参"));
}

TEST(CalibRepository, ReadyForScanEmptyRectifyTable) {
    ReadyReport rep;
    ASSERT_TRUE(writeMutatedReadyCheck("t08_empty_rect.json", [](nlohmann::json& j) {
        j["tempTables"]["rectify"]["table"] = nlohmann::json::array();
    }, rep));
    EXPECT_FALSE(rep.ready);
    EXPECT_TRUE(missingHas(rep, "立体温度表"));
}

TEST(CalibRepository, ReadyForScanEmptyPlaneMapTable) {
    ReadyReport rep;
    ASSERT_TRUE(writeMutatedReadyCheck("t08_empty_pm.json", [](nlohmann::json& j) {
        j["planeMap"]["tempTable"]["table"] = nlohmann::json::array();
    }, rep));
    EXPECT_FALSE(rep.ready);
    EXPECT_TRUE(missingHas(rep, "激光档表"));
}

TEST(CalibRepository, ReadyForScanMissingImageSize) {
    // 手工构造去掉 meta.imageSize 的仓库档（load 路径绕过 write 组 meta）
    nlohmann::json doc = samplePayloadJson();
    doc["meta"] = {{"version", 1}};
    const std::string path = tmpPath("t08_no_size.json");
    {
        std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
        ofs << doc.dump();
    }
    CalibrationRepository repo;
    ASSERT_TRUE(repo.load(path).success);
    ReadyReport rep;
    repo.readyForScan(rep);
    EXPECT_FALSE(rep.ready);
    EXPECT_TRUE(missingHas(rep, "imageSize"));
    fs::remove(path);
}

TEST(CalibRepository, ReadyForScanFullPayloadReady) {
    CalibrationRepository repo;
    ASSERT_TRUE(loadSample(repo, "t08_full.json"));
    ReadyReport rep;
    EXPECT_TRUE(repo.readyForScan(rep).success);
    EXPECT_TRUE(rep.ready);
    EXPECT_TRUE(rep.missing.empty());
    fs::remove(tmpPath("t08_full.json"));
}

// —— 工厂档适配（方案 A）：factory_calib.camera_calib.v1 → 主工程内部档 ——
TEST(CalibRepository, FactorySchemaAdaptReadyExceptLaser) {
    // 独立子目录（laser_calib.json 是同目录自动合并约定——隔离防跨用例残留串扰）
    const std::string dir = tmpPath("t09_fc_dir");
    fs::create_directories(dir);
    const std::string camPath = dir + "/camera_calib.json";
    // 构造最小工厂档（键路径照抄 factory_calib module1_camera/calib_io.cpp assembleFull）
    nlohmann::json fc = {
        {"schema", "factory_calib.camera_calib.v1"},
        {"imageSize", nlohmann::json::array({2048, 1536})},
        {"intrinsic",
         {{"left", {{"camera_matrix", mat3(1839)}, {"dist_coeffs", nlohmann::json::array({nlohmann::json::array({0, 0, 0, 0, 0})})}}},
          {"right", {{"camera_matrix", mat3(1841)}, {"dist_coeffs", nlohmann::json::array({nlohmann::json::array({0, 0, 0, 0, 0})})}}}}},
        {"extrinsic", {{"R", mat3(1)},
                       {"T", nlohmann::json::array({nlohmann::json::array({-40}),
                                                    nlohmann::json::array({0}),
                                                    nlohmann::json::array({0})})}}},
        {"rectify", {{"R1", mat3(1)}, {"R2", mat3(1)},
                     {"P1", mat34(1)}, {"P2", mat34(1)}, {"Q", mat44(1)}}},
        {"stereoRectifyTempTable",
         {{"tableSize", 2},
          {"table", nlohmann::json::array({
              {{"temperature", 22.4}, {"R1", mat3(1)}, {"R2", mat3(1)},
               {"P1", mat34(1)}, {"P2", mat34(1)}, {"Q", mat44(1)},
               {"compensatedCameraMatrixL", mat3(1839)}},   // 工厂多余键——应被忽略
               {{"temperature", 22.6}, {"R1", mat3(1)}, {"R2", mat3(1)},
                {"P1", mat34(1)}, {"P2", mat34(1)}, {"Q", mat44(1)}}})}}}};
    {
        std::ofstream ofs(camPath, std::ios::binary | std::ios::trunc);
        ofs << fc.dump();
    }
    // 无 laser_calib.json：相机侧五项过、激光档表如实报缺
    CalibrationRepository repo;
    ASSERT_TRUE(repo.load(camPath).success);
    ReadyReport rep;
    EXPECT_TRUE(repo.readyForScan(rep).success);
    EXPECT_FALSE(rep.ready);
    EXPECT_EQ(rep.missing.size(), 1u);
    EXPECT_TRUE(missingHas(rep, "激光档表"));

    // typed getters：适配后矩阵可取、尺寸正确、温补表 2 档
    StereoData st = repo.stereo();
    ASSERT_FALSE(st.cameraMatrixL.empty());
    EXPECT_EQ(st.imageSize, cv::Size(2048, 1536));
    StereoTempTable tt = repo.stereoTempTable();
    ASSERT_EQ(tt.tiers.size(), 2u);
    EXPECT_DOUBLE_EQ(tt.tiers[0].tempC, 22.4);
    fs::remove_all(dir);
}

TEST(CalibRepository, FactorySchemaLaserMergeMakesReady) {
    // 相机档＋同目录 laser_calib.json（含 tempTable.table）→ 六项全过
    nlohmann::json fc = {
        {"schema", "factory_calib.camera_calib.v1"},
        {"imageSize", nlohmann::json::array({2048, 1536})},
        {"intrinsic",
         {{"left", {{"camera_matrix", mat3(1839)}, {"dist_coeffs", nlohmann::json::array({nlohmann::json::array({0, 0, 0, 0, 0})})}}},
          {"right", {{"camera_matrix", mat3(1841)}, {"dist_coeffs", nlohmann::json::array({nlohmann::json::array({0, 0, 0, 0, 0})})}}}}},
        {"extrinsic", {{"R", mat3(1)},
                       {"T", nlohmann::json::array({nlohmann::json::array({-40}),
                                                    nlohmann::json::array({0}),
                                                    nlohmann::json::array({0})})}}},
        {"rectify", {{"R1", mat3(1)}, {"R2", mat3(1)},
                     {"P1", mat34(1)}, {"P2", mat34(1)}, {"Q", mat44(1)}}},
        {"stereoRectifyTempTable",
         {{"tableSize", 1},
          {"table", nlohmann::json::array({
              {{"temperature", 22.5}, {"R1", mat3(1)}, {"R2", mat3(1)},
               {"P1", mat34(1)}, {"P2", mat34(1)}, {"Q", mat44(1)}}})}}}};
    const std::string dir2 = tmpPath("t10_fc_dir");
    fs::create_directories(dir2);
    const std::string camPath = dir2 + "/camera_calib.json";
    const std::string laserPath = dir2 + "/laser_calib.json";
    {
        std::ofstream ofs(camPath, std::ios::binary | std::ios::trunc);
        ofs << fc.dump();
        // 工厂激光档：顶层对象 tempTable.table（显式 put——避 nlohmann 花括号初始化
        // 把单键内层解析成数组的歧义坑）
        nlohmann::json laser = nlohmann::json::object();
        nlohmann::json tier = nlohmann::json::object();
        tier["temperature"] = 22.5;
        tier["deltaT"] = 0.0;
        tier["totalPairs"] = 128;
        laser["tempTable"]["table"] = nlohmann::json::array({tier});
        std::ofstream lfs(laserPath, std::ios::binary | std::ios::trunc);
        lfs << laser.dump();
    }
    CalibrationRepository repo;
    ASSERT_TRUE(repo.load(camPath).success);
    ReadyReport rep;
    EXPECT_TRUE(repo.readyForScan(rep).success);
    EXPECT_TRUE(rep.ready);                     // 六项全过
    EXPECT_TRUE(rep.missing.empty());
    PlaneMapTempTableRef pm = repo.planeMapTiers();
    ASSERT_EQ(pm.tiers.size(), 1u);
    EXPECT_DOUBLE_EQ(pm.tiers[0].tempC, 22.5);
    fs::remove_all(dir2);
}

TEST(CalibRepository, ClearMakesNotReady) {
    CalibrationRepository repo;
    ASSERT_TRUE(loadSample(repo, "t08_clear.json"));
    ReadyReport rep;
    repo.readyForScan(rep);
    EXPECT_TRUE(rep.ready);
    repo.clear();
    repo.readyForScan(rep);
    EXPECT_FALSE(rep.ready);
    fs::remove(tmpPath("t08_clear.json"));
}
