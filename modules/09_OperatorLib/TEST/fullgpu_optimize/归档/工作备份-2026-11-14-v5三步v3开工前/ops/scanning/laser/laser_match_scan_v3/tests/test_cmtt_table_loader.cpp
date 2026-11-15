// test_cmtt_table_loader.cpp — CMTT sidecar → 单档 LaserPlaneMapTempTable 转换器单测
//
// 覆盖: 合成 CMPU（CurveMapGenerator 小场景 64×48）双档 CMTT → 就近档单表转换 /
//       逐分量对拍（EnumerateRow 独立重放 × 表内 rowStep: [uL·s, row·s, uR·s, lid]）/
//       nearestTier 选档（另一温度另一档 / 等距偏低温）/ tierIdx·tierTemp 回传 /
//       错误路径（文件不存在 / 坏 magic / 无档空表 / 就近档 missing）。
// 场景: 小像面合成输入（同 test_curve_map SyntheticScene 构造法缩放,
//       近水平抛物线 × 25, 单位虚拟相机）, 生成秒级, 可进 CI。

#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>

#include "../cmtt_table_loader.h"
#include "curve_map.h"
#include "cmtt_container.h"

using namespace calib;

namespace {

constexpr int kLines = 25;
constexpr const char* kSamplePath = "test_cmtt_table_loader_sample.cmtt";

CurveMapInput makeMiniInput(double fScale) {
    CurveMapInput in;
    const double f = 64.0 * fScale;
    const double cx = 32.0, cy = 24.0;
    in.f = f;
    in.principalPoint = cv::Point2d(cx, cy);
    in.baseline = 8.0;
    in.imageSize = cv::Size(64, 48);
    in.virtualK = cv::Matx33d(f, 0, cx, 0, f, cy, 0, 0, 1);
    in.virtualT = cv::Vec3d(4.0, 0.8, 0.5);
    in.roi = cv::Rect(8, 4, 48, 40);

    in.curves.resize(kLines);
    for (int i = 0; i < kLines; ++i) {
        const double v0 = 4.0 + i * 1.6;
        const double sag = 2.0 + 1.8 * i / (kLines - 1);
        const double hs = 24.0;
        const double a = sag / (hs * hs);
        const double b = -2.0 * sag * cx / (hs * hs);
        const double c = v0 + sag * cx * cx / (hs * hs);
        double co[6] = {a, 0.0, 0.0, b, -1.0, c};
        double nrm = 0;
        for (double v : co) nrm += v * v;
        nrm = std::sqrt(nrm);
        for (int k = 0; k < 6; ++k)
            in.curves[static_cast<size_t>(i)].coeffs[k] = co[k] / nrm;
        in.curves[static_cast<size_t>(i)].pointCount = 0;
    }
    return in;
}

std::vector<uint8_t> generateTierBlob(double fScale, uint32_t* entryCount) {
    CurveMapParams p;
    p.epipolarRowStep = 0.7f;
    p.depthMin = 100.0f;
    p.depthMax = 700.0f;
    CurveMapGenerator gen(p);
    CurveMapResult r = gen.Generate(makeMiniInput(fScale));
    EXPECT_TRUE(r.success) << r.message;
    *entryCount = r.entryCount;
    return r.tableBytes;
}

bool writeBytes(const char* path, const std::vector<uint8_t>& bytes) {
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) return false;
    ofs.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    return ofs.good();
}

std::vector<uint8_t> readBytes(const char* path) {
    std::ifstream ifs(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(ifs)),
                                std::istreambuf_iterator<char>());
}

// 样例 CMTT: 档温 {22.0, 22.5, 25.0(missing)}, 头 base=25.0/step=0.5（生产同构）
std::vector<uint8_t> buildSampleCmtt(std::vector<uint32_t>* entryCounts) {
    uint32_t ec0 = 0, ec1 = 0;
    auto blob0 = generateTierBlob(1.0, &ec0);
    auto blob1 = generateTierBlob(1.005, &ec1);
    CmttBuilder b;
    b.setHeader(250, 5, 0.7f, 100.f, 700.f, cv::Rect(8, 4, 48, 40), 6, 63, 0xBEEFULL);
    b.addTier(220, blob0, ec0, 0x00);
    b.addTier(225, blob1, ec1, 0x00);
    b.addTier(250, {}, 0, kCmttFlagMissing);
    *entryCounts = {ec0, ec1, 0};
    return b.build();
}

// 独立重放对拍: EnumerateRow 两段式全行展开 × 表内 rowStep → 期望 Vec4f 序列
void expectMatMatchesBlob(const cv::Mat& m, const std::vector<uint8_t>& blob) {
    ASSERT_EQ(m.type(), CV_32FC4);
    ASSERT_EQ(m.cols, 1);
    ASSERT_TRUE(m.isContinuous());

    CurveMapTable cmt;
    std::string lerr;
    ASSERT_TRUE(cmt.Load(blob.data(), blob.size(), lerr)) << lerr;
    const float rowStep = cmt.rowStep();

    std::vector<cv::Vec4f> expect;
    std::vector<CurveMapEntry> buf(64);
    for (uint32_t row = 0; row < cmt.rowCount(); ++row) {
        int n = 0;
        while ((n = cmt.EnumerateRow(static_cast<int>(row), buf.data(),
                                     static_cast<int>(buf.size()))) < 0) {
            buf.resize(buf.size() * 2);
        }
        for (int i = 0; i < n; ++i) {
            const CurveMapEntry& e = buf[static_cast<size_t>(i)];
            expect.emplace_back(e.uL * rowStep, e.row * rowStep, e.uR * rowStep,
                                static_cast<float>(e.lid));
        }
    }
    ASSERT_EQ(m.rows, static_cast<int>(expect.size()));
    ASSERT_EQ(m.rows, static_cast<int>(cmt.entryCount()));
    for (int i = 0; i < m.rows; ++i)
        ASSERT_EQ(m.at<cv::Vec4f>(i, 0), expect[static_cast<size_t>(i)]) << "row " << i;
}

class CmttTableLoaderTest : public ::testing::Test {
protected:
    static std::vector<uint8_t> s_bytes;
    static std::vector<uint32_t> s_ecs;

    static void SetUpTestSuite() {
        s_bytes = buildSampleCmtt(&s_ecs);
        EXPECT_TRUE(writeBytes(kSamplePath, s_bytes));
    }
    static void TearDownTestSuite() { std::remove(kSamplePath); }
};

std::vector<uint8_t> CmttTableLoaderTest::s_bytes;
std::vector<uint32_t> CmttTableLoaderTest::s_ecs;

} // namespace

TEST_F(CmttTableLoaderTest, LoadNearestTierAndReplay) {
    std::string err;
    int tierIdx = -2;
    double tierTemp = -999.0;
    float outRowStep = -1.0f;
    auto table = LoadTempTableFromCmtt(kSamplePath, 22.5, err, &tierIdx,
                                       &tierTemp, &outRowStep);

    ASSERT_NE(table, nullptr) << err;
    EXPECT_TRUE(err.empty());
    EXPECT_EQ(tierIdx, 1);
    EXPECT_DOUBLE_EQ(tierTemp, 22.5);
    EXPECT_FLOAT_EQ(outRowStep, 0.7f);

    ASSERT_EQ(table->table.size(), size_t(1));
    EXPECT_NE(table->table.find(tierTemp), table->table.end());
    EXPECT_DOUBLE_EQ(table->referenceTemperature, 22.5);
    EXPECT_DOUBLE_EQ(table->tempStep, 0.5);

    const LaserPlaneMap* pm = table->lookup(tierTemp);
    ASSERT_NE(pm, nullptr);
    EXPECT_DOUBLE_EQ(pm->temperature, 22.5);
    EXPECT_GT(pm->totalPairs, 0);
    EXPECT_EQ(pm->totalPairs, static_cast<int>(s_ecs[1]));

    CmttReader rr;
    ASSERT_TRUE(rr.load(s_bytes.data(), s_bytes.size()));
    EXPECT_EQ(rr.entryCount(1), static_cast<uint32_t>(pm->totalPairs));
    auto blob = rr.tierBytes(1);
    ASSERT_TRUE(blob.has_value());
    expectMatMatchesBlob(pm->leftToRightMap, *blob);
}

TEST_F(CmttTableLoaderTest, OtherTempSelectsOtherTier) {
    std::string err;
    int tierIdx = -2;
    double tierTemp = -999.0;
    auto table = LoadTempTableFromCmtt(kSamplePath, 22.2, err, &tierIdx, &tierTemp);

    ASSERT_NE(table, nullptr) << err;
    EXPECT_EQ(tierIdx, 0);
    EXPECT_DOUBLE_EQ(tierTemp, 22.0);

    const LaserPlaneMap* pm = table->lookup(tierTemp);
    ASSERT_NE(pm, nullptr);
    EXPECT_EQ(pm->totalPairs, static_cast<int>(s_ecs[0]));
    EXPECT_EQ(pm->leftToRightMap.rows, static_cast<int>(s_ecs[0]));

    CmttReader rr;
    ASSERT_TRUE(rr.load(s_bytes.data(), s_bytes.size()));
    auto blob = rr.tierBytes(0);
    ASSERT_TRUE(blob.has_value());
    expectMatMatchesBlob(pm->leftToRightMap, *blob);

    // 等距（22.25 距 22.0/22.5 各 0.25）→ 偏低温侧档 0
    err.clear();
    table = LoadTempTableFromCmtt(kSamplePath, 22.25, err, &tierIdx, &tierTemp);
    ASSERT_NE(table, nullptr) << err;
    EXPECT_EQ(tierIdx, 0);
    EXPECT_DOUBLE_EQ(tierTemp, 22.0);
}

TEST_F(CmttTableLoaderTest, MissingNearestTierFails) {
    std::string err;
    auto table = LoadTempTableFromCmtt(kSamplePath, 25.0, err);
    EXPECT_EQ(table, nullptr);
    EXPECT_FALSE(err.empty());
}

TEST_F(CmttTableLoaderTest, CorruptedPayloadCrcFails) {
    // 翻转载荷区一字节（不动 96B 头/索引/索引内 CRC 字段）→ 懒校验拒绝:
    // 索引完好 load 通过, tierBytes CRC 不符 → nullptr + err
    auto bytes = s_bytes;
    const size_t payloadBase = sizeof(CmttHeader) + 3 * sizeof(CmttTierMeta);
    ASSERT_GT(bytes.size(), payloadBase + 10);
    bytes[payloadBase + 10] ^= 0xFF;   // 档 0 载荷内（档 0 blob 数千字节）
    ASSERT_TRUE(writeBytes("test_cmtt_table_loader_crc.cmtt", bytes));

    std::string err;
    auto table = LoadTempTableFromCmtt("test_cmtt_table_loader_crc.cmtt", 22.2, err);
    EXPECT_EQ(table, nullptr);
    EXPECT_FALSE(err.empty());
    std::remove("test_cmtt_table_loader_crc.cmtt");
}

TEST_F(CmttTableLoaderTest, IndependentSemanticAnchors) {
    // 独立锚点（不依赖 EnumerateRow 重放, 防双侧同错）: 场景真值直接取自合成输入——
    //   rowStep=0.7, roi=(8,4,48,40), 像面 64×48
    //   xL ∈ roi.x 向外半格余量（栅格量化 round 可向 ROI 边外溢半格）
    //   yL = k·rowStep（k 整数）且落图像高内; uR < xL（视差恒正, 栅格域 d≥1）
    std::string err;
    auto table = LoadTempTableFromCmtt(kSamplePath, 22.5, err);
    ASSERT_NE(table, nullptr) << err;
    const LaserPlaneMap* pm = table->lookup(22.5);
    ASSERT_NE(pm, nullptr);
    const cv::Mat& m = pm->leftToRightMap;
    ASSERT_GT(m.rows, 0);

    const float s = 0.7f;
    const float roiX = 8.0f, roiW = 48.0f;
    const float H = 48.0f;
    const float slack = s * 0.5f + 1e-4f;
    for (int i = 0; i < m.rows; ++i) {
        const cv::Vec4f v = m.at<cv::Vec4f>(i, 0);
        EXPECT_GE(v[0], roiX - slack) << "row " << i;
        EXPECT_LT(v[0], roiX + roiW + slack) << "row " << i;
        const float k = v[1] / s;
        EXPECT_NEAR(k, std::round(k), 1e-3f) << "row " << i;
        EXPECT_GE(v[1], 0.0f) << "row " << i;
        EXPECT_LT(v[1], H) << "row " << i;
        EXPECT_LT(v[2], v[0]) << "row " << i;
        EXPECT_GT(v[3], 0.0f) << "row " << i;
        EXPECT_LE(v[3], static_cast<float>(kLines)) << "row " << i;
    }
}

TEST(CmttTableLoader, NonexistentFileFails) {
    std::string err = "prior";
    auto table = LoadTempTableFromCmtt("no_such_dir_xyz/no_such_file.cmtt", 22.0, err);
    EXPECT_EQ(table, nullptr);
    EXPECT_FALSE(err.empty());
    EXPECT_NE(err, "prior");
}

TEST(CmttTableLoader, BadMagicFails) {
    std::vector<uint32_t> ecs;
    auto bytes = buildSampleCmtt(&ecs);
    bytes[0] = 'X';
    ASSERT_TRUE(writeBytes("test_cmtt_table_loader_badmagic.cmtt", bytes));
    std::string err;
    auto table = LoadTempTableFromCmtt("test_cmtt_table_loader_badmagic.cmtt", 22.0, err);
    EXPECT_EQ(table, nullptr);
    EXPECT_FALSE(err.empty());
    std::remove("test_cmtt_table_loader_badmagic.cmtt");
}

TEST(CmttTableLoader, EmptyTierTableFails) {
    CmttBuilder b;
    b.setHeader(250, 5, 0.7f, 100.f, 700.f, cv::Rect(8, 4, 48, 40), 6, 63, 1ULL);
    ASSERT_TRUE(writeBytes("test_cmtt_table_loader_empty.cmtt", b.build()));
    std::string err;
    auto table = LoadTempTableFromCmtt("test_cmtt_table_loader_empty.cmtt", 22.0, err);
    EXPECT_EQ(table, nullptr);
    EXPECT_FALSE(err.empty());
    std::remove("test_cmtt_table_loader_empty.cmtt");
}
