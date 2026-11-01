// ============================================================================
// test_postprocess_normal.cpp — E 五阶段·阶段0 法线重算（NormalStage 实接
// 09 laser_cloud_normal_cpu 适配）。真值断言对齐算子精度：单位球面解析法线
// （点位置归一化）夹角全量 < 10°、中位数 < 3°；空云 fail 不崩；预填垃圾
// 法线被整体重算覆盖（数量==点数、单位长度）。
// ============================================================================
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "pipelines/postprocess/NormalStage.h"
#include "pipelines/postprocess/PostProcessPipeline.h"

using namespace Scanner::pipeline;

namespace {

constexpr double kPi = 3.14159265358979323846;

/// 单位球面均匀采样（Fibonacci 螺旋，确定性）——解析法线 = 点位置归一化
std::vector<float> makeUnitSphereXyz(size_t n) {
    std::vector<float> xyz;
    xyz.reserve(n * 3);
    const double ga = kPi * (3.0 - std::sqrt(5.0));
    for (size_t i = 0; i < n; ++i) {
        const double y = 1.0 - 2.0 * (static_cast<double>(i) + 0.5) / static_cast<double>(n);
        const double r = std::sqrt(std::max(0.0, 1.0 - y * y));
        const double th = ga * static_cast<double>(i);
        xyz.push_back(static_cast<float>(r * std::cos(th)));
        xyz.push_back(static_cast<float>(y));
        xyz.push_back(static_cast<float>(r * std::sin(th)));
    }
    return xyz;
}

/// 无符号夹角（度）。PCA 法线方向符号不定（特征向量 ± 等价），验证法向轴
/// 取 min(θ, 180°−θ)，同 09 test_laser_cloud_normal_cpu 球面断言口径。
double unsignedAngleDeg(double nx, double ny, double nz,
                        double px, double py, double pz) {
    const double len = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (len < 1e-12) return 180.0;
    double dot = (nx * px + ny * py + nz * pz) / len;   // p 已归一
    dot = std::max(-1.0, std::min(1.0, std::fabs(dot)));
    return std::acos(dot) * 180.0 / kPi;
}

} // namespace

// ============================================================================
// 1：球面法线精度——1000 点单位球 → 法线与解析值（点位置归一化）无符号
//    夹角全量 < 10°、中位数 < 3°（对齐 09 算子 PCA 精度）
// ============================================================================
TEST(NormalStageTest, SphereNormalsAccurate) {
    MeshData m;
    m.xyz = makeUnitSphereXyz(1000);
    NormalStage stage;
    CancelToken cancel;
    auto res = stage.run(m, cancel);
    ASSERT_TRUE(res.success) << res.message;

    ASSERT_EQ(m.normals.size(), m.xyz.size());
    std::vector<double> angles;
    angles.reserve(m.pointCount());
    for (size_t i = 0; i < m.pointCount(); ++i) {
        const double px = m.xyz[3 * i], py = m.xyz[3 * i + 1], pz = m.xyz[3 * i + 2];
        const double pr = std::sqrt(px * px + py * py + pz * pz);
        ASSERT_GT(pr, 0.9);
        angles.push_back(unsignedAngleDeg(m.normals[3 * i], m.normals[3 * i + 1],
                                          m.normals[3 * i + 2], px / pr, py / pr, pz / pr));
    }
    std::sort(angles.begin(), angles.end());
    const double median = angles[angles.size() / 2];
    EXPECT_LT(median, 3.0) << "法线中位误差应 < 3°，实测 " << median << "°";
    for (double a : angles)
        EXPECT_LT(a, 10.0) << "存在法线误差 ≥ 10° 的点";
}

// ============================================================================
// 2：空云 fail——空 xyz → fail（不崩），不写任何法线
// ============================================================================
TEST(NormalStageTest, EmptyCloudFails) {
    MeshData m;
    NormalStage stage;
    CancelToken cancel;
    auto res = stage.run(m, cancel);
    EXPECT_FALSE(res.success);
    EXPECT_TRUE(res.isFault());
    EXPECT_TRUE(m.normals.empty()) << "失败不应写任何法线";
}

// ============================================================================
// 3：垃圾法线被覆盖——预填 12345 → run 后整体重算（数量==点数、单位长度）
// ============================================================================
TEST(NormalStageTest, NormalsOverwritten) {
    MeshData m;
    m.xyz = makeUnitSphereXyz(1000);
    m.normals.assign(m.xyz.size(), 12345.0f);
    NormalStage stage;
    CancelToken cancel;
    auto res = stage.run(m, cancel);
    ASSERT_TRUE(res.success) << res.message;

    ASSERT_EQ(m.normals.size(), m.xyz.size()) << "法线数量 == 点数";
    for (size_t i = 0; i < m.pointCount(); ++i) {
        const double nx = m.normals[3 * i], ny = m.normals[3 * i + 1], nz = m.normals[3 * i + 2];
        const double len = std::sqrt(nx * nx + ny * ny + nz * nz);
        EXPECT_NEAR(len, 1.0, 1e-3) << "第 " << i << " 点法线应为单位向量，实测模长 " << len;
    }
}
