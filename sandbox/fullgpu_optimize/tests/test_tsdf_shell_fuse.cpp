// ============================================================================
// test_tsdf_shell_fuse.cpp — TSDF+MC 抽薄壳备用算子测试
// ============================================================================
// 覆盖: 参数 JSON 往返 / 解析场景(厚平面→薄壳, 球→水密闭壳) / 逐帧模式+
// 关键帧门 / 确定性 / 真实数据端到端(有数据才跑, 产物落盘)。
// ============================================================================
#include <gtest/gtest.h>

#include <cstdio>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "tsdf_shell_fuse.h"
#include "tsdf_shell_ply.h"

#ifndef SCR_SOURCE_DIR
#define SCR_SOURCE_DIR "."
#endif

using calib::TsdfShellFuseCpu;
using calib::TsdfShellFuseParams;
using calib::TsdfShellStats;
using calib::computeCloudToMeshStats;

namespace {

constexpr double kPi = 3.14159265358979323846;

// 确定性 LCG（12 均值近似高斯, σ=1）
struct Lcg {
    std::mt19937 eng;
    explicit Lcg(uint32_t seed) : eng(seed) {}
    double uniform() { return std::uniform_real_distribution<>(0.0, 1.0)(eng); }
    double gauss() {
        double s = 0;
        for (int i = 0; i < 12; ++i) s += uniform();
        return s - 6.0;
    }
};

// 厚平面: z=0 上 x,y∈[-20,20] 步 0.5, z 加 σ=noiseSigma 高斯噪声
std::vector<cv::Point3f> makeThickPlane(double noiseSigma, uint32_t seed) {
    Lcg rng(seed);
    std::vector<cv::Point3f> pts;
    for (double x = -20.0; x <= 20.001; x += 0.5)
        for (double y = -20.0; y <= 20.001; y += 0.5)
            pts.emplace_back((float)x, (float)y, (float)(rng.gauss() * noiseSigma));
    return pts;
}

// Fibonacci 球 + 径向噪声（法线=径向, 已知真值）
void makeSphere(double radius, int n, double noiseSigma, uint32_t seed,
                std::vector<cv::Point3f>& pts, std::vector<cv::Point3f>& normals) {
    Lcg rng(seed);
    pts.clear();
    normals.clear();
    const double ga = kPi * (3.0 - std::sqrt(5.0));
    for (int i = 0; i < n; ++i) {
        const double z = 1.0 - 2.0 * (i + 0.5) / n;
        const double rxy = std::sqrt(std::max(0.0, 1.0 - z * z));
        const double th = ga * i;
        const double rr = radius + rng.gauss() * noiseSigma;
        const double ux = rxy * std::cos(th), uy = rxy * std::sin(th), uz = z;
        pts.emplace_back((float)(rr * ux), (float)(rr * uy), (float)(rr * uz));
        normals.emplace_back((float)ux, (float)uy, (float)uz);
    }
}

}  // namespace

// ---------------------------------------------------------------------
TEST(TsdfShellFuse, ParamsJsonRoundTrip) {
    TsdfShellFuseParams p;
    p.voxelSize = 0.37f;
    p.truncFactor = 2.5f;
    p.mcCornerStep = 3;
    p.keyframeRotDeg = 0.1f;
    p.normalKnn = 24;
    const auto j = p.toJson();
    const auto q = TsdfShellFuseParams::fromJson(j);
    EXPECT_FLOAT_EQ(q.voxelSize, p.voxelSize);
    EXPECT_FLOAT_EQ(q.truncFactor, p.truncFactor);
    EXPECT_EQ(q.mcCornerStep, p.mcCornerStep);
    EXPECT_FLOAT_EQ(q.keyframeRotDeg, p.keyframeRotDeg);
    EXPECT_EQ(q.normalKnn, p.normalKnn);
    EXPECT_NO_THROW(q.validate());
    SUCCEED();
}

// ---------------------------------------------------------------------
// 厚平面 → 薄壳: 噪声 σ=0.15mm 的厚点云应抽出 |z| 很小的一张面
TEST(TsdfShellFuse, ThickPlaneToThinShell) {
    const auto pts = makeThickPlane(0.15, 20261120u);
    TsdfShellFuseParams p;
    p.voxelSize = 0.5f;
    TsdfShellFuseCpu fuse(p);
    ASSERT_TRUE(fuse.addCloudAutoNormals(pts));
    const auto mesh = fuse.extractMesh();
    ASSERT_GT(mesh.triangleCount, 2000) << "平面应切出大量三角形";
    ASSERT_GT(mesh.vertexCount, 1000);
    // 薄壳判据: 顶点 z 分布紧（σ_in=0.15, 格 0.5 → 顶点 |z| 应 <0.6mm）
    double sz = 0, maxz = 0;
    for (int i = 0; i < mesh.vertexCount; ++i) {
        const double z = std::fabs(mesh.vertices[3 * i + 2]);
        sz += z;
        maxz = std::max(maxz, z);
    }
    sz /= mesh.vertexCount;
    EXPECT_LT(sz, 0.30) << "壳顶点平均 |z| 应显著小于噪声厚度的 2σ";
    EXPECT_LT(maxz, 0.60);
    // 厚度量化: 输入点到壳的统计
    const auto st = computeCloudToMeshStats(mesh, pts, fuse.stats(), 0.5f);
    EXPECT_GT(st.meanPointToMesh, 0.0);
    EXPECT_LT(st.meanPointToMesh, 0.6);
}

// ---------------------------------------------------------------------
// 球 → 水密闭壳: 已知法线; 检查封闭性(每棱恰 2 面)+欧拉示性数 2+半径精度
TEST(TsdfShellFuse, SphereClosedWatertight) {
    std::vector<cv::Point3f> pts, nrm;
    makeSphere(20.0, 20000, 0.03, 7u, pts, nrm);
    TsdfShellFuseParams p;
    p.voxelSize = 0.5f;
    TsdfShellFuseCpu fuse(p);
    ASSERT_TRUE(fuse.addCloudWithNormals(pts, nrm));
    TsdfShellFuseCpu::McDiag diag;
    const auto mesh = fuse.extractMeshDiag(&diag);
    printf("[SphereDiag] visited=%d cornerMiss=%d rayGate=%d skipIncomplete=%d "
           "cubeGate=%d l1Drop=%d -> tris=%d\n",
           diag.cubesVisited, diag.cornerMissing, diag.cornerRayGate,
           diag.cubesSkippedIncomplete, diag.cubesGate, diag.trianglesL1Dropped,
           mesh.triangleCount);
    // 场剖面探针: 沿 +z 轴从外向球心扫（球面 r=20 与 +z 轴交点附近）
    printf("[SphereProfile] z : tsdf (weight)\n");
    for (double z = 22.0; z >= 17.0; z -= 0.25) {
        float s = 0, w = 0;
        float r3[3] = {0, 0, 0};
        if (fuse.probeVoxel(0.0, 0.0, z, s, w, r3))
            printf("  %6.2f : %+7.3f (%.0f)\n", z, s, w);
        else
            printf("  %6.2f : ---\n", z);
    }
    ASSERT_GT(mesh.triangleCount, 5000);
    // 封闭流形: 无界棱
    std::map<std::pair<uint32_t, uint32_t>, int> edgeCount;
    for (int t = 0; t < mesh.triangleCount; ++t) {
        const uint32_t a = mesh.indices[3 * t], b = mesh.indices[3 * t + 1],
                       c = mesh.indices[3 * t + 2];
        for (auto pr : {std::make_pair(std::min(a, b), std::max(a, b)),
                        std::make_pair(std::min(b, c), std::max(b, c)),
                        std::make_pair(std::min(c, a), std::max(c, a))})
            ++edgeCount[pr];
    }
    int boundary = 0, nonManifold = 0;
    for (const auto& kv : edgeCount) {
        if (kv.second == 1) ++boundary;
        else if (kv.second != 2) ++nonManifold;
    }
    EXPECT_EQ(boundary, 0) << "闭壳不应有边界棱";
    EXPECT_EQ(nonManifold, 0) << "流形网格每棱应恰 2 面";
    // 欧拉示性数 = V − E + F = 2（亏格 0）
    const long V = mesh.vertexCount, E = (long)edgeCount.size(), F = mesh.triangleCount;
    EXPECT_EQ(V - E + F, 2) << "球壳应为亏格 0";
    // 半径精度: 顶点到球心距离 ≈ 20（噪声 σ=0.03 + MC 内插误差 → <0.15）
    double sumR = 0, maxErr = 0;
    for (int i = 0; i < mesh.vertexCount; ++i) {
        const double r = std::sqrt((double)mesh.vertices[3 * i] * mesh.vertices[3 * i] +
                                   (double)mesh.vertices[3 * i + 1] * mesh.vertices[3 * i + 1] +
                                   (double)mesh.vertices[3 * i + 2] * mesh.vertices[3 * i + 2]);
        sumR += r;
        maxErr = std::max(maxErr, std::fabs(r - 20.0));
    }
    EXPECT_NEAR(sumR / mesh.vertexCount, 20.0, 0.06);
    EXPECT_LT(maxErr, 0.15);
}

// ---------------------------------------------------------------------
// 逐帧模式 + 关键帧门: 移动位姿融合, 重复位姿跳帧
TEST(TsdfShellFuse, FrameModeAndKeyframeGate) {
    TsdfShellFuseParams p;
    p.voxelSize = 0.5f;
    p.useKeyframeGate = true;
    TsdfShellFuseCpu fuse(p);
    // 世界系平面片 z=0, 传感器在其上方沿 x 移动
    auto makeFrame = [](double txDeg, double txx, double ty, double tz) {
        const double th = kPi / 180.0 * txDeg;
        cv::Matx33d R(std::cos(th), -std::sin(th), 0, std::sin(th), std::cos(th), 0, 0,
                      0, 1);
        cv::Vec3d T(txx, ty, tz);
        std::vector<cv::Point3f> local;
        for (double x = -10.0; x <= 10.001; x += 0.8)
            for (double y = -10.0; y <= 10.001; y += 0.8) {
                const cv::Vec3d w(x, y, 0.0);
                const cv::Vec3d l = R.t() * (w - T);  // world→local
                local.emplace_back((float)l(0), (float)l(1), (float)l(2));
            }
        return std::make_tuple(R, T, local);
    };
    for (int k = 0; k < 5; ++k) {
        auto [R, T, local] = makeFrame(8.0 * k, -20.0 + 10.0 * k, 0.0, 300.0);
        ASSERT_TRUE(fuse.addFrame(local, R, T));
    }
    EXPECT_EQ(fuse.stats().fusedFrames, 5);
    EXPECT_EQ(fuse.stats().skippedFrames, 0);
    // 重复位姿（位移 0, 转角 0）→ 跳帧
    {
        auto [R, T, local] = makeFrame(32.0, 20.0, 0.0, 300.0);
        ASSERT_TRUE(fuse.addFrame(local, R, T));
        EXPECT_EQ(fuse.stats().skippedFrames, 1);
        EXPECT_EQ(fuse.stats().fusedFrames, 5);
    }
    const auto mesh = fuse.extractMesh();
    ASSERT_GT(mesh.triangleCount, 1000);
    // 单侧观测平面: 顶点 z 应贴近 0
    double maxz = 0;
    for (int i = 0; i < mesh.vertexCount; ++i)
        maxz = std::max(maxz, std::fabs((double)mesh.vertices[3 * i + 2]));
    EXPECT_LT(maxz, 0.8) << "单面平面壳应薄";
}

// ---------------------------------------------------------------------
// 确定性: 同输入两次全流程逐位一致
TEST(TsdfShellFuse, Determinism) {
    auto runOnce = [] {
        std::vector<cv::Point3f> pts, nrm;
        makeSphere(15.0, 8000, 0.05, 99u, pts, nrm);
        TsdfShellFuseParams p;
        p.voxelSize = 0.4f;
        TsdfShellFuseCpu fuse(p);
        fuse.addCloudWithNormals(pts, nrm);
        return fuse.extractMesh();
    };
    const auto a = runOnce();
    const auto b = runOnce();
    ASSERT_EQ(a.vertices.size(), b.vertices.size());
    ASSERT_EQ(a.indices.size(), b.indices.size());
    EXPECT_EQ(0, std::memcmp(a.vertices.data(), b.vertices.data(),
                             a.vertices.size() * sizeof(float)));
    EXPECT_EQ(0, std::memcmp(a.indices.data(), b.indices.data(),
                             a.indices.size() * sizeof(uint32_t)));
}

// ---------------------------------------------------------------------
// 真实数据端到端: out_fuse_test.ply（119833 点代表点制融合云, 无厚度消减）
// → 抽出薄壳网格并落盘。数据缺失则 SKIP。
TEST(TsdfShellFuse, RealDataE2E) {
    const std::string inPath = std::string(SCR_SOURCE_DIR) + "/out_fuse_test.ply";
    std::vector<cv::Point3f> pts, nrm;
    bool hasN = false;
    if (!calib::tsdfply::loadPlyPoints(inPath, pts, nrm, hasN)) {
        GTEST_SKIP() << "真实数据 " << inPath << " 不存在, 跳过端到端";
    }
    ASSERT_GT(pts.size(), 10000u) << "真实云规模异常";
    TsdfShellFuseParams p;
    p.voxelSize = 0.5f;
    p.normalKnn = 16;
    TsdfShellFuseCpu fuse(p);
    ASSERT_TRUE(hasN ? fuse.addCloudWithNormals(pts, nrm)
                     : fuse.addCloudAutoNormals(pts));
    const auto mesh = fuse.extractMesh();
    EXPECT_GT(mesh.triangleCount, 1000) << "真实数据应抽出足量三角形";
    EXPECT_GT(mesh.vertexCount, 500);
    auto st0 = fuse.stats();
    st0.cubesCut = mesh.cubesCut;
    st0.triangleCount = mesh.triangleCount;
    st0.vertexCount = mesh.vertexCount;
    st0.voxelCount = mesh.voxelCount;
    const auto st = computeCloudToMeshStats(mesh, pts, st0, 0.5f);
    // 产物落盘（沙盒根 out_tsdf_shell.ply / out_tsdf_shell_stats.json）
    const std::string outPly = std::string(SCR_SOURCE_DIR) + "/out_tsdf_shell.ply";
    const std::string outJson = std::string(SCR_SOURCE_DIR) + "/out_tsdf_shell_stats.json";
    EXPECT_TRUE(calib::tsdfply::savePlyMesh(outPly, mesh));
    {
        std::ofstream f(outJson);
        f << st.toJson().dump(2) << "\n";
    }
    printf("[RealDataE2E] pts=%zu tris=%d verts=%d voxels=%d "
           "pointToMesh mean=%.3f p95=%.3f max=%.3f mm\n",
           pts.size(), mesh.triangleCount, mesh.vertexCount, mesh.voxelCount,
           st.meanPointToMesh, st.p95PointToMesh, st.maxPointToMesh);
    SUCCEED();
}
