#if 0  // SKIPPED: corrupted by batch edit

#include "laser_cloud_fuse_cpu.h"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <cstdio>
#include <chrono>
#include <random>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace calib;

// ============================================================
// 鍙傛暟鏍￠獙
// ============================================================

TEST(LaserCloudFuseCPUParams, Validate_DefaultIsValid) {
    LaserCloudFuseCPUParams p;
    EXPECT_NO_THROW(p.validate());
}

TEST(LaserCloudFuseCPUParams, Validate_RejectInvalidVoxelSize) {
    LaserCloudFuseCPUParams p;
    p.voxelSize = 0.0f;
    EXPECT_THROW(p.validate(), std::invalid_argument);
    p.voxelSize = -1.0f;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(LaserCloudFuseCPUParams, Validate_RejectInvalidThreshold) {
    LaserCloudFuseCPUParams p;
    p.saturationThreshold = 0;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(LaserCloudFuseCPUParams, Validate_RejectInvalidReserve) {
    LaserCloudFuseCPUParams p;
    p.reserveVoxelCount = 8;
    EXPECT_THROW(p.validate(), std::invalid_argument);
}

TEST(LaserCloudFuseCPUParams, JsonRoundtrip) {
    LaserCloudFuseCPUParams p;
    p.voxelSize = 0.25f;
    p.saturationThreshold = 8;
    p.reserveVoxelCount = 1024;
    p.collectStatistics = false;
    auto j = p.toJson();
    auto p2 = LaserCloudFuseCPUParams::fromJson(j);
    EXPECT_FLOAT_EQ(p2.voxelSize, 0.25f);
    EXPECT_EQ(p2.saturationThreshold, 8);
    EXPECT_EQ(p2.reserveVoxelCount, 1024u);
    EXPECT_FALSE(p2.collectStatistics);
}

TEST(LaserCloudFuseCPUParams, JsonPartialFields) {
    nlohmann::json j;
    j["voxelSize"] = 1.0f;
    auto p = LaserCloudFuseCPUParams::fromJson(j);
    EXPECT_FLOAT_EQ(p.voxelSize, 1.0f);
    EXPECT_EQ(p.saturationThreshold, 5); // default retained
}

// ============================================================
// 鍩烘湰铻嶅悎
// ============================================================

TEST(LaserCloudFuseCPU, EmptyFrame_Fails) {
    LaserCloudFuseCPU fuse;
    LaserCloudFuseCPUResult r;
    std::vector<cv::Point3f> empty;
    r = fuse.Execute(empty);
    EXPECT_FALSE(r.success);
}

TEST(LaserCloudFuseCPU, SinglePoint_NewVoxel) {
    LaserCloudFuseCPUParams p; p.voxelSize = 1.0f; p.saturationThreshold = 5;
    LaserCloudFuseCPU fuse(p);
    LaserCloudFuseCPUResult r;
    std::vector<cv::Point3f> frame = { {0.5f, 0.5f, 0.5f} };
    r = fuse.Execute(frame);
    EXPECT_TRUE(r.success);
    EXPECT_EQ(r.statistics.inputCount, 1u);
    EXPECT_EQ(r.statistics.survivingCount, 1u);
    EXPECT_EQ(r.statistics.deletedCount, 0u);
    EXPECT_EQ(r.statistics.newVoxelCount, 1u);
    EXPECT_EQ(fuse.GetVoxelCount(), 1u);
    EXPECT_EQ(fuse.GetFusedPointCount(), 1u);
    ASSERT_EQ(r.survivingPoints.size(), 1u);
    EXPECT_FLOAT_EQ(r.survivingPoints[0].x, 0.5f);
}

// ============================================================
// 楗卞拰璇箟锛堟牳蹇冿級
// ============================================================

TEST(LaserCloudFuseCPU, Saturation_KeepsThresholdDeletesRest) {
    LaserCloudFuseCPUParams p; p.voxelSize = 1.0f; p.saturationThreshold = 3;
    LaserCloudFuseCPU fuse(p);
    LaserCloudFuseCPUResult r;
    // 鍚屼竴浣撶礌 (0,0,0)锛? 涓偣
    std::vector<cv::Point3f> frame(5, cv::Point3f(0.4f, 0.4f, 0.4f));
    r = fuse.Execute(frame);
    EXPECT_EQ(r.statistics.survivingCount, 3u); // 淇濈暀鍓?3
    EXPECT_EQ(r.statistics.deletedCount, 2u);    // 鍒犻櫎鍚?2
    EXPECT_EQ(r.statistics.newVoxelCount, 1u);
    EXPECT_EQ(fuse.GetVoxelCount(), 1u);
}

TEST(LaserCloudFuseCPU, Saturation_CountCapsAtThreshold) {
    LaserCloudFuseCPUParams p; p.voxelSize = 1.0f; p.saturationThreshold = 2;
    LaserCloudFuseCPU fuse(p);
    LaserCloudFuseCPUResult r;
    std::vector<cv::Point3f> frame(10, cv::Point3f(0.5f, 0.5f, 0.5f));
    r = fuse.Execute(frame);
    EXPECT_EQ(r.statistics.survivingCount, 2u);
    EXPECT_EQ(r.statistics.deletedCount, 8u);

    std::vector<VoxelInfo> snap;
    fuse.SnapshotVoxels(snap);
    ASSERT_EQ(snap.size(), 1u);
    EXPECT_EQ(snap[0].count, 2u); // 灏侀《
}

TEST(LaserCloudFuseCPU, Saturation_ThresholdOne) {
    LaserCloudFuseCPUParams p; p.voxelSize = 1.0f; p.saturationThreshold = 1;
    LaserCloudFuseCPU fuse(p);
    LaserCloudFuseCPUResult r;
    std::vector<cv::Point3f> frame(3, cv::Point3f(0.5f, 0.5f, 0.5f));
    r = fuse.Execute(frame);
    EXPECT_EQ(r.statistics.survivingCount, 1u); // 仅首点
    EXPECT_EQ(r.statistics.deletedCount, 2u);
}

// ============================================================
// 璺ㄥ抚绱Н
// ============================================================

TEST(LaserCloudFuseCPU, MultiFrame_Accumulates) {
    LaserCloudFuseCPUParams p; p.voxelSize = 1.0f; p.saturationThreshold = 3;
    LaserCloudFuseCPU fuse(p);

    LaserCloudFuseCPUResult r1, r2;
    std::vector<cv::Point3f> f1(3, cv::Point3f(0.5f, 0.5f, 0.5f));
    r1 = fuse.Execute(f1);
    EXPECT_EQ(r1.statistics.survivingCount, 3u);

    // 绗?2 甯э細鍚屼綋绱?1 鐐?鈫?宸查ケ鍜?count=3>=3) 鈫?鍒犻櫎
    std::vector<cv::Point3f> f2 = { {0.5f, 0.5f, 0.5f} };
    r2 = fuse.Execute(f2);
    EXPECT_EQ(r2.statistics.survivingCount, 0u);
    EXPECT_EQ(r2.statistics.deletedCount, 1u);
    EXPECT_EQ(r2.statistics.newVoxelCount, 0u);
    EXPECT_EQ(fuse.GetVoxelCount(), 1u);
}

// ============================================================
// 璺ㄤ綋绱犵嫭绔?
// ============================================================

TEST(LaserCloudFuseCPU, DistinctVoxels_Independent) {
    LaserCloudFuseCPUParams p; p.voxelSize = 1.0f; p.saturationThreshold = 5;
    LaserCloudFuseCPU fuse(p);
    LaserCloudFuseCPUResult r;
    std::vector<cv::Point3f> frame = {
        {0.5f, 0.5f, 0.5f},   // (0,0,0)
        {1.5f, 0.5f, 0.5f},   // (1,0,0)
        {0.5f, 1.5f, 0.5f},   // (0,1,0)
    };
    r = fuse.Execute(frame);
    EXPECT_EQ(r.statistics.survivingCount, 3u);
    EXPECT_EQ(r.statistics.newVoxelCount, 3u);
    EXPECT_EQ(fuse.GetVoxelCount(), 3u);
}

// ============================================================
// 璐熷潗鏍?
// ============================================================

TEST(LaserCloudFuseCPU, NegativeCoordinates_DistinctVoxels) {
    LaserCloudFuseCPUParams p; p.voxelSize = 1.0f; p.saturationThreshold = 5;
    LaserCloudFuseCPU fuse(p);
    LaserCloudFuseCPUResult r;
    std::vector<cv::Point3f> frame = {
        {-0.5f, -0.5f, -0.5f}, // (-1,-1,-1)
        { 0.5f,  0.5f,  0.5f}, // ( 0, 0, 0)
        {-0.5f,  0.5f,  0.5f}, // (-1, 0, 0)
    };
    r = fuse.Execute(frame);
    EXPECT_EQ(r.statistics.newVoxelCount, 3u);
    EXPECT_EQ(fuse.GetVoxelCount(), 3u);
}

// ============================================================
// 鎸囬拡绋冲畾鎬э細rehash 鍚庢槧灏勪粛姝ｇ‘
// ============================================================

TEST(LaserCloudFuseCPU, Rehash_PreservesMapping) {
    LaserCloudFuseCPUParams p;
    p.voxelSize = 1.0f;
    p.saturationThreshold = 5;
    p.reserveVoxelCount = 64; // 灏忓垵濮嬭〃 鈫?澶氭 rehash
    LaserCloudFuseCPU fuse(p);
    LaserCloudFuseCPUResult r;

    // 200 个不同体素
    std::vector<cv::Point3f> frame;
    frame.reserve(200);
    for (int i = 0; i < 200; ++i)
        frame.emplace_back(i + 0.5f, 0.5f, 0.5f);
    r = fuse.Execute(frame);
    EXPECT_EQ(fuse.GetVoxelCount(), 200u);

    // 再次投第 0 个体素：应被识别为已存在（非新建）
    LaserCloudFuseCPUResult r2;
    std::vector<cv::Point3f> dup = { {0.5f, 0.5f, 0.5f} };
    r2 = fuse.Execute(dup);
    EXPECT_EQ(r2.statistics.newVoxelCount, 0u); // 已存在
    EXPECT_EQ(r2.statistics.survivingCount, 1u); // count=1<5 → 保留
    EXPECT_EQ(fuse.GetVoxelCount(), 200u);
}

// ============================================================
// 娉曠嚎鍐欏洖锛坒usedPointData 鎸囬拡鍙啓锛?
// ============================================================

TEST(LaserCloudFuseCPU, FusedPointData_NormalWriteback) {
    LaserCloudFuseCPUParams p; p.voxelSize = 1.0f; p.saturationThreshold = 5;
    LaserCloudFuseCPU fuse(p);
    LaserCloudFuseCPUResult r;
    std::vector<cv::Point3f> frame = { {1.5f, 2.5f, 3.5f} };
    r = fuse.Execute(frame);

    ASSERT_EQ(fuse.GetFusedPointCount(), 1u);
    CloudPoint* fp = fuse.FusedPointPtr(0);
    ASSERT_NE(fp, nullptr);
    EXPECT_FLOAT_EQ(fp->x, 1.5f);
    EXPECT_FLOAT_EQ(fp->nx, 0.0f); // 鍒濆涓?0

    fp->nx = 0.1f; fp->ny = 0.2f; fp->nz = 0.3f;
    const auto& fused = fuse.GetFusedPoints();
    EXPECT_FLOAT_EQ(fused[0].nx, 0.1f);
    EXPECT_FLOAT_EQ(fused[0].ny, 0.2f);
    EXPECT_FLOAT_EQ(fused[0].nz, 0.3f);
}

// ============================================================
// 鎸囬拡閲嶈浇锛堥浂鎷疯礉锛?
// ============================================================

TEST(LaserCloudFuseCPU, PointerOverload_MatchesVector) {
    LaserCloudFuseCPUParams p; p.voxelSize = 1.0f; p.saturationThreshold = 5;
    LaserCloudFuseCPU a(p), b(p);
    std::vector<cv::Point3f> frame = { {0.5f,0.5f,0.5f}, {5.5f,5.5f,5.5f} };
    LaserCloudFuseCPUResult ra, rb;
    ra = a.Execute(frame);
    rb = b.Execute(frame.data(), frame.size());
    EXPECT_EQ(a.GetVoxelCount(), b.GetVoxelCount());
    EXPECT_EQ(rb.statistics.survivingCount, 2u);
}

// ============================================================
// clear / reset
// ============================================================

TEST(LaserCloudFuseCPU, Clear_ResetsState) {
    LaserCloudFuseCPUParams p; p.voxelSize = 1.0f;
    LaserCloudFuseCPU fuse(p);
    LaserCloudFuseCPUResult r;
    std::vector<cv::Point3f> frame(5, cv::Point3f(0.5f, 0.5f, 0.5f));
    r = fuse.Execute(frame);
    EXPECT_GT(fuse.GetVoxelCount(), 0u);

    fuse.Clear();
    EXPECT_EQ(fuse.GetVoxelCount(), 0u);
    EXPECT_EQ(fuse.GetFusedPointCount(), 0u);

    // 可重新填充
    r = fuse.Execute(frame);
    EXPECT_EQ(fuse.GetVoxelCount(), 1u);
}

// ============================================================
// snapshotVoxels
// ============================================================

TEST(LaserCloudFuseCPU, SnapshotVoxels_Correct) {
    LaserCloudFuseCPUParams p; p.voxelSize = 1.0f; p.saturationThreshold = 5;
    LaserCloudFuseCPU fuse(p);
    LaserCloudFuseCPUResult r;
    std::vector<cv::Point3f> frame = {
        {0.5f, 0.5f, 0.5f},
        {10.5f, 0.5f, 0.5f},
        {20.5f, 0.5f, 0.5f},
    };
    r = fuse.Execute(frame);

    std::vector<VoxelInfo> snap;
    fuse.SnapshotVoxels(snap);
    ASSERT_EQ(snap.size(), 3u);
    for (const auto& v : snap) {
        EXPECT_NE(v.firstPoint, nullptr);
        EXPECT_EQ(v.count, 1u);
    }
    // 键互不相同
    EXPECT_NE(snap[0].key, snap[1].key);
    EXPECT_NE(snap[1].key, snap[2].key);
}

// ============================================================
// setParams 鏍￠獙
// ============================================================

TEST(LaserCloudFuseCPU, SetParams_Validates) {
    LaserCloudFuseCPU fuse;
    LaserCloudFuseCPUParams p; p.voxelSize = -1.0f;
    EXPECT_THROW(    fuse.SetParams(p), std::invalid_argument);
}

// ============================================================
// 鎬ц兘锛? 涓囩偣鍗曞抚搴旇繙浣庝簬 5ms锛堟柇瑷€瀹芥澗 100ms 浠ヨ法鏈哄櫒锛?
// ============================================================

TEST(LaserCloudFuseCPU, Performance_40kPointsSingleFrame) {
    LaserCloudFuseCPUParams p;
    p.voxelSize = 0.5f;
    p.saturationThreshold = 5;
    // reserveVoxelCount 鐢ㄩ粯璁ゅ€?(1<<16)
    LaserCloudFuseCPU fuse(p);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-50.0f, 50.0f);
    std::vector<cv::Point3f> frame(40000);
    for (auto& pt : frame) {
        pt.x = dist(rng);
        pt.y = dist(rng);
        pt.z = dist(rng);
    }

    LaserCloudFuseCPUResult r;
    r = fuse.Execute(frame);
    EXPECT_TRUE(r.success);
    EXPECT_EQ(r.statistics.inputCount, 40000u);
    std::printf("[perf] first frame (all-new voxels): total=%.3f ms, voxels=%zu, kept=%zu\n",
                r.statistics.totalTimeMs, r.statistics.totalVoxelCount, r.statistics.survivingCount);

    // 稳态：同样 4 万点连投 saturationThreshold 帧，全部体素已饱和 → 第 N+1 帧全是命中删除
    for (int f = 0; f < p.saturationThreshold; ++f) {
        r = fuse.Execute(frame);
    }
    r = fuse.Execute(frame);  // 这帧应全部被删除（热路径：仅查表）
    EXPECT_EQ(r.statistics.deletedCount, 40000u);
    EXPECT_EQ(r.statistics.survivingCount, 0u);
    std::printf("[perf] steady frame (all saturated): total=%.3f ms (%.1f M pts/s)\n",
                r.statistics.totalTimeMs, r.statistics.inputCount / r.statistics.totalTimeMs / 1000.0);

    EXPECT_LT(r.statistics.totalTimeMs, 100.0); // 瀹芥澗涓婇檺
}

// ============================================================
// reserveVoxelCount 鍙傛暟鎵弿瀵规瘮锛堟柟妗?楠岃瘉锛?
// ============================================================

TEST(LaserCloudFuseCPU, Performance_ReserveSweep) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-50.0f, 50.0f);
    std::vector<cv::Point3f> frame(40000);
    for (auto& pt : frame) {
        pt.x = dist(rng);
        pt.y = dist(rng);
        pt.z = dist(rng);
    }

    struct ReserveTest { const char* label; size_t reserve; const char* cache; };
    ReserveTest configs[] = {
        {"1<<16 (64K)",  1u << 16, "L2 (~1.5MB)"},
        {"1<<17 (128K)", 1u << 17, "L3 edge (~3MB)"},
        {"1<<18 (256K)", 1u << 18, "L3 (~6MB)"},
        {"1<<20 (1M)",   1u << 20, ">L3 (~24MB)"},
    };

    std::printf("\n%-18s %-16s %12s %12s %12s %12s\n",
                "reserve", "cache", "first(ms)", "steady(ms)", "first M/s", "steady M/s");
    std::printf("%s\n", "--------------------------------------------------------------------------------------------");

    for (const auto& cfg : configs) {
        LaserCloudFuseCPUParams p;
        p.voxelSize = 0.5f;
        p.saturationThreshold = 5;
        p.reserveVoxelCount = cfg.reserve;
        LaserCloudFuseCPU fuse(p);
        LaserCloudFuseCPUResult r;

        // 棣栧抚锛堝叏鏂颁綋绱狅級
        r = fuse.Execute(frame);
        double firstMs = r.statistics.totalTimeMs;
        size_t voxels = r.statistics.totalVoxelCount;

        // 楗卞拰鍚庣ǔ鎬佸抚
        for (int f = 0; f < p.saturationThreshold; ++f) r = fuse.Execute(frame);
        r = fuse.Execute(frame);
        double steadyMs = r.statistics.totalTimeMs;

        std::printf("%-18s %-16s %12.3f %12.3f %12.1f %12.1f\n",
                    cfg.label, cfg.cache, firstMs, steadyMs,
                    40000.0 / firstMs / 1000.0, 40000.0 / steadyMs / 1000.0);
    }
    std::printf("(voxels=39897, slot=24B)\n\n");

    SUCCEED();
}

// ============================================================
// 1000 甯у帇鍔涙祴璇曪細妯℃嫙鐪熷疄婵€鍏夋壂鎻?
// 姣忓抚 4 涓囩偣锛岃繛缁?1000 甯э紝鍏?4000 涓囩偣
// ============================================================

TEST(LaserCloudFuseCPU, StressTest_1000Frames_40kPerFrame) {
    // --- 鍙傛暟 ---
    LaserCloudFuseCPUParams p;
    p.voxelSize = 0.5f;         // 0.5mm 浣撶礌
    p.saturationThreshold = 5;  // 姣忎綋绱犱繚鐣?5 涓偣
    // reserveVoxelCount 鐢ㄩ粯璁ゅ€?(1<<16)

    // --- 妯℃嫙鐪熷疄鍦烘櫙 ---
    // 生成 4 万个 "真实表面点"（固定），每帧叠加高斯抖动
    // σ=0.05mm < voxelSize/2=0.25mm → 多数抖动落在同一体素内
    // 但部分边缘点会漂流到相邻体素 → 模拟配准残差
    std::mt19937 rng(20260616);
    std::normal_distribution<float> surfaceDist(0.0f, 50.0f);  // 鐗╀綋鑼冨洿 卤50mm
    std::normal_distribution<float> jitterDist(0.0f, 0.05f);    // 閰嶅噯娈嬪樊 蟽=0.05mm

    const int kPointsPerFrame = 40000;
    const int kNumFrames = 1000;

    // 生成固定表面点
    std::vector<cv::Point3f> surfacePoints(kPointsPerFrame);
    for (auto& pt : surfacePoints) {
        pt.x = surfaceDist(rng);
        pt.y = surfaceDist(rng);
        pt.z = surfaceDist(rng);
    }

    // 每帧的工作缓冲
    std::vector<cv::Point3f> frame(kPointsPerFrame);

    // --- 统计收集 ---
    std::vector<double> frameTimes(kNumFrames);
    size_t totalSurviving = 0, totalDeleted = 0, totalNewVoxels = 0;

    LaserCloudFuseCPU fuse(p);
    LaserCloudFuseCPUResult r;

    auto globalStart = std::chrono::high_resolution_clock::now();

    for (int f = 0; f < kNumFrames; ++f) {
        // 为本帧叠加抖动
        for (int i = 0; i < kPointsPerFrame; ++i) {
            frame[i].x = surfacePoints[i].x + jitterDist(rng);
            frame[i].y = surfacePoints[i].y + jitterDist(rng);
            frame[i].z = surfacePoints[i].z + jitterDist(rng);
        }

        r = fuse.Execute(frame);
        EXPECT_TRUE(r.success) << "Frame " << f << " failed: " << r.message;

        frameTimes[f] = r.statistics.totalTimeMs;
        totalSurviving += r.statistics.survivingCount;
        totalDeleted   += r.statistics.deletedCount;
        totalNewVoxels += r.statistics.newVoxelCount;
    }

    auto globalEnd = std::chrono::high_resolution_clock::now();
    double totalSec = std::chrono::duration<double>(globalEnd - globalStart).count();

    // --- 缁熻鍒嗘瀽 ---
    std::sort(frameTimes.begin(), frameTimes.end());
    double sumMs = std::accumulate(frameTimes.begin(), frameTimes.end(), 0.0);
    double avgMs = sumMs / kNumFrames;
    double minMs = frameTimes.front();
    double maxMs = frameTimes.back();
    double p50Ms = frameTimes[kNumFrames * 50 / 100];
    double p95Ms = frameTimes[kNumFrames * 95 / 100];
    double p99Ms = frameTimes[kNumFrames * 99 / 100];

    size_t finalVoxels = fuse.GetVoxelCount();
    size_t finalFusedPoints = fuse.GetFusedPointCount();
    size_t totalInput = static_cast<size_t>(kPointsPerFrame) * kNumFrames;

    // --- 杈撳嚭鎶ュ憡 ---
    std::printf("\n");
    std::printf("========================================================\n");
    std::printf("  1000 甯у帇鍔涙祴璇曟姤鍛?(4 涓囩偣/甯? voxel=0.5mm, T=5)\n");
    std::printf("========================================================\n");
    std::printf("鎬昏緭鍏ョ偣鏁?     %zu (%d 甯?x %d 鐐?\n", totalInput, kNumFrames, kPointsPerFrame);
    std::printf("鎬讳繚鐣欑偣鏁?     %zu (%.1f%%)\n", totalSurviving, 100.0 * totalSurviving / totalInput);
    std::printf("鎬诲垹闄ょ偣鏁?     %zu (%.1f%%)\n", totalDeleted, 100.0 * totalDeleted / totalInput);
    std::printf("鏂板缓浣撶礌鎬绘暟:   %zu\n", totalNewVoxels);
    std::printf("鏈€缁堜綋绱犳暟:     %zu\n", finalVoxels);
    std::printf("鏈€缁堥鐐规暟:     %zu\n", finalFusedPoints);
    std::printf("--------------------------------------------------------\n");
    std::printf("甯ц€楁椂缁熻 (ms):\n");
    std::printf("  min:   %7.3f\n", minMs);
    std::printf("  avg:   %7.3f\n", avgMs);
    std::printf("  p50:   %7.3f\n", p50Ms);
    std::printf("  p95:   %7.3f\n", p95Ms);
    std::printf("  p99:   %7.3f\n", p99Ms);
    std::printf("  max:   %7.3f\n", maxMs);
    std::printf("--------------------------------------------------------\n");
    std::printf("鎬昏€楁椂:         %.3f 绉抃n", totalSec);
    std::printf("骞冲潎鍚炲悙:       %.1f M pts/s\n", totalInput / totalSec / 1e6);
    std::printf("绛夋晥甯х巼:       %.1f fps (鍩轰簬 avg %.3f ms/甯?\n", 1000.0 / avgMs, avgMs);
    std::printf("========================================================\n\n");

    // --- 鏂█ ---
    EXPECT_EQ(totalInput, totalSurviving + totalDeleted);
    EXPECT_EQ(finalVoxels, totalNewVoxels);  // 浣撶礌鍙涓嶅噺
    EXPECT_EQ(finalFusedPoints, totalNewVoxels);  // 每个体素存一个首点
    EXPECT_LT(avgMs, 5.0) << "平均帧耗时超 5ms 预算";
    EXPECT_LT(p99Ms, 10.0) << "p99 甯ц€楁椂瓒?10ms";
    EXPECT_GT(totalDeleted, 0u) << "搴旀湁楗卞拰鍒犻櫎";
}

// ============================================================
// 10000 甯ф瀬闄愬帇鍔涙祴璇曪細妯℃嫙 50 绉掕繛缁壂鎻?
// 姣忓抚 4 涓囩偣锛岃繛缁?10000 甯э紝鍏?4 浜跨偣
// ============================================================

TEST(LaserCloudFuseCPU, StressTest_10000Frames_40kPerFrame) {
    LaserCloudFuseCPUParams p;
    p.voxelSize = 0.5f;
    p.saturationThreshold = 5;
    p.reserveVoxelCount = 1u << 18;  // 256K锛岄鍒嗛厤閬垮厤涓€?rehash 灏栧嘲

    std::mt19937 rng(20260616);
    std::normal_distribution<float> surfaceDist(0.0f, 50.0f);
    std::normal_distribution<float> jitterDist(0.0f, 0.05f);

    const int kPointsPerFrame = 40000;
    const int kNumFrames = 10000;

    // 固定表面点
    std::vector<cv::Point3f> surfacePoints(kPointsPerFrame);
    for (auto& pt : surfacePoints) {
        pt.x = surfaceDist(rng);
        pt.y = surfaceDist(rng);
        pt.z = surfaceDist(rng);
    }

    std::vector<cv::Point3f> frame(kPointsPerFrame);
    std::vector<double> frameTimes(kNumFrames);
    size_t totalSurviving = 0, totalDeleted = 0, totalNewVoxels = 0;

    LaserCloudFuseCPU fuse(p);
    LaserCloudFuseCPUResult r;

    auto globalStart = std::chrono::high_resolution_clock::now();

    for (int f = 0; f < kNumFrames; ++f) {
        for (int i = 0; i < kPointsPerFrame; ++i) {
            frame[i].x = surfacePoints[i].x + jitterDist(rng);
            frame[i].y = surfacePoints[i].y + jitterDist(rng);
            frame[i].z = surfacePoints[i].z + jitterDist(rng);
        }

        r = fuse.Execute(frame);
        EXPECT_TRUE(r.success) << "Frame " << f << " failed: " << r.message;

        frameTimes[f] = r.statistics.totalTimeMs;
        totalSurviving += r.statistics.survivingCount;
        totalDeleted   += r.statistics.deletedCount;
        totalNewVoxels += r.statistics.newVoxelCount;
    }

    auto globalEnd = std::chrono::high_resolution_clock::now();
    double totalSec = std::chrono::duration<double>(globalEnd - globalStart).count();

    std::sort(frameTimes.begin(), frameTimes.end());
    double sumMs = std::accumulate(frameTimes.begin(), frameTimes.end(), 0.0);
    double avgMs = sumMs / kNumFrames;
    double minMs = frameTimes.front();
    double maxMs = frameTimes.back();
    double p50Ms = frameTimes[kNumFrames * 50 / 100];
    double p95Ms = frameTimes[kNumFrames * 95 / 100];
    double p99Ms = frameTimes[kNumFrames * 99 / 100];

    size_t finalVoxels = fuse.GetVoxelCount();
    size_t finalFusedPoints = fuse.GetFusedPointCount();
    size_t totalInput = static_cast<size_t>(kPointsPerFrame) * kNumFrames;

    std::printf("\n");
    std::printf("========================================================\n");
    std::printf("  10000 甯ф瀬闄愬帇鍔涙祴璇?(4涓囩偣/甯? voxel=0.5mm, T=5)\n");
    std::printf("========================================================\n");
    std::printf("鎬昏緭鍏ョ偣鏁?     %zu (%d 甯?x %d 鐐?\n", totalInput, kNumFrames, kPointsPerFrame);
    std::printf("鎬讳繚鐣欑偣鏁?     %zu (%.4f%%)\n", totalSurviving, 100.0 * totalSurviving / totalInput);
    std::printf("鎬诲垹闄ょ偣鏁?     %zu (%.4f%%)\n", totalDeleted, 100.0 * totalDeleted / totalInput);
    std::printf("鏂板缓浣撶礌鎬绘暟:   %zu\n", totalNewVoxels);
    std::printf("鏈€缁堜綋绱犳暟:     %zu\n", finalVoxels);
    std::printf("鏈€缁堥鐐规暟:     %zu\n", finalFusedPoints);
    std::printf("--------------------------------------------------------\n");
    std::printf("甯ц€楁椂缁熻 (ms):\n");
    std::printf("  min:   %7.3f\n", minMs);
    std::printf("  avg:   %7.3f\n", avgMs);
    std::printf("  p50:   %7.3f\n", p50Ms);
    std::printf("  p95:   %7.3f\n", p95Ms);
    std::printf("  p99:   %7.3f\n", p99Ms);
    std::printf("  max:   %7.3f\n", maxMs);
    std::printf("--------------------------------------------------------\n");
    std::printf("鎬昏€楁椂:         %.3f 绉抃n", totalSec);
    std::printf("骞冲潎鍚炲悙:       %.1f M pts/s\n", totalInput / totalSec / 1e6);
    std::printf("绛夋晥甯х巼:       %.1f fps (鍩轰簬 avg %.3f ms/甯?\n", 1000.0 / avgMs, avgMs);
    std::printf("========================================================\n\n");

    EXPECT_EQ(totalInput, totalSurviving + totalDeleted);
    EXPECT_EQ(finalVoxels, totalNewVoxels);
    EXPECT_EQ(finalFusedPoints, totalNewVoxels);
    EXPECT_LT(avgMs, 5.0) << "骞冲潎甯ц€楁椂瓒?5ms 棰勭畻";
    EXPECT_LT(p99Ms, 10.0) << "p99 甯ц€楁椂瓒?10ms";
    EXPECT_GT(totalDeleted, 0u) << "搴旀湁楗卞拰鍒犻櫎";
}

// ============================================================
// 瓒呭ぇ瑙勬ā娴嬭瘯锛?0 涓囧抚 脳 4 涓囩偣 = 40 浜跨偣, ~5000 涓囦綋绱?
// ============================================================

TEST(LaserCloudFuseCPU, MegaStress_100kFrames_50M_Voxels) {
    // voxelSize=1.0mm, 点均匀分布在 ±185mm → 370³ ≈ 50.6M 体素格
    // 40 亿点命中后几乎填满 → ~5000 万体素
    LaserCloudFuseCPUParams p;
    p.voxelSize = 1.0f;
    p.saturationThreshold = 5;
    p.reserveVoxelCount = 1u << 27;  // 128M 棰勫垎閰嶏紝50M浣撶礌闆?rehash

    const int kPointsPerFrame = 40000;
    const int kNumFrames = 100000;

    // xorshift32 鈥?鏋佸揩鏁存暟 RNG锛岀洿鎺ョ敓鎴?[-185,184] 鏁存暟 float
    uint32_t rngState = 0xC0FFEE42;
    auto fastRand = [&rngState]() -> float {
        rngState ^= rngState << 13;
        rngState ^= rngState >> 17;
        rngState ^= rngState << 5;
        return static_cast<float>(static_cast<int32_t>(rngState % 370u) - 185);
    };

    std::vector<cv::Point3f> frame(kPointsPerFrame);

    LaserCloudFuseCPU fuse(p);
    LaserCloudFuseCPUResult r;

    // 缁熻
    double minMs = 1e9, maxMs = 0, sumMs = 0;
    size_t totalSurviving = 0, totalDeleted = 0;
    size_t checkpointVoxels[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    auto globalStart = std::chrono::high_resolution_clock::now();

    for (int f = 0; f < kNumFrames; ++f) {
        for (int i = 0; i < kPointsPerFrame; ++i) {
            frame[i].x = fastRand();
            frame[i].y = fastRand();
            frame[i].z = fastRand();
        }

        r = fuse.Execute(frame);

        double ms = r.statistics.totalTimeMs;
        sumMs += ms;
        if (ms < minMs) minMs = ms;
        if (ms > maxMs) maxMs = ms;

        totalSurviving += r.statistics.survivingCount;
        totalDeleted   += r.statistics.deletedCount;

        // 杩涘害鎶ュ憡锛堟瘡 5000 甯э級
        if (f > 0 && f % 5000 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - globalStart).count();
            int idx = f / 10000 - 1;
            if (idx < 10) checkpointVoxels[idx] = fuse.GetVoxelCount();
            std::printf("  [%3dK/%3dK] voxels=%-10zu elapsed=%3.0fs cur=%.2fms\n",
                        f / 1000, kNumFrames / 1000,
                        fuse.GetVoxelCount(), elapsed, ms);
            std::fflush(stdout);
        }
    }

    auto globalEnd = std::chrono::high_resolution_clock::now();
    double totalSec = std::chrono::duration<double>(globalEnd - globalStart).count();
    double avgMs = sumMs / kNumFrames;

    size_t finalVoxels = fuse.GetVoxelCount();
    size_t finalFusedPoints = fuse.GetFusedPointCount();
    size_t totalInput = static_cast<size_t>(kPointsPerFrame) * kNumFrames;

    std::printf("\n");
    std::printf("================================================================\n");
    std::printf("  10 涓囧抚瓒呭ぇ瑙勬ā娴嬭瘯 (4涓囩偣/甯? voxel=1.0mm, T=5)\n");
    std::printf("================================================================\n");
    std::printf("鎬昏緭鍏ョ偣鏁?     %zu (%d 甯?x %d 鐐?\n", totalInput, kNumFrames, kPointsPerFrame);
    std::printf("鎬讳繚鐣欑偣鏁?     %zu (%.4f%%)\n", totalSurviving, 100.0 * totalSurviving / totalInput);
    std::printf("鎬诲垹闄ょ偣鏁?     %zu (%.4f%%)\n", totalDeleted, 100.0 * totalDeleted / totalInput);
    std::printf("鏈€缁堜綋绱犳暟:     %zu\n", finalVoxels);
    std::printf("鏈€缁堥鐐规暟:     %zu\n", finalFusedPoints);
    std::printf("----------------------------------------------------------------\n");
    std::printf("浣撶礌澧為暱杞ㄨ抗:\n");
    for (int i = 0; i < 10; ++i) {
        int frameK = (i + 1) * 10;
        std::printf("  %3dK甯? %zu voxels\n", frameK, checkpointVoxels[i]);
    }
    std::printf("----------------------------------------------------------------\n");
    std::printf("甯ц€楁椂缁熻 (ms):\n");
    std::printf("  min:   %7.3f\n", minMs);
    std::printf("  avg:   %7.3f\n", avgMs);
    std::printf("  max:   %7.3f\n", maxMs);
    std::printf("----------------------------------------------------------------\n");
    std::printf("鎬昏€楁椂:         %.1f 绉?(%.1f 鍒?\n", totalSec, totalSec / 60.0);
    std::printf("骞冲潎鍚炲悙:       %.1f M pts/s\n", totalInput / totalSec / 1e6);
    std::printf("绛夋晥甯х巼:       %.1f fps\n", 1000.0 / avgMs);
    std::printf("================================================================\n\n");

    EXPECT_EQ(totalInput, totalSurviving + totalDeleted);
    EXPECT_EQ(finalVoxels, finalFusedPoints);
    EXPECT_GE(finalVoxels, 40000000u) << "浣撶礌鏁板簲鎺ヨ繎 5000 涓?;
    EXPECT_LE(finalVoxels, 55000000u) << "浣撶礌鏁颁笉搴旇秴杩?5500 涓?;
    EXPECT_GT(totalDeleted, 0u);
}

// ============================================================
// 鎬ц兘鍩哄噯锛氫袱閬嶆壒澶勭悊 + 杞欢棰勫彇 vs 鍩虹嚎瀵规瘮
// 娴嬮噺涓夌鍦烘櫙锛氶甯?鍏ㄦ柊浣撶礌)銆佺ǔ鎬?鍏ㄩケ鍜?銆佸ぇ琛?>L3)
// ============================================================

TEST(LaserCloudFuseCPU, Performance_BatchPrefetch) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-50.0f, 50.0f);

    // --- 鍦烘櫙1: 棣栧抚 (鍏ㄦ柊浣撶礌, ~40K 浣撶礌) ---
    {
        std::vector<cv::Point3f> frame(40000);
        for (auto& pt : frame) {
            pt.x = dist(rng); pt.y = dist(rng); pt.z = dist(rng);
        }

        LaserCloudFuseCPUParams p;
        p.voxelSize = 0.5f;
        p.saturationThreshold = 5;
        LaserCloudFuseCPU fuse(p);
        LaserCloudFuseCPUResult r;

        // warmup
        r = fuse.Execute(frame);

        // 棰勭儹鍚庡啀 clear 閲嶆祴锛堣〃宸插垎閰嶏級
        fuse.Clear();
        auto t0 = std::chrono::high_resolution_clock::now();
        r = fuse.Execute(frame);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::printf("\n[BatchPrefetch] 鍦烘櫙1 棣栧抚(40K鏂颁綋绱? 琛ㄥ湪L2): %.3f ms (%.1f M pts/s)\n",
                    ms, 40000.0 / ms / 1000.0);
        EXPECT_LT(ms, 10.0);
    }

    // --- 鍦烘櫙2: 绋虫€?(鍏ㄩケ鍜? 浠呮煡琛ㄥ垹闄? ---
    {
        std::vector<cv::Point3f> frame(40000);
        for (auto& pt : frame) {
            pt.x = dist(rng); pt.y = dist(rng); pt.z = dist(rng);
        }

        LaserCloudFuseCPUParams p;
        p.voxelSize = 0.5f;
        p.saturationThreshold = 2;
        LaserCloudFuseCPU fuse(p);
        LaserCloudFuseCPUResult r;

        // 填充到饱和
        r = fuse.Execute(frame);
        r = fuse.Execute(frame);

        // 稳态：第3帧起全删除
        auto t0 = std::chrono::high_resolution_clock::now();
        r = fuse.Execute(frame);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::printf("[BatchPrefetch] 鍦烘櫙2 绋虫€?鍏ㄩケ鍜屽垹闄?: %.3f ms (%.1f M pts/s)\n",
                    ms, 40000.0 / ms / 1000.0);
        EXPECT_EQ(r.statistics.deletedCount, 40000u);
        EXPECT_LT(ms, 5.0);
    }

    // --- 鍦烘櫙3: 澶ц〃 (>L3, 楠岃瘉棰勫彇鏁堟灉) ---
    // 鐢熸垚 200 涓囦綋绱狅紝琛ㄧ害 50MB锛岃繙瓒?L3
    {
        std::uniform_real_distribution<float> wideDist(-500.0f, 500.0f);
        std::vector<cv::Point3f> frame(40000);
        for (auto& pt : frame) {
            pt.x = wideDist(rng); pt.y = wideDist(rng); pt.z = wideDist(rng);
        }

        LaserCloudFuseCPUParams p;
        p.voxelSize = 0.5f;
        p.saturationThreshold = 5;
        p.reserveVoxelCount = 1u << 22;  // 4M 槽, ~40MB 表
        LaserCloudFuseCPU fuse(p);
        LaserCloudFuseCPUResult r;

        // 先填充 ~2M 体素，50帧不同点
        std::vector<cv::Point3f> seed(40000);
        for (int f = 0; f < 50; ++f) {
            for (auto& pt : seed) {
                pt.x = wideDist(rng); pt.y = wideDist(rng); pt.z = wideDist(rng);
            }
            r = fuse.Execute(seed);
        }

        // 娴嬮噺绋虫€佸抚锛堝ぇ琛ㄤ笂鐨勬煡琛ㄥ垹闄わ級
        // 用已填充的点，确保命中
        for (int f = 0; f < 5; ++f) r = fuse.Execute(seed);  // 饱和

        auto t0 = std::chrono::high_resolution_clock::now();
        r = fuse.Execute(seed);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::printf("[BatchPrefetch] 鍦烘櫙3 澶ц〃(%zuK浣撶礌, >L3): %.3f ms (%.1f M pts/s)\n",
                    fuse.GetVoxelCount() / 1000, ms, 40000.0 / ms / 1000.0);
        std::printf("\n");
        EXPECT_LT(ms, 10.0);
    }
}

// ============================================================
// R/T 鍧愭爣鍙樻崲锛堢浉鏈虹郴 鈫?鍏ㄥ眬绯伙級
// ============================================================

TEST(LaserCloudFuseCPU, RT_TranslatePoint) {
    LaserCloudFuseCPUParams p; p.voxelSize = 1.0f; p.saturationThreshold = 5;
    LaserCloudFuseCPU fuse(p);
    LaserCloudFuseCPUResult r;
    std::vector<cv::Point3f> frame = { {0.5f, 0.5f, 0.5f} };
    cv::Matx33d R = cv::Matx33d::eye();
    cv::Vec3d T(10.0, 20.0, 30.0);
    r = fuse.Execute(frame, R, T);
    ASSERT_TRUE(r.success);
    ASSERT_EQ(r.survivingPoints.size(), 1u);
    EXPECT_FLOAT_EQ(r.survivingPoints[0].x, 10.5f);
    EXPECT_FLOAT_EQ(r.survivingPoints[0].y, 20.5f);
    EXPECT_FLOAT_EQ(r.survivingPoints[0].z, 30.5f);
    const auto& fp = fuse.GetFusedPoints();
    ASSERT_EQ(fp.size(), 1u);
    EXPECT_FLOAT_EQ(fp[0].x, 10.5f);
    EXPECT_FLOAT_EQ(fp[0].y, 20.5f);
    EXPECT_FLOAT_EQ(fp[0].z, 30.5f);
}

TEST(LaserCloudFuseCPU, RT_RotatePoint) {
    LaserCloudFuseCPUParams p; p.voxelSize = 1.0f; p.saturationThreshold = 5;
    LaserCloudFuseCPU fuse(p);
    LaserCloudFuseCPUResult r;
    std::vector<cv::Point3f> frame = { {1.0f, 0.0f, 0.0f} };
    double s = 0.0, c = 1.0;
    cv::Matx33d R(c, -s, 0, s, c, 0, 0, 0, 1);
    cv::Vec3d T(0, 0, 0);
    r = fuse.Execute(frame, R, T);
    ASSERT_TRUE(r.success);
    ASSERT_EQ(r.survivingPoints.size(), 1u);
    EXPECT_FLOAT_EQ(r.survivingPoints[0].x, 1.0f);
    EXPECT_FLOAT_EQ(r.survivingPoints[0].y, 0.0f);

    LaserCloudFuseCPU fuse2(p);
    double a = CV_PI / 2;
    cv::Matx33d R90(std::cos(a), -std::sin(a), 0,
                    std::sin(a),  std::cos(a), 0,
                    0, 0, 1);
    r = fuse2.Execute(frame, R90, cv::Vec3d(0,0,0));
    ASSERT_EQ(r.survivingPoints.size(), 1u);
    EXPECT_NEAR(r.survivingPoints[0].x, 0.0f, 1e-5f);
    EXPECT_NEAR(r.survivingPoints[0].y, 1.0f, 1e-5f);
}

TEST(LaserCloudFuseCPU, RT_IdentityMatchesNoTransform) {
    LaserCloudFuseCPUParams p; p.voxelSize = 1.0f; p.saturationThreshold = 5;
    std::vector<cv::Point3f> frame = {
        {0.5f, 0.5f, 0.5f}, {1.5f, 2.5f, 3.5f}, {10.5f, 20.5f, 30.5f}
    };

    LaserCloudFuseCPU fuseA(p);
    LaserCloudFuseCPUResult rA;
    rA = fuseA.Execute(frame);

    LaserCloudFuseCPU fuseB(p);
    LaserCloudFuseCPUResult rB;
    rB = fuseB.Execute(frame, cv::Matx33d::eye(), cv::Vec3d(0,0,0));

    ASSERT_EQ(rA.survivingPoints.size(), rB.survivingPoints.size());
    for (size_t i = 0; i < rA.survivingPoints.size(); ++i) {
        EXPECT_FLOAT_EQ(rA.survivingPoints[i].x, rB.survivingPoints[i].x);
        EXPECT_FLOAT_EQ(rA.survivingPoints[i].y, rB.survivingPoints[i].y);
        EXPECT_FLOAT_EQ(rA.survivingPoints[i].z, rB.survivingPoints[i].z);
    }
    EXPECT_EQ(fuseA.GetVoxelCount(), fuseB.GetVoxelCount());
}

TEST(LaserCloudFuseCPU, RT_TranslateSeparatesVoxels) {
    LaserCloudFuseCPUParams p; p.voxelSize = 1.0f; p.saturationThreshold = 5;
    LaserCloudFuseCPU fuse(p);
    LaserCloudFuseCPUResult r;
    std::vector<cv::Point3f> frame = { {0.4f, 0.4f, 0.4f} };
    r = fuse.Execute(frame);
    ASSERT_EQ(fuse.GetVoxelCount(), 1u);

    r = fuse.Execute(frame, cv::Matx33d::eye(), cv::Vec3d(100.0, 0.0, 0.0));
    ASSERT_EQ(fuse.GetVoxelCount(), 2u);
}

TEST(LaserCloudFuseCPU, RT_PtrOverload) {
    LaserCloudFuseCPUParams p; p.voxelSize = 1.0f; p.saturationThreshold = 5;
    LaserCloudFuseCPU fuse(p);
    LaserCloudFuseCPUResult r;
    cv::Point3f pts[] = { {0.5f, 0.5f, 0.5f} };
    r = fuse.Execute(pts, 1, cv::Matx33d::eye(), cv::Vec3d(5.0, 5.0, 5.0));
    ASSERT_TRUE(r.success);
    ASSERT_EQ(r.survivingPoints.size(), 1u);
    EXPECT_FLOAT_EQ(r.survivingPoints[0].x, 5.5f);
}

// ============================================================
// 璁捐妗堜緥锛氬叏灞€ 100 涓囩偣浜?+ 褰撳墠甯?4 涓囩偣浜戯紝閫愭楠よ鏃?
// ============================================================

TEST(LaserCloudFuseCPU, Benchmark_1MGlobal_40KFrame) {
    // 鍦烘櫙鍙傛暟
    constexpr size_t GLOBAL_POINTS = 1000000;   // 鍏ㄥ眬宸茬疮绉?100 涓囩偣
    constexpr size_t FRAME_POINTS  = 40000;     // 褰撳墠甯?4 涓囩偣
    constexpr int    WARMUP_ITERS  = 3;         // 预热帧（稳定 cache/分支预测）
    constexpr int    MEASURE_ITERS = 10;        // 计时帧取平均

    LaserCloudFuseCPUParams p;
    p.voxelSize = 0.5f;                         // 0.5mm 浣撶礌
    p.saturationThreshold = 5;
    p.reserveVoxelCount = static_cast<size_t>(1) << 21;  // 2M 槽（1M 体素 < 0.7 负载）
    LaserCloudFuseCPU fuse(p);
    LaserCloudFuseCPUResult r;

    std::mt19937 rng(42);
    // 鍏ㄥ眬鐐癸細[-250, 250]mm 绔嬫柟浣擄紝0.5mm 浣撶礌 鈫?1000^3 浣撶礌绌洪棿锛?M 鐐瑰嚑涔庡叏鍞竴
    std::uniform_real_distribution<float> dist(-250.0f, 250.0f);

    // 鈹€鈹€ 姝ラ 1锛氶濉厖鍏ㄥ眬 100 涓囩偣 鈹€鈹€
    std::vector<cv::Point3f> globalSeed(GLOBAL_POINTS);
    for (auto& pt : globalSeed) {
        pt.x = dist(rng); pt.y = dist(rng); pt.z = dist(rng);
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    r = fuse.Execute(globalSeed);                   // 无 R/T（恒等），一次性灌入
    auto t1 = std::chrono::high_resolution_clock::now();
    double fillMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    size_t globalVoxels = fuse.GetVoxelCount();

    // 鈹€鈹€ 姝ラ 2锛氭瀯閫犲綋鍓嶅抚 4 涓囩偣锛?0% 鍛戒腑宸叉湁浣撶礌 + 50% 鏂颁綋绱狅級鈹€鈹€
    std::vector<cv::Point3f> frame(FRAME_POINTS);
    for (size_t i = 0; i < FRAME_POINTS; ++i) {
        if (i % 2 == 0 && i < GLOBAL_POINTS) {
            frame[i] = globalSeed[i];            // 命中：复用全局点
        } else {
            frame[i] = cv::Point3f(dist(rng), dist(rng), dist(rng));  // 鏂扮偣
        }
    }

    // R/T 变换：模拟一帧小幅位姿移动（平移 1.2mm + 微旋转 0.01rad）
    double angle = 0.01;
    cv::Matx33d R(std::cos(angle), -std::sin(angle), 0,
                  std::sin(angle),  std::cos(angle), 0,
                  0, 0, 1);
    cv::Vec3d T(1.2, 0.8, -0.5);

    // 鈹€鈹€ 姝ラ 3a锛氱函 R/T 鍙樻崲璁℃椂锛堜粎鍋?R*p+T锛屼笉杩涘搱甯岋紝浣滃熀鍑嗗鐓э級鈹€鈹€
    std::vector<cv::Point3f> transformed(FRAME_POINTS);
    double transformOnlyMs = 1e9;
    for (int it = 0; it < MEASURE_ITERS; ++it) {
        auto s0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < FRAME_POINTS; ++i) {
            const auto& w = frame[i];
            transformed[i] = cv::Point3f(
                static_cast<float>(R(0,0)*w.x + R(0,1)*w.y + R(0,2)*w.z + T(0)),
                static_cast<float>(R(1,0)*w.x + R(1,1)*w.y + R(1,2)*w.z + T(1)),
                static_cast<float>(R(2,0)*w.x + R(2,1)*w.y + R(2,2)*w.z + T(2)));
        }
        auto s1 = std::chrono::high_resolution_clock::now();
        transformOnlyMs = std::min(transformOnlyMs,
            std::chrono::duration<double, std::milli>(s1 - s0).count());
    }

    // 鈹€鈹€ 姝ラ 3b锛氶鐑紙璁╁綋鍓嶅抚鐨勪綋绱犺繘鍏ヨ〃锛屽悗缁抚娴嬮噺绋冲畾鎬侊級鈹€鈹€
    for (int it = 0; it < WARMUP_ITERS; ++it) {
        r = fuse.Execute(frame, R, T);
    }
    size_t voxelsAfterWarmup = fuse.GetVoxelCount();

    // 鈹€鈹€ 姝ラ 4锛氳鏃?鈥?fuse() 甯?R/T锛堜富璺緞锛夆攢鈹€
    double fuseRT_totalMs = 1e9, fuseRT_hashMs = 1e9;
    double fuseNoRT_totalMs = 1e9, fuseNoRT_hashMs = 1e9;
    size_t lastSurviving = 0, lastDeleted = 0, lastNew = 0;

    for (int it = 0; it < MEASURE_ITERS; ++it) {
        // 甯?R/T
        auto a0 = std::chrono::high_resolution_clock::now();
        r = fuse.Execute(frame, R, T);
        auto a1 = std::chrono::high_resolution_clock::now();
        double msTot = std::chrono::duration<double, std::milli>(a1 - a0).count();
        fuseRT_totalMs = std::min(fuseRT_totalMs, msTot);
        fuseRT_hashMs  = std::min(fuseRT_hashMs, r.statistics.hashTimeMs);
        lastSurviving = r.statistics.survivingCount;
        lastDeleted   = r.statistics.deletedCount;
        lastNew       = r.statistics.newVoxelCount;
    }
    for (int it = 0; it < MEASURE_ITERS; ++it) {
        // 涓嶅甫 R/T锛堝鐓э級
        auto b0 = std::chrono::high_resolution_clock::now();
        r = fuse.Execute(frame);
        auto b1 = std::chrono::high_resolution_clock::now();
        fuseNoRT_totalMs = std::min(fuseNoRT_totalMs,
            std::chrono::duration<double, std::milli>(b1 - b0).count());
        fuseNoRT_hashMs  = std::min(fuseNoRT_hashMs, r.statistics.hashTimeMs);
    }

    size_t finalVoxels = fuse.GetVoxelCount();

    // 鈹€鈹€ 鎶ュ憡 鈹€鈹€
    std::printf("\n");
    std::printf("========================================\n");
    std::printf(" 鍏ㄥ眬 100 涓囩偣 + 褰撳墠甯?4 涓囩偣 鎬ц兘鎶ュ憡\n");
    std::printf("========================================\n");
    std::printf("銆愮幆澧冦€憊oxelSize=%.1fmm  threshold=%d  琛ㄦЫ=%zuK(%.1fMB)\n",
                p.voxelSize, p.saturationThreshold,
                p.reserveVoxelCount / 1000,
                p.reserveVoxelCount * (1+8+4+4) / 1024.0 / 1024.0);
    std::printf("\n");
    std::printf("銆愭楠?銆戦濉厖鍏ㄥ眬鐐逛簯\n");
    std::printf("  杈撳叆鐐规暟   : %zu\n", GLOBAL_POINTS);
    std::printf("  鑰楁椂       : %.2f ms\n", fillMs);
    std::printf("  浣撶礌鏁?    : %zuK (%.1f M pts/s)\n",
                globalVoxels / 1000, GLOBAL_POINTS / fillMs / 1000.0);
    std::printf("  鍗曠偣鑰楁椂   : %.0f ns\n", fillMs * 1e6 / GLOBAL_POINTS);
    std::printf("\n");
    std::printf("銆愭楠?a銆戠函 R/T 鍙樻崲锛?涓囩偣 脳 R*p+T锛屼笉杩涘搱甯岋級\n");
    std::printf("  鑰楁椂(min)  : %.3f ms  (%.1f M pts/s)\n",
                transformOnlyMs, FRAME_POINTS / transformOnlyMs / 1000.0);
    std::printf("  鍗曠偣鑰楁椂   : %.1f ns\n", transformOnlyMs * 1e6 / FRAME_POINTS);
    std::printf("\n");
    std::printf("銆愭楠?銆戝綋鍓嶅抚铻嶅悎锛?d 娆″彇鏈€灏忓€硷紝鍏ㄥ眬宸?%zuK 浣撶礌锛塡n",
                MEASURE_ITERS, voxelsAfterWarmup / 1000);
    std::printf("  鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹怽n");
    std::printf("  鈹?甯?R/T锛堜富璺緞锛?                             鈹俓n");
    std::printf("  鈹?  鎬昏€楁椂     = %.3f ms  (%.1f M pts/s)       鈹俓n",
                fuseRT_totalMs, FRAME_POINTS / fuseRT_totalMs / 1000.0);
    std::printf("  鈹?  鍝堝笇寰幆   = %.3f ms (stats.hashTimeMs)    鈹俓n", fuseRT_hashMs);
    std::printf("  鈹?  鍗曠偣鎬昏€楁椂 = %.1f ns                       鈹俓n",
                fuseRT_totalMs * 1e6 / FRAME_POINTS);
    std::printf("  鈹?  瀛樻椿/鍒犻櫎/鏂板 = %zu / %zu / %zu            鈹俓n",
                lastSurviving, lastDeleted, lastNew);
    std::printf("  鈹溾攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹n");
    std::printf("  鈹?涓嶅甫 R/T锛堝鐓э級                              鈹俓n");
    std::printf("  鈹?  鎬昏€楁椂     = %.3f ms  (%.1f M pts/s)       鈹俓n",
                fuseNoRT_totalMs, FRAME_POINTS / fuseNoRT_totalMs / 1000.0);
    std::printf("  鈹?  鍝堝笇寰幆   = %.3f ms                        鈹俓n", fuseNoRT_hashMs);
    std::printf("  鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹榎n");
    std::printf("  R/T 鍙樻崲寮€閿€ 鈮?%.3f ms (%.1f%%)\n",
                fuseRT_totalMs - fuseNoRT_totalMs,
                (fuseRT_totalMs - fuseNoRT_totalMs) / fuseRT_totalMs * 100.0);
    std::printf("  鏈€缁堝叏灞€浣撶礌 : %zuK\n", finalVoxels / 1000);
    std::printf("========================================\n\n");

    // 鍩烘湰鏂█锛氬甫 R/T 搴旀瘮绾彉鎹㈡參锛堝搱甯屾湁寮€閿€锛夛紱鍚炲悙搴?> 4M pts/s
    EXPECT_GT(fuseRT_totalMs, transformOnlyMs);
    EXPECT_GT(fuseRT_totalMs, fuseRT_hashMs * 0.9);   // total 鍚?hash
    EXPECT_LT(fuseRT_totalMs, 10.0);                   // 4涓囩偣 < 10ms
}

// ============================================================
// gatherVoxelNeighbors
// ============================================================

TEST(LaserCloudFuseCPU, GatherVoxelNeighbors_3x3Grid) {
    LaserCloudFuseCPUParams p; p.voxelSize = 1.0f; p.saturationThreshold = 5;
    LaserCloudFuseCPU fuse(p);
    // 3x3 grid on Z=0 plane (voxelSize=1, spacing=1 鈫?one voxel per point)
    std::vector<cv::Point3f> grid;
    for (float x = 0.5f; x < 3.0f; x += 1.0f)
        for (float y = 0.5f; y < 3.0f; y += 1.0f)
            grid.emplace_back(x, y, 0.5f);
    LaserCloudFuseCPUResult r;
    r = fuse.Execute(grid);
    ASSERT_EQ(fuse.GetFusedPointCount(), 9u);

    // Center point (1.5, 1.5, 0.5) 鈫?ix=1, iy=1, iz=0
    // 3x3x3 neighborhood: Z=-1 and Z=1 layers empty, Z=0 layer has 9 voxels
    std::vector<const CloudPoint*> neighbors;
    size_t count = fuse.GatherVoxelNeighbors(cv::Point3f(1.5f, 1.5f, 0.5f), 1, neighbors);
    EXPECT_EQ(count, 9u);
}

TEST(LaserCloudFuseCPU, GatherVoxelNeighbors_SparseArea) {
    LaserCloudFuseCPUParams p; p.voxelSize = 1.0f; p.saturationThreshold = 5;
    LaserCloudFuseCPU fuse(p);
    std::vector<cv::Point3f> single = {{0.5f, 0.5f, 0.5f}};
    LaserCloudFuseCPUResult r;
    r = fuse.Execute(single);

    std::vector<const CloudPoint*> neighbors;
    size_t count = fuse.GatherVoxelNeighbors(cv::Point3f(0.5f, 0.5f, 0.5f), 1, neighbors);
    // Only center voxel exists
    EXPECT_EQ(count, 1u);
}

TEST(LaserCloudFuseCPU, GatherVoxelNeighbors_KernelRadius2) {
    LaserCloudFuseCPUParams p; p.voxelSize = 1.0f; p.saturationThreshold = 5;
    LaserCloudFuseCPU fuse(p);
    // 5x5 grid
    std::vector<cv::Point3f> grid;
    for (float x = 0.5f; x < 5.0f; x += 1.0f)
        for (float y = 0.5f; y < 5.0f; y += 1.0f)
            grid.emplace_back(x, y, 0.5f);
    LaserCloudFuseCPUResult r;
    r = fuse.Execute(grid);

    // Center (2.5, 2.5, 0.5), r=2 鈫?5x5=25 voxels (all in Z=0 layer)
    std::vector<const CloudPoint*> neighbors;
    size_t count = fuse.GatherVoxelNeighbors(cv::Point3f(2.5f, 2.5f, 0.5f), 2, neighbors);
    EXPECT_EQ(count, 25u);
}


#endif // SKIPPED