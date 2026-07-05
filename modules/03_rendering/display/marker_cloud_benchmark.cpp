// ============================================================
// marker_cloud_benchmark.cpp — 标记点云融合+渲染 性能剖面
//
// 场景：20 个当前帧标记点 → 逐帧融合 → 500 个全局标记点 → 渲染
// 计时步骤：
//   1. marker_cloud_fuse_cpu::fuse() 每帧耗时（R/T变换 + 体素哈希去重）
//   2. renderer.update() 耗时（存引用，零开销）
//   3. renderer.flush() 耗时（展点 + OSG VBO 填充）
// ============================================================

#include "marker_cloud_renderer.h"
#include "core/marker/marker_cloud_fuse_cpu/marker_cloud_fuse_cpu.h"

#include <chrono>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>

using namespace calib;

// ---- 计时辅助 ----
using Clock = std::chrono::high_resolution_clock;

template<typename T>
static double ns(const T& d) { return std::chrono::duration<double, std::nano>(d).count(); }
template<typename T>
static double us(const T& d) { return ns(d) / 1000.0; }
template<typename T>
static double ms(const T& d) { return ns(d) / 1'000'000.0; }

// ============================================================
// 生成 N 个唯一标记点（Fibonacci 球面分布）
// ============================================================
static std::vector<MarkerFuseInput> generateMarkers(size_t total, float sphereR, float whiteR)
{
    std::vector<MarkerFuseInput> pts(total);
    float goldenRatio = (1.0f + std::sqrt(5.0f)) * 0.5f;

    for (size_t i = 0; i < total; ++i) {
        float theta = 2.0f * 3.14159265f * (float)i / goldenRatio;
        float phi   = std::acos(1.0f - 2.0f * ((float)i + 0.5f) / (float)total);
        float nx = std::sin(phi) * std::cos(theta);
        float ny = std::sin(phi) * std::sin(theta);
        float nz = std::cos(phi);

        pts[i].x = nx * sphereR;
        pts[i].y = ny * sphereR;
        pts[i].z = nz * sphereR;
        pts[i].nx = nx;
        pts[i].ny = ny;
        pts[i].nz = nz;
        pts[i].whiteRadius = whiteR;
    }
    return pts;
}

// ============================================================
// main
// ============================================================
int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    const size_t FRAME_MARKERS = 20;
    const size_t TOTAL_MARKERS = 500;
    const float  SPHERE_RADIUS = 150.0f;
    const float  WHITE_RADIUS  = 3.0f;

    printf("==========================================================\n");
    printf("  标记点云 融合+渲染 性能剖面\n");
    printf("==========================================================\n");
    printf("  当前帧标记点:  %zu 个\n", FRAME_MARKERS);
    printf("  目标全局标记点: %zu 个\n", TOTAL_MARKERS);
    printf("  去重方式:      体素哈希 (0.5mm voxel, threshold=99)\n\n");

    // ---- 生成全部 500 个标记点（相机坐标系，作为 MarkerFuseInput）----
    printf("[1] 生成 %zu 个唯一标记点... ", TOTAL_MARKERS);
    auto allMarkers = generateMarkers(TOTAL_MARKERS, SPHERE_RADIUS, WHITE_RADIUS);
    printf("OK\n");

    // ---- 初始化融合器 + 渲染器 ----
    printf("[2] 初始化 MarkerCloudFuseCPU + MarkerCloudRenderer... ");
    MarkerCloudFuseCPU fuse;
    MarkerCloudRenderer renderer(1024);
    printf("OK\n\n");

    // ---- R/T ----
    cv::Matx33d Rmat(1,0,0, 0,1,0, 0,0,1);
    cv::Vec3d Tvec(0,0,0);

    // ============================================================
    // [3] 逐帧融合计时
    // ============================================================
    printf("[3] 逐帧 fuse() 计时 (体素哈希融合)\n");
    printf("    --------------------------------------------------\n");
    printf("    帧   |  输入点  |  耗时(us)  |  新体素 |  全局总数\n");
    printf("    --------------------------------------------------\n");

    size_t framesNeeded = (TOTAL_MARKERS + FRAME_MARKERS - 1) / FRAME_MARKERS;
    std::vector<double> fuseTimes_us;
    size_t markerIdx = 0;

    for (size_t f = 0; f < framesNeeded; ++f) {
        std::vector<MarkerFuseInput> frame;
        frame.reserve(FRAME_MARKERS);
        for (size_t i = 0; i < FRAME_MARKERS && markerIdx < TOTAL_MARKERS; ++i, ++markerIdx) {
            frame.push_back(allMarkers[markerIdx]);
        }

        auto t0 = Clock::now();
        auto result = fuse.Execute(frame, Rmat, Tvec);
        auto t1 = Clock::now();

        double dt_us = us(t1 - t0);
        fuseTimes_us.push_back(dt_us);

        printf("    %3zu   |   %3zu    |  %8.1f    |  %5zu  |   %zu\n",
               f + 1, frame.size(), dt_us,
               result.statistics.newVoxelCount, fuse.GetFusedPointCount());
    }

    double sumFuse = 0, minFuse = 1e9, maxFuse = 0;
    for (double v : fuseTimes_us) { sumFuse += v; minFuse = std::min(minFuse, v); maxFuse = std::max(maxFuse, v); }

    printf("    --------------------------------------------------\n");
    printf("    fuse 统计 (%zu 帧):\n", fuseTimes_us.size());
    printf("      总耗时:  %.3f ms\n", sumFuse / 1000.0);
    printf("      平均:    %.1f us/帧\n", sumFuse / (double)fuseTimes_us.size());
    printf("      最快:    %.1f us\n", minFuse);
    printf("      最慢:    %.1f us\n", maxFuse);
    printf("      最终全局: %zu 点\n\n", fuse.GetFusedPointCount());

    // ============================================================
    // [4] renderer.update() + flush() 计时
    // ============================================================
    printf("[4] 渲染器 计时\n");
    double flushAvg_us = 0;
    {
        const auto& fusedPoints = fuse.GetFusedPoints();

        // update() — 仅存引用
        auto t0 = Clock::now();
        renderer.update(fusedPoints);
        auto t1 = Clock::now();
        printf("    update(%zu 点): %.1f us (纯存引用, 零拷贝)\n",
               fusedPoints.size(), us(t1 - t0));

        // flush() — 展点 + VBO 写入
        const int F_RUNS = 20;
        double sumF = 0, minF = 1e9, maxF = 0;
        for (int i = 0; i < F_RUNS; ++i) {
            auto ts = Clock::now();
            renderer.flush();
            auto te = Clock::now();
            double d = us(te - ts);
            sumF += d; minF = std::min(minF, d); maxF = std::max(maxF, d);
        }
        flushAvg_us = sumF / F_RUNS;
        printf("    flush(%zu 点): avg=%.1f us, min=%.1f us, max=%.1f us  (%d 次)\n",
               fusedPoints.size(), flushAvg_us, minF, maxF, F_RUNS);
        printf("    每点: %.1f ns\n\n", flushAvg_us * 1000.0 / fusedPoints.size());
    }

    // ============================================================
    // [5] 展点数学微基准
    // ============================================================
    printf("[5] expandMarkerPointCPU 展点数学 (纯 CPU)\n");
    {
        const auto& fusedPoints = fuse.GetFusedPoints();
        ExpandVertex vBuf[4];
        const int N = fusedPoints.size() > 0 ? (int)fusedPoints.size() : 1;

        auto t0 = Clock::now();
        const int RUNS = 1000;
        for (int r = 0; r < RUNS; ++r) {
            for (int i = 0; i < N; ++i) {
                const auto& p = fusedPoints[i];
                float outerR = p.whiteRadius * MarkerCloudRenderer::RING_RATIO;
                expandMarkerPointCPU(vBuf, p.x, p.y, p.z, p.nx, p.ny, p.nz, outerR);
            }
        }
        auto t1 = Clock::now();
        printf("    expandMarkerPointCPU: %.1f ns/点  (%d 点, %d 次)\n\n",
               ns(t1 - t0) / (double)(RUNS * N), N, RUNS);
    }

    // ============================================================
    // [6] 汇总
    // ============================================================
    printf("==========================================================\n");
    printf("  汇总报告\n");
    printf("==========================================================\n");
    printf("  场景: %zu 点/帧 × %zu 帧 → %zu 全局标记点\n",
           FRAME_MARKERS, fuseTimes_us.size(), fuse.GetFusedPointCount());
    printf("\n");
    printf("  步骤                    耗时             说明\n");
    printf("  -----------------------------------------------------\n");
    printf("  fuse (单帧均值)         %5.0f us         体素哈希融合(R/T+去重)\n",
           sumFuse / (double)fuseTimes_us.size());
    printf("  fuse (合计)             %5.1f ms         %zu 帧\n",
           sumFuse / 1000.0, fuseTimes_us.size());
    printf("  update                   ~0 us           存引用\n");
    printf("  flush (%zu点)            %5.0f us        展点+VBO\n",
           fuse.GetFusedPointCount(), flushAvg_us);
    printf("  -----------------------------------------------------\n");
    printf("  全路径 (fuse+update+flush): ~%.1f ms\n",
           sumFuse / 1000.0 + flushAvg_us / 1000.0);
    printf("\n");
    printf("  数据通路: fuse → vector<MarkerCloudPoint> → update(存引用) → flush(展点)\n");
    printf("  融合-02M 为全局标记点唯一权威源, 渲染器零冗余消费。\n");
    printf("==========================================================\n");

    return 0;
}
