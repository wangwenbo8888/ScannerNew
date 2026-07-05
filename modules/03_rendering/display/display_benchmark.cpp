// ============================================================
// display_benchmark.cpp — 点云显示渲染性能基准测试
//
// 测量各步骤耗时：
//   1. CUDA 展点 kernel
//   2. D2H 下载
//   3. OSG 数组填充
//   4. OSG 单帧渲染
//   5. update() 总耗时
// ============================================================

#include "scanner_viewer.h"
#include "laser_cloud_renderer.h"
#include "marker_cloud_renderer.h"
#include "point_expand_kernel.h"
#include "laser_cloud_fuse_cuda.h"
#include "core/marker/marker_cloud_fuse_cpu/marker_cloud_fuse_cpu.h"

#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <cstdio>
#include <cmath>
#include <random>

using namespace calib;

// 计时辅助
using Clock = std::chrono::high_resolution_clock;
template<typename T>
static double ms(const T& d) {
    return std::chrono::duration<double, std::milli>(d).count();
}

#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        printf("CUDA ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return 1; \
    } \
} while(0)

// ============================================================
// 生成合成点云（球面上 N 个点，法线朝外）
// ============================================================
static void generateSyntheticPoints(
    float* h_xyz, float* h_normal, size_t N, float radius)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(0, 1);

    for (size_t i = 0; i < N; ++i) {
        float theta = u(rng) * 3.14159265f * 2;
        float phi   = std::acos(1.0f - 2.0f * u(rng));
        float nx = std::sin(phi) * std::cos(theta);
        float ny = std::sin(phi) * std::sin(theta);
        float nz = std::cos(phi);

        h_xyz[i*3]   = nx * radius;
        h_xyz[i*3+1] = ny * radius;
        h_xyz[i*3+2] = nz * radius;

        h_normal[i*3]   = nx;
        h_normal[i*3+1] = ny;
        h_normal[i*3+2] = nz;
    }
}

// ============================================================
// main
// ============================================================
int main(int argc, char** argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);  // 无缓冲，崩溃时也能看到输出
    setvbuf(stderr, NULL, _IONBF, 0);
    // 测试参数
    size_t N = (argc > 1) ? std::stoull(argv[1]) : 1000000;  // 默认 100 万
    float  voxelSize = 0.5f;
    float  sphereRadius = 100.0f;

    printf("=== 点云显示渲染性能基准 ===\n");
    printf("点数: %zu (%.1fK)\n", N, N / 1000.0);
    printf("体素边长: %.2f\n", voxelSize);
    printf("GPU: %s\n\n", "Quadro RTX 5000");

    // ---- 1. 生成合成数据 ----
    printf("[1] 生成合成点云... ");
    std::vector<float> h_xyz(N * 3), h_normal(N * 3);
    generateSyntheticPoints(h_xyz.data(), h_normal.data(), N, sphereRadius);
    printf("OK (%zu 点)\n", N);

    // ---- 2. 上传到 GPU ----
    printf("[2] 上传到 GPU... ");
    float *d_xyz = nullptr, *d_normal = nullptr;
    CUDA_CHECK(cudaMalloc(&d_xyz, N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_normal, N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_xyz, h_xyz.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_normal, h_normal.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
    printf("OK\n");

    // ---- 3. 单独测试展点 kernel ----
    printf("[3] 展点 kernel 计时...\n");
    {
        size_t vertCount = N * 4;
        float *d_outPos, *d_outUv, *d_outNormal;
        CUDA_CHECK(cudaMalloc(&d_outPos, vertCount * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_outUv, vertCount * 2 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_outNormal, vertCount * 3 * sizeof(float)));

        // warmup
        launchExpandLaserPoints(d_outPos, d_outUv, d_outNormal,
                                d_xyz, d_normal, voxelSize * 0.5f, N, nullptr);
        cudaDeviceSynchronize();

        // 计时（CUDA events）
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        const int RUNS = 10;
        cudaEventRecord(start);
        for (int i = 0; i < RUNS; ++i) {
            launchExpandLaserPoints(d_outPos, d_outUv, d_outNormal,
                                    d_xyz, d_normal, voxelSize * 0.5f, N, nullptr);
        }
        cudaEventRecord(stop);
        cudaDeviceSynchronize();

        float msTotal = 0;
        cudaEventElapsedTime(&msTotal, start, stop);
        float msAvg = msTotal / RUNS;
        printf("    kernel: %.3f ms/帧 (%d 次平均)\n", msAvg, RUNS);

        // D2H 下载计时
        std::vector<float> h_outPos(vertCount * 3);
        cudaEventRecord(start);
        for (int i = 0; i < RUNS; ++i) {
            cudaMemcpyAsync(h_outPos.data(), d_outPos,
                           vertCount * 3 * sizeof(float),
                           cudaMemcpyDeviceToHost);
        }
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&msTotal, start, stop);
        printf("    D2H 下载 pos (%zu MB): %.3f ms/帧\n",
               vertCount * 3 * sizeof(float) / (1024*1024), msTotal / RUNS);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_outPos);
        cudaFree(d_outUv);
        cudaFree(d_outNormal);
    }

    // ---- 4. 构造 DeviceContext ----
    LaserCloudFuseDeviceContext ctx;
    ctx.d_fusedXyz    = d_xyz;
    ctx.d_fusedNormal = d_normal;
    ctx.voxelSize     = voxelSize;
    ctx.fusedPointCount = N;

    // ---- 5. 初始化 ScannerViewer ----
    printf("[4] 初始化 ScannerViewer... ");
    ScannerViewer viewer(N, 256);
    viewer.init(1280, 720);
    printf("OK\n\n");

    // ---- 6. 测试 LaserCloudRenderer::update() ----
    printf("[5] LaserCloudRenderer::update() 计时\n");
    {
        const int RUNS = 20;
        double totalMs = 0;
        double minMs = 1e9, maxMs = 0;

        // warmup
        viewer.laserRenderer()->update(ctx);

        for (int i = 0; i < RUNS; ++i) {
            auto t0 = Clock::now();
            viewer.laserRenderer()->update(ctx);
            auto t1 = Clock::now();
            double d = ms(t1 - t0);
            totalMs += d;
            minMs = std::min(minMs, d);
            maxMs = std::max(maxMs, d);
        }
        printf("    update 总计: avg=%.3f ms, min=%.3f ms, max=%.3f ms (%d 次)\n",
               totalMs / RUNS, minMs, maxMs, RUNS);
        printf("    （仅存储指针，实际展点在 frame() draw 时 GPU 零拷贝完成）\n");
    }

    // ---- 7. 测试 OSG 单帧渲染 ----
    printf("[6] OSG 单帧渲染计时（含 视锥+背面剔除 + GPU 展点 + 绘制）\n");
    {
        const int RUNS = 60;
        double totalMs = 0;
        double minMs = 1e9, maxMs = 0;

        for (int i = 0; i < RUNS; ++i) {
            auto t0 = Clock::now();
            viewer.frame();
            auto t1 = Clock::now();
            double d = ms(t1 - t0);
            totalMs += d;
            minMs = std::min(minMs, d);
            maxMs = std::max(maxMs, d);
        }
        printf("    frame(): avg=%.2f ms, min=%.2f ms, max=%.2f ms (%d 次)\n",
               totalMs / RUNS, minMs, maxMs, RUNS);
        printf("    → 理论 FPS: %.0f\n", 1000.0 / (totalMs / RUNS));
    }

    // ---- 8. 端到端（update + frame） ----
    printf("[7] 端到端（update + frame，全 GPU 零拷贝）计时\n");
    {
        const int RUNS = 30;
        double totalMs = 0;

        for (int i = 0; i < RUNS; ++i) {
            auto t0 = Clock::now();
            viewer.laserRenderer()->update(ctx);
            viewer.frame();
            auto t1 = Clock::now();
            totalMs += ms(t1 - t0);
        }
        double avg = totalMs / RUNS;
        printf("    avg=%.2f ms/帧 → 理论 FPS: %.0f\n", avg, 1000.0 / avg);
    }

    // ---- 9. 标记点全 CPU 路径测试 ----
    printf("\n[8] 标记点全 CPU 路径（fuse + update + flush + render）\n");
    {
        // 模拟 100 帧扫描，每帧 50 个标记点，相机移动
        const int   FRAMES    = 100;
        const int   MARKERS   = 50;
        const float RADIUS    = 3.0f;

        std::vector<MarkerFuseInput> frameMarkers(MARKERS);
        for (int i = 0; i < MARKERS; ++i) {
            float a = i * 0.3f;
            frameMarkers[i].x = std::cos(a) * sphereRadius * 1.1f;
            frameMarkers[i].y = std::sin(a) * sphereRadius * 1.1f;
            frameMarkers[i].z = 0;
            frameMarkers[i].nx = 0; frameMarkers[i].ny = 0; frameMarkers[i].nz = 1;
            frameMarkers[i].whiteRadius = RADIUS;
        }

        cv::Matx33d Rmat(1,0,0, 0,1,0, 0,0,1);
        cv::Vec3d Tvec(0,0,0);

        MarkerCloudFuseCPU fuse;

        // 首帧融合
        auto t0 = Clock::now();
        auto result = fuse.Execute(frameMarkers, Rmat, Tvec);
        auto t1 = Clock::now();

        // 模拟多帧融合（每帧略微偏移，部分重复）
        auto t2 = Clock::now();
        for (int f = 0; f < FRAMES; ++f) {
            cv::Vec3d TT(0.1f * f, 0, 0);
            result = fuse.Execute(frameMarkers, Rmat, TT);
        }
        auto t3 = Clock::now();

        printf("    fuse 首帧(%d 点): %.3f ms\n",
               MARKERS, ms(t1 - t0));
        printf("    fuse %d 帧(每帧 %d 点): avg=%.3f ms/帧, 全局累积=%zu 点\n",
               FRAMES, MARKERS, ms(t3 - t2) / FRAMES,
               fuse.GetFusedPointCount());

        // 渲染器 update + flush
        const auto& fusedPoints = fuse.GetFusedPoints();
        auto t4 = Clock::now();
        viewer.markerRenderer()->update(fusedPoints);
        viewer.markerRenderer()->flush();
        auto t5 = Clock::now();
        printf("    update+flush(%zu 点): %.3f ms\n",
               fusedPoints.size(), ms(t5 - t4));

        // 渲染
        auto t6 = Clock::now();
        viewer.frame();
        auto t7 = Clock::now();
        printf("    frame() 含标记点渲染: %.3f ms\n", ms(t7 - t6));

        printf("    标记点全路径: fuse=%.3f + update+flush=%.3f + frame=%.3f = %.3f ms/frame\n",
               ms(t3-t2)/FRAMES, ms(t5-t4), ms(t7-t6),
               ms(t3-t2)/FRAMES + ms(t5-t4) + ms(t7-t6));
    }

    // ---- 报告 ----
    printf("\n=== 测试完成 ===\n");

    cudaFree(d_xyz);
    cudaFree(d_normal);
    return 0;
}
