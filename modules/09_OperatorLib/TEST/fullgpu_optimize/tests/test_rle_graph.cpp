/**
 * @file test_rle_graph.cpp
 * @brief 【批C·RLE 入图】槽图单元级验收（T1/T2/T3/T4，规格=二轮-R2规格RLE入图 §6）
 *
 *   T1 图/逐发逐位对拍（5 组合成掩膜：空/满/随机/超窗/超 cap；组间交叉复跑）
 *   T2 超窗注入（图窗口拷＋gpuRleTailCopy 拼装 vs 逐发整包）
 *   T3 尺寸变更重建（不同 H×W 各自成图皆正确——近似覆盖；执行器级签名重建
 *      由 T5/对拍门覆盖）
 *   T4 事件节点外部消费（evRle/evGate record 入图；不支持则 GSKIP 非 FAIL——
 *      K1 裁决点；退守变体在案）
 */

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include <algorithm>
#include <cstdlib>
#include <random>
#include <vector>

#include "gpu_accum.h"

namespace {

constexpr int kH = 64, kW = 1024, kCap = 256, kWin = 100;

cv::Mat MakeMask(int kind) {
    cv::Mat m(kH, kW, CV_8UC1, cv::Scalar(0));
    std::mt19937 rng(12345 + kind);
    switch (kind) {
        case 0: break;                                    // 空
        case 1: m.setTo(255); break;                      // 满（每行 1 行程）
        case 2: {                                         // 随机 ~2 行程/行
            for (int y = 0; y < kH; ++y)
                for (int r = 0; r < 2; ++r) {
                    const int x0 = static_cast<int>(rng() % (kW - 8));
                    const int len = 3 + static_cast<int>(rng() % 5);
                    for (int x = x0; x < x0 + len && x < kW; ++x) m.at<uchar>(y, x) = 255;
                }
            break;
        }
        case 3: {                                         // 每行 8 行程 → 512 > cap（钳位）
            for (int y = 0; y < kH; ++y)
                for (int r = 0; r < 8; ++r) {
                    const int x0 = (kW / 8) * r + 4;
                    m.at<uchar>(y, x0) = 255;
                    m.at<uchar>(y, x0 + 1) = 255;
                }
            break;
        }
        default: {                                        // 每行 3 行程 → 192 ∈ (win, cap)
            for (int y = 0; y < kH; ++y)
                for (int r = 0; r < 3; ++r) {
                    const int x0 = (kW / 3) * r + 2;
                    m.at<uchar>(y, x0) = 255;
                }
            break;
        }
    }
    return m;
}

struct RleFixture {
    cv::cuda::GpuMat dMaskL, dMaskR, bufL, bufR;
    int* cntPin = nullptr;    // 2
    int* runsPin = nullptr;   // 6*cap
    cv::cuda::Stream stream;
    cudaEvent_t evRle = nullptr, evGate = nullptr;

    RleFixture() {
        bufL.create(1, 3 * kCap, CV_32SC1);
        bufR.create(1, 3 * kCap, CV_32SC1);
        cudaMallocHost(&cntPin, 2 * sizeof(int));
        cudaMallocHost(&runsPin, 6 * kCap * sizeof(int));
        cudaEventCreateWithFlags(&evRle, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&evGate, cudaEventDisableTiming);
    }
    ~RleFixture() {
        cudaEventDestroy(evRle);
        cudaEventDestroy(evGate);
        cudaFreeHost(cntPin);
        cudaFreeHost(runsPin);
    }
    void Upload(const cv::Mat& hL, const cv::Mat& hR) {
        dMaskL.upload(hL, stream);
        dMaskR.upload(hR, stream);
    }
    std::vector<int> EagerWindowed() {
        std::fill(cntPin, cntPin + 2, -1);
        std::fill(runsPin, runsPin + 6 * kCap, -7);
        EXPECT_TRUE(scr::gpuMaskRleBothWindowed(dMaskL, dMaskR, bufL, bufR, kCap, kWin,
                                                cntPin, runsPin, stream));
        stream.waitForCompletion();
        return std::vector<int>(runsPin, runsPin + 6 * kCap);
    }
    // 图路径：构建＋发射＋图外常规 record（退守变体定案）＋事件同步
    bool GraphWindowed(std::vector<int>& runsOut, int* cntOut, cudaGraph_t* gKeep = nullptr,
                       cudaGraphExec_t* geKeep = nullptr) {
        std::fill(cntPin, cntPin + 2, -1);
        std::fill(runsPin, runsPin + 6 * kCap, -7);
        cudaGraph_t g = nullptr;
        cudaGraphExec_t ge = nullptr;
        if (!scr::gpuRleBuildSlotGraph(dMaskL, dMaskR, bufL, bufR, kCap, kWin, cntPin,
                                       runsPin, stream, &g, &ge))
            return false;
        cudaStream_t cs = cv::cuda::StreamAccessor::getStream(stream);
        if (cudaGraphLaunch(ge, cs) != cudaSuccess
            || cudaEventRecord(evRle, cs) != cudaSuccess
            || cudaEventRecord(evGate, cs) != cudaSuccess) {
            cudaGraphExecDestroy(ge);
            cudaGraphDestroy(g);
            return false;
        }
        EXPECT_EQ(cudaSuccess, cudaEventSynchronize(evRle));   // T4：图后 record 外部消费
        EXPECT_EQ(cudaSuccess, cudaEventSynchronize(evGate));
        runsOut.assign(runsPin, runsPin + 6 * kCap);
        if (cntOut) cntOut[0] = cntPin[0], cntOut[1] = cntPin[1];
        if (gKeep) *gKeep = g;
        else cudaGraphDestroy(g);
        if (geKeep) *geKeep = ge;
        else cudaGraphExecDestroy(ge);
        return true;
    }
};

TEST(RleSlotGraph, T1GraphVsEagerBitwiseFiveGroups) {
    RleFixture f;
    for (int round = 0; round < 2; ++round) {          // 组间交叉复跑（含首轮暂存分配）
        for (int kind = 0; kind < 5; ++kind) {
            const cv::Mat hL = MakeMask(kind);
            const cv::Mat hR = MakeMask(kind + 1);     // 左右异组交叉
            f.Upload(hL, hR);
            std::vector<int> eager = f.EagerWindowed();
            std::vector<int> graph;
            int cnt[2] = {0, 0};
            ASSERT_TRUE(f.GraphWindowed(graph, cnt)) << "图构建/发射失败（退守变体下不应发生）";
            ASSERT_EQ(eager.size(), graph.size());
            for (size_t i = 0; i < eager.size(); ++i)
                ASSERT_EQ(eager[i], graph[i])
                    << "round=" << round << " kind=" << kind << " int#" << i;
        }
    }
}

TEST(RleSlotGraph, T2OverWindowTailAssembly) {
    RleFixture f;
    const cv::Mat hL = MakeMask(4);                    // 192 行程 ∈ (win=100, cap=256)
    const cv::Mat hR = MakeMask(4);
    f.Upload(hL, hR);
    // 基准＝逐发整包（非窗口版）
    std::fill(f.cntPin, f.cntPin + 2, -1);
    std::fill(f.runsPin, f.runsPin + 6 * kCap, -7);
    EXPECT_TRUE(scr::gpuMaskRleBoth(f.dMaskL, f.dMaskR, f.bufL, f.bufR, kCap,
                                    f.cntPin, f.runsPin, f.stream));
    f.stream.waitForCompletion();
    const std::vector<int> full(f.runsPin, f.runsPin + 6 * kCap);
    const int nL = f.cntPin[0], nR = f.cntPin[1];
    ASSERT_GT(nL, kWin);
    ASSERT_GT(nR, kWin);
    ASSERT_LE(nL, kCap);

    // 图窗口拷＋尾部补拷拼装
    std::vector<int> graph;
    int cnt[2] = {0, 0};
    ASSERT_TRUE(f.GraphWindowed(graph, cnt)) << "图构建/发射失败";
    EXPECT_EQ(nL, cnt[0]);
    EXPECT_EQ(nR, cnt[1]);
    scr::gpuRleTailCopy(f.bufL, f.bufR, kCap, kWin, cnt[0], cnt[1], f.runsPin, f.evRle);
    const std::vector<int> assembled(f.runsPin, f.runsPin + 6 * kCap);
    // [0, n) 段逐位一致（n 之外窗口版不拷＝语义允许陈旧）
    for (int eye = 0; eye < 2; ++eye) {
        const int n = eye == 0 ? nL : nR;
        const int* baseE = full.data() + eye * 3 * kCap;
        const int* baseA = assembled.data() + eye * 3 * kCap;
        for (int i = 0; i < 3 * n; ++i)
            ASSERT_EQ(baseE[i], baseA[i]) << "eye=" << eye << " int#" << i;
    }
}

TEST(RleSlotGraph, T3SizeChangeRebuild) {
    RleFixture f;
    // 第一尺寸：64×1024
    {
        const cv::Mat hL = MakeMask(2), hR = MakeMask(3);
        f.Upload(hL, hR);
        std::vector<int> eager = f.EagerWindowed();
        std::vector<int> graph;
        int cnt[2] = {0, 0};
        ASSERT_TRUE(f.GraphWindowed(graph, cnt)) << "K1 不支持";
        EXPECT_EQ(eager, graph);
    }
    // 第二尺寸：128×512（暂存增长路径＋新图正确）
    {
        cv::Mat hL(128, 512, CV_8UC1, cv::Scalar(0));
        cv::Mat hR(128, 512, CV_8UC1, cv::Scalar(0));
        for (int y = 0; y < 128; ++y) {
            hL.at<uchar>(y, 8) = 255;
            hR.at<uchar>(y, 16) = 255;
            hR.at<uchar>(y, 17) = 255;
        }
        f.Upload(hL, hR);
        std::vector<int> eager = f.EagerWindowed();
        std::vector<int> graph;
        int cnt[2] = {0, 0};
        ASSERT_TRUE(f.GraphWindowed(graph, cnt));
        EXPECT_EQ(128, cnt[0]);
        EXPECT_EQ(128, cnt[1]);
        EXPECT_EQ(eager, graph);
    }
}

TEST(RleSlotGraph, T4EventsOutsideGraphExternallyConsumed) {
    // 【退守变体定案 2026-10-28】K1 探针：legacy record 捕获期静默丢弃、External
    // 节点与并发图脆弱——事件一律图外常规 record（与 1B/B4/sep 图同模式）。
    // 本测验证该模式的图外消费全链：evSync/evQuery/跨流 wait/复射重记录。
    RleFixture f;
    const cv::Mat hL = MakeMask(1), hR = MakeMask(2);
    f.Upload(hL, hR);
    cudaGraph_t g = nullptr;
    cudaGraphExec_t ge = nullptr;
    std::vector<int> graph;
    int cnt[2] = {0, 0};
    ASSERT_TRUE(f.GraphWindowed(graph, cnt, &g, &ge));
    EXPECT_EQ(64, cnt[0]);
    ASSERT_GT(cnt[1], 0);
    // 复射（exec 复用）＋重记录＋再消费（稳态逐帧模式）
    cudaStream_t cs = cv::cuda::StreamAccessor::getStream(f.stream);
    ASSERT_EQ(cudaSuccess, cudaGraphLaunch(ge, cs));
    ASSERT_EQ(cudaSuccess, cudaEventRecord(f.evRle, cs));
    ASSERT_EQ(cudaSuccess, cudaEventRecord(f.evGate, cs));
    ASSERT_EQ(cudaSuccess, cudaEventSynchronize(f.evRle));
    EXPECT_EQ(cudaSuccess, cudaEventQuery(f.evRle));
    EXPECT_EQ(64, f.cntPin[0]);
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
}

} // namespace
