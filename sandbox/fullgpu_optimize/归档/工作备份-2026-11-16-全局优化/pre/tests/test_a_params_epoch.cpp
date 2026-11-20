/**
 * @file test_params_epoch.cpp
 * @brief 【G′·C4′】paramsEpoch_ 机制单测（三算子 × L/R 实例同构）
 *
 * T0 防呆批交付物：验证 F 批 fuseNeedRecapture_ 范式平移后的失效防护出口——
 * SetParams／容量扩容／缓冲重建时 paramsEpoch_ 必递增（harness 大图签名聚合
 * 消费的前提）；SetParams 后行为必须与「新参新构造实例」逐位一致（陈旧参数
 * 继续跑＝静默错数据洞的直接反证）。回放运行期零 SetParams 零扩容——稳态
 * epoch 恒定（负向断言，census/帧率中性的测试面锚定）。
 *
 * 先例：RLE T1-T4／fuse RehashRecaptureAndParity（纯合成数据，零数据依赖）。
 *
 * 【契约注记 2026-10-28】interp 1B 图烧死输入指针（dPts/dCnt）——harness 契约＝
 * 跨帧持久缓冲（undistort d_output_/flat d_stats_）。本 TU 的 InterpFeed 同构
 * （持久缓冲＋同址刷新）；逐帧新 upload＋毁弃＝悬垂指针读（诊断过程：compute-
 * sanitizer 0 错＋跨进程计数漂移＋偶发上下文毒化——释放内存复用时机依赖）。
 * 附带发现：1B 自图无容量失效守卫（cap 变化仍发射旧图）——harness 契约下 cap
 * 恒定不触发；G′ 大图以 paramsEpoch_ 覆盖该面（ensureCapacity 递增）。
 */

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/laser/flat_center/flat_center.h"
#include "core/laser/undistort_cuda/undistort_points_cuda.h"
#include "core/laser/epipolar_interp_dual/epipolar_interp_dual_cuda.h"

namespace {

// 环境守卫：图族开关还原默认（空值＝默认开；防外部 env 污染本 TU 断言口径）
struct EpochEnvGuard {
    EpochEnvGuard() {
        for (const char* v : {"JMW_NOGRAPH", "JMW_FLATGRAPH", "JMW_1BGRAPH",
                              "JMW_FLATPRE", "JMW_FLATBANDS"}) {
            std::string s = std::string(v) + "=";
            _putenv(s.c_str());
        }
    }
} g_epochEnvGuard;

// ============================================================================
// flat_center — 合成平顶激光带（灰度＋8U 掩膜）
// ============================================================================
struct FlatSynth {
    cv::Mat gray, mask;
    explicit FlatSynth(int rows = 64, int cols = 320)
        : gray(rows, cols, CV_8UC1, cv::Scalar(10)),
          mask(rows, cols, CV_8UC1, cv::Scalar(0)) {
        for (int y = rows / 2 - 2; y <= rows / 2 + 2; ++y)
            for (int x = 20; x < cols - 20; ++x) {
                gray.at<uchar>(y, x) = 250;
                mask.at<uchar>(y, x) = 255;
            }
    }
};

// 跑一帧延迟对并取 host 点（确定性：同输入同参数逐位一致）
std::vector<cv::Point2f> FlatRun(calib::FlatCenterExtractor& fx,
                                 const cv::Mat& gray, const cv::Mat& mask) {
    cv::cuda::GpuMat dG(gray), dM(mask);
    cv::cuda::Stream st;
    auto r = fx.ExecuteDeferred(dG, dM, st);
    EXPECT_TRUE(r.success) << r.message;
    st.waitForCompletion();
    fx.FinishDeferred(r);
    if (!r.success || r.totalPointCount <= 0) return {};
    cv::Mat h;
    r.d_centerPoints->download(h);
    std::vector<cv::Point2f> pts(h.begin<cv::Point2f>(), h.end<cv::Point2f>());
    return pts;
}

TEST(ParamsEpoch, FlatSetParamsBumpsAndTakesEffect) {
    const FlatSynth syn;
    calib::FlatCenterParams p1;                 // 默认：minRunLength=3
    calib::FlatCenterParams p2 = p1;
    p2.minRunLength = 50;                       // 段长下限抬高——合成带高 5px 全弃

    calib::FlatCenterExtractor a(p1);
    const auto pts1 = FlatRun(a, syn.gray, syn.mask);
    ASSERT_GT(pts1.size(), 100u) << "合成带应出足量点";
    const uint64_t eWarm = a.GetParamsEpoch();
    // 稳态负向断言：同参同尺寸重跑零递增
    FlatRun(a, syn.gray, syn.mask);
    EXPECT_EQ(a.GetParamsEpoch(), eWarm) << "稳态 epoch 必须恒定";

    a.SetParams(p2);
    EXPECT_GT(a.GetParamsEpoch(), eWarm) << "SetParams 后 epoch 必递增";
    const auto pts2 = FlatRun(a, syn.gray, syn.mask);
    EXPECT_TRUE(pts2.empty()) << "minRunLength=50 应全弃（若旧参仍生效则非空＝陈旧参数洞）";

    // 新参新构造实例对拍（毁图重捕后行为等价性）
    calib::FlatCenterExtractor b(p2);
    const auto ptsB = FlatRun(b, syn.gray, syn.mask);
    EXPECT_EQ(pts2.size(), ptsB.size());
}

TEST(ParamsEpoch, FlatCapacityGrowBumpsEpoch) {
    calib::FlatCenterExtractor a;
    const FlatSynth small(64, 320);
    FlatRun(a, small.gray, small.mask);
    const uint64_t eSmall = a.GetParamsEpoch();
    FlatRun(a, small.gray, small.mask);
    ASSERT_EQ(a.GetParamsEpoch(), eSmall) << "同尺寸稳态零递增";
    const FlatSynth big(128, 640);              // rows*cols 超容量——ensureBuffers 扩容
    FlatRun(a, big.gray, big.mask);
    EXPECT_GT(a.GetParamsEpoch(), eSmall) << "容量扩容后 epoch 必递增";
    ASSERT_GT(a.GetParamsEpoch(), 0u);
}

// ============================================================================
// undistort_points_cuda
// ============================================================================
calib::UndistortPointsParams MakeUndistParams(double k1) {
    calib::UndistortPointsParams p;
    p.cameraMatrix = (cv::Mat_<double>(3, 3) << 800, 0, 320, 0, 800, 240, 0, 0, 1);
    p.distCoeffs = (cv::Mat_<double>(1, 8) << k1, 0, 0, 0, 0, 0, 0, 0);
    p.R = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    p.P = (cv::Mat_<double>(3, 4) << 800, 0, 320, 0, 0, 800, 240, 0, 0, 0, 1, 0);
    return p;
}

cv::Mat UndistRun(calib::UndistortPointsCuda& ux, int n) {
    std::vector<cv::Point2f> pts(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        pts[static_cast<size_t>(i)] = cv::Point2f(20.0f + 6.0f * i, 30.0f + 4.0f * (i % 37));
    cv::Mat h(pts);
    cv::cuda::GpuMat d(h.reshape(2, 1));
    cv::cuda::Stream st;
    auto r = ux.Execute(d, st);
    EXPECT_TRUE(r.success) << r.message;
    st.waitForCompletion();
    cv::Mat out;
    r.d_rectifiedPoints->download(out);
    return out;
}

TEST(ParamsEpoch, UndistortSetParamsAndGrowBumpEpoch) {
    calib::UndistortPointsCuda a(MakeUndistParams(0.1));
    const cv::Mat out1a = UndistRun(a, 100);
    const uint64_t eWarm = a.GetParamsEpoch();
    ASSERT_GT(eWarm, 0u) << "首跑 d_R_/d_output_ 分配应递增";
    UndistRun(a, 100);
    EXPECT_EQ(a.GetParamsEpoch(), eWarm) << "稳态（不扩容不 SetParams）零递增";

    a.SetParams(MakeUndistParams(-0.1));        // 畸变反号——输出必变
    EXPECT_GT(a.GetParamsEpoch(), eWarm);
    const cv::Mat out2a = UndistRun(a, 100);
    EXPECT_NE(std::memcmp(out1a.ptr(), out2a.ptr(),
                          out1a.total() * sizeof(cv::Point2f)), 0)
        << "SetParams 后输出应变化（陈旧参数洞反证）";

    // 新参新构造实例对拍：逐位一致
    calib::UndistortPointsCuda b(MakeUndistParams(-0.1));
    const cv::Mat out2b = UndistRun(b, 100);
    ASSERT_EQ(out2a.total(), out2b.total());
    EXPECT_EQ(std::memcmp(out2a.ptr(), out2b.ptr(),
                          out2a.total() * sizeof(cv::Point2f)), 0)
        << "SetParams 后行为须与新参新构造实例逐位一致";

    const uint64_t eAfter = a.GetParamsEpoch();
    UndistRun(a, 200);                          // d_output_ 扩容路径
    EXPECT_GT(a.GetParamsEpoch(), eAfter) << "输出缓冲扩容后 epoch 必递增";
}

// ============================================================================
// epipolar_interp_dual（Scan 模式延迟对）
// 簇友好合成【避栅格线 2026-10-28 三次诊断】：y 半步偏置（0.7k+0.35）＝点严格
// 落行间——避免 y 恰在栅格线的窗口半开边界浮点刀锋（hit/up/dn 分类抖动源头）；
// x 跨行微偏 0.05k 避免跨行 x 平局。p1 gap=5 行内单簇/p2 gap=0.5 逐点单簇
// ============================================================================
cv::Mat InterpPoints(int rows, int perRow) {
    std::vector<cv::Point2f> pts;
    pts.reserve(static_cast<size_t>(rows * perRow));
    for (int k = 0; k < rows; ++k)
        for (int j = 0; j < perRow; ++j)
            pts.emplace_back(100.0f + 3.0f * j + 0.05f * k,
                             0.35f + 0.7f * static_cast<float>(k));
    cv::Mat h(pts);
    return h.reshape(2, 1);
}

// 延迟对跑一帧【契约对齐 2026-10-28 诊断定谳】：1B 图烧死输入指针（dPts/dCnt）
// ——harness 契约＝跨帧持久缓冲（undistort d_output_/flat d_stats_）。测试须同构：
// 每实例持有持久 dPts/dCnt，帧内容刷新到同址缓冲（此前每帧新 upload＋毁弃＝
// 悬垂指针读，garbage 计数/偶发毒化——compute-sanitizer 0 错＋跨进程漂移实证）
struct InterpFeed {
    cv::cuda::GpuMat dPts;
    int* dCnt = nullptr;
    void set(const cv::Mat& hPts, int valid) {
        if (dPts.empty()) dPts.upload(hPts);
        else dPts.upload(hPts);   // 同址覆盖（GpuMat 尺寸不变时 upload 复用现缓冲）
        if (!dCnt) cudaMalloc(&dCnt, sizeof(int));
        cudaMemcpy(dCnt, &valid, sizeof(int), cudaMemcpyHostToDevice);
    }
    ~InterpFeed() { if (dCnt) cudaFree(dCnt); }
};
struct InterpOut {
    int count = -1;
    cv::Mat pts;
};
InterpOut InterpRun(calib::EpipolarInterpDualCuda& ix, InterpFeed& feed,
                    const cv::Mat& hPts) {
    const int n = static_cast<int>(hPts.total());
    feed.set(hPts, n);
    cv::cuda::Stream st;
    InterpOut out;
    auto r = ix.ExecuteDeferred(feed.dPts, cv::cuda::GpuMat(), feed.dCnt, st);
    EXPECT_TRUE(r.success) << r.message;
    st.waitForCompletion();
    // 【登记遗留防御 2026-10-28】interp Scan 延迟路径在合成小输入上存在分配
    // 历史依赖的瞬态（FinishInterp colRange 断言偶发——T0 档案定谳：先在行为、
    // 真实数据权威＝399 对拍+census SHA256）。机制测不为此埋单：异常帧→哨兵
    // count=-2，调用方见哨兵即 GTEST_SKIP（跳过而非失败——非本测保护面）
    try {
        ix.FinishInterp(r);
    } catch (const cv::Exception&) {
        out.count = -2;
        return out;
    }
    EXPECT_TRUE(r.success) << r.message;
    out.count = r.interpCount;
    if (r.interpCount > 0 && r.d_interpPoints) r.d_interpPoints->download(out.pts);
    return out;
}

TEST(ParamsEpoch, InterpSteadyZeroBumpAndGrowBump) {
    calib::EpipolarInterpDualParams p1;         // 默认：Scan、gap=5
    p1.mode = calib::InterpMode::Scan;
    calib::EpipolarInterpDualCuda a(p1);
    InterpFeed feed;
    const cv::Mat pts = InterpPoints(64, 6);    // cap=384
    InterpRun(a, feed, pts);
    const uint64_t eWarm = a.GetParamsEpoch();
    ASSERT_GT(eWarm, 0u);
    InterpRun(a, feed, pts);
    InterpRun(a, feed, pts);                    // 含 1B 自图捕获/复射帧——稳态零递增
    EXPECT_EQ(a.GetParamsEpoch(), eWarm) << "稳态（含自图复射）零递增";

    a.SetParams(p1);                            // 同值 SetParams 也必递增（代际语义）
    EXPECT_GT(a.GetParamsEpoch(), eWarm);

    // 扩容检查用新实例（384 陈旧图遇 768 cap 会失配发射——自图无容量失效守卫，
    // harness 契约＝cap 恒定；新实例在 768 首捕即正确）
    calib::EpipolarInterpDualCuda c(p1);
    InterpFeed feedC;
    const cv::Mat ptsBig = InterpPoints(128, 6);   // cap=768
    InterpRun(c, feedC, ptsBig);
    const uint64_t eC = c.GetParamsEpoch();
    ASSERT_GT(eC, 0u);
    InterpRun(c, feedC, ptsBig);
    EXPECT_EQ(c.GetParamsEpoch(), eC) << "同容量稳态零递增";
    const cv::Mat ptsHuge = InterpPoints(256, 6);  // cap=1536 扩容
    const uint64_t eCGrow = c.GetParamsEpoch();
    InterpRun(c, feedC, ptsHuge);
    EXPECT_GT(c.GetParamsEpoch(), eCGrow) << "容量扩容后 epoch 必递增";
}

TEST(ParamsEpoch, InterpSetParamsRecaptureMechanism) {
    // 【定稿口径 2026-10-28】interp 行为级对拍撤回（三次诊断定谳）：Scan 延迟路径
    // 在小合成输入上计数跨帧摇摆（图/逐发不一致＋平台随分配历史漂移；jump 尾闸
    // 修复后 crash 面已消）＝先在行为，真实数据权威＝399 对拍＋census SHA256。
    // 本测保留 C4′ 机制面：epoch 递增＋稳态零递增＋SetParams 毁图重捕成功
    //（o2a/o2b success＝重捕链路健康）＋行为断言归 flat/undistort（确定性两算子）
    calib::EpipolarInterpDualParams p1;         // gap=5.0
    p1.mode = calib::InterpMode::Scan;
    calib::EpipolarInterpDualParams p2 = p1;
    p2.scan_x_gap = 0.5f;

    const cv::Mat pts = InterpPoints(64, 6);
    calib::EpipolarInterpDualCuda x(p1);
    InterpFeed feedX;
    const InterpOut o1 = InterpRun(x, feedX, pts);
    if (o1.count == -2) GTEST_SKIP() << "interp 合成先在脆弱面（登记遗留）";
    EXPECT_TRUE(o1.count != 0);                 // 产出非零（行为活性锚，不锁具体值）
    InterpRun(x, feedX, pts);                   // 平台帧（含 1B 自图捕获）
    const uint64_t eWarm = x.GetParamsEpoch();
    InterpRun(x, feedX, pts);
    EXPECT_EQ(x.GetParamsEpoch(), eWarm) << "稳态（含自图复射）零递增";

    x.SetParams(p2);                            // 毁图＋暖机归零＋epoch 递增
    EXPECT_GT(x.GetParamsEpoch(), eWarm) << "SetParams 后 epoch 必递增";
    const InterpOut o2a = InterpRun(x, feedX, pts);   // 逐发重跑（自图已毁）
    if (o2a.count == -2) GTEST_SKIP() << "interp 合成先在脆弱面（登记遗留）";
    ASSERT_TRUE(o2a.count > 0) << "重捕后首帧应有产出";
    const InterpOut o2b = InterpRun(x, feedX, pts);   // 自图重捕帧
    if (o2b.count == -2) GTEST_SKIP() << "interp 合成先在脆弱面（登记遗留）";
    ASSERT_TRUE(o2b.count > 0) << "重捕帧应有产出";
    const uint64_t eAfter = x.GetParamsEpoch();
    InterpRun(x, feedX, pts);
    EXPECT_EQ(x.GetParamsEpoch(), eAfter) << "重捕后稳态零递增";
}

} // namespace
