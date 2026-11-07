// gpu_stress.cpp — 纯 GPU 算子计时／满载测试
// 口径（2026-10-17 重写）：
//   0) 一帧图像：PNG 解码一次、上传一次（d_grayL/R 常驻显存）
//   1) 全链跑一遍物化各算子输入（KEEP 持有者，输入 GpuMat 全程不动）
//   2) 逐算子隔离循环 1000 次：循环内零同步，每圈 cudaEvent 对记录；
//      事件时长＝GPU 流上真实占用（含算子内部固有的同步停顿），
//      wall/圈 − 事件中位 ＝ 宿主侧开销（缓冲分配/发射间隙）
//   3) 全链持续循环 N 次：无每圈尾部同步（算子内部固有同步除外），
//      每 1000 圈一批记事件对→热稳态逐算子耗时；全程监控利用率
#include <chrono>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>

#include "calib_loader.h"
#include "frame_source.h"
#include "scr_config.h"

#include "scanning/preprocess/mask_separation/laser_markingpoint_mask_separation_cuda.h"
#include "core/vision/ccl/region_analyze_cuda.h"
#include "core/laser/steger_fast/steger_fast.h"
#include "core/laser/undistort_cuda/undistort_points_cuda.h"
#include "core/laser/epipolar_interp_dual/epipolar_interp_dual_cuda.h"
#include "scanning/laser/laser_match_scan_v3/laser_match_scan_v3_cuda.h"
#include "core/laser/laser_reconstruct/laser_reconstruct_cuda.h"

#include <thread>
#include <atomic>
#include <cuda_runtime.h>

// 粘性错误探针：定位哪个算子调用留下了 CUDA 错误
static void probeErr(const char* tag) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[probe] %-16s <- %s (%d)\n", tag, cudaGetErrorString(e), (int)e);
        fflush(stderr);
    }
}

// ── GPU 利用率监控线程（nvidia-smi 100ms 采样）────────────────────────
struct GpuMon {
    std::atomic<bool> run{false};
    std::vector<int> utils, clocks;
    std::thread th;
    void start() {
        run = true;
        th = std::thread([this]() {
            while (run) {
                FILE* fp = _popen(
                    "nvidia-smi --query-gpu=utilization.gpu,clocks.sm --format=csv,noheader",
                    "r");
                if (fp) {
                    char buf[256];
                    if (fgets(buf, sizeof(buf), fp)) {
                        int u = -1, c = -1;
                        sscanf(buf, "%d", &u);
                        char* p = strchr(buf, ',');
                        if (p) sscanf(p + 1, "%d", &c);
                        if (u >= 0) {
                            utils.push_back(u);
                            if (c > 0) clocks.push_back(c);
                        }
                    }
                    _pclose(fp);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });
    }
    void stop() { run = false; if (th.joinable()) th.join(); }
    void report(const char* tag) {
        if (utils.empty()) { printf("  (%s: 无样本)\n", tag); return; }
        std::vector<int> su = utils; std::sort(su.begin(), su.end());
        std::vector<int> sc = clocks; std::sort(sc.begin(), sc.end());
        double avg = 0; for (int u : utils) avg += u;
        avg /= utils.size();
        int hi = 0; for (int u : utils) if (u >= 80) hi++;
        printf("  [%s] 利用率: min=%d%% P50=%d%% P90=%d%% max=%d%% 平均=%.1f%% "
               "满载(≥80%%)=%.1f%%  核频: min=%d P50=%d max=%d MHz\n",
               tag, su[0], su[su.size()/2], su[su.size()*9/10], su.back(), avg,
               hi*100.0/utils.size(),
               sc.empty() ? 0 : sc[0], sc.empty() ? 0 : sc[sc.size()/2],
               sc.empty() ? 0 : sc.back());
    }
    void reset() { utils.clear(); clocks.clear(); }
};

static float medianOf(std::vector<float>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// 解析 v3 阶段串 " 相名=毫秒 …" → 聚合表
static void parseBreakdown(const std::string& s,
                           std::map<std::string, std::vector<float>>& out) {
    const char* p = s.c_str();
    while (*p) {
        char name[64];
        double val = 0;
        int len = 0;
        if (sscanf(p, " %63[^=]=%lf%n", name, &val, &len) >= 2 && len > 0) {
            out[name].push_back((float)val);
            p += len;
        } else {
            ++p;
        }
    }
}

int main(int argc, char** argv) {
    std::string cfgPath = "modules/09_operatorlib/TEST/speed_optimize/config.json";
    int iters = 10000;
    int frameIdx = 0;
    int perOpN = 1000;
    bool useCcl = true;   // --noccl：ccl 不参与（量激光链自身口径）
    bool dualMode = false;   // --dual：双流双线程全链（WDDM 跨帧重叠可行性实验）
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--config")) cfgPath = argv[++i];
        else if (!strcmp(argv[i], "--iter")) iters = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--frame")) frameIdx = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--perop")) perOpN = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "--noccl")) useCcl = false;
        else if (!strcmp(argv[i], "--dual")) dualMode = true;
    }
    spdlog::set_level(spdlog::level::warn);  // 计时口径：静默 info，保留 warn（分配诊断）
    if (!getenv("JMW_STRESS_NOSTEG")) _putenv("JMW_STEGTIME=1");  // 【算子内部计时】steger 分步事件
    printf("=== 纯 GPU 算子计时/满载测试（含算子内部计时）===\n");
    printf("帧: %d | 隔离循环: %d 次/算子 | 全链: %d 次\n\n", frameIdx, perOpN, iters);

    // 1. 加载配置＋标定＋一帧（CPU 预处理仅此一次）
    scr::ScrConfig cfg;
    std::string err;
    scr::ScrConfig::fromFile(cfgPath, cfg, err);
    scr::CalibData calib;
    scr::loadCalib(cfg, calib, err);
    auto src = scr::makeFrameSource(cfg, err);
    if (!src || frameIdx >= (int)src->cycles.size()) {
        fprintf(stderr, "帧源错误\n");
        return 1;
    }
    cv::Mat grayL, grayR;
    src->readFrame(src->cycles[frameIdx].laserIdx, true, grayL, err);
    src->readFrame(src->cycles[frameIdx].laserIdx, false, grayR, err);
    printf("帧加载(仅一次): %dx%d\n", grayL.cols, grayL.rows);

    // 2. 创建 6 个 GPU 算子
    cv::cuda::Stream stream;
    auto sepL = std::make_unique<calib::LaserMarkingSeparationCUDA>(
        calib::LaserMarkingSeparationParams::fromJson(cfg.opParams("mask_separation")));
    auto sepR = std::make_unique<calib::LaserMarkingSeparationCUDA>(
        calib::LaserMarkingSeparationParams::fromJson(cfg.opParams("mask_separation")));
    auto stgL = std::make_unique<calib::StegerExtractorFast>(
        calib::StegerParams::fromJson(cfg.opParams("steger_fast")));
    auto stgR = std::make_unique<calib::StegerExtractorFast>(
        calib::StegerParams::fromJson(cfg.opParams("steger_fast")));
    // 【测量修正】生产路径不开宿主点采集（harness 每帧设 false；此前工具漏设，
    // 导致 Out 段含已砍掉的 D2H＋同步假账——反对派代理 2026-10-17 抓出）
    stgL->SetCollectHostPoints(false);
    stgR->SetCollectHostPoints(false);
    auto udL = std::make_unique<calib::UndistortPointsCuda>([&]() {
        calib::UndistortPointsParams p;
        p.cameraMatrix = calib.K1c.empty() ? calib.K1 : calib.K1c;
        p.distCoeffs = calib.D1; p.R = calib.R1; p.P = calib.P1;
        return p;
    }());
    auto udR = std::make_unique<calib::UndistortPointsCuda>([&]() {
        calib::UndistortPointsParams p;
        p.cameraMatrix = calib.K2c.empty() ? calib.K2 : calib.K2c;
        p.distCoeffs = calib.D2; p.R = calib.R2;
        cv::Mat P2z = calib.P2.clone();
        P2z.at<double>(0, 3) = 0.0;   // 右目 P(0,3) 置零约定
        p.P = P2z;
        return p;
    }());
    auto ipL = std::make_unique<calib::EpipolarInterpDualCuda>([&]() {
        auto p = calib::EpipolarInterpDualParams::fromJson(cfg.opParams("epipolar_interp_dual"));
        p.mode = calib::InterpMode::Scan;
        return p;
    }());
    auto ipR = std::make_unique<calib::EpipolarInterpDualCuda>([&]() {
        auto p = calib::EpipolarInterpDualParams::fromJson(cfg.opParams("epipolar_interp_dual"));
        p.mode = calib::InterpMode::Scan;
        return p;
    }());
    auto mt = std::make_unique<calib::LaserMatchScanCudaV3>(
        calib::LaserMatchScanParamsV3::fromJson(cfg.opParams("laser_match_scan_v3")));
    mt->SetTempTable(calib.laserTable);
    mt->SetCurrentTemperature(cfg.runtime.nominalTemperature);
    auto rc = std::make_unique<calib::LaserReconstructCuda>(
        calib::LaserReconstructParams::fromJson(cfg.opParams("laser_reconstruct")));
    // 第 7 个 GPU 算子：ccl（吃标记点掩膜，服务标记点链——生产中与激光链并行/
    // 争用同一块卡；本工具按串行口径量它）
    auto cclL = std::make_unique<calib::RegionAnalyzerCUDA>(
        calib::RegionAnalyzerParams::fromJson(cfg.opParams("ccl")));
    auto cclR = std::make_unique<calib::RegionAnalyzerCUDA>(
        calib::RegionAnalyzerParams::fromJson(cfg.opParams("ccl")));

    // 3. 图像上传一次，常驻显存
    cv::cuda::GpuMat d_grayL, d_grayR;
    d_grayL.upload(grayL, stream);
    d_grayR.upload(grayR, stream);
    stream.waitForCompletion();

    // 4. 物化各算子输入（KEEP 持有者，全程存活；输入对象不再变动）
    auto sepKL = sepL->Execute(d_grayL, stream);  probeErr("sepKL");
    auto sepKR = sepR->Execute(d_grayR, stream);  probeErr("sepKR");
    auto stgKL = stgL->Execute(d_grayL, *sepKL.d_laserMask, stream, calib::GroupMode::Flat);  probeErr("stgKL");
    auto stgKR = stgR->Execute(d_grayR, *sepKR.d_laserMask, stream, calib::GroupMode::Flat);  probeErr("stgKR");
    auto udKL = udL->Execute(*stgKL.d_centerPoints, *stgKL.d_line_ids, stream);  probeErr("udKL");
    auto udKR = udR->Execute(*stgKR.d_centerPoints, *stgKR.d_line_ids, stream);  probeErr("udKR");
    auto ipKL = ipL->Execute(*udKL.d_rectifiedPoints, *udKL.d_line_ids, stream);  probeErr("ipKL");
    auto ipKR = ipR->Execute(*udKR.d_rectifiedPoints, *udKR.d_line_ids, stream);  probeErr("ipKR");
    auto mtK = mt->Execute(*ipKL.d_interpPoints, *ipKL.d_interp_line_ids,
                           *ipKR.d_interpPoints, *ipKR.d_interp_line_ids, stream);
    auto rcK = rc->Execute(*mtK.d_matched_left, *mtK.d_matched_right,
                           *mtK.d_matched_line_ids, calib.Q, stream);
    auto cclKL = useCcl ? cclL->Execute(sepKL.d_markingPointMask, stream)
                        : calib::RegionAnalysisResult{};
    if (useCcl) probeErr("cclKL");
    auto cclKR = useCcl ? cclR->Execute(sepKR.d_markingPointMask, stream)
                        : calib::RegionAnalysisResult{};
    if (useCcl) probeErr("cclKR");
    stream.waitForCompletion();
    if (!mtK.success) { fprintf(stderr, "物化失败: %s\n", mtK.message.c_str()); return 1; }
    printf("物化: matched=%d 点L=%d 点R=%d 组件L=%d 组件R=%d\n\n", mtK.matchedCount,
           (int)stgKL.d_centerPoints->cols, (int)stgKR.d_centerPoints->cols,
           (int)cclKL.components.size(), (int)cclKR.components.size());

    // 再预热若干次（建 CUDA Graph／稳住显存分配）
    for (int w = 0; w < 10; ++w) {
        auto a = sepL->Execute(d_grayL, stream);
        auto b = sepR->Execute(d_grayR, stream);
        if (useCcl) {
            auto c0 = cclL->Execute(a.d_markingPointMask, stream);
            auto c0r = cclR->Execute(b.d_markingPointMask, stream);
        }
        auto c = stgL->Execute(d_grayL, *a.d_laserMask, stream, calib::GroupMode::Flat);
        auto d = stgR->Execute(d_grayR, *b.d_laserMask, stream, calib::GroupMode::Flat);
        auto e = udL->Execute(*c.d_centerPoints, *c.d_line_ids, stream);
        auto f = udR->Execute(*d.d_centerPoints, *d.d_line_ids, stream);
        auto g = ipL->Execute(*e.d_rectifiedPoints, *e.d_line_ids, stream);
        auto h = ipR->Execute(*f.d_rectifiedPoints, *f.d_line_ids, stream);
        auto i2 = mt->Execute(*g.d_interpPoints, *g.d_interp_line_ids,
                              *h.d_interpPoints, *h.d_interp_line_ids, stream);
        auto j = rc->Execute(*i2.d_matched_left, *i2.d_matched_right,
                             *i2.d_matched_line_ids, calib.Q, stream);
        stream.waitForCompletion();
    }

    cudaStream_t cs = cv::cuda::StreamAccessor::getStream(stream);

    // 5. 逐算子隔离循环：零同步＋事件对
    printf("=== 逐算子隔离计时（各 %d 次，循环内零同步）===\n", perOpN);
    printf("%-26s %10s %8s %8s %10s %10s\n", "算子", "GPU中位", "min", "max",
           "wall/圈", "宿主开销");
    struct Row { const char* name; double ev, wall; };
    std::vector<Row> rows;

    auto benchOp = [&](const char* name, auto&& fn) {
        std::vector<cudaEvent_t> evS(perOpN), evE(perOpN);
        for (int i = 0; i < perOpN; ++i) {
            cudaEventCreate(&evS[i]); cudaEventCreate(&evE[i]);
        }
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < perOpN; ++i) {
            cudaEventRecord(evS[i], cs);
            fn();
            cudaEventRecord(evE[i], cs);
        }
        stream.waitForCompletion();
        auto t1 = std::chrono::steady_clock::now();
        std::vector<float> s(perOpN);
        for (int i = 0; i < perOpN; ++i) {
            cudaEventElapsedTime(&s[i], evS[i], evE[i]);
            cudaEventDestroy(evS[i]); cudaEventDestroy(evE[i]);
        }
        float med = medianOf(s);
        double wall = std::chrono::duration<double, std::milli>(t1 - t0).count() / perOpN;
        printf("%-26s %9.3fms %7.3f %7.3f %9.3f %9.3fms\n", name, med, s[0],
               s.back(), wall, wall - med);
        rows.push_back({name, med, wall});
        return med;
    };

    GpuMon mon;
    mon.start();

    double isoSum = 0;
    std::map<std::string, std::vector<float>> v3PhIso;   // 隔离循环内 v3 分解（链上对照）
    isoSum += benchOp("mask_separation(L+R)", [&] {
        auto a = sepL->Execute(d_grayL, stream);   // 设备直输（生产口径：帧常驻显存）
        auto b = sepR->Execute(d_grayR, stream);
    });
    if (useCcl)
        isoSum += benchOp("ccl(L+R)", [&] {
            auto a = cclL->Execute(sepKL.d_markingPointMask, stream);
            auto b = cclR->Execute(sepKR.d_markingPointMask, stream);
        });
    isoSum += benchOp("steger_fast(L+R)", [&] {
        auto a = stgL->Execute(d_grayL, *sepKL.d_laserMask, stream, calib::GroupMode::Flat);
        auto b = stgR->Execute(d_grayR, *sepKR.d_laserMask, stream, calib::GroupMode::Flat);
    });
    isoSum += benchOp("undistort_cuda(L+R)", [&] {
        auto a = udL->Execute(*stgKL.d_centerPoints, *stgKL.d_line_ids, stream);
        auto b = udR->Execute(*stgKR.d_centerPoints, *stgKR.d_line_ids, stream);
    });
    isoSum += benchOp("epipolar_interp(L+R)", [&] {
        auto a = ipL->Execute(*udKL.d_rectifiedPoints, *udKL.d_line_ids, stream);
        auto b = ipR->Execute(*udKR.d_rectifiedPoints, *udKR.d_line_ids, stream);
    });
    isoSum += benchOp("laser_match_v3", [&] {
        auto r = mt->Execute(*ipKL.d_interpPoints, *ipKL.d_interp_line_ids,
                             *ipKR.d_interpPoints, *ipKR.d_interp_line_ids, stream);
        parseBreakdown(mt->LastTimingBreakdown(), v3PhIso);   // 隔离态内部分解
    });
    isoSum += benchOp("laser_reconstruct", [&] {
        auto r = rc->Execute(*mtK.d_matched_left, *mtK.d_matched_right,
                             *mtK.d_matched_line_ids, calib.Q, stream);
    });
    mon.stop();
    printf("%-26s %9.3fms\n", "隔离算子合计(GPU中位)", isoSum);
    mon.report("隔离阶段");

    // ====================================================================
    // 【150FPS·方案 C 可行性】双流双线程全链：两套算子各一条流各一线程，
    // 各跑 iters 圈完整链（同帧同参）——量 WDDM 下跨流内核重叠的真实聚合吞吐
    // ====================================================================
    if (dualMode) {
        printf("\n=== 双流双线程全链（各 %d 圈/lane）===\n", iters);
        cv::cuda::Stream streamB;
        cv::cuda::GpuMat d_grayLB, d_grayRB;
        d_grayLB.upload(grayL, streamB);
        d_grayRB.upload(grayR, streamB);
        auto sepLB = std::make_unique<calib::LaserMarkingSeparationCUDA>(
            calib::LaserMarkingSeparationParams::fromJson(cfg.opParams("mask_separation")));
        auto sepRB = std::make_unique<calib::LaserMarkingSeparationCUDA>(
            calib::LaserMarkingSeparationParams::fromJson(cfg.opParams("mask_separation")));
        auto stgLB = std::make_unique<calib::StegerExtractorFast>(
            calib::StegerParams::fromJson(cfg.opParams("steger_fast")));
        auto stgRB = std::make_unique<calib::StegerExtractorFast>(
            calib::StegerParams::fromJson(cfg.opParams("steger_fast")));
        stgLB->SetCollectHostPoints(false);
        stgRB->SetCollectHostPoints(false);
        auto udLB = std::make_unique<calib::UndistortPointsCuda>([&]() {
            calib::UndistortPointsParams p;
            p.cameraMatrix = calib.K1c.empty() ? calib.K1 : calib.K1c;
            p.distCoeffs = calib.D1; p.R = calib.R1; p.P = calib.P1;
            return p;
        }());
        auto udRB = std::make_unique<calib::UndistortPointsCuda>([&]() {
            calib::UndistortPointsParams p;
            p.cameraMatrix = calib.K2c.empty() ? calib.K2 : calib.K2c;
            p.distCoeffs = calib.D2; p.R = calib.R2;
            cv::Mat P2z = calib.P2.clone();
            P2z.at<double>(0, 3) = 0.0;
            p.P = P2z;
            return p;
        }());
        auto ipLB = std::make_unique<calib::EpipolarInterpDualCuda>([&]() {
            auto p = calib::EpipolarInterpDualParams::fromJson(
                cfg.opParams("epipolar_interp_dual"));
            p.mode = calib::InterpMode::Scan;
            return p;
        }());
        auto ipRB = std::make_unique<calib::EpipolarInterpDualCuda>([&]() {
            auto p = calib::EpipolarInterpDualParams::fromJson(
                cfg.opParams("epipolar_interp_dual"));
            p.mode = calib::InterpMode::Scan;
            return p;
        }());
        auto mtB = std::make_unique<calib::LaserMatchScanCudaV3>(
            calib::LaserMatchScanParamsV3::fromJson(cfg.opParams("laser_match_scan_v3")));
        mtB->SetTempTable(calib.laserTable);
        mtB->SetCurrentTemperature(cfg.runtime.nominalTemperature);
        auto rcB = std::make_unique<calib::LaserReconstructCuda>(
            calib::LaserReconstructParams::fromJson(cfg.opParams("laser_reconstruct")));
        // lane B 预热（建图/稳显存）
        streamB.waitForCompletion();
        for (int w = 0; w < 10; ++w) {
            auto a = sepLB->Execute(d_grayLB, streamB);
            auto b = sepRB->Execute(d_grayRB, streamB);
            auto c = stgLB->Execute(d_grayLB, *a.d_laserMask, streamB, calib::GroupMode::Flat);
            auto d = stgRB->Execute(d_grayRB, *b.d_laserMask, streamB, calib::GroupMode::Flat);
            auto e = udLB->Execute(*c.d_centerPoints, *c.d_line_ids, streamB);
            auto f = udRB->Execute(*d.d_centerPoints, *d.d_line_ids, streamB);
            auto g = ipLB->Execute(*e.d_rectifiedPoints, *e.d_line_ids, streamB);
            auto h = ipRB->Execute(*f.d_rectifiedPoints, *f.d_line_ids, streamB);
            auto i2 = mtB->Execute(*g.d_interpPoints, *g.d_interp_line_ids,
                                   *h.d_interpPoints, *h.d_interp_line_ids, streamB);
            auto j = rcB->Execute(*i2.d_matched_left, *i2.d_matched_right,
                                  *i2.d_matched_line_ids, calib.Q, streamB);
            streamB.waitForCompletion();
        }
        auto laneFn = [&](bool isB) {
            cv::cuda::Stream& s = isB ? streamB : stream;
            cv::cuda::GpuMat &gl = isB ? d_grayLB : d_grayL;
            cv::cuda::GpuMat &gr = isB ? d_grayRB : d_grayR;
            auto& sL = isB ? sepLB : sepL;
            auto& sR = isB ? sepRB : sepR;
            auto& tL = isB ? stgLB : stgL;
            auto& tR = isB ? stgRB : stgR;
            auto& uL = isB ? udLB : udL;
            auto& uR = isB ? udRB : udR;
            auto& pL = isB ? ipLB : ipL;
            auto& pR = isB ? ipRB : ipR;
            auto& m  = isB ? mtB : mt;
            auto& rr = isB ? rcB : rc;
            for (int i = 0; i < iters; ++i) {
                auto a = sL->Execute(gl, s);
                auto b = sR->Execute(gr, s);
                // dual 模式两 lane 均不跑 ccl（生产口径 ccl 已在 CPU——方案 A）
                auto c = tL->Execute(gl, *a.d_laserMask, s, calib::GroupMode::Flat);
                auto d = tR->Execute(gr, *b.d_laserMask, s, calib::GroupMode::Flat);
                auto e = uL->Execute(*c.d_centerPoints, *c.d_line_ids, s);
                auto f = uR->Execute(*d.d_centerPoints, *d.d_line_ids, s);
                auto g = pL->Execute(*e.d_rectifiedPoints, *e.d_line_ids, s);
                auto h = pR->Execute(*f.d_rectifiedPoints, *f.d_line_ids, s);
                auto i2 = m->Execute(*g.d_interpPoints, *g.d_interp_line_ids,
                                     *h.d_interpPoints, *h.d_interp_line_ids, s);
                auto j = rr->Execute(*i2.d_matched_left, *i2.d_matched_right,
                                     *i2.d_matched_line_ids, calib.Q, s);
            }
            s.waitForCompletion();
        };
        mon.reset(); mon.start();
        auto td0 = std::chrono::steady_clock::now();
        std::thread thB([&] { laneFn(true); });
        laneFn(false);
        thB.join();
        auto td1 = std::chrono::steady_clock::now();
        mon.stop();
        const double wallMs =
            std::chrono::duration<double, std::milli>(td1 - td0).count();
        printf("双 lane 总耗时 %.1f ms | 聚合 %.3f ms/圈（=%d 圈×2 lane）| "
               "等效吞吐 %.1f FPS\n",
               wallMs, wallMs / (2.0 * iters), iters,
               2000.0 / (wallMs / iters));
        mon.report("双流持续");
        printf("\n结论口径：单流全链单圈（--iter 单跑）vs 聚合/圈——比值 <1.3× 则"
               "双流无净收益（WDDM 串行化），≥1.5× 值得上 GPU 双 lane。\n");
        return 0;
    }

    // 6. 全链持续循环：无每圈尾部同步；每 1000 圈一批事件对（热稳态分算子）
    printf("\n=== 全链持续循环（%d 次，无尾部同步）===\n", iters);
    const int batch = iters < 1000 ? iters : 1000;
    const int nBatch = (iters + batch - 1) / batch;   // ≥1
    const int nGrp = useCcl ? 7 : 6;   // mask[/ccl]/steger/undist/interp/v3/rc
    std::vector<cudaEvent_t> bS[7], bE[7];
    for (int b = 0; b < nBatch; ++b)
        for (int k = 0; k < nGrp; ++k) {
            cudaEvent_t e; cudaEventCreate(&e); bS[k].push_back(e);
            cudaEventCreate(&e); bE[k].push_back(e);
        }

    mon.reset(); mon.start();
    auto t0 = std::chrono::steady_clock::now();
    int lastMatched = -1;
    // 【算子内部计时】聚合表：v3 阶段串 / steger 分步（L+R 合计，7 项）/ interp 阶段（L/R 分表）
    std::map<std::string, std::vector<float>> v3Ph;
    std::map<std::string, std::vector<float>> ipPhL, ipPhR;
    std::vector<float> stPh[7];
    std::vector<float> stDp[5];   // 【批0】去重细分：D1/D2+3/D4/D5/D6
    for (int i = 0; i < iters; ++i) {
        if (iters > 40 && (i % (iters / 20) == 0)) {
            printf("\r进度: %d/%d", i, iters); fflush(stdout);
        }
        int b = i / batch, k = 0;
        if (b >= nBatch) b = nBatch - 1;   // 尾批余量（batch 整除时不会走到）
        cudaEventRecord(bS[k][b], cs);
        auto sA = sepL->Execute(d_grayL, stream);
        auto sB = sepR->Execute(d_grayR, stream);
        cudaEventRecord(bE[k][b], cs); k++;
        if (useCcl) {
            cudaEventRecord(bS[k][b], cs);
            auto ccA = cclL->Execute(sA.d_markingPointMask, stream);
            auto ccB = cclR->Execute(sB.d_markingPointMask, stream);
            cudaEventRecord(bE[k][b], cs); k++;
        }
        cudaEventRecord(bS[k][b], cs);
        auto gA = stgL->Execute(d_grayL, *sA.d_laserMask, stream, calib::GroupMode::Flat);
        auto gB = stgR->Execute(d_grayR, *sB.d_laserMask, stream, calib::GroupMode::Flat);
        {
            float la[7], lb[7];
            stgL->LastBenchMs(la); stgR->LastBenchMs(lb);
            for (int j = 0; j < 7; ++j) stPh[j].push_back(la[j] + lb[j]);
            float da[5], db[5];
            stgL->LastDedupMs(da); stgR->LastDedupMs(db);
            for (int j = 0; j < 5; ++j) stDp[j].push_back(da[j] + db[j]);
        }
        cudaEventRecord(bE[k][b], cs); k++;
        cudaEventRecord(bS[k][b], cs);
        auto uA = udL->Execute(*gA.d_centerPoints, *gA.d_line_ids, stream);
        auto uB = udR->Execute(*gB.d_centerPoints, *gB.d_line_ids, stream);
        cudaEventRecord(bE[k][b], cs); k++;
        cudaEventRecord(bS[k][b], cs);
        auto iA = ipL->Execute(*uA.d_rectifiedPoints, *uA.d_line_ids, stream);
        auto iB = ipR->Execute(*uB.d_rectifiedPoints, *uB.d_line_ids, stream);
        parseBreakdown(ipL->LastTimingBreakdown(), ipPhL);
        parseBreakdown(ipR->LastTimingBreakdown(), ipPhR);
        cudaEventRecord(bE[k][b], cs); k++;
        cudaEventRecord(bS[k][b], cs);
        auto mM = mt->Execute(*iA.d_interpPoints, *iA.d_interp_line_ids,
                              *iB.d_interpPoints, *iB.d_interp_line_ids, stream);
        parseBreakdown(mt->LastTimingBreakdown(), v3Ph);
        cudaEventRecord(bE[k][b], cs); k++;
        cudaEventRecord(bS[k][b], cs);
        auto rR = rc->Execute(*mM.d_matched_left, *mM.d_matched_right,
                              *mM.d_matched_line_ids, calib.Q, stream);
        cudaEventRecord(bE[k][b], cs);
        lastMatched = mM.matchedCount;
    }
    stream.waitForCompletion();
    auto t1 = std::chrono::steady_clock::now();
    mon.stop();
    printf("\r进度: %d/%d\n", iters, iters);

    double wallMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("总耗时: %.1f ms | 单圈: %.3f ms | 末圈 matched=%d\n",
           wallMs, wallMs / iters, lastMatched);
    mon.report("全链持续");

    // 热稳态逐算子（各批均值 → 跨批中位）
    static const char* names7[7] = { "mask_sep(L+R)", "ccl(L+R)", "steger(L+R)", "undistort(L+R)",
                                     "interp(L+R)", "match_v3", "reconstruct" };
    static const char* names6[6] = { "mask_sep(L+R)", "steger(L+R)", "undistort(L+R)",
                                     "interp(L+R)", "match_v3", "reconstruct" };
    const char* const* names = useCcl ? names7 : names6;
    printf("\n=== 热稳态逐算子（%d 批×%d 圈均值 → 中位）===\n", nBatch, batch);
    double hotSum = 0;
    for (int k = 0; k < nGrp; ++k) {
        std::vector<float> perBatch(nBatch);
        for (int b = 0; b < nBatch; ++b)
            cudaEventElapsedTime(&perBatch[b], bS[k][b], bE[k][b]);
        float med = medianOf(perBatch);
        hotSum += med;
        printf("  %-18s %8.3f ms/圈\n", names[k], med);
        for (int b = 0; b < nBatch; ++b) {
            cudaEventDestroy(bS[k][b]); cudaEventDestroy(bE[k][b]);
        }
    }
    printf("  %-18s %8.3f ms/圈（含算子间间隙）\n", "合计", hotSum);
    printf("  全链单圈实测      %8.3f ms → 算子间+发射间隙 %.3f ms\n",
           wallMs / iters, wallMs / iters - hotSum);
    printf("\n隔离(冷)合计 %.3f vs 热稳态合计 %.3f（降频/争用系数 %.2f×）\n",
           isoSum, hotSum, hotSum / isoSum);

    // 7. 【算子内部计时】v3 阶段＋steger 分步（热稳态全程逐圈聚合的中位）
    printf("\n=== 算子内部计时（全链循环逐圈聚合，%d 圈取中位）===\n", iters);
    if (!v3Ph.empty()) {
        float v3Total = 0;
        if (v3Ph.count("total")) v3Total = medianOf(v3Ph["total"]);
        printf("[laser_match_v3 内部·host计时口径]\n");
        for (auto& kv : v3Ph) {
            float m = medianOf(kv.second);
            printf("  %-22s %8.3f ms  (%4.1f%%)\n", kv.first.c_str(), m,
                   v3Total > 0 ? m / v3Total * 100 : 0.f);
        }
        // 【验证】逐批 v3 total 中位（看渐进爬升 vs 平稳——热降频判据）
        if (v3Ph.count("total") && v3Ph["total"].size() >= static_cast<size_t>(nBatch)) {
            printf("[v3 逐批 total 中位（批=%d 圈）]\n  ", batch);
            for (int b = 0; b < nBatch; ++b) {
                std::vector<float> seg(v3Ph["total"].begin() + b * batch,
                                       v3Ph["total"].begin() + (b + 1) * batch);
                printf("B%d=%.2f  ", b + 1, medianOf(seg));
            }
            printf("\n");
        }
    }
    if (!v3PhIso.empty()) {
        float v3Total = 0;
        if (v3PhIso.count("total")) v3Total = medianOf(v3PhIso["total"]);
        printf("[laser_match_v3 内部·隔离态（单独循环）]\n");
        for (auto& kv : v3PhIso) {
            float m = medianOf(kv.second);
            printf("  %-22s %8.3f ms  (%4.1f%%)\n", kv.first.c_str(), m,
                   v3Total > 0 ? m / v3Total * 100 : 0.f);
        }
    }
    if (!stPh[0].empty()) {
        static const char* sn[7] = { "Total(事件)", "Row3", "Col5", "Hessian",
                                     "Dedup", "Group(净)", "Out" };
        float tot = medianOf(stPh[0]);
        printf("[steger_fast 内部·CUDA事件口径·L+R 合计]\n");
        for (int j = 0; j < 7; ++j) {
            float m = medianOf(stPh[j]);
            printf("  %-22s %8.3f ms  (%4.1f%%)\n", sn[j], m,
                   tot > 0 ? m / tot * 100 : 0.f);
        }
        if (!stDp[0].empty()) {
            static const char* dn[5] = { "D1 排序+gather", "D2+D3 查找", "D4 传播",
                                         "D5 排序2", "D6 段归约" };
            float dsum = 0;
            for (int j = 0; j < 5; ++j) dsum += medianOf(stDp[j]);
            printf("[steger 去重链细分·L+R 合计]\n");
            for (int j = 0; j < 5; ++j) {
                float m = medianOf(stDp[j]);
                printf("  %-22s %8.3f ms  (%4.1f%%)\n", dn[j], m,
                       dsum > 0 ? m / dsum * 100 : 0.f);
            }
        }
    }
    if (!ipPhL.empty() && !ipPhR.empty()) {
        float ipTot = 0;
        if (ipPhL.count("total")) ipTot = medianOf(ipPhL["total"]) + medianOf(ipPhR["total"]);
        printf("[epipolar_interp 内部·host计时口径·L+R 合计]\n");
        std::map<std::string, std::vector<float>>* a = &ipPhL;
        std::map<std::string, std::vector<float>>* b2 = &ipPhR;
        for (auto& kv : *a) {
            if (kv.first == "total" || !b2->count(kv.first)) continue;
            float m = medianOf(kv.second) + medianOf((*b2)[kv.first]);
            printf("  %-22s %8.3f ms  (%4.1f%%)\n", kv.first.c_str(), m,
                   ipTot > 0 ? m / ipTot * 100 : 0.f);
        }
        if (ipPhL.count("total"))
            printf("  %-22s %8.3f ms  (100%%)\n", "total",
                   medianOf(ipPhL["total"]) + medianOf(ipPhR["total"]));
    }

    // 8. 【算子内部计时】mask_separation 诊断段（JMW_SEPTIME：事件逐核口径，
    //    走 legacy 路径——总量≠生产 Graph 路径，但各步相对占比可参照）
    {
        const int diagN = 200;
        _putenv("JMW_SEPTIME=1");
        for (int i = 0; i < 10; ++i) {  // 诊断路径预热
            auto a = sepL->Execute(d_grayL, stream);
            auto b = sepR->Execute(d_grayR, stream);
            stream.waitForCompletion();
        }
        std::vector<float> mUp, mS1, mS2, mS3, mS4, mS5, mTot;
        for (int i = 0; i < diagN; ++i) {
            auto a = sepL->Execute(grayL, stream);
            auto b = sepR->Execute(grayR, stream);
            stream.waitForCompletion();
            mUp.push_back((float)(a.timings.upload_ms + b.timings.upload_ms));
            mS1.push_back((float)(a.timings.step1_gaussian_threshold_ms + b.timings.step1_gaussian_threshold_ms));
            mS2.push_back((float)(a.timings.step2_noise_erode_ms + b.timings.step2_noise_erode_ms));
            mS3.push_back((float)(a.timings.step3_laser_extract_ms + b.timings.step3_laser_extract_ms));
            mS4.push_back((float)(a.timings.step4_marking_extract_ms + b.timings.step4_marking_extract_ms));
            mS5.push_back((float)(a.timings.step5_edge_dilate_ms + b.timings.step5_edge_dilate_ms));
            mTot.push_back((float)(a.timings.total_pipeline_ms + b.timings.total_pipeline_ms));
        }
        _putenv("JMW_SEPTIME=");
        float mt2 = medianOf(mTot);
        printf("[mask_separation 内部·诊断路径(逐核事件+同步)·L+R 合计·%d 次]\n", diagN);
        printf("  %-22s %8.3f ms  (100%%)\n", "Total(诊断)", mt2);
        printf("  %-22s %8.3f ms  (%4.1f%%)\n", "up 上传", medianOf(mUp), medianOf(mUp) / mt2 * 100);
        printf("  %-22s %8.3f ms  (%4.1f%%)\n", "s1 高斯+阈值", medianOf(mS1), medianOf(mS1) / mt2 * 100);
        printf("  %-22s %8.3f ms  (%4.1f%%)\n", "s2 噪点腐蚀", medianOf(mS2), medianOf(mS2) / mt2 * 100);
        printf("  %-22s %8.3f ms  (%4.1f%%)\n", "s3 激光提取", medianOf(mS3), medianOf(mS3) / mt2 * 100);
        printf("  %-22s %8.3f ms  (%4.1f%%)\n", "s4 标记点提取", medianOf(mS4), medianOf(mS4) / mt2 * 100);
        printf("  %-22s %8.3f ms  (%4.1f%%)\n", "s5 边缘膨胀", medianOf(mS5), medianOf(mS5) / mt2 * 100);
    }
    return 0;
}
