/**
 * @file op_time_probe.cpp
 * @brief 算子耗时探针 —— 零落盘·可循环的长跑计时工具（tools/ 家族）
 *
 * 目的：对全量数据集（默认 399 帧）反复循环跑完整管线，**不向硬盘写任何管线
 *       产物**（不建 outDir、无 dump/report/timing/图像落盘），只在内存累计
 *       每算子（harness 步骤）每帧毫秒；结束时输出统计表。
 *
 * 用法（fullgpu_optimize 目录下）：
 *   op_time_probe.exe                              # 全量 399 帧 × 1 遍，只打终端
 *   op_time_probe.exe --loops 5                    # 399 帧 × 5 遍连续长跑
 *   op_time_probe.exe --cycles 50 --loops 10       # 截断 50 帧 × 10 遍（快速压测）
 *   op_time_probe.exe --warmup 10 --loops 3        # 开头 10 帧预热（不进统计）
 *   op_time_probe.exe --out 归档/xxx/耗时.csv      # 统计表另存（默认零落盘）
 *
 * 参数：
 *   --config <json>    配置（默认 ScrConfig::defaultConfigPath()）
 *   --data-dir <dir>   覆盖数据目录
 *   --cycles N         0=全量（默认）；>0 截断到前 N 帧
 *   --loops N          数据集循环遍数（默认 1；回绕处配准降级属预期——测速口径）
 *   --warmup N         预热帧数（-1=取 config 值；仅整个测试开头执行，不进统计）
 *   --out <csv>        统计结果另存 CSV；不给则完全不写硬盘
 *   --per-frame        额外逐帧逐算子打印（调试用，量大慎开）
 *
 * 输出：
 *   [1] 逐轮概览：每轮帧数 / 轮墙钟 / 帧均值 / 等效 FPS（看轮间漂移——
 *       本机存在 3.8↔16ms 乘性环境漂移在案，跨轮对比只看趋势）
 *   [2] 逐算子表：n / mean / median / p95 / max / 失败数（ms，按 mean 降序）
 *   [3] 帧合计（Σ算子）分布 + 帧墙钟总计
 *
 * 口径：串行 harness 全链（同 speed_replay serial 模式）；算子粒度＝步骤表
 *   （steger_fast / mask_separation / laser_match_scan_v3 / registration /
 *    laser_reconstruct / …）。V5 后段默认开，其耗时包含在 laser_match 步内
 *   （细分解用 JMW_V5BD=1 跑 speed_replay 查看）。
 */

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "calib_loader.h"
#include "frame_source.h"
#include "harness.h"
#include "scr_config.h"

namespace {

double MedianOf(std::vector<double> v) {
    if (v.empty()) return 0.0;
    const size_t n = v.size();
    std::nth_element(v.begin(), v.begin() + static_cast<long>(n / 2), v.end());
    return v[n / 2];
}

double P95Of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    const size_t k = static_cast<size_t>(0.95 * static_cast<double>(v.size() - 1));
    std::nth_element(v.begin(), v.begin() + static_cast<long>(k), v.end());
    return v[k];
}

} // namespace

int main(int argc, char** argv) {
    std::string configPath;
    std::string dataDirOverride;
    std::string outCsv;
    int cyclesArg = 0;       // 0=全量
    int loops = 1;
    int warmupArg = -1;      // -1=取 config
    bool perFrame = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "缺少 " << what << " 参数值" << std::endl;
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--config") configPath = next("--config");
        else if (a == "--data-dir") dataDirOverride = next("--data-dir");
        else if (a == "--cycles") cyclesArg = std::atoi(next("--cycles").c_str());
        else if (a == "--loops") loops = std::atoi(next("--loops").c_str());
        else if (a == "--warmup") warmupArg = std::atoi(next("--warmup").c_str());
        else if (a == "--out") outCsv = next("--out");
        else if (a == "--per-frame") perFrame = true;
        else if (a == "--help" || a == "-h") {
            std::printf("用法见文件头注释。核心：op_time_probe [--cycles N] [--loops N] "
                        "[--warmup N] [--out csv] [--config json] [--data-dir dir]\n");
            return 0;
        } else {
            std::cerr << "未知参数: " << a << std::endl;
            return 1;
        }
    }
    if (loops < 1) { std::cerr << "--loops 须 >=1" << std::endl; return 1; }
    if (cyclesArg < 0) { std::cerr << "--cycles 须 >=0（0=全量）" << std::endl; return 1; }
    spdlog::set_level(spdlog::level::err);   // 计时跑降噪（同 --quiet）

    // —— 配置（零落盘：outDir 指向占位符，不创建目录；本工具不调用任何 dump）——
    if (configPath.empty()) configPath = scr::ScrConfig::defaultConfigPath();
    scr::ScrConfig cfg;
    std::string err;
    if (!scr::ScrConfig::fromFile(configPath, cfg, err)) {
        std::cerr << "[config] " << err << std::endl;
        return 1;
    }
    if (!dataDirOverride.empty()) cfg.paths.dataDir = dataDirOverride;
    cfg.paths.outDir = "(op_time_probe 零落盘)";
    cfg.runtime.dumpFullFrames = 0;
    cfg.runtime.visDump = false;
    cfg.runtime.markerOnly = false;
    cfg.runtime.matchImageDump = false;
    if (!cfg.paths.allExist(err)) {
        std::cerr << "[data] " << err << std::endl;
        return 1;
    }

    // —— 装配：标定 / 帧源 / harness（同 main 串行模式）——
    scr::CalibData calib;
    if (!scr::loadCalib(cfg, calib, err)) {
        std::cerr << "[calib] " << err << std::endl;
        return 1;
    }
    auto src = scr::makeFrameSource(cfg, err);
    if (!src) {
        std::cerr << "[frame] " << err << std::endl;
        return 1;
    }
    int passCount = cfg.runtime.maxCycles > 0
                        ? std::min<int>(cfg.runtime.maxCycles,
                                        static_cast<int>(src->cycles.size()))
                        : static_cast<int>(src->cycles.size());
    if (cyclesArg > 0) passCount = std::min(passCount, cyclesArg);
    if (passCount <= 0) { std::cerr << "[data] 无帧可跑" << std::endl; return 1; }

    // —— --loops N：数据集循环 N 遍（同 main --repeat：追加 cycle 拷贝）——
    if (loops > 1) {
        src->cycles.resize(static_cast<size_t>(passCount));
        const std::vector<scr::Cycle> onePass = src->cycles;
        for (int r = 1; r < loops; ++r)
            src->cycles.insert(src->cycles.end(), onePass.begin(), onePass.end());
    }
    const int total = passCount * loops;
    const int warmup = warmupArg >= 0
                           ? std::min(warmupArg, total > 0 ? total - 1 : 0)
                           : std::min(cfg.runtime.warmupCycles,
                                      total > 0 ? total - 1 : 0);

    scr::Harness harness(cfg, calib, *src, err);
    if (!err.empty()) {
        std::cerr << "[harness] " << err << std::endl;
        return 1;
    }

    std::printf("[op_time_probe] 帧=%d × 遍=%d（合计 %d）预热=%d 串行全链 零落盘%s\n",
                passCount, loops, total, warmup,
                getenv("JMW_LMV5") && getenv("JMW_LMV5")[0] == '0' ? ""
                                                                   : " V5默认开");

    // —— 主循环：只累计计时，不落盘 ——
    std::map<std::string, std::vector<double>> stepAll;      // 算子 → 每帧 ms
    std::map<std::string, long long> stepFail;               // 算子 → 步骤失败次数
    std::vector<double> frameStepSum;                        // 每帧 Σ算子 ms
    std::vector<double> frameWallMs;                         // 每帧墙钟 ms（帧率口径）
    std::vector<int> frameStepCount;                         // 每帧步骤数（空帧甄别）
    struct LoopStat {
        int loop, frames, fails, matched0;
        double wallMs, stepSumMs;
    };
    std::vector<LoopStat> loopStats;
    const auto tAll0 = std::chrono::steady_clock::now();

    scr::CycleProducts prod;
    int failedCycles = 0;
    for (int c = 0; c < total; ++c) {
        const bool isWarmup = c < warmup;
        const auto tF0 = std::chrono::steady_clock::now();
        if (!harness.runCycle(c, prod, !isWarmup, err)) {
            std::fprintf(stderr, "[run] cycle %d 读取失败: %s\n", c, err.c_str());
            ++failedCycles;
            continue;
        }
        const double wallMs =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - tF0).count();
        if (isWarmup) continue;

        const int loop = c / passCount;
        if (static_cast<int>(loopStats.size()) < loop + 1)
            loopStats.push_back({loop, 0, 0, 0, 0.0, 0.0});
        LoopStat& ls = loopStats[static_cast<size_t>(loop)];
        ++ls.frames;
        ls.wallMs += wallMs;
        frameWallMs.push_back(wallMs);
        frameStepCount.push_back(static_cast<int>(prod.stepMs.size()));

        double sum = 0.0;
        {
            // 同名步骤一帧可能记多条：laser_match_scan_v3＝GPU 前段(发卡)＋CPU 后段
            // (V5) 各一条——按出现顺序拆成 #1/#2 分列统计（#1=前段 #2=后段），
            // 其余单条步骤保持原名。n=帧数、mean=该段帧耗时。
            std::map<std::string, int> seen;
            std::map<std::string, double> perName;
            for (const auto& kv : prod.stepMs) {
                sum += kv.second;
                const int k = ++seen[kv.first];
                if (k == 1) perName[kv.first] += kv.second;
                else {
                    char buf[160];
                    std::snprintf(buf, sizeof(buf), "%s#%d", kv.first.c_str(), k);
                    perName[buf] += kv.second;
                }
            }
            for (const auto& kv : perName) stepAll[kv.first].push_back(kv.second);
        }
        ls.stepSumMs += sum;
        frameStepSum.push_back(sum);
        bool anyFail = false;
        for (const auto& kv : prod.stepError) {
            if (!kv.second.empty()) {
                ++stepFail[kv.first];
                anyFail = true;
            }
        }
        if (anyFail) ++ls.fails;
        if (prod.laserMatched == 0) ++ls.matched0;
        if (perFrame) {
            std::printf("[frame %d] wall=%.2fms |", c, wallMs);
            for (const auto& kv : prod.stepMs)
                std::printf(" %s=%.2f", kv.first.c_str(), kv.second);
            std::printf("\n");
        }
    }
    const double totalWallMs =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - tAll0).count();

    // —— [1] 逐轮概览 ——
    std::printf("\n[1] 逐轮概览（回绕帧配准降级属预期）\n");
    std::printf("    %-4s %-7s %-11s %-11s %-9s %-6s %-8s\n",
                "轮", "帧数", "轮墙钟s", "帧均值ms", "等效FPS", "失败帧", "零配对帧");
    for (const auto& ls : loopStats) {
        std::printf("    %-4d %-7d %-11.1f %-11.2f %-9.1f %-6d %-8d\n",
                    ls.loop, ls.frames, ls.wallMs / 1000.0,
                    ls.frames > 0 ? ls.wallMs / ls.frames : 0.0,
                    ls.wallMs > 0 ? ls.frames / (ls.wallMs / 1000.0) : 0.0,
                    ls.fails, ls.matched0);
    }

    // —— [1b] 帧率统计（逐帧间隔→FPS；口径＝串行全链，非产线五链并行）——
    if (!frameWallMs.empty()) {
        // 剔除无实质处理的帧（数据集尾部空帧墙钟≈0 会把均值拉爆）：
        // 只统计墙钟 ≥0.5ms 且算子步骤数 ≥10 的帧
        std::vector<double> fps;
        int excluded = 0;
        for (size_t i = 0; i < frameWallMs.size(); ++i) {
            const double w = frameWallMs[i];
            const bool real = w >= 0.5 && frameStepCount.size() > i &&
                              frameStepCount[i] >= 10;
            if (!real) { ++excluded; continue; }
            fps.push_back(1000.0 / w);
        }
        std::vector<double> srt = fps;
        std::sort(srt.begin(), srt.end());
        const size_t n = srt.size();
        auto q = [&](double p) {
            return srt[static_cast<size_t>(p * static_cast<double>(n - 1))];
        };
        double mean = 0; for (double v : fps) mean += v; mean /= static_cast<double>(n);
        double var = 0; for (double v : fps) var += (v - mean) * (v - mean);
        var /= static_cast<double>(n);
        std::printf("[1b] 帧率（串行全链口径；n=%zu 帧，剔除空帧 %d）\n", n, excluded);
        std::printf("    mean=%.1f median=%.1f p5=%.1f p95=%.1f min=%.1f max=%.1f FPS"
                    "  变异系数cv=%.1f%%\n",
                    mean, q(0.5), q(0.05), q(0.95), srt.front(), srt.back(),
                    mean > 0 ? 100.0 * std::sqrt(var) / mean : 0.0);
        // 逐秒桶（首帧起 1s 分桶——秒级凹陷/卡顿直接可见；同口径剔除空帧）
        std::printf("    逐秒FPS:");
        {
            double acc = 0.0; int cnt = 0, sec = 0;
            for (size_t i = 0; i < frameWallMs.size(); ++i) {
                const double w = frameWallMs[i];
                if (!(w >= 0.5 && frameStepCount.size() > i &&
                      frameStepCount[i] >= 10))
                    continue;
                acc += w; ++cnt;
                if (acc >= 1000.0) {
                    std::printf(" %d:%.0f", sec++, 1000.0 * cnt / acc);
                    acc = 0.0; cnt = 0;
                }
            }
            if (cnt > 0) std::printf(" %d:%.0f", sec, cnt > 0 && acc > 0 ? 1000.0 * cnt / acc : 0.0);
        }
        std::printf("\n");
    }

    // —— [2] 逐算子表（按 mean 降序）——
    struct Row {
        std::string name;
        size_t n;
        double mean, med, p95, mx;
        long long fails;
    };
    std::vector<Row> rows;
    for (const auto& kv : stepAll) {
        const std::vector<double>& v = kv.second;
        double s = 0.0, m = 0.0;
        for (double x : v) { s += x; m = std::max(m, x); }
        Row r;
        r.name = kv.first;
        r.n = v.size();
        r.mean = s / static_cast<double>(v.size());
        r.med = MedianOf(v);
        r.p95 = P95Of(v);
        r.mx = m;
        r.fails = stepFail.count(kv.first) ? stepFail[kv.first] : 0;
        rows.push_back(r);
    }
    std::sort(rows.begin(), rows.end(),
              [](const Row& a, const Row& b) { return a.mean > b.mean; });
    std::printf("\n[2] 逐算子耗时（ms；n=帧数 失败=步骤失败次数）\n");
    std::printf("    %-24s %-7s %-9s %-9s %-9s %-9s %-6s\n",
                "算子(步骤)", "n", "mean", "median", "p95", "max", "失败");
    for (const auto& r : rows)
        std::printf("    %-24s %-7zu %-9.3f %-9.3f %-9.3f %-9.3f %-6lld\n",
                    r.name.c_str(), r.n, r.mean, r.med, r.p95, r.mx, r.fails);

    // —— [3] 帧合计分布 ——
    std::printf("\n[3] 帧合计 Σ算子（ms，不含 warmup）\n");
    if (!frameStepSum.empty()) {
        std::vector<double> fs = frameStepSum;
        std::printf("    n=%zu mean=%.2f median=%.2f p95=%.2f max=%.2f | "
                    "全测试墙钟=%.1fs（含装配）失败帧=%d\n",
                    fs.size(), [&]{ double s=0; for(double x:fs)s+=x; return s/fs.size(); }(),
                    MedianOf(fs), P95Of(fs),
                    [&]{ double m=0; for(double x:fs)m=std::max(m,x); return m; }(),
                    totalWallMs / 1000.0, failedCycles);
    }

    // —— 可选 CSV（默认零落盘；--out 才写这一个文件）——
    if (!outCsv.empty()) {
        std::ofstream f(outCsv);
        if (!f) {
            std::cerr << "[out] 无法写入 " << outCsv << std::endl;
            return 1;
        }
        f << "section,loop_or_op,n,mean_ms,median_ms,p95_ms,max_ms,fails,frames\n";
        for (const auto& ls : loopStats)
            f << "loop," << ls.loop << "," << ls.frames << ","
              << (ls.frames ? ls.wallMs / ls.frames : 0.0) << ",,,,"
              << ls.fails << "," << ls.frames << "\n";
        for (const auto& r : rows)
            f << "op," << r.name << "," << r.n << "," << r.mean << "," << r.med
              << "," << r.p95 << "," << r.mx << "," << r.fails << ",\n";
        f << "frame_total,," << frameStepSum.size() << ",,,," << failedCycles
          << ",\n";
        std::printf("\n[out] 统计已存 %s\n", outCsv.c_str());
    }
    return failedCycles > 0 ? 0 : 0;
}
