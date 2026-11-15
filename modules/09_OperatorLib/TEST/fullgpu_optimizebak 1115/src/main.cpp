// main.cpp — scan_replay CLI 前端
//
// 用法：
//   scan_replay.exe [--config <json>] [--cycles N] [--step all|name[,name…]]
//                   [--dump-dir <dir>] [--temp-compare t1,t2] [--list-steps] [--help]
// 退出码：0 正常；1 配置/数据/标定错误；2 半数以上 cycle 失败。
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "cuda_census.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>   // 【产线三雷】定时器/电源检查（NOMINMAX 防 min/max 宏污染）
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")

#include "artifacts.h"
#include "calib_loader.h"
#include "frame_source.h"
#include "harness.h"
#include "scr_config.h"
#include "stages.h"

#include <cuda_runtime.h>   // 【诊断臂】JMW_BLOCKSYNC：cudaSetDeviceFlags

namespace {

void printHelp() {
    std::cout <<
        "scan_replay — 扫描链真实数据回放（16 算子 v3 链）\n"
        "  --config <json>      配置文件（默认 exe 旁 config.json → 源码模板）\n"
        "  --cycles N           截断回放 cycle 数（0=全部；默认取 config）\n"
        "  --repeat N           数据集循环 N 遍（默认 1；帧号全局连续编，回绕处\n"
        "                       位姿跳变由配准降级链吸收——测速/帧率稳定性专用，\n"
        "                       勿用于精度对拍）\n"
        "  --fps-cap N          限帧 N FPS（默认 0=不限）：逐帧节拍压到固定速率，\n"
        "                       散热预算内恒速运行防降频触发（稳定优先）\n"
        "  --step all|a,b,…     全链 or 指定步骤（前置仍执行，仅筛选 dump）\n"
        "  --dump-dir <dir>     产物目录（覆盖 config paths.outDir）\n"
        "  --markers-only       只跑标记点链（跳过激光链与配准），逐 cycle 仅落\n"
        "                       标记点圆心坐标＋法线 CSV（cycle_XXX_markers.csv）\n"
        "  --lean               精简模式：只输出全局点云 PLY＋timing.csv，\n"
        "                       跳过逐帧 dump/report/一致性（配合 --dump-global-ply）\n"
        "  --vis                分步可视化：整跑强制串行，第 3/中位/倒数第 3 帧\n"
        "                       落全量逐步产物（PNG/CSV/PLY）到 vis_out/run_<时间戳>/，\n"
        "                       自动带全局点云 PLY；与 --markers-only 互斥，优先于 --lean\n"
        "  --dump-global-ply <f> 全帧跑完后导出全局标记点＋全局激光合并 PLY\n"
        "  --temp-compare t1,t2 双名义温各跑一遍出对比（诊断）\n"
        "  --quiet              静默算子日志（spdlog→err；计时跑推荐）\n"
        "  --list-steps         列出全部步骤名\n"
        "  --help               本帮助\n";
}

std::string fmtVal(double v, int prec = 3) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(prec) << v;
    return os.str();
}

} // namespace

int main(int argc, char** argv) {
    // CUDA API 协作普查（JMW_CENSUS=1 启用；须在一切 CUDA 调用前初始化）
    jmw::census::init();
    jmw::census::setStage("main");
    // 【诊断臂·2026-10-29】JMW_BLOCKSYNC=1：进程级阻塞式同步（cudaDeviceSchedule-
    // BlockingSync——stream/event Synchronize 由自旋改挂起唤醒）。须在 CUDA 上下文
    // 创建前设置。用途＝1b 排水自旋假设验证＋减负选项①（阻塞同步）评估；
    // 默认关＝零行为变化。ABBA 口径见 逐核负载画像 档 §四。
    if (const char* bs = std::getenv("JMW_BLOCKSYNC")) {
        if (bs[0] == '1') {
            const cudaError_t fe = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
            fprintf(stderr,
                    "[fc:blocksync] 阻塞式同步臂启用 ret=%d(%s)\n",
                    static_cast<int>(fe), cudaGetErrorString(fe));
        }
    }
    // 【产线三雷之一·2026-10-24】定时器分辨率 1ms：净机（无浏览器持有者）默认
    // tick=15.6ms，marker/fuse 的 1ms 超时轮询实等 15.6-31ms→FPS 塌方 ~64 档。
    // 本进程级设置（Win10 2004+ 只影响本进程）；atexit 归还。
    timeBeginPeriod(1);
    atexit([] { timeEndPeriod(1); });
    // 【产线三雷之三】电源前置检查：电池供电（DC）时处理器最小状态降 5%＋核心
    // 停放启用——性能与方差口径全变。仅告警不阻断。
    {
        SYSTEM_POWER_STATUS ps;
        if (GetSystemPowerStatus(&ps) && ps.ACLineStatus != 1)
            fprintf(stderr, "[power] 警告：电池供电运行（ACLineStatus=%d）——"
                            "处理器降频/核心停放将劣化性能，测速与产线须交流电\n",
                    static_cast<int>(ps.ACLineStatus));
    }

    std::string configPath;
    int cyclesOverride = -1;
    int repeatCount = 1;        // --repeat N：数据集循环 N 遍（帧率稳定性测量口径）
    double fpsCap = 0.0;        // --fps-cap N：限帧（>0 时逐帧节拍压到 N FPS）
    std::string stepArg = "all";
    std::string dumpDirOverride;
    std::string tempCompare;
    std::string globalPlyPath;   // --dump-global-ply：全帧跑完后导出全局标记点 PLY
    bool listSteps = false;
    bool markersOnly = false;
    bool visFlag = false;        // --vis：分步可视化总开关（置 cfg.runtime.visDump）
    bool leanMode = false;       // --lean：只输出全局点云＋计时，跳过逐帧 dump/报告
    // 【产线口径 2026-10-19·用户裁定】JMW_PROD=1：跳过回放测试专用记账（逐帧
    // 控制台行；harness 侧行残差统计等）——产线不消费这些输出
    const bool prodProfileRun = (getenv("JMW_PROD") != nullptr);
    bool quiet = false;          // --quiet：spdlog→err（计时跑降噪）
    std::string toRawDir;        // --to-raw <dir>：dataDir → raw 转换后退出（方案 B）
    std::string dataDirOverride; // --data-dir <dir>：覆盖 config paths.dataDir
    std::string cmpA, cmpB;      // --compare-report <a.csv> <b.csv>：跨执行器对拍后退出

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "缺少 " << what << " 参数值" << std::endl;
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--help" || a == "-h") {
            printHelp();
            return 0;
        }
        if (a == "--list-steps") {
            listSteps = true;
        } else if (a == "--config") {
            configPath = next("--config");
        } else if (a == "--cycles") {
            cyclesOverride = std::stoi(next("--cycles"));
        } else if (a == "--repeat") {
            repeatCount = std::stoi(next("--repeat"));
            if (repeatCount < 1) {
                std::cerr << "--repeat 须 >=1" << std::endl;
                return 1;
            }
        } else if (a == "--fps-cap") {
            fpsCap = std::stod(next("--fps-cap"));
            if (fpsCap < 0) {
                std::cerr << "--fps-cap 须 >=0" << std::endl;
                return 1;
            }
        } else if (a == "--step") {
            stepArg = next("--step");
        } else if (a == "--dump-dir") {
            dumpDirOverride = next("--dump-dir");
        } else if (a == "--markers-only") {
            markersOnly = true;
        } else if (a == "--vis") {
            visFlag = true;
        } else if (a == "--dump-global-ply") {
            globalPlyPath = next("--dump-global-ply");
        } else if (a == "--lean") {
            leanMode = true;
        } else if (a == "--temp-compare") {
            tempCompare = next("--temp-compare");
        } else if (a == "--quiet") {
            quiet = true;
        } else if (a == "--to-raw") {
            toRawDir = next("--to-raw");
        } else if (a == "--data-dir") {
            dataDirOverride = next("--data-dir");
        } else if (a == "--compare-report") {
            cmpA = next("--compare-report");
            cmpB = next("--compare-report");
        } else {
            std::cerr << "未知参数: " << a << "（--help 查看用法）" << std::endl;
            return 1;
        }
    }

    if (listSteps) {
        for (const auto& s : scr::stageTable())
            std::cout << s.name << (s.perEye ? "  [per-eye]" : "") << "\n";
        return 0;
    }
    if (quiet) spdlog::set_level(spdlog::level::err);

    // —— 步骤筛选 ——
    std::vector<std::string> onlySteps;
    if (stepArg != "all") {
        std::stringstream ss(stepArg);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            if (!scr::isValidStage(tok)) {
                std::cerr << "未知步骤名: " << tok << "（--list-steps 查看）" << std::endl;
                return 1;
            }
            onlySteps.push_back(tok);
        }
        if (onlySteps.empty()) onlySteps.push_back("all");
    }

    // —— 配置 ——
    if (configPath.empty())
        configPath = scr::ScrConfig::defaultConfigPath();
    scr::ScrConfig cfg;
    std::string err;
    if (!scr::ScrConfig::fromFile(configPath, cfg, err)) {
        std::cerr << "[config] " << err << std::endl;
        return 1;
    }
    if (cyclesOverride >= 0) cfg.runtime.maxCycles = cyclesOverride;
    if (!dumpDirOverride.empty()) cfg.paths.outDir = dumpDirOverride;
    if (!dataDirOverride.empty()) cfg.paths.dataDir = dataDirOverride;
    if (markersOnly) cfg.runtime.markerOnly = true;

    // —— --vis：分步可视化总开关（JSON runtime.visDump 亦可开；CLI 优先）——
    // 整跑强制串行（harness 五链/pipeline 分支旁路），仅 selectVisFrames 选中
    // 帧落全量 16 步产物；产物归拢 vis_out/run_<时间戳>/，自动带全局点云 PLY。
    if (visFlag) cfg.runtime.visDump = true;
    if (cfg.runtime.visDump) {
        if (markersOnly) {
            std::cerr << "--vis 与 --markers-only 互斥" << std::endl;
            return 1;
        }
        leanMode = false;   // --vis 优先于 --lean（选中帧逐帧 dump 不能被 lean 跳过）
        if (dumpDirOverride.empty()) {
            std::time_t now = std::time(nullptr);
            std::tm tmv;
            localtime_s(&tmv, &now);
            char buf[32];
            std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tmv);
            cfg.paths.outDir = (std::filesystem::path(SCR_SOURCE_DIR) / "vis_out"
                                / ("run_" + std::string(buf))).string();
        }
        if (globalPlyPath.empty())
            globalPlyPath = (std::filesystem::path(cfg.paths.outDir)
                             / "global_cloud.ply").string();
    }

    // —— 方案 B：PNG → raw 一次性转换后退出 ——
    if (!toRawDir.empty()) {
        if (!scr::convertToRaw(cfg.paths.dataDir, toRawDir, err)) {
            std::cerr << "[to-raw] " << err << std::endl;
            return 1;
        }
        return 0;
    }

    // —— CompareRuns：跨执行器 report.csv 对拍（P1 交付；容差口径=设计 §十一.1）——
    if (!cmpA.empty() && !cmpB.empty()) {
        auto loadRows = [](const std::string& path) {
            std::map<int, std::vector<std::string>> rows;
            std::ifstream f(path);
            if (!f) return rows;
            std::string line;
            std::getline(f, line);   // header
            while (std::getline(f, line)) {
                if (line.empty()) continue;
                std::stringstream ss(line);
                std::vector<std::string> cols;
                std::string tok;
                while (std::getline(ss, tok, ',')) cols.push_back(tok);
                if (cols.size() < 8) continue;
                const int cycle = std::atoi(cols[0].c_str());
                if (!rows.count(cycle)) rows[cycle] = cols;   // 历史双写行取首现
            }
            return rows;
        };
        const auto A = loadRows(cmpA);
        const auto B = loadRows(cmpB);
        if (A.empty() || B.empty()) {
            std::cerr << "[compare] 文件为空或不可读" << std::endl;
            return 1;
        }
        auto num = [](const std::vector<std::string>& c, size_t i) {
            return i < c.size() ? std::atof(c[i].c_str()) : -1e18;
        };
        int bad = 0, checked = 0, regSrcFlips = 0;
        const int regSrcFlipBudget = std::max(1, static_cast<int>(A.size()) / 100);
        for (const auto& [cycle, a] : A) {
            auto it = B.find(cycle);
            if (it == B.end()) {
                std::cout << "[compare] cycle " << cycle << " 仅存在于 A" << std::endl;
                ++bad;
                continue;
            }
            const auto& b = it->second;
            ++checked;
            auto fail = [&](const char* col, double va, double vb) {
                std::cout << "[compare] cycle " << cycle << " " << col
                          << " A=" << va << " B=" << vb << std::endl;
                ++bad;
            };
            // README §8.2 抖动口径：标记点 ±1（ccl 边界抖动）、配对 ±2、
            // regSource 翻转仅容忍 ≤1% 帧（抖动帧光流/兜底边界翻转——pipe 自比对同款）
            const bool srcFlip = (a[3] != b[3]);
            if (srcFlip) ++regSrcFlips;
            if (std::abs(num(a,1) - num(b,1)) > 2) fail("markersDetected", num(a,1), num(b,1));
            if (std::abs(num(a,2) - num(b,2)) > 1) fail("markersValid", num(a,2), num(b,2));
            if (!srcFlip && std::abs(num(a,4) - num(b,4)) > 2)
                fail("regMatched", num(a,4), num(b,4));   // 翻转帧 0↔N 属结构差，跳过
            // dt/drot：抖动帧 RANSAC 解局部发散后回归（pipe 自比对实测 Δdt 达 0.6mm，
            // hold/fuse 翻转帧结构性差达数 mm）——翻转帧跳过；其余相对容差
            if (!srcFlip) {
                if (std::abs(num(a,5) - num(b,5)) > std::max(2.5, 0.2 * std::abs(num(a,5))))
                    fail("dt_mm", num(a,5), num(b,5));
                if (std::abs(num(a,6) - num(b,6)) > std::max(0.4, 0.1 * std::abs(num(a,6))))
                    fail("drot_deg", num(a,6), num(b,6));
            }
            for (size_t col : {size_t(7), size_t(8)}) {
                const double ref = num(a, col), got = num(b, col);
                if (std::abs(ref - got) > std::max(4.0, 0.01 * std::abs(ref)))
                    fail(col == 7 ? "laserMatched" : "laserPoints3D", ref, got);
            }
            if (std::abs(num(a,9) - num(b,9)) > 0.01) fail("rowResid", num(a,9), num(b,9));
            if (std::abs(num(a,10) - num(b,10)) > 0.1) fail("dispMed", num(a,10), num(b,10));
        }
        for (const auto& [cycle, _] : B)
            if (!A.count(cycle)) {
                std::cout << "[compare] cycle " << cycle << " 仅存在于 B" << std::endl;
                ++bad;
            }
        if (regSrcFlips > regSrcFlipBudget) {
            std::cout << "[compare] regSource 翻转帧数 " << regSrcFlips
                      << " 超抖动预算 " << regSrcFlipBudget << std::endl;
            ++bad;
        }
        std::cout << "[compare] checked=" << checked << " diffs=" << bad
                  << " regSrcFlips=" << regSrcFlips
                  << " → " << (bad == 0 ? "PASS" : "FAIL") << std::endl;
        return bad == 0 ? 0 : 1;
    }

    if (!cfg.paths.allExist(err)) {
        std::cerr << "[data] " << err << "\n（数据全引用不拷——路径经 config paths 节配置）"
                  << std::endl;
        return 1;
    }
    std::error_code ec;
    std::filesystem::create_directories(cfg.paths.outDir, ec);

    // —— 温对比模式：不同名义温各跑一遍，各自出 timing/report ——
    std::vector<double> temps{cfg.runtime.nominalTemperature};
    if (!tempCompare.empty()) {
        temps.clear();
        std::stringstream ss(tempCompare);
        std::string tok;
        while (std::getline(ss, tok, ','))
            try {
                temps.push_back(std::stod(tok));
            } catch (...) {
                std::cerr << "--temp-compare 格式非法: " << tempCompare << std::endl;
                return 1;
            }
    }

    const auto tAll0 = std::chrono::steady_clock::now();

    for (double temp : temps) {
        scr::ScrConfig cfgT = cfg;
        cfgT.runtime.nominalTemperature = temp;
        std::string tag;
        if (temps.size() > 1) {
            std::ostringstream os;
            os << "_t" << temp;
            tag = os.str();
        }

        // —— 标定/帧源/harness ——
        scr::CalibData calib;
        if (!scr::loadCalib(cfgT, calib, err)) {
            std::cerr << "[calib] " << err << std::endl;
            return 1;
        }
        auto src = scr::makeFrameSource(cfgT, err);
        if (!src) {
            std::cerr << "[frame] " << err << std::endl;
            return 1;
        }
        // —— --repeat N：截断后的数据集循环 N 遍（追加 cycle 拷贝，帧号全局
        //    连续——dump/report/timing/prefetch 零改动）。maxCycles 截断折算进
        //    每遍后置 0（=全量），使 fivechain/pipeline 执行器内部 total 口径
        //    与主循环一致。回绕处（末帧→首帧）位姿跳变由配准降级链（hold/fuse）
        //    自然吸收——测速/稳定性口径专用，勿用于精度对拍。
        int total = cfgT.runtime.maxCycles > 0
                        ? std::min<int>(cfgT.runtime.maxCycles, src->cycles.size())
                        : static_cast<int>(src->cycles.size());
        if (repeatCount > 1 && total > 0) {
            src->cycles.resize(static_cast<size_t>(total));
            const std::vector<scr::Cycle> onePass = src->cycles;
            for (int r = 1; r < repeatCount; ++r)
                src->cycles.insert(src->cycles.end(), onePass.begin(), onePass.end());
            total *= repeatCount;
            cfgT.runtime.maxCycles = 0;
        }
        scr::Harness harness(cfgT, calib, *src, err);
        if (!err.empty()) {
            std::cerr << "[harness] " << err << std::endl;
            return 1;
        }

        const int warmup = std::min(cfgT.runtime.warmupCycles, total > 0 ? total - 1 : 0);

        // —— --vis 帧集（0 基）：第 3/中位/倒数第 3 帧，仅这些帧落全量产物 ——
        std::set<int> visFrames;
        if (cfgT.runtime.visDump)
            for (int v : scr::selectVisFrames(total)) visFrames.insert(v);
        if (!visFrames.empty()) {
            std::cout << "[vis] 分步可视化帧（0 基）:";
            for (int v : visFrames) std::cout << ' ' << v;
            std::cout << "\n[vis] 产物目录: " << cfgT.paths.outDir << std::endl;
        }

        const std::string timingCsv =
            (std::filesystem::path(cfgT.paths.outDir) / ("timing" + tag + ".csv")).string();
        const std::string reportCsv =
            (std::filesystem::path(cfgT.paths.outDir) / ("report" + tag + ".csv")).string();
        if (!cfg.runtime.markerOnly) {
            scr::ensureCsvHeader(timingCsv, "cycle,step,ms,source");
            scr::ensureCsvHeader(reportCsv,
                                 "cycle,markersDetected,markersValid,regSource,regMatched,"
                                 "dt_mm,drot_deg,laserMatched,laserPoints3D,rowResidMedPx,"
                                 "dispMedPx");
        }
        // 【150FPS】常开 CSV 句柄（原逐行 open-append-close 在 150FPS 口径下
        // 每帧 16 次开关≈数十 ms——管道瓶颈）；[cycle] 控制台行缓冲批量刷
        std::ofstream timingOfs, reportOfs;
        if (!cfg.runtime.markerOnly) {
            timingOfs.open(timingCsv, std::ios::app);
            reportOfs.open(reportCsv, std::ios::app);
        }
        std::ostringstream cycleLog;

        int failed = 0;
        double consistencyMax = -1.0;
        double prevR[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        double prevT[3] = {0, 0, 0};
        bool havePrev = false;
        std::map<std::string, std::vector<double>> stepMsAll;
        // 【150FPS】稳态吞吐测量：逐帧完成时间戳（剔除 warmup）→ 稳态 FPS
        std::vector<std::chrono::steady_clock::time_point> frameDoneTs;
jmw::census::noteFrames(0);  // 占位：循环结束处以实数覆写（census v2 稳态口径）
        std::vector<int> frameDoneCyc;   // 同步的正式 cycle 号（帧率分轮归属用）
        const auto paceT0 = std::chrono::steady_clock::now();   // --fps-cap 节拍基准

        for (int c = 0; c < total; ++c) {
            const bool isWarmup = c < warmup;
            const auto cycT0 = std::chrono::steady_clock::now();
            scr::CycleProducts prod;
            if (!harness.runCycle(c, prod, !isWarmup, err)) {
                std::cerr << "[run] cycle " << c << " 读取失败: " << err << std::endl;
                ++failed;
                continue;
            }
            if (getenv("JMW_FRAMET")) {
                std::vector<std::pair<std::string, double>> ms(prod.stepMs.begin(),
                                                              prod.stepMs.end());
                std::sort(ms.begin(), ms.end(),
                          [](auto& a, auto& b) { return a.second > b.second; });
                std::string top;
                for (size_t k = 0; k < ms.size() && k < 4; ++k)
                    top += " " + ms[k].first + "=" + fmtVal(ms[k].second, 1);
                fprintf(stderr, "[frame-wall] cycle %d: %.2f ms |%s\n", c,
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - cycT0).count(),
                        top.c_str());
            }
            const bool cycleOk = prod.stepError.empty();
            if (!cycleOk) ++failed;
            if (!isWarmup) {
                frameDoneTs.push_back(std::chrono::steady_clock::now());
                frameDoneCyc.push_back(c);
            }

            // markers-only：只落标记点圆心坐标＋法线 CSV，跳过其余一切 dump/记账
            if (cfgT.runtime.markerOnly) {
                std::ostringstream ns;
                ns << "cycle_" << std::setw(3) << std::setfill('0') << c << "_markers.csv";
                const auto csvPath = std::filesystem::path(cfgT.paths.outDir) / ns.str();
                std::ofstream ofs(csvPath);
                int n = 0;
                if (ofs) {
                    ofs << std::fixed << std::setprecision(6);
                    ofs << "marker,centerX_mm,centerY_mm,centerZ_mm,"
                           "normalX,normalY,normalZ\n";
                    for (size_t i = 0; i < prod.conic.markers.size(); ++i) {
                        const auto& m = prod.conic.markers[i];
                        if (!m.validCircle) continue;
                        ofs << i << ',' << m.centerX << ',' << m.centerY << ',' << m.centerZ
                            << ',' << m.normalX << ',' << m.normalY << ',' << m.normalZ
                            << '\n';
                        ++n;
                    }
                }
                std::cout << "[cycle " << c << "] " << (cycleOk ? "OK " : "DEGRADED ")
                          << "markers valid=" << n << " -> "
                          << csvPath.filename().string() << std::endl;
                continue;
            }

            // —— lean 模式：跳过逐帧 dump / 一致性 / report——只输出计时＋全局点云
            if (!leanMode) {
                // 双链一致性（首个成功 undistort_cpu 的 cycle 一次性记录）
                if (consistencyMax < 0.0 && prod.stepSuccess.count("undistort_cpu")
                    && prod.stepSuccess.at("undistort_cpu")) {
                    std::string detail;
                    consistencyMax = harness.checkConsistency(prod, detail);
                    std::cout << "[consistency] " << detail << std::endl;
                    nlohmann::json cj;
                    cj["maxAbsDiff"] = consistencyMax;
                    cj["detail"] = detail;
                    scr::saveJson((std::filesystem::path(cfgT.paths.outDir)
                                   / ("consistency" + tag + ".json")).string(), cj);
                }

                // dump（预热 cycle 不落全量；dumpFullFrames 只对正式 cycle 计数；
                // --vis 选中帧无条件全量——与 dumpFullFrames 语义正交叠加。
                // vis 模式非选中帧连 summary 也不落——整跑 399 个空壳目录属噪音）
                const bool full = (!isWarmup
                                   && c < warmup + cfgT.runtime.dumpFullFrames)
                                  || visFrames.count(c) > 0;
                const std::vector<std::string>* dumpFilter =
                    onlySteps.empty() ? nullptr : &onlySteps;
                if (!cfgT.runtime.visDump || visFrames.count(c) > 0)
                    harness.dumpCycleFiltered(c, prod, full, dumpFilter);
            }

            // 端到端 P0：逐帧累积全局激光点云（hold/skip 帧含沿用 R/T）——
            // 提前到 timing 落盘前，使「融合」步骤计入 timing.csv（第 16 步）
            if (!globalPlyPath.empty())
                harness.accumulateGlobalLaser(prod);

            if (isWarmup) continue;

            // timing.csv（常开句涵直写——150FPS 口径免逐行开关）
            for (const auto& kv : prod.stepMs) {
                std::string srcTag = "external";
                if (prod.stepSource.count(kv.first)) srcTag = prod.stepSource.at(kv.first);
                if (timingOfs)
                    timingOfs << c << "," << kv.first << "," << fmtVal(kv.second)
                              << "," << srcTag << "\n";
                stepMsAll[kv.first].push_back(kv.second);
            }

            // report.csv（R/T 增量）—— lean 模式跳过
            if (!leanMode) {
                double dtMm = -1.0, drotDeg = -1.0;
            if (havePrev) {
                dtMm = std::sqrt((prod.T[0] - prevT[0]) * (prod.T[0] - prevT[0])
                                 + (prod.T[1] - prevT[1]) * (prod.T[1] - prevT[1])
                                 + (prod.T[2] - prevT[2]) * (prod.T[2] - prevT[2]));
                const cv::Matx33d dR = cv::Matx33d(prod.R) * cv::Matx33d(prevR).t();
                const double trace = dR(0, 0) + dR(1, 1) + dR(2, 2);
                const double cosHalf = std::min(1.0, std::max(-1.0, (trace - 1.0) * 0.5));
                drotDeg = std::acos(cosHalf) * 180.0 / CV_PI;
            }
            std::copy(std::begin(prod.R), std::end(prod.R), std::begin(prevR));
            std::copy(std::begin(prod.T), std::end(prod.T), std::begin(prevT));
            havePrev = true;

            {
                std::ostringstream row;
                row << c << ',' << prod.markersDetected << ',' << prod.markersValid << ','
                    << prod.regSource << ',' << prod.regMatched << ',' << fmtVal(dtMm, 4)
                    << ',' << fmtVal(drotDeg, 4) << ',' << prod.laserMatched << ','
                    << prod.laserPoints3D << ',' << fmtVal(prod.rowResidMedPx, 4) << ','
                    << fmtVal(prod.dispMedPx, 4);
                if (reportOfs) reportOfs << row.str() << "\n";
            }
            }   // if (!leanMode) report.csv 闭合

            // 【150FPS】[cycle] 行缓冲（每 32 帧批量刷——逐行 endl 在管道下
            // 每帧数十 ms 控制台往返）。【产线口径 2026-10-19】逐帧控制台行＝
            // 回放测试专用（用户裁定产线不打印）——JMW_PROD=1 整体跳过
            if (!prodProfileRun) {
                cycleLog << "[cycle " << c << "] " << (cycleOk ? "OK " : "DEGRADED ")
                         << "markers=" << prod.markersDetected << "/" << prod.markersValid
                         << " reg=" << prod.regSource << "(" << prod.regMatched << ")"
                         << " laser=" << prod.laserMatched << "/" << prod.laserPoints3D
                         << " rowResid=" << prod.rowResidMedPx << "px\n";
                if ((c + 1) % 32 == 0 || c == total - 1) {
                    std::cout << cycleLog.str() << std::flush;
                    cycleLog.str("");
                }
            }

            // —— --fps-cap N：限帧节拍——把逐帧节奏压到 N FPS（deadline 对齐循环
            //    起点）。散热预算内恒速运行，防触发 CPU/GPU 降频——对应产线相机
            //    按固定帧率给图的口径；0=不限速。sleep_until 依赖 main 已设的
            //    1ms 定时器分辨率。
            if (fpsCap > 0.0) {
                const auto deadline =
                    paceT0
                    + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                        std::chrono::duration<double>((c + 1) / fpsCap));
                std::this_thread::sleep_until(deadline);
            }
        }

        // —— 全局标记点＋全局激光点云导出（--dump-global-ply <file>）——
        if (!globalPlyPath.empty()) {
            const auto& gms = harness.globalMarkerSet();
            std::ofstream ply(globalPlyPath);
            if (ply) {
                ply << "ply\nformat ascii 1.0\n";
                ply << "element vertex " << gms.positions.size() << "\n";
                ply << "property float x\nproperty float y\nproperty float z\n";
                ply << "property float nx\nproperty float ny\nproperty float nz\n";
                ply << "end_header\n";
                ply << std::fixed << std::setprecision(6);
                for (size_t i = 0; i < gms.positions.size(); ++i) {
                    ply << gms.positions[i].x << " " << gms.positions[i].y << " "
                        << gms.positions[i].z << " " << gms.normals[i](0) << " "
                        << gms.normals[i](1) << " " << gms.normals[i](2) << "\n";
                }
                std::cout << "\n[global-ply] 全局标记点 " << gms.positions.size()
                          << " 个 → " << globalPlyPath << std::endl;
            }
            // 端到端 P0：全局激光合并点云（同目录、_laser 后缀）
            const std::string laserPly = globalPlyPath.substr(0,
                globalPlyPath.find_last_of('.')) + "_laser.ply";
            const size_t nLaser = harness.saveGlobalLaserPly(laserPly);
            if (nLaser > 0)
                std::cout << "[global-ply] 全局激光点 " << nLaser
                          << " 个 → " << laserPly << std::endl;
            // 【五链 P1】体素融合对账行（C5 定容/C1 耗时数据源）
            std::cout << "[fuse:voxel] 帧数=" << harness.fuseFrameCount()
                      << " Σ输入=" << harness.fuseInputTotal()
                      << " 唯一体素=" << harness.fuseVoxelCount()
                      << " 融合点=" << harness.fuseFusedPointCount()
                      << " 扩容=" << harness.fuseRehashCount()
                      << " 越限=" << harness.fuseOverloadCount() << std::endl;
        }

        // —— 稳态吞吐（首帧含流水线填充延迟，剔除首尾各 1 帧取中位口径）——
        if (frameDoneTs.size() >= 4) {
            std::vector<double> inv;
            for (size_t i = 1; i < frameDoneTs.size(); ++i)
                inv.push_back(std::chrono::duration<double>(
                    frameDoneTs[i] - frameDoneTs[i - 1]).count());
            std::sort(inv.begin(), inv.end());
            const double med = inv[inv.size() / 2];
            const double span = std::chrono::duration<double>(
                frameDoneTs.back() - frameDoneTs.front()).count();
            std::cout << "\n===== 吞吐 =====\n";
            std::cout << "帧间完成间隔中位: " << fmtVal(med * 1000.0, 2) << " ms"
                      << " → 中位口径 " << fmtVal(1.0 / med, 1) << " FPS\n";
            std::cout << "全程口径: " << fmtVal((frameDoneTs.size() - 1) / span, 1)
                      << " FPS（" << frameDoneTs.size() << " 正式帧 / "
                      << fmtVal(span, 2) << "s，含首帧填充）\n";
        }

        // —— 帧率稳定性（--repeat 长跑口径）：逐帧完成间隔 → 瞬时 FPS 序列＋
        //    分轮/全程统计。统计样本剔除首尾各 1 个间隔（流水线填充/收尾尖峰），
        //    序列文件 fps.csv 全量保留；分轮边界间隔属连续真实节奏，保留。
        if (frameDoneTs.size() >= 4) {
            struct FpsRow { int cycle, pass, srcCycle; double ms, fps; };
            std::vector<FpsRow> rows;
            const int passFrames =
                std::max(1, total / std::max(1, repeatCount));   // 每遍帧数
            for (size_t i = 1; i < frameDoneTs.size(); ++i) {
                const double ms = std::chrono::duration<double, std::milli>(
                                      frameDoneTs[i] - frameDoneTs[i - 1]).count();
                const int cyc = frameDoneCyc[i];
                rows.push_back({cyc, cyc / passFrames, cyc % passFrames, ms,
                                1000.0 / ms});
            }
            // fps.csv（逐帧全量序列）
            {
                std::ofstream ofs((std::filesystem::path(cfgT.paths.outDir)
                                   / ("fps" + tag + ".csv")).string());
                if (ofs) {
                    ofs << "cycle,pass,src_cycle,interval_ms,fps\n"
                        << std::fixed << std::setprecision(3);
                    for (const auto& r : rows)
                        ofs << r.cycle << ',' << r.pass << ',' << r.srcCycle << ','
                            << r.ms << ',' << r.fps << '\n';
                }
            }
            // 统计样本（剔首尾间隔）＋按轮分组
            const std::vector<FpsRow> stat(rows.begin() + 1, rows.end() - 1);
            std::map<int, std::vector<double>> passFps;
            std::vector<double> allFps;
            for (const auto& r : stat) {
                passFps[r.pass].push_back(r.fps);
                allFps.push_back(r.fps);
            }
            auto medOf = [](std::vector<double> v) {
                std::sort(v.begin(), v.end());
                return v[v.size() / 2];
            };
            // 全程统计（总体标准差口径——含机器噪声）
            double mean = 0.0;
            for (double x : allFps) mean += x;
            mean /= static_cast<double>(allFps.size());
            double var = 0.0;
            for (double x : allFps) var += (x - mean) * (x - mean);
            var /= static_cast<double>(allFps.size());
            const double sd = std::sqrt(var);
            std::vector<double> srt = allFps;
            std::sort(srt.begin(), srt.end());
            const double p5 = srt[srt.size() * 5 / 100];
            const double p95 = srt[srt.size() * 95 / 100];
            int bestP = -1, worstP = -1;
            double bestMed = -1.0, worstMed = -1.0;
            for (const auto& kv : passFps) {
                const double m = medOf(kv.second);
                if (bestMed < 0 || m > bestMed) { bestMed = m; bestP = kv.first; }
                if (worstMed < 0 || m < worstMed) { worstMed = m; worstP = kv.first; }
            }
            // fps_summary.csv（分轮表＋全程行＋稳定性扩展指标）
            {
                auto statLine = [](const std::vector<double>& v) {
                    std::vector<double> s = v;
                    std::sort(s.begin(), s.end());
                    double sum = 0.0;
                    for (double x : s) sum += x;
                    const double m = sum / static_cast<double>(s.size());
                    double vr = 0.0;
                    for (double x : s) vr += (x - m) * (x - m);
                    vr /= static_cast<double>(s.size());
                    std::ostringstream os;
                    os << std::fixed << std::setprecision(2) << s[s.size() / 2] << ','
                       << m << ',' << std::sqrt(vr) << ',' << s.front() << ','
                       << s.back();
                    return os.str();
                };
                std::ofstream ofs((std::filesystem::path(cfgT.paths.outDir)
                                   / ("fps_summary" + tag + ".csv")).string());
                if (ofs) {
                    ofs << "pass,frames,median_fps,mean_fps,std_fps,min_fps,max_fps\n";
                    for (const auto& kv : passFps)
                        ofs << kv.first << ',' << kv.second.size() << ','
                            << statLine(kv.second) << '\n';
                    ofs << "ALL," << allFps.size() << ',' << statLine(allFps) << '\n';
                    ofs << std::fixed << std::setprecision(2);
                    ofs << "cv_pct," << 100.0 * sd / mean << "\n";
                    ofs << "p5_fps," << p5 << "\n";
                    ofs << "p95_fps," << p95 << "\n";
                    ofs << "best_pass," << bestP << "\n";
                    ofs << "worst_pass," << worstP << "\n";
                    ofs << "drift_fps," << (bestMed >= 0 ? bestMed - worstMed : 0.0)
                        << "\n";
                }
            }
            std::cout << "\n===== 帧率稳定性（剔 warmup＋首尾间隔）=====\n";
            std::cout << "全程 " << allFps.size() << " 间隔: mean "
                      << fmtVal(mean, 1) << " / median "
                      << fmtVal(srt[srt.size() / 2], 1) << " / std "
                      << fmtVal(sd, 2) << "（CV " << fmtVal(100.0 * sd / mean, 2)
                      << "%）/ min " << fmtVal(srt.front(), 1) << " / P5 "
                      << fmtVal(p5, 1) << " / P95 " << fmtVal(p95, 1) << " / max "
                      << fmtVal(srt.back(), 1) << " FPS\n";
            if (repeatCount > 1) {
                std::cout << "分轮中位 FPS:";
                for (const auto& kv : passFps)
                    std::cout << " r" << kv.first << "=" << fmtVal(medOf(kv.second), 1);
                std::cout << "\n最优轮 r" << bestP << "=" << fmtVal(bestMed, 1)
                          << " / 最差轮 r" << worstP << "=" << fmtVal(worstMed, 1)
                          << " / 漂移 " << fmtVal(bestMed - worstMed, 1) << " FPS（"
                          << fmtVal(100.0 * (bestMed - worstMed) / bestMed, 1)
                          << "%）\n";
            }
            // 每秒实时帧率：以首个正式帧完成时刻为 0 点，1s 分桶计完成帧数
            //（桶内完成帧数＝该秒吞吐 FPS；末桶不足 1s 仅记录，不入 min/中位）
            {
                std::vector<int> perSec;
                for (const auto& ts : frameDoneTs) {
                    const size_t b = static_cast<size_t>(
                        std::chrono::duration<double>(ts - frameDoneTs[0]).count());
                    if (b >= perSec.size()) perSec.resize(b + 1, 0);
                    ++perSec[b];
                }
                std::ofstream ofs((std::filesystem::path(cfgT.paths.outDir)
                                   / ("fps_per_second" + tag + ".csv")).string());
                if (ofs) {
                    ofs << "sec,fps\n";
                    for (size_t k = 0; k < perSec.size(); ++k)
                        ofs << k << ',' << perSec[k] << '\n';
                }
                if (perSec.size() > 1) {
                    std::vector<int> full(perSec.begin(), perSec.end() - 1);
                    std::sort(full.begin(), full.end());
                    std::cout << "每秒实时帧率 → fps_per_second.csv（" << perSec.size()
                              << " 秒；完整秒 min " << full.front() << " / 中位 "
                              << full[full.size() / 2] << " / max " << full.back()
                              << "；末桶不足 1s）\n";
                }
            }
            std::cout << "序列 fps.csv ＋ 汇总 fps_summary.csv → "
                      << cfgT.paths.outDir << "\n";
        }

        // —— 汇总 ——
        std::cout << "\n===== 汇总" << (tag.empty() ? "" : " (" + tag + ")") << " =====\n";
        std::cout << "cycles: " << total << " (warmup " << warmup << "), failed: "
                  << failed << "\n";
        std::cout << "一致性 max|Δ|: " << consistencyMax << "\n";
        // 端到端 P2：配准轨迹统计（hold 假零剔除）＋库健康度
        {
            std::map<std::string, int> regCount;
            std::vector<double> dtNonZero, drotNonZero;
            // 逐帧重跑 report.csv 数据太重——直接用 stepMsAll 同款模式：读 CSV
            // 简化版：汇总时从 prevR/prevT 已有数据算（此处用 report 数据代理）
            std::ifstream rpt(reportCsv);
            if (rpt) {
                std::string line;
                std::getline(rpt, line);   // header
                while (std::getline(rpt, line)) {
                    // cycle,markersDetected,markersValid,regSource,regMatched,dt_mm,drot_deg,...
                    std::stringstream ss(line);
                    std::string tok;
                    std::vector<std::string> cols;
                    while (std::getline(ss, tok, ',')) cols.push_back(tok);
                    if (cols.size() < 8) continue;
                    regCount[cols[3]]++;
                    const double dt = std::stod(cols[5]);
                    const double dr = std::stod(cols[6]);
                    if (dt > 0.001) dtNonZero.push_back(dt);   // 剔 hold 假零
                    if (dr > 0.001) drotNonZero.push_back(dr);
                }
            }
            std::cout << "配准路径分布: ";
            for (const auto& kv : regCount)
                std::cout << kv.first << "=" << kv.second << " ";
            std::cout << "\n";
            auto stats = [](std::vector<double>& v, const char* name) {
                if (v.empty()) return;
                std::sort(v.begin(), v.end());
                std::cout << "  " << name << " P50=" << fmtVal(v[v.size()/2], 3)
                          << " P95=" << fmtVal(v[v.size()*95/100], 3)
                          << " max=" << fmtVal(v.back(), 3) << "\n";
            };
            if (!dtNonZero.empty() || !drotNonZero.empty()) {
                std::cout << "帧间增量（hold 假零已剔）:\n";
                stats(dtNonZero, "位移(mm)");
                stats(drotNonZero, "转角(°)");
            }
            std::cout << "全局地图: 标记点 " << harness.librarySize()
                      << " / 候选池 " << harness.candidatePoolSize() << "\n";
        }
        if (!cfg.runtime.markerOnly) {
            std::cout << "各步中位耗时(ms):\n";
            for (const auto& kv : stepMsAll) {
                std::vector<double> s = kv.second;
                std::sort(s.begin(), s.end());
                std::cout << "  " << std::left << std::setw(24) << kv.first << std::right
                          << fmtVal(s[s.size() / 2], 2) << "\n";
            }
        }
        std::cout << "产物目录: " << cfgT.paths.outDir << "\n";

        if (total > 0 && failed > (total - warmup) / 2) {
            std::cerr << "失败 cycle 过半（" << failed << "/" << total << "）" << std::endl;
            return 2;
        }
        // census v2 稳态口径：在内层作用域以实数覆写（init 分配由 C1 剔除）
        jmw::census::noteFrames(static_cast<long>(frameDoneTs.size()));
    }

    const auto sec = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - tAll0).count();
    std::cout << "总耗时 " << fmtVal(sec, 1) << "s" << std::endl;
    return 0;
}
