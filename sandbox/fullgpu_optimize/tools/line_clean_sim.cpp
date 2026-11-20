/**
 * @file line_clean_sim.cpp
 * @brief 补丁⑰·line_clean 阶段0 离线仿真——只读 res.csv 残差库，零代码改动算子
 *
 * 输入：--match-img 产物 frame_NNN_res.csv（列 lid,xL,yL,xR,res；res=⑫输出的
 *       赢家卡带符号残差——与⑰算子消费的同一数据口径，结论可直接迁移）。
 * 逻辑：与 laser_match_scan_v5.cpp 的⑰主块严格同构——按 lid 分桶、桶内
 *       (yL,原序) 排序、row 差>gap_rows 切 run、run≥min_run 滑窗 Hampel
 *       （win 中值+MAD×1.4826，σ_floor 下限）、k 门标记、单线率帽整线回退。
 * 输出：go/no-go 判据所需的全部统计——剔点率/按线分布/幅值直方图/σ̂ 分布/
 *       run 长度分布/率帽触发/逐帧 top 异常帧。
 *
 * 用法：line_clean_sim <res_dir> [k=3.5] [gap_rows=3] [cap=0.05] [rowStep=0.67]
 *                     [win=5] [min_run=5] [sigmaFloor=0.3]
 *
 * ⚠ 沙箱定位：阶段0 观测工具（方案 v3 路线图）；厚度改善≥2% 的另一半判据
 *   需阶段1 ON 臂真跑管线，本工具不覆盖（无标定表不可离线重建 3D）。
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// —— 与算子⑰同构的参数（默认值＝方案常量）——
struct SimParams {
    double k = 3.5;
    int gap_rows = 3;
    double cap = 0.05;
    double rowStep = 0.67;
    int win = 5;
    int min_run = 5;
    double sigmaFloor = 0.3;
};

// roundf 半值远离零（正数域与算子 froundAwayI 等价）
inline int rowOf(double y, double rowStep) {
    const double q = y / rowStep;
    return static_cast<int>(q >= 0.0 ? q + 0.5 : q - 0.5);
}

// 小数组插入排序（同算子 lcSortSmall——确定性来源）
void sortSmall(float* v, int n) {
    for (int a = 1; a < n; ++a) {
        const float key = v[a];
        int b = a - 1;
        while (b >= 0 && v[b] > key) { v[b + 1] = v[b]; --b; }
        v[b + 1] = key;
    }
}

struct Pt {
    float y;
    float res;
    int row;
    int seq;   // 原输入序（排序平票锚）
    int lidOf; // 所属线号（分桶冗余存一份，统计用）
};

struct FrameStat {
    long long pts = 0;
    long long gross = 0;    // 毛剔（率帽撤销前）
    long long net = 0;      // 净剔（率帽撤销后）
    int fallbackLines = 0;
};

// 一帧的逐线处理（桶=lid；返回净剔/毛剔，填充全局统计钩子由调用方聚合）
struct Tally {
    long long gross = 0, net = 0;
    int fallback = 0;
    std::vector<int> netByLid;              // [1..32]
    std::vector<long long> devHist;         // 0.5px 桶（dev/0.5 取整，帽 24 桶）
    std::vector<long long> resHist;         // |res| 0.5px 桶（帽 24 桶）
    std::vector<double> sigmas;             // 每点 σ̂（抽样存储判 P50/P90）
    long long runsLt5 = 0, runs5to59 = 0, runsGe60 = 0;
    long long edgeBand = 0;                 // dev∈[kσ,(k+1.5)σ) 可疑误杀带
    Tally() : netByLid(33, 0), devHist(25, 0), resHist(25, 0) {}
};

void processBucket(std::vector<Pt>& v, const SimParams& sp, Tally& t,
                   FrameStat& fs) {
    if (v.size() >= static_cast<size_t>(sp.min_run))
        (v.size() >= 60) ? ++t.runsGe60 : ++t.runs5to59;
    else
        ++t.runsLt5;
    std::sort(v.begin(), v.end(), [](const Pt& a, const Pt& b) {
        return (a.row != b.row) ? (a.row < b.row) : (a.seq < b.seq);
    });
    const int n = static_cast<int>(v.size());
    const int half = sp.win / 2;
    std::vector<unsigned char> mark(static_cast<size_t>(n), 0);
    std::vector<float> win(static_cast<size_t>(2 * half + 1));
    int runBeg = 0;
    for (int i1 = 1; i1 <= n; ++i1) {
        const bool cut = (i1 == n) ||
            (v[static_cast<size_t>(i1)].row - v[static_cast<size_t>(i1 - 1)].row >
             sp.gap_rows);
        if (!cut) continue;
        if (i1 - runBeg >= sp.min_run) {
            for (int j = runBeg; j < i1; ++j) {
                const int lo = std::max(runBeg, j - half);
                const int hi = std::min(i1 - 1, j + half);
                const int wn = hi - lo + 1;
                for (int q = lo; q <= hi; ++q)
                    win[static_cast<size_t>(q - lo)] = v[static_cast<size_t>(q)].res;
                sortSmall(win.data(), wn);
                const float med = win[static_cast<size_t>(wn / 2)];
                for (int q = lo; q <= hi; ++q)
                    win[static_cast<size_t>(q - lo)] =
                        std::fabs(v[static_cast<size_t>(q)].res - med);
                sortSmall(win.data(), wn);
                const float sig = static_cast<float>(std::max(
                    1.4826 * win[static_cast<size_t>(wn / 2)], sp.sigmaFloor));
                const double dev = std::fabs(v[static_cast<size_t>(j)].res - med);
                if (t.sigmas.size() < 4000000)
                    t.sigmas.push_back(sig);
                if (dev > sp.k * sig) {
                    mark[static_cast<size_t>(j)] = 1;
                    ++t.gross;
                    const int dh = std::min(24, static_cast<int>(dev / 0.5));
                    ++t.devHist[static_cast<size_t>(dh)];
                    const double ar = std::fabs(v[static_cast<size_t>(j)].res);
                    const int rh = std::min(24, static_cast<int>(ar / 0.5));
                    ++t.resHist[static_cast<size_t>(rh)];
                    if (dev < (sp.k + 1.5) * sig) ++t.edgeBand;
                }
            }
        }
        runBeg = i1;
    }
    int markCnt = 0;
    for (int q = 0; q < n; ++q) markCnt += mark[static_cast<size_t>(q)];
    if (markCnt > 0 &&
        static_cast<double>(markCnt) > sp.cap * static_cast<double>(n)) {
        ++t.fallback;                     // 整线回退（故障安全）
        ++fs.fallbackLines;
    } else {
        for (int q = 0; q < n; ++q)
            if (mark[static_cast<size_t>(q)]) {
                ++t.net;
                fs.net += 1;
                const int lid = v[static_cast<size_t>(q)].lidOf;
                ++t.netByLid[static_cast<size_t>(std::min(32, std::max(1, lid)))];
            }
    }
    fs.gross += markCnt;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s <res_dir> [k=3.5] [gap_rows=3] [cap=0.05] [rowStep=0.67]"
            " [win=5] [min_run=5] [sigmaFloor=0.3]\n", argv[0]);
        return 2;
    }
    SimParams sp;
    if (argc > 2) sp.k = atof(argv[2]);
    if (argc > 3) sp.gap_rows = atoi(argv[3]);
    if (argc > 4) sp.cap = atof(argv[4]);
    if (argc > 5) sp.rowStep = atof(argv[5]);
    if (argc > 6) sp.win = atoi(argv[6]);
    if (argc > 7) sp.min_run = atoi(argv[7]);
    if (argc > 8) sp.sigmaFloor = atof(argv[8]);

    // 枚举 frame_*_res.csv
    std::vector<std::string> files;
    for (int f = 0; f <= 999; ++f) {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "frame_%03d_res.csv", f);
        const std::string p = std::string(argv[1]) + "/" + nm;
        FILE* probe = std::fopen(p.c_str(), "rb");
        if (probe) { std::fclose(probe); files.push_back(p); }
    }
    if (files.empty()) {
        std::fprintf(stderr, "no frame_*_res.csv under %s\n", argv[1]);
        return 2;
    }

    Tally t;
    long long totalPts = 0;
    std::vector<std::pair<int, long long>> perFrameNet;   // (frameIdx, net)
    char line[512];
    for (size_t fi = 0; fi < files.size(); ++fi) {
        FILE* fp = std::fopen(files[fi].c_str(), "rb");
        if (!fp) continue;
        if (!std::fgets(line, sizeof(line), fp)) { std::fclose(fp); continue; } // 表头
        // 按 lid 分桶（哈希到 33 桶）
        std::vector<Pt> buckets[33];
        FrameStat fs;
        int seq = 0;
        while (std::fgets(line, sizeof(line), fp)) {
            int lid = 0; double xL = 0, yL = 0, xR = 0, res = 0;
            if (std::sscanf(line, "%d,%lf,%lf,%lf,%lf", &lid, &xL, &yL, &xR, &res) != 5)
                continue;
            if (lid < 1 || lid > 32) continue;
            Pt p;
            p.y = static_cast<float>(yL);
            p.res = static_cast<float>(res);
            p.row = rowOf(yL, sp.rowStep);
            p.seq = seq++;
            p.lidOf = lid;
            buckets[lid].push_back(p);
        }
        std::fclose(fp);
        for (int l = 1; l <= 32; ++l) {
            if (buckets[l].empty()) continue;
            fs.pts += static_cast<long long>(buckets[l].size());
            processBucket(buckets[l], sp, t, fs);
        }
        totalPts += fs.pts;
        perFrameNet.emplace_back(static_cast<int>(fi), fs.net);
    }

    // σ̂ 分位数（抽样数组）
    std::vector<double>& s = t.sigmas;
    std::sort(s.begin(), s.end());
    const auto sq = [&](double q) -> double {
        if (s.empty()) return -1.0;
        return s[static_cast<size_t>(q * static_cast<double>(s.size() - 1))];
    };

    // 逐帧 top10 异常帧
    std::sort(perFrameNet.begin(), perFrameNet.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    std::printf("== line_clean 阶段0 离线仿真（与⑰算子同构）==\n");
    std::printf("输入: %s (%zu 帧)\n", argv[1], files.size());
    std::printf("参数: k=%.2f gap_rows=%d cap=%.3f rowStep=%.2f win=%d min_run=%d sigma_floor=%.2f\n",
                sp.k, sp.gap_rows, sp.cap, sp.rowStep, sp.win, sp.min_run, sp.sigmaFloor);
    std::printf("\n-- go/no-go 主判据 --\n");
    std::printf("总点数        : %lld\n", totalPts);
    std::printf("毛剔点数      : %lld (%.4f%%)\n", t.gross,
                100.0 * static_cast<double>(t.gross) / static_cast<double>(totalPts));
    std::printf("净剔点数      : %lld (%.4f%%)  ← 剔点率判据(0.3%%~3%%)\n", t.net,
                100.0 * static_cast<double>(t.net) / static_cast<double>(totalPts));
    std::printf("率帽回退线数  : %d (帧×线累计)\n", t.fallback);
    std::printf("边缘带占比    : %.1f%% (dev∈[kσ,(k+1.5)σ) 可疑误杀带;毛剔为分母)\n",
                t.gross ? 100.0 * static_cast<double>(t.edgeBand) /
                              static_cast<double>(t.gross) : 0.0);
    std::printf("\n-- 按线净剔分布(1..25) --\n");
    for (int l = 1; l <= 25; ++l)
        std::printf("L%-2d:%-7lld ", l,
                    static_cast<long long>(t.netByLid[static_cast<size_t>(l)]));
    std::printf("\n\n-- 被剔点 dev 幅值直方图(0.5px桶,末桶=12px+) --\n");
    for (size_t b = 0; b < t.devHist.size(); ++b)
        if (t.devHist[b])
            std::printf("%.1f-%.1fpx: %lld\n", 0.5 * b, 0.5 * (b + 1), t.devHist[b]);
    std::printf("\n-- 被剔点 |res| 直方图(0.5px桶;>2px≈错配级,<1px≈抖动级) --\n");
    for (size_t b = 0; b < t.resHist.size(); ++b)
        if (t.resHist[b])
            std::printf("%.1f-%.1fpx: %lld\n", 0.5 * b, 0.5 * (b + 1), t.resHist[b]);
    std::printf("\n-- σ̂ 分布(×1.4826 后,含下限) --\n");
    std::printf("P10=%.3f P50=%.3f P90=%.3f px (σ_floor=%.2f)\n",
                sq(0.10), sq(0.50), sq(0.90), sp.sigmaFloor);
    std::printf("\n-- run 长度分布 --\n");
    std::printf("<5(豁免): %lld  5..59: %lld  >=60: %lld (帧×线×run)\n",
                t.runsLt5, t.runs5to59, t.runsGe60);
    std::printf("\n-- 逐帧净剔 top10(异常帧探针) --\n");
    for (size_t i = 0; i < perFrameNet.size() && i < 10; ++i)
        std::printf("frame_%03d: %lld\n", perFrameNet[i].first, perFrameNet[i].second);
    return 0;
}
