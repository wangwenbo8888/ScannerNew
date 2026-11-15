// frame_attrib.cpp — 离面点逐帧归因（查 BUG 一次性工具，不入构建树）
// 输入：JMW_FRAME_DUMPS 目录（rt.csv＋f%03d.ply 相机系点）
// 流程：全帧变换到全局系→RANSAC 拟合主平面→逐帧统计离面点→按 regSource/奇偶聚合
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

struct Frame {
    int cyc;
    std::string reg;
    double R[9], T[3];
    std::vector<float> pts;   // 相机系 xyz
    // 统计
    size_t n = 0, off = 0;
    double offFrac = 0, dMax = 0, dSumAbs = 0;
};

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: frame_attrib <dumpDir> [--th MM]\n"; return 1; }
    std::string dir = argv[1];
    double th = 50.0;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--th" && i + 1 < argc) th = std::atof(argv[++i]);

    std::vector<Frame> frames;
    {
        std::ifstream rt(dir + "/rt.csv");
        if (!rt) { std::cerr << "no rt.csv\n"; return 1; }
        std::string line;
        while (std::getline(rt, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::vector<std::string> toks;
            std::string tok;
            while (std::getline(ss, tok, ',')) toks.push_back(tok);
            if (toks.size() < 14) continue;
            Frame f;
            f.cyc = std::atoi(toks[0].c_str());
            f.reg = toks[1];
            for (int i = 0; i < 9; ++i) f.R[i] = std::atof(toks[2 + i].c_str());
            for (int i = 0; i < 3; ++i) f.T[i] = std::atof(toks[11 + i].c_str());
            frames.push_back(std::move(f));
        }
    }
    std::cout << "rt rows: " << frames.size() << " | th=" << th << "mm\n";

    // —— 读各帧 PLY（相机系）并变换到全局系（=R*pt+T）——
    std::vector<double> gx, gy, gz;      // 全局系全部点（离面分析）
    std::vector<int> owner;              // 点→帧下标
    for (auto& f : frames) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%s/f%03d.ply", dir.c_str(), f.cyc);
        std::ifstream ply(buf);
        if (!ply) continue;
        std::string line;
        long long nDecl = 0;
        while (std::getline(ply, line)) {
            if (line.rfind("element vertex", 0) == 0)
                nDecl = std::atoll(line.c_str() + 14);
            if (line == "end_header") break;
        }
        const size_t base = gx.size();
        double x = 0, y = 0, z = 0;
        while (ply >> x >> y >> z) {
            const double X = f.R[0] * x + f.R[1] * y + f.R[2] * z + f.T[0];
            const double Y = f.R[3] * x + f.R[4] * y + f.R[5] * z + f.T[1];
            const double Z = f.R[6] * x + f.R[7] * y + f.R[8] * z + f.T[2];
            gx.push_back(X); gy.push_back(Y); gz.push_back(Z);
            owner.push_back(-1);   // 下面回填
        }
        const size_t cnt = gx.size() - base;
        for (size_t i = base; i < gx.size(); ++i) owner[i] = f.cyc;
        f.n = cnt;
        (void)nDecl;
    }
    const size_t n = gx.size();
    std::cout << "global pts: " << n << "\n";

    // —— RANSAC 主平面 ——
    std::mt19937 rng(2026);
    double bn[3] = {0, 0, 1}, bd = 0;
    size_t bestIn = 0;
    std::uniform_int_distribution<size_t> pick(0, n - 1);
    for (int it = 0; it < 2500; ++it) {
        const size_t a = pick(rng), b = pick(rng), c = pick(rng);
        double ux = gx[b] - gx[a], uy = gy[b] - gy[a], uz = gz[b] - gz[a];
        double vx = gx[c] - gx[a], vy = gy[c] - gy[a], vz = gz[c] - gz[a];
        double nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
        const double len = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (len < 1e-6) continue;
        nx /= len; ny /= len; nz /= len;
        const double d = -(nx * gx[a] + ny * gy[a] + nz * gz[a]);
        size_t in = 0;
        for (size_t s = 0; s < n; s += 11)
            if (std::fabs(nx * gx[s] + ny * gy[s] + nz * gz[s] + d) <= 3.0) ++in;
        if (in > bestIn) { bestIn = in; bn[0] = nx; bn[1] = ny; bn[2] = nz; bd = d; }
    }
    std::printf("plane: n=(%.4f %.4f %.4f) d=%.2f\n", bn[0], bn[1], bn[2], bd);

    // —— 逐帧统计 ——
    std::map<int, Frame*> byCyc;
    for (auto& f : frames) byCyc[f.cyc] = &f;
    std::vector<double> sd(n);
    for (size_t i = 0; i < n; ++i) {
        sd[i] = bn[0] * gx[i] + bn[1] * gy[i] + bn[2] * gz[i] + bd;
        Frame* f = byCyc.count(owner[i]) ? byCyc[owner[i]] : nullptr;
        if (!f) continue;
        if (std::fabs(sd[i]) > th) { ++f->off; f->dSumAbs += std::fabs(sd[i]); }
        f->dMax = std::max(f->dMax, std::fabs(sd[i]));
    }
    for (auto& f : frames) f.offFrac = f.n ? 100.0 * f.off / f.n : 0.0;

    // —— 帧清单（按离面点数降序 top 40）——
    std::vector<Frame*> ranked;
    for (auto& f : frames) ranked.push_back(&f);
    std::sort(ranked.begin(), ranked.end(),
              [](const Frame* a, const Frame* b) { return a->off > b->off; });
    std::cout << "top frames by off-plane(>" << th << "mm) count:\n";
    std::cout << "  cyc reg    npts   off  frac%   dMax  meanAbs\n";
    for (int i = 0; i < 40 && i < (int)ranked.size() && ranked[i]->off > 0; ++i) {
        const Frame* f = ranked[i];
        std::printf("  %3d %-5s %6zu %6zu %6.2f %6.0f %6.0f\n", f->cyc, f->reg.c_str(),
                    f->n, f->off, f->offFrac, f->dMax,
                    f->off ? f->dSumAbs / f->off : 0.0);
    }

    // —— 按 regSource 聚合 ——
    std::cout << "aggregate by regSource:\n";
    std::map<std::string, std::pair<size_t, size_t>> agg;   // reg→(总点, 离面点)
    for (const auto& f : frames) {
        agg[f.reg].first += f.n;
        agg[f.reg].second += f.off;
    }
    for (const auto& kv : agg)
        std::printf("  %-6s pts=%8zu off=%8zu (%5.2f%%)\n", kv.first.c_str(),
                    kv.second.first, kv.second.second,
                    kv.second.first ? 100.0 * kv.second.second / kv.second.first : 0.0);

    // —— 按奇偶聚合（帧号奇偶＝激光管号）——
    size_t evenPts = 0, evenOff = 0, oddPts = 0, oddOff = 0;
    for (const auto& f : frames) {
        if (f.cyc % 2 == 0) { evenPts += f.n; evenOff += f.off; }
        else                { oddPts += f.n;  oddOff += f.off; }
    }
    std::printf("parity: even(偶管) pts=%zu off=%zu (%.2f%%) | odd(奇管) pts=%zu off=%zu (%.2f%%)\n",
                evenPts, evenOff, evenPts ? 100.0 * evenOff / evenPts : 0.0,
                oddPts, oddOff, oddPts ? 100.0 * oddOff / oddPts : 0.0);

    // —— 离面帧的帧号区间分布（离面点占比>5% 的帧）——
    std::cout << "frames with offFrac>5%: ";
    int shown = 0;
    for (const auto& f : frames) {
        if (f.offFrac > 5.0) {
            std::cout << f.cyc << "(" << f.reg << ") ";
            if (++shown >= 60) { std::cout << "..."; break; }
        }
    }
    std::cout << "\n";
    return 0;
}
