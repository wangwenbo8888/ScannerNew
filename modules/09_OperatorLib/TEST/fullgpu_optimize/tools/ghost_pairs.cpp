// ghost_pairs.cpp — 鬼影点对归因 v2：视差→主面/鬼影分桶＋线号/y/x 分布
// 用法：ghost_pairs.exe <l4dumpDir> <cycle>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: ghost_pairs <dir> <cycle>\n"; return 1; }
    const std::string dir = argv[1];
    const int cyc = std::atoi(argv[2]);
    char buf[512];

    auto readP2 = [&](const char* side, std::vector<double>& xs,
                      std::vector<double>& ys) {
        std::snprintf(buf, sizeof(buf), "%s/cycle_%03d/L4-match_%s.csv",
                      dir.c_str(), cyc, side);
        std::ifstream f(buf);
        if (!f) { std::cerr << "open fail " << buf << "\n"; return false; }
        std::string line;
        std::getline(f, line);   // header
        std::string a, b;
        while (std::getline(f, line)) {
            std::stringstream ss(line);
            if (std::getline(ss, a, ',') && std::getline(ss, b, ',')) {
                xs.push_back(std::atof(a.c_str()));
                ys.push_back(std::atof(b.c_str()));
            }
        }
        return true;
    };
    std::vector<double> xl, yl, xr, yr;
    if (!readP2("L", xl, yl) || !readP2("R", xr, yr)) return 1;
    std::vector<int> lid;
    {
        std::snprintf(buf, sizeof(buf), "%s/cycle_%03d/L4t-match-tracks.csv",
                      dir.c_str(), cyc);
        std::ifstream f(buf);
        std::string line;
        std::getline(f, line);
        std::string a, b, c, d;
        while (std::getline(f, line)) {
            std::stringstream ss(line);
            if (std::getline(ss, a, ',') && std::getline(ss, b, ',')
                && std::getline(ss, c, ',') && std::getline(ss, d, ','))
                lid.push_back(std::atoi(c.c_str()));
        }
    }
    const size_t n = std::min(xl.size(), xr.size());
    std::cout << "cycle " << cyc << " pairs=" << n << " lids=" << lid.size() << "\n";
    if (n < 10) return 1;

    // —— 视差数组（唯一事实源）——
    std::vector<double> disp(n);
    for (size_t i = 0; i < n; ++i) disp[i] = xl[i] - xr[i];

    // —— 1px 桶直方图 → 主面众数 ——
    std::map<int, int> dh;
    for (double d : disp) ++dh[(int)std::floor(d)];
    int modeK = 0, modeC = 0;
    for (auto& kv : dh)
        if (kv.second > modeC) { modeC = kv.second; modeK = kv.first; }
    const double dispMode = modeK + 0.5;
    std::cout << "disp mode=" << dispMode << "px (f=" << modeC << ")\n";
    std::cout << "disp histogram (1px buckets, counts>30):\n";
    for (auto& kv : dh)
        if (kv.second > 30) std::printf("  [%d %d)px : %d\n", kv.first, kv.first + 1, kv.second);

    // —— 主面/鬼影分桶：主面＝|disp-mode|<=3px；鬼影＝disp<mode-10px ——
    size_t nMain = 0, nGhost = 0;
    for (double d : disp) {
        if (std::fabs(d - dispMode) <= 3.0) ++nMain;
        else if (d < dispMode - 10.0) ++nGhost;
    }
    std::printf("main(|d-mode|<=3px)=%zu  ghost(d<mode-10px)=%zu  other=%zu\n",
                nMain, nGhost, n - nMain - nGhost);

    // —— 鬼影分布：线号 / y 带 / 左点 x ——
    std::map<int, int> gLid, gY, aLid;
    double gX0 = 1e30, gX1 = -1e30, gY0 = 1e30, gY1 = -1e30;
    double gD0 = 1e30, gD1 = -1e30;
    for (size_t i = 0; i < n; ++i) {
        if (i < lid.size()) ++aLid[lid[i]];
        if (disp[i] >= dispMode - 10.0) continue;
        if (i < lid.size()) ++gLid[lid[i]];
        ++gY[(int)std::floor(yl[i] / 100) * 100];
        gX0 = std::min(gX0, xl[i]); gX1 = std::max(gX1, xl[i]);
        gY0 = std::min(gY0, yl[i]); gY1 = std::max(gY1, yl[i]);
        gD0 = std::min(gD0, disp[i]); gD1 = std::max(gD1, disp[i]);
    }
    std::printf("ghost disp[%.1f %.1f]px  xL[%.0f %.0f] y[%.0f %.0f]\n",
                gD0, gD1, gX0, gX1, gY0, gY1);
    auto dumpTop = [](const char* tag, std::map<int, int>& m) {
        std::vector<std::pair<int, int>> v(m.begin(), m.end());
        std::sort(v.begin(), v.end(),
                  [](auto& a, auto& b) { return a.second > b.second; });
        std::cout << tag << " (top 12): ";
        for (int i = 0; i < 12 && i < (int)v.size(); ++i)
            std::cout << v[i].first << ":" << v[i].second << " ";
        std::cout << "\n";
    };
    dumpTop("ghost lid", gLid);
    dumpTop("ALL   lid", aLid);
    std::cout << "ghost y-band: ";
    for (auto& kv : gY) std::cout << kv.first << ":" << kv.second << " ";
    std::cout << "\n";

    // —— 鬼影对里的『右点 x - 预测』粗核验：右点是否与其它线重叠 ——
    // （输出鬼影对的 lid 集与主面对 lid 集的差＝哪几条线贡献了鬼影）
    return 0;
}
