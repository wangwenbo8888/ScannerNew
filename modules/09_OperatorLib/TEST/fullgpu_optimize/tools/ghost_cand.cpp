// ghost_cand.cpp — 判别：鬼影对的正确视差位置是否存在右目候选点
// 用法：ghost_cand.exe <l4dumpDir> <cycle> <dispMain>
// 读 L4-match_L/R.csv（匹配对）＋L3-interp_R.csv（右目全部插值点）
// 对鬼影对（disp<dispMain-30）：在 (xL-dispMain, yL) ±8px 窗内找右目点→有/无候选
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
    if (argc < 4) { std::cerr << "usage: ghost_cand <dir> <cycle> <dispMain>\n"; return 1; }
    const std::string dir = argv[1];
    const int cyc = std::atoi(argv[2]);
    const double dispMain = std::atof(argv[3]);
    char buf[512];

    auto readP2 = [&](const std::string& path, std::vector<double>& xs,
                      std::vector<double>& ys) {
        std::ifstream f(path);
        if (!f) { std::cerr << "open fail " << path << "\n"; return false; }
        std::string line, a, b;
        std::getline(f, line);
        while (std::getline(f, line)) {
            std::stringstream ss(line);
            if (std::getline(ss, a, ',') && std::getline(ss, b, ',')) {
                xs.push_back(std::atof(a.c_str()));
                ys.push_back(std::atof(b.c_str()));
            }
        }
        return true;
    };
    std::vector<double> xl, yl, xr, yr, ix, iy;
    std::snprintf(buf, sizeof(buf), "%s/cycle_%03d/L4-match_L.csv", dir.c_str(), cyc);
    if (!readP2(buf, xl, yl)) return 1;
    std::snprintf(buf, sizeof(buf), "%s/cycle_%03d/L4-match_R.csv", dir.c_str(), cyc);
    if (!readP2(buf, xr, yr)) return 1;
    std::snprintf(buf, sizeof(buf), "%s/cycle_%03d/L3-interp_R.csv", dir.c_str(), cyc);
    if (!readP2(buf, ix, iy)) return 1;
    std::cout << "cycle " << cyc << " pairs=" << xl.size()
              << " interpR=" << ix.size() << " dispMain=" << dispMain << "\n";

    // 右目点按 y 建桶加速（±8px 窗）
    const double kWin = 8.0;
    std::map<int, std::vector<size_t>> yb;
    for (size_t j = 0; j < ix.size(); ++j)
        yb[(int)std::floor(iy[j] / kWin)].push_back(j);

    size_t nGhost = 0, withCand = 0, noCand = 0;
    double dxHas = 0;
    for (size_t i = 0; i + 1 < xl.size() + 1 && i < xl.size(); ++i) {
        const double d = xl[i] - xr[i];
        if (d >= dispMain - 30.0) continue;
        ++nGhost;
        const double wantX = xl[i] - dispMain;
        const double wantY = yl[i];
        bool has = false;
        double best = 1e30;
        const int b0 = (int)std::floor((wantY - kWin) / kWin);
        const int b1 = (int)std::floor((wantY + kWin) / kWin);
        for (int b = b0; b <= b1; ++b) {
            auto it = yb.find(b);
            if (it == yb.end()) continue;
            for (size_t j : it->second) {
                const double dy = std::fabs(iy[j] - wantY);
                const double dx = std::fabs(ix[j] - wantX);
                if (dy <= kWin && dx <= kWin) {
                    has = true;
                    best = std::min(best, dx);
                }
            }
        }
        if (has) { ++withCand; dxHas += best; }
        else ++noCand;
    }
    std::printf("ghost pairs=%zu  correct-pos candidate PRESENT=%zu (%.1f%%)  ABSENT=%zu (%.1f%%)\n",
                nGhost, withCand, nGhost ? 100.0 * withCand / nGhost : 0.0,
                noCand, nGhost ? 100.0 * noCand / nGhost : 0.0);
    if (withCand)
        std::printf("mean |cand.x - wantX| of present ones = %.2f px\n", dxHas / withCand);
    return 0;
}
