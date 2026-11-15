// ghost_track.cpp — 鬼影对的段(track)归属＋线号×段交叉表
// 用法：ghost_track.exe <l4dumpDir> <cycle> <dispMain>
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
    if (argc < 4) { std::cerr << "usage: ghost_track <dir> <cycle> <dispMain>\n"; return 1; }
    const std::string dir = argv[1];
    const int cyc = std::atoi(argv[2]);
    const double dispMain = std::atof(argv[3]);
    char buf[512];

    auto readCsv = [&](const std::string& path, int cols,
                       std::vector<std::vector<double>>& out) {
        std::ifstream f(path);
        if (!f) { std::cerr << "open fail " << path << "\n"; return false; }
        std::string line;
        std::getline(f, line);   // header
        while (std::getline(f, line)) {
            std::stringstream ss(line);
            std::vector<double> row;
            std::string tok;
            while (std::getline(ss, tok, ',') && row.size() < (size_t)cols)
                row.push_back(std::atof(tok.c_str()));
            if (row.size() == (size_t)cols) out.push_back(row);
        }
        return true;
    };
    std::vector<std::vector<double>> L, R, T;
    std::snprintf(buf, sizeof(buf), "%s/cycle_%03d/L4-match_L.csv", dir.c_str(), cyc);
    if (!readCsv(buf, 2, L)) return 1;
    std::snprintf(buf, sizeof(buf), "%s/cycle_%03d/L4-match_R.csv", dir.c_str(), cyc);
    if (!readCsv(buf, 2, R)) return 1;
    std::snprintf(buf, sizeof(buf), "%s/cycle_%03d/L4t-match-tracks.csv", dir.c_str(), cyc);
    if (!readCsv(buf, 4, T)) return 1;
    const size_t n = std::min(L.size(), R.size());
    std::cout << "cycle " << cyc << " pairs=" << n << " tracks=" << T.size() << "\n";

    // —— 每 track：总对数 / 鬼影对数 / lid 直方图 ——
    std::map<long long, std::map<long long, int>> trLid;      // track→lid→count
    std::map<long long, int> trGhost, trAll;
    for (size_t i = 0; i < n; ++i) {
        const long long tr = (i < T.size()) ? (long long)T[i][2 + 1] : -1;   // 第4列 track
        const long long lid = (i < T.size()) ? (long long)T[i][2] : -1;      // 第3列 lid
        const double d = L[i][0] - R[i][0];
        ++trLid[tr][lid];
        ++trAll[tr];
        if (d < dispMain - 30.0) ++trGhost[tr];
    }
    // —— track 概览：含鬼影的 track 全列（鬼影降序）——
    std::vector<std::pair<long long, int>> withGhost;
    for (auto& kv : trGhost) if (kv.second > 0) withGhost.push_back({kv.first, kv.second});
    std::sort(withGhost.begin(), withGhost.end(),
              [](auto& a, auto& b) { return a.second > b.second; });
    std::cout << "tracks containing ghost pairs: " << withGhost.size()
              << " (of " << trAll.size() << " tracks total)\n";
    // —— 全 track 表：tr | lid | n | xL 范围 | y 范围（供跨帧对比）——
    std::map<long long, double> trX0, trX1, trY0, trY1;
    for (size_t i = 0; i < n; ++i) {
        const long long tr = (i < T.size()) ? (long long)T[i][3] : -1;
        if (!trAll.count(tr)) continue;
        if (!trX0.count(tr)) { trX0[tr] = trX1[tr] = L[i][0]; trY0[tr] = trY1[tr] = L[i][1]; }
        else {
            trX0[tr] = std::min(trX0[tr], L[i][0]); trX1[tr] = std::max(trX1[tr], L[i][0]);
            trY0[tr] = std::min(trY0[tr], L[i][1]); trY1[tr] = std::max(trY1[tr], L[i][1]);
        }
    }
    std::cout << "ALL tracks: tr|lid|n|xL-range|y-range|ghost\n";
    for (auto& kv : trAll) {
        long long lid = -1;
        int best = 0;
        for (auto& lv : trLid[kv.first])
            if (lv.second > best) { best = lv.second; lid = lv.first; }
        std::printf("  tr%-6lld lid=%-3d n=%-5zu xL[%6.0f %6.0f] y[%5.0f %5.0f] g=%d\n",
                    kv.first, lid, kv.second, trX0[kv.first], trX1[kv.first],
                    trY0[kv.first], trY1[kv.first], trGhost.count(kv.first) ? trGhost[kv.first] : 0);
    }
    std::cout << "track | ghost/total | lid histogram\n";
    for (int k = 0; k < 15 && k < (int)withGhost.size(); ++k) {
        const long long tr = withGhost[k].first;
        std::cout << "  tr" << tr << " | " << trGhost[tr] << "/" << trAll[tr] << " | ";
        std::vector<std::pair<long long, int>> lv(trLid[tr].begin(), trLid[tr].end());
        std::sort(lv.begin(), lv.end(),
                  [](auto& a, auto& b) { return a.second > b.second; });
        for (int j = 0; j < 8 && j < (int)lv.size(); ++j)
            std::cout << "lid" << lv[j].first << ":" << lv[j].second << " ";
        std::cout << "\n";
    }
    // —— 鬼影 track 是否与主面共存：track 内鬼影占比 ——
    int mixed = 0, pureGhost = 0;
    for (auto& kv : trGhost) {
        if (kv.second == 0) continue;
        if (kv.second >= trAll[kv.first]) ++pureGhost;
        else ++mixed;
    }
    std::printf("ghost tracks: pure(all-ghost)=%d  mixed(部分鬼影)=%d\n", pureGhost, mixed);
    return 0;
}
