// cmtt_probe.cpp — CMTT 表反查：鬼影点的线号曲线覆盖/预测核对（查 BUG 工具）
// 用法：cmtt_probe.exe <sidecar.bin> <tempC> <l4dumpDir> <cycle> <dispMain>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include "cmtt_table_loader.h"

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "usage: cmtt_probe <sidecar.bin> <tempC> <l4dumpDir> <cycle> <dispMain>\n";
        return 1;
    }
    const char* bin = argv[1];
    const double tempC = std::atof(argv[2]);
    const std::string dir = argv[3];
    const int cyc = std::atoi(argv[4]);
    const double dispMain = std::atof(argv[5]);

    std::string err;
    int tierIdx = -1;
    double tierTemp = 0;
    float rowStep = 0;
    auto tbl = calib::LoadTempTableFromCmtt(bin, tempC, err, &tierIdx, &tierTemp, &rowStep);
    if (!tbl) { std::cerr << "load fail: " << err << "\n"; return 1; }
    std::printf("CMTT loaded: tier=%d temp=%.1f rowStep=%.3f tiers=%d\n",
                tierIdx, tierTemp, rowStep, (int)tbl->table.size());
    const calib::LaserPlaneMap* m = tbl->lookup(tempC);
    if (!m || m->leftToRightMap.empty()) { std::cerr << "tier empty\n"; return 1; }
    const cv::Mat& mp = m->leftToRightMap;   // N×1 CV_32FC4 或 N×4
    const int N = mp.rows * mp.cols;
    const bool fc4 = mp.type() == CV_32FC4 || (mp.channels() == 4);
    std::cout << "map entries=" << N << (fc4 ? " [FC4]" : " [N4]") << "\n";

    // —— 逐线号覆盖统计 ——
    struct Cover { float x0 = 1e30f, x1 = -1e30f, y0 = 1e30f, y1 = -1e30f; int n = 0; };
    std::map<int, Cover> cov;
    auto each = [&](auto&& fn) {
        for (int i = 0; i < N; ++i) {
            float xL, yL, uR, lidF;
            if (fc4) {
                const cv::Vec4f& v = mp.at<cv::Vec4f>(i);
                xL = v[0]; yL = v[1]; uR = v[2]; lidF = v[3];
            } else {
                xL = mp.at<float>(i, 0); yL = mp.at<float>(i, 1);
                uR = mp.at<float>(i, 2); lidF = mp.at<float>(i, 3);
            }
            fn(xL, yL, uR, (int)(lidF + 0.5f));
        }
    };
    each([&](float xL, float yL, float, int lid) {
        Cover& c = cov[lid];
        c.x0 = std::min(c.x0, xL); c.x1 = std::max(c.x1, xL);
        c.y0 = std::min(c.y0, yL); c.y1 = std::max(c.y1, yL);
        ++c.n;
    });
    std::cout << "lid coverage (xL range / y range / entries):\n";
    for (auto& kv : cov)
        std::printf("  lid=%2d x[%6.1f %6.1f] y[%6.1f %6.1f] n=%d\n",
                    kv.first, kv.second.x0, kv.second.x1, kv.second.y0,
                    kv.second.y1, kv.second.n);

    // —— 读该帧匹配对（含 lid/track）——
    auto readCsv = [&](const std::string& path, int cols,
                       std::vector<std::vector<double>>& out) {
        std::ifstream f(path);
        if (!f) { std::cerr << "open fail " << path << "\n"; return false; }
        std::string line;
        std::getline(f, line);
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
    char buf[512];
    std::vector<std::vector<double>> L, R, T;
    std::snprintf(buf, sizeof(buf), "%s/cycle_%03d/L4-match_L.csv", dir.c_str(), cyc);
    if (!readCsv(buf, 2, L)) return 1;
    std::snprintf(buf, sizeof(buf), "%s/cycle_%03d/L4-match_R.csv", dir.c_str(), cyc);
    if (!readCsv(buf, 2, R)) return 1;
    std::snprintf(buf, sizeof(buf), "%s/cycle_%03d/L4t-match-tracks.csv", dir.c_str(), cyc);
    readCsv(buf, 4, T);
    const size_t n = std::min(L.size(), R.size());

    // —— 表索引：按 (lid) → vector<(xL,yL,uR)>，二分查 xL ——
    std::map<int, std::vector<std::array<float, 3>>> byLid;
    each([&](float xL, float yL, float uR, int lid) {
        byLid[lid].push_back({xL, yL, uR});
    });
    for (auto& kv : byLid)
        std::sort(kv.second.begin(), kv.second.end(),
                  [](auto& a, auto& b) { return a[0] < b[0]; });

    // 对鬼影对（抽样 300）：
    //   ① 投票线号 ℓ 的表预测 uR(ℓ,xL,y) vs 实选 xR
    //   ② 哪个线号的预测最接近正确伙伴位置 xL−dispMain
    const double kVL = 0.7, kRowY = 3.0;
    size_t tested = 0, votedHit = 0, noEntryVoted = 0;
    std::map<int, int> bestLidHist;
    double bestOffSum = 0;
    for (size_t i = 0; i < n && tested < 300; ++i) {
        const double d = L[i][0] - R[i][0];
        if (d >= dispMain - 30.0) continue;
        const double xL = L[i][0], y = L[i][1];
        const double xR = R[i][0];
        const int voted = (i < T.size()) ? (int)(T[i][2] + 0.5) : -1;
        // ① voted lid 预测
        auto query = [&](int lid, double* uRout) {
            auto it = byLid.find(lid);
            if (it == byLid.end()) return false;
            const auto& v = it->second;
            auto lb = std::lower_bound(v.begin(), v.end(), (float)(xL - kVL),
                                       [](auto& a, float b) { return a[0] < b; });
            float best = -1e30f;
            bool found = false;
            for (auto j = lb; j != v.end() && j->at(0) <= xL + kVL; ++j)
                if (std::fabs(j->at(1) - y) <= kRowY) {
                    if (!found || std::fabs(j->at(2) - xR) < std::fabs(best - xR)) {
                        best = j->at(2);
                        found = true;
                    }
                    uRout && (*uRout = best);
                    return true;   // 简化：首个近邻行即返回
                }
            if (found && uRout) *uRout = best;
            return found;
        };
        double uRv = 0;
        if (query(voted, &uRv)) {
            if (std::fabs(uRv - xR) <= 8.0) ++votedHit;
        } else ++noEntryVoted;
        // ② 找预测最接近正确位置的线号
        const double wantU = xL - dispMain;
        int bestLid = -1;
        double bestOff = 1e30;
        for (auto& kv : byLid) {
            const auto& v = kv.second;
            auto lb = std::lower_bound(v.begin(), v.end(), (float)(xL - kVL),
                                       [](auto& a, float b) { return a[0] < b; });
            for (auto j = lb; j != v.end() && j->at(0) <= xL + kVL; ++j) {
                if (std::fabs(j->at(1) - y) > kRowY) continue;
                const double off = std::fabs(j->at(2) - wantU);
                if (off < bestOff) { bestOff = off; bestLid = kv.first; }
                break;
            }
        }
        if (bestLid >= 0 && bestOff <= 8.0) {
            ++bestLidHist[bestLid];
            bestOffSum += bestOff;
        }
        ++tested;
    }
    std::printf("\nghost sampled=%zu\n", tested);
    std::printf("voted-lid table hit (|uR_pred - xR|<=8px): %zu (%.1f%%) | no-entry: %zu\n",
                votedHit, tested ? 100.0 * votedHit / tested : 0, noEntryVoted);
    std::cout << "lid whose curve passes correct partner (|uR-want|<=8px), histogram:\n";
    std::vector<std::pair<int, int>> bv(bestLidHist.begin(), bestLidHist.end());
    std::sort(bv.begin(), bv.end(), [](auto& a, auto& b) { return a.second > b.second; });
    for (auto& kv : bv) std::printf("  lid=%d : %d (meanOff=%.2fpx)\n", kv.first, kv.second,
                                    bestOffSum / std::max<size_t>(1, bestLidHist.size()));
    return 0;
}
