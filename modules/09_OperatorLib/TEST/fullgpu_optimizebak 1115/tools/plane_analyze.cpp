// plane_analyze.cpp — 点云离面分析一次性诊断工具（查 BUG 用，不入构建树）
// 用法：plane_analyze.exe <in.ply> [outlier_out.ply] [--dist MM]
// 输出：RANSAC 主平面参数、逐阈值离面占比、带符号距离直方图、远点空间聚类
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
#include <tuple>
#include <vector>

struct Pt { float x, y, z; };

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: plane_analyze <in.ply> [outlier_out.ply] [--dist MM]\n";
        return 1;
    }
    const char* inPath = argv[1];
    const char* outPath = argc >= 3 && argv[2][0] != '-' ? argv[2] : nullptr;
    double distTh = 100.0;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--dist" && i + 1 < argc)
            distTh = std::atof(argv[++i]);
    }
    std::ifstream f(inPath);
    if (!f) { std::cerr << "open fail: " << inPath << "\n"; return 1; }
    std::string line;
    long long nDecl = 0;
    while (std::getline(f, line)) {
        if (line.rfind("element vertex", 0) == 0)
            nDecl = std::atoll(line.c_str() + std::string("element vertex").size());
        if (line == "end_header") break;
    }
    std::vector<Pt> pts;
    pts.reserve(static_cast<size_t>(nDecl));
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        Pt p;
        if (ss >> p.x >> p.y >> p.z) pts.push_back(p);
    }
    const size_t n = pts.size();
    std::cout << "points: " << n << " (declared " << nDecl << ")\n";
    if (n < 100) return 1;

    // —— 包围盒 ——
    Pt lo = pts[0], hi = pts[0];
    for (const auto& p : pts) {
        lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
        hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
    }
    std::printf("bbox: x[%.1f %.1f] y[%.1f %.1f] z[%.1f %.1f]\n",
                lo.x, hi.x, lo.y, hi.y, lo.z, hi.z);

    // —— RANSAC 平面拟合（迭代 3000，内点阈 3mm；平面：n·x+d=0）——
    std::mt19937 rng(12345);
    double bestN[3] = {0, 0, 1}, bestD = 0;
    size_t bestIn = 0;
    std::uniform_int_distribution<size_t> pick(0, n - 1);
    const double kInlierMm = 3.0;
    for (int it = 0; it < 3000; ++it) {
        const Pt& a = pts[pick(rng)];
        const Pt& b = pts[pick(rng)];
        const Pt& c = pts[pick(rng)];
        double ux = b.x - a.x, uy = b.y - a.y, uz = b.z - a.z;
        double vx = c.x - a.x, vy = c.y - a.y, vz = c.z - a.z;
        double nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz_ = ux * vy - uy * vx;
        const double len = std::sqrt(nx * nx + ny * ny + nz_ * nz_);
        if (len < 1e-6) continue;
        nx /= len; ny /= len; nz_ /= len;
        const double d = -(nx * a.x + ny * a.y + nz_ * a.z);
        size_t in = 0;
        for (size_t s = 0; s < n; s += 7) {   // 1/7 抽样计数
            const Pt& p = pts[s];
            if (std::fabs(nx * p.x + ny * p.y + nz_ * p.z + d) <= kInlierMm) ++in;
        }
        if (in > bestIn) {
            bestIn = in;
            bestN[0] = nx; bestN[1] = ny; bestN[2] = nz_; bestD = d;
        }
    }
    // 全量精算内点
    std::vector<double> sd(n);
    size_t in3 = 0;
    for (size_t i = 0; i < n; ++i) {
        const Pt& p = pts[i];
        sd[i] = bestN[0] * p.x + bestN[1] * p.y + bestN[2] * p.z + bestD;
        if (std::fabs(sd[i]) <= kInlierMm) ++in3;
    }
    std::printf("plane: n=(%.4f %.4f %.4f) d=%.2f | inlier(3mm)=%.1f%%\n",
                bestN[0], bestN[1], bestN[2], bestD, 100.0 * in3 / n);

    // —— 逐阈值离面占比 ——
    for (double th : {3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0}) {
        size_t c = 0;
        for (double v : sd) if (std::fabs(v) > th) ++c;
        std::printf("|d|>%6.0fmm: %8zu (%5.2f%%)\n", th, c, 100.0 * c / n);
    }

    // —— 带符号距离直方图（对数桶）——
    std::cout << "signed-dist histogram (mm buckets, log-ish):\n";
    const double edges[] = {-4000,-2000,-1000,-500,-200,-100,-50,-20,-10,-5,-3,
                             3,5,10,20,50,100,200,500,1000,2000,4000};
    const int nB = sizeof(edges) / sizeof(edges[0]) - 1;
    std::vector<size_t> hist(nB + 1, 0);
    for (double v : sd) {
        int b = 0;
        while (b < nB && v >= edges[b + 1]) ++b;
        ++hist[std::min(b, nB)];
    }
    for (int b = 0; b <= nB; ++b) {
        char loS[32], hiS[32];
        if (b == 0) std::snprintf(loS, 32, "-inf");
        else std::snprintf(loS, 32, "%.0f", edges[b]);
        if (b == nB) std::snprintf(hiS, 32, "+inf");
        else std::snprintf(hiS, 32, "%.0f", edges[b + 1]);
        std::printf("  [%8s %8s): %9zu (%5.2f%%)\n", loS, hiS, hist[b],
                    100.0 * hist[b] / n);
    }

    // —— 远点空间聚类（体素 10mm 网格连通域，|d|>distTh）——
    std::map<std::tuple<int,int,int>, int> vox;   // 体素→簇号
    std::vector<int> clusterOf;
    std::vector<std::vector<size_t>> clusters;
    for (size_t i = 0; i < n; ++i) {
        if (std::fabs(sd[i]) <= distTh) continue;
        const auto key = std::make_tuple((int)std::floor(pts[i].x / 10.0),
                                         (int)std::floor(pts[i].y / 10.0),
                                         (int)std::floor(pts[i].z / 10.0));
        auto it = vox.find(key);
        if (it == vox.end()) { vox[key] = (int)clusters.size(); clusters.emplace_back(); it = vox.find(key); }
        const int cid = it->second;
        // 26 邻域并入（简化：只归并入第一个已有邻居簇）
        for (int dx = -1; dx <= 1; ++dx) for (int dy = -1; dy <= 1; ++dy) for (int dz = -1; dz <= 1; ++dz) {
            if (!dx && !dy && !dz) continue;
            auto nit = vox.find(std::make_tuple(std::get<0>(key) + dx,
                                                std::get<1>(key) + dy,
                                                std::get<2>(key) + dz));
            if (nit != vox.end() && nit->second != cid) {
                // 合并：把大簇号并入小簇号（惰性，够诊断用）
                const int from = std::max(cid, nit->second), to = std::min(cid, nit->second);
                if (from != to) {
                    for (auto& kv : vox) if (kv.second == from) kv.second = to;
                }
            }
        }
        clusters[vox[key]].push_back(i);
    }
    // 重编簇号并统计（vox[] 已含合并后的簇号；按最终簇号计数）
    std::vector<std::pair<size_t, int>> csize;   // (大小, 簇号)
    {
        std::map<int, size_t> sz;
        for (size_t i = 0; i < n; ++i) {
            if (std::fabs(sd[i]) <= distTh) continue;
            const auto key = std::make_tuple((int)std::floor(pts[i].x / 10.0),
                                             (int)std::floor(pts[i].y / 10.0),
                                             (int)std::floor(pts[i].z / 10.0));
            ++sz[vox[key]];
        }
        for (auto& kv : sz) csize.push_back({kv.second, kv.first});
        std::sort(csize.rbegin(), csize.rend());
    }
    std::cout << "far-point clusters (|d|>" << distTh << "mm, 10mm voxel CC, top 12):\n";
    int shown = 0;
    for (const auto& cs : csize) {
        if (cs.first < 50) break;
        // 找该簇的包围盒与距离范围
        double x0 = 1e30, x1 = -1e30, y0 = 1e30, y1 = -1e30, z0 = 1e30, z1 = -1e30;
        double d0 = 1e30, d1 = -1e30;
        size_t cnt = 0;
        for (size_t i = 0; i < n; ++i) {
            if (std::fabs(sd[i]) <= distTh) continue;
            const auto key = std::make_tuple((int)std::floor(pts[i].x / 10.0),
                                             (int)std::floor(pts[i].y / 10.0),
                                             (int)std::floor(pts[i].z / 10.0));
            if (vox[key] != cs.second) continue;
            ++cnt;
            x0 = std::min(x0, (double)pts[i].x); x1 = std::max(x1, (double)pts[i].x);
            y0 = std::min(y0, (double)pts[i].y); y1 = std::max(y1, (double)pts[i].y);
            z0 = std::min(z0, (double)pts[i].z); z1 = std::max(z1, (double)pts[i].z);
            d0 = std::min(d0, sd[i]); d1 = std::max(d1, sd[i]);
        }
        std::printf("  cluster#%d n=%zu bbox x[%.0f %.0f] y[%.0f %.0f] z[%.0f %.0f] d[%.0f %.0f]\n",
                    ++shown, cnt, x0, x1, y0, y1, z0, z1, d0, d1);
        if (shown >= 12) break;
    }

    // —— 输出远点子集（带符号距离当 intensity 供查看器着色）——
    if (outPath) {
        std::ofstream o(outPath);
        size_t m = 0;
        for (size_t i = 0; i < n; ++i) if (std::fabs(sd[i]) > distTh) ++m;
        o << "ply\nformat ascii 1.0\nelement vertex " << m << "\n"
          << "property float x\nproperty float y\nproperty float z\nproperty float d\nend_header\n";
        o << std::fixed << std::setprecision(3);
        for (size_t i = 0; i < n; ++i)
            if (std::fabs(sd[i]) > distTh)
                o << pts[i].x << ' ' << pts[i].y << ' ' << pts[i].z << ' ' << sd[i] << '\n';
        std::cout << "outlier subset -> " << outPath << " (" << m << " pts)\n";
    }
    return 0;
}
