// steger_dedup_merge.cpp — 位移式去重实现（算法与 steger_dedup_probe 验证版一致）
#include "steger_dedup_merge.h"

#include <opencv2/cudaarithm.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <unordered_map>

namespace steger_dedup {
namespace {

struct Pt { float x, y; };

constexpr float kPairDist = 1.6f;
constexpr float kAlongGate = 0.5f;
constexpr float kNormalGate = 1.5f;
constexpr float kTangentDot = 0.5f;
constexpr float kNbrRadius = 2.5f;
constexpr int kNbrMax = 10;
constexpr int kClusterMax = 8;
constexpr float kClusterSpan = 2.5f;

struct Ctx {
    std::vector<float> tx, ty;
    std::vector<char> valid;
};

static long long CellKey(int x, int y) {
    return ((long long)x << 20) | (long long)y;
}

static Ctx EstimateTangents(const std::vector<Pt>& p) {
    const int n = (int)p.size();
    std::unordered_map<long long, std::vector<int>> grid;
    grid.reserve(n * 2);
    for (int i = 0; i < n; ++i)
        grid[CellKey((int)std::floor(p[i].x), (int)std::floor(p[i].y))].push_back(i);
    Ctx c;
    c.tx.assign(n, 1.f); c.ty.assign(n, 0.f); c.valid.assign(n, 0);
    for (int i = 0; i < n; ++i) {
        int cx = (int)std::floor(p[i].x), cy = (int)std::floor(p[i].y);
        std::vector<std::pair<float, int>> nbr;
        for (int gx = cx - 3; gx <= cx + 3; ++gx)
            for (int gy = cy - 3; gy <= cy + 3; ++gy) {
                auto it = grid.find(CellKey(gx, gy));
                if (it == grid.end()) continue;
                for (int j : it->second) {
                    if (j == i) continue;
                    float dx = p[j].x - p[i].x, dy = p[j].y - p[i].y;
                    float d2 = dx * dx + dy * dy;
                    if (d2 <= kNbrRadius * kNbrRadius) nbr.push_back({d2, j});
                }
            }
        if (nbr.size() < 2) continue;
        std::sort(nbr.begin(), nbr.end());
        int take = (int)std::min<size_t>(nbr.size(), kNbrMax);
        float sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
        for (int k = 0; k < take; ++k) {
            float dx = p[nbr[k].second].x - p[i].x;
            float dy = p[nbr[k].second].y - p[i].y;
            sx += dx; sy += dy; sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
        }
        float mx = sx / take, my = sy / take;
        float vxx = sxx / take - mx * mx, vyy = syy / take - my * my, vxy = sxy / take - mx * my;
        float tr = vxx + vyy, det = vxx * vyy - vxy * vxy;
        float disc = tr * tr / 4 - det;
        float l1 = tr / 2 + std::sqrt(std::max(0.f, disc));
        float l2 = tr / 2 - std::sqrt(std::max(0.f, disc));
        if (l1 <= 1e-6f || l1 < 4 * std::max(l2, 1e-9f)) continue;
        float ang = 0.5f * std::atan2(2 * vxy, vxx - vyy);
        float tX = std::cos(ang), tY = std::sin(ang);
        if (tX < 0 || (tX == 0 && tY < 0)) { tX = -tX; tY = -tY; }
        c.tx[i] = tX; c.ty[i] = tY; c.valid[i] = 1;
    }
    return c;
}

static int CountDupPairs(const std::vector<Pt>& p, const Ctx& c) {
    std::unordered_map<long long, std::vector<int>> grid;
    for (int i = 0; i < (int)p.size(); ++i)
        grid[CellKey((int)std::floor(p[i].x), (int)std::floor(p[i].y))].push_back(i);
    int pairs = 0;
    for (int i = 0; i < (int)p.size(); ++i) {
        if (!c.valid[i]) continue;
        int cx = (int)std::floor(p[i].x), cy = (int)std::floor(p[i].y);
        for (int gx = cx - 2; gx <= cx + 2; ++gx)
            for (int gy = cy - 2; gy <= cy + 2; ++gy) {
                auto it = grid.find(CellKey(gx, gy));
                if (it == grid.end()) continue;
                for (int j : it->second) {
                    if (j <= i || !c.valid[j]) continue;
                    float dx = p[j].x - p[i].x, dy = p[j].y - p[i].y;
                    if (dx * dx + dy * dy > kPairDist * kPairDist) continue;
                    float along = std::fabs(dx * c.tx[i] + dy * c.ty[i]);
                    float nrm = std::fabs(-dx * c.ty[i] + dy * c.tx[i]);
                    if (along <= kAlongGate && nrm <= kNormalGate) ++pairs;
                }
            }
    }
    return pairs;
}

}  // namespace

std::vector<cv::Point2f> mergePoints(const std::vector<cv::Point2f>& in, Stats* st) {
    std::vector<Pt> p(in.size());
    for (size_t i = 0; i < in.size(); ++i) p[i] = Pt{in[i].x, in[i].y};

    Stats s;
    s.inCount = (int)p.size();
    Ctx c = EstimateTangents(p);
    for (int i = 0; i < (int)p.size(); ++i) if (!c.valid[i]) ++s.bypassTangent;

    std::vector<int> parent(p.size());
    for (size_t i = 0; i < p.size(); ++i) parent[i] = (int)i;
    std::function<int(int)> find = [&](int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    auto uni = [&](int a, int b) { a = find(a); b = find(b); if (a != b) parent[a] = b; };

    std::unordered_map<long long, std::vector<int>> grid;
    for (int i = 0; i < (int)p.size(); ++i)
        grid[CellKey((int)std::floor(p[i].x), (int)std::floor(p[i].y))].push_back(i);
    for (int i = 0; i < (int)p.size(); ++i) {
        if (!c.valid[i]) continue;
        int cx = (int)std::floor(p[i].x), cy = (int)std::floor(p[i].y);
        for (int gx = cx - 2; gx <= cx + 2; ++gx)
            for (int gy = cy - 2; gy <= cy + 2; ++gy) {
                auto it = grid.find(CellKey(gx, gy));
                if (it == grid.end()) continue;
                for (int j : it->second) {
                    if (j <= i || !c.valid[j]) continue;
                    float dx = p[j].x - p[i].x, dy = p[j].y - p[i].y;
                    if (dx * dx + dy * dy > kPairDist * kPairDist) continue;
                    float along = std::fabs(dx * c.tx[i] + dy * c.ty[i]);
                    float nrm = std::fabs(-dx * c.ty[i] + dy * c.tx[i]);
                    float tdot = c.tx[i] * c.tx[j] + c.ty[i] * c.ty[j];
                    if (along <= kAlongGate && nrm <= kNormalGate && std::fabs(tdot) >= kTangentDot)
                        uni(i, j);
                }
            }
    }
    std::unordered_map<int, std::vector<int>> clusters;
    for (size_t i = 0; i < p.size(); ++i) clusters[find((int)i)].push_back((int)i);

    std::vector<cv::Point2f> out;
    out.reserve(p.size());
    for (auto& kv : clusters) {
        auto& mem = kv.second;
        if (mem.size() == 1 || mem.size() > (size_t)kClusterMax) {
            for (int m : mem) out.push_back(in[m]);
            continue;
        }
        float tx = c.tx[mem[0]], ty = c.ty[mem[0]];
        float nmin = 1e9f, nmax = -1e9f;
        for (int m : mem) {
            float pr = -p[m].x * ty + p[m].y * tx;
            nmin = std::min(nmin, pr); nmax = std::max(nmax, pr);
        }
        if (nmax - nmin > kClusterSpan) {
            for (int m : mem) out.push_back(in[m]);
            continue;
        }
        if (mem.size() == 2) {
            out.push_back(cv::Point2f(0.5f * (p[mem[0]].x + p[mem[1]].x),
                                      0.5f * (p[mem[0]].y + p[mem[1]].y)));
        } else {
            std::vector<std::pair<float, int>> byN;
            for (int m : mem) byN.push_back({-p[m].x * ty + p[m].y * tx, m});
            std::sort(byN.begin(), byN.end());
            if (byN.size() % 2) out.push_back(in[byN[byN.size() / 2].second]);
            else {
                const Pt& a = p[byN[byN.size() / 2 - 1].second];
                const Pt& b = p[byN[byN.size() / 2].second];
                out.push_back(cv::Point2f(0.5f * (a.x + b.x), 0.5f * (a.y + b.y)));
            }
        }
        ++s.mergedClusters;
        s.mergedPoints += (int)mem.size() - 1;
    }
    s.outCount = (int)out.size();
    std::vector<Pt> outPts(out.size());
    for (size_t i = 0; i < out.size(); ++i) outPts[i] = Pt{out[i].x, out[i].y};
    Ctx c2 = EstimateTangents(outPts);
    s.residDupPairs = CountDupPairs(outPts, c2);
    if (st) *st = s;
    return out;
}

bool apply(calib::StegerResult& r, cv::cuda::Stream& stream, std::string& msg) {
    if (!r.d_centerPoints || r.d_centerPoints->empty() || r.centerPoints.empty()) {
        msg = "empty input";
        return false;
    }
    std::vector<cv::Point2f> in;
    int label = r.centerPoints.begin()->first;
    for (auto& kv : r.centerPoints)
        for (auto& q : kv.second) in.push_back(q);

    Stats st;
    std::vector<cv::Point2f> out = mergePoints(in, &st);
    msg = std::to_string(st.inCount) + "->" + std::to_string(st.outCount)
        + " (merged " + std::to_string(st.mergedPoints)
        + ", bypass " + std::to_string(st.bypassTangent)
        + ", residDup " + std::to_string(st.residDupPairs) + ")";
    if (st.mergedPoints == 0) return true;   // 无合并，原样保留

    // 重建 GPU 双输出（Flat 单标签语义）
    cv::Mat hPts(1, (int)out.size(), CV_32FC2, out.data());
    auto dcp = std::make_shared<cv::cuda::GpuMat>();
    dcp->create(1, (int)out.size(), CV_32FC2);
    dcp->upload(hPts, stream);
    cv::Mat hIds(1, (int)out.size(), CV_32SC1, cv::Scalar(label));
    auto dpl = std::make_shared<cv::cuda::GpuMat>();
    dpl->create(1, (int)out.size(), CV_32SC1);
    dpl->upload(hIds, stream);
    stream.waitForCompletion();

    r.d_centerPoints = dcp;
    r.d_line_ids = dpl;
    r.centerPoints.clear();
    // ⚠ outCount 须在 std::move(out) 之前取值：MSVC 下 moved-from vector 为空，
    // 先 move 再读 out.size() 会把 totalPointCount 置 0（DedupAlign 恒等式桥踩中）
    r.centerPoints.emplace(label, std::move(out));
    r.totalPointCount = st.outCount;
    return true;
}

}  // namespace steger_dedup
