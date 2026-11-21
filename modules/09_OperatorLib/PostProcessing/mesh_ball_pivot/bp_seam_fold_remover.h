#pragma once
// ============================================================================
// bp_seam_fold_remover.h — 折缝（朝向翻转边）剔除器
// （PCMP seam_fold_remover.h 忠实移植；debug 转储块裁剪，算法逐行同源）
//
// 契约（沙盒原注释保留）：
//   * 只删三角形，绝不删顶点（索引对齐保持）
//   * 折缝拓扑判定：cnt==2 边两面同向穿越（d1==d2）＝翻转
//   * 开边（cnt==1）与非流形边（cnt>=3）不是折缝候选
//   * 投票：ref=n(v0)+n(v1)+n(v2)，cross=面叉积，s=归一点积；
//     少数派整带删除；弃权多数/双营全有/平票按积分规则裁决
//   * cap: max_remove>0 ? max_remove : max(16, tris/400)
// ============================================================================
#include <Eigen/Dense>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace calib {
namespace bp {
namespace seamfold {

struct SeamFoldStats {
    int bands = 0;
    int del = 0;
    int abstain = 0;
    int cap_hit = 0;
    int nm_edges = 0;
    int nm_del = 0;
    int refused_quorum = 0;
    int refused_empty = 0;
    int refused_tie = 0;
    int removed_bands = 0;
};

struct SeamFoldParams {
    bool   report_only = true;
    double min_dot     = 0.10;
    int    max_remove  = 0;
    double ref_eps     = 1e-12;
    double cross_eps   = 1e-12;
    bool   debug = false;
};

namespace detail {

struct EdgeInfo {
    int cnt = 0;
    int t1 = -1, t2 = -1;
    int8_t d1 = 0, d2 = 0;
};

inline std::uint64_t ekey(int a, int b) {
    return a < b ? (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint32_t>(b)
                 : (static_cast<std::uint64_t>(b) << 32) | static_cast<std::uint32_t>(a);
}

inline int ufFind(std::vector<int>& parent, int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

}  // namespace detail

inline SeamFoldStats removeSeamFolds(const std::vector<Eigen::Vector3d>& pts,
                                     const std::vector<Eigen::Vector3d>& nrm,
                                     std::vector<Eigen::Vector3i>& tris,
                                     const SeamFoldParams& p = {}) {
    SeamFoldStats st;
    const std::size_t F = tris.size();
    if (F == 0) return st;

    // ---- 1) 边表（自边 a==b 跳过；零面积面保留边）----
    std::unordered_map<std::uint64_t, detail::EdgeInfo> ec;
    ec.reserve(F * 3);
    for (std::size_t fi = 0; fi < F; ++fi) {
        const Eigen::Vector3i& t = tris[fi];
        for (int k = 0; k < 3; ++k) {
            int a = t[k], b = t[(k + 1) % 3];
            if (a == b) continue;
            detail::EdgeInfo& e = ec[detail::ekey(a, b)];
            ++e.cnt;
            if (e.t1 < 0) {
                e.t1 = static_cast<int>(fi);
                e.d1 = (a < b) ? 1 : -1;
            } else if (e.t2 < 0) {
                e.t2 = static_cast<int>(fi);
                e.d2 = (a < b) ? 1 : -1;
            }
        }
    }

    // ---- 2) 折缝边（cnt==2 且同向）+ 并查集带 + 积分度 ----
    std::vector<int> parent(F);
    for (std::size_t i = 0; i < F; ++i) parent[i] = static_cast<int>(i);
    std::vector<char> inBand(F, 0);
    std::vector<int> healthyDeg(F, 0), openDeg(F, 0);
    int flipEdges = 0;
    for (const auto& kv : ec) {
        const detail::EdgeInfo& e = kv.second;
        if (e.cnt == 1) {
            if (e.t1 >= 0) openDeg[static_cast<std::size_t>(e.t1)]++;
            continue;
        }
        if (e.cnt != 2 || e.t2 < 0) continue;
        if (e.d1 != e.d2) {
            healthyDeg[static_cast<std::size_t>(e.t1)]++;
            healthyDeg[static_cast<std::size_t>(e.t2)]++;
            continue;
        }
        ++flipEdges;
        int ra = detail::ufFind(parent, e.t1);
        int rb = detail::ufFind(parent, e.t2);
        if (ra != rb) parent[ra] = rb;
        inBand[static_cast<std::size_t>(e.t1)] = 1;
        inBand[static_cast<std::size_t>(e.t2)] = 1;
    }
    for (const auto& kv : ec)
        if (kv.second.cnt >= 3) ++st.nm_edges;
    if (flipEdges == 0 && st.nm_edges == 0) return st;

    // ---- 3) 分带成员（＋ nm 边面表）----
    std::unordered_map<std::uint64_t, std::vector<int>> nme2f;
    if (st.nm_edges > 0) {
        for (std::size_t fi = 0; fi < F; ++fi) {
            const Eigen::Vector3i& t = tris[fi];
            for (int k = 0; k < 3; ++k) {
                int a = t[k], b = t[(k + 1) % 3];
                if (a == b) continue;
                std::uint64_t k2 = detail::ekey(a, b);
                if (ec[k2].cnt >= 3) nme2f[k2].push_back(static_cast<int>(fi));
            }
        }
    }
    std::unordered_map<int, std::vector<int>> bandsRoot;
    for (std::size_t fi = 0; fi < F; ++fi) {
        if (!inBand[fi]) continue;
        bandsRoot[detail::ufFind(parent, static_cast<int>(fi))]
            .push_back(static_cast<int>(fi));
    }

    const int cap = p.max_remove > 0
                        ? p.max_remove
                        : std::max(16, static_cast<int>(F / 400));

    // ---- 4) 逐带投票 + 裁决 ----
    std::vector<char> dead(F, 0);
    std::vector<int> roots;
    roots.reserve(bandsRoot.size());
    for (const auto& kv : bandsRoot) roots.push_back(kv.first);
    std::sort(roots.begin(), roots.end());
    for (int root : roots) {
        const std::vector<int>& members = bandsRoot[root];
        ++st.bands;
        std::vector<int> pos, neg;
        int abst = 0;
        for (int fi : members) {
            const Eigen::Vector3i& t = tris[static_cast<std::size_t>(fi)];
            Eigen::Vector3d ref = nrm[static_cast<std::size_t>(t[0])] +
                                  nrm[static_cast<std::size_t>(t[1])] +
                                  nrm[static_cast<std::size_t>(t[2])];
            Eigen::Vector3d cr =
                (pts[static_cast<std::size_t>(t[1])] - pts[static_cast<std::size_t>(t[0])])
                    .cross(pts[static_cast<std::size_t>(t[2])] -
                           pts[static_cast<std::size_t>(t[0])]);
            double refN = ref.norm(), crN = cr.norm();
            if (refN < p.ref_eps || crN < p.cross_eps) {
                ++abst;
                continue;
            }
            double s = cr.dot(ref) / (crN * refN);
            if (s >= p.min_dot) pos.push_back(fi);
            else if (s <= -p.min_dot) neg.push_back(fi);
            else ++abst;
        }
        st.abstain += abst;
        const int total = static_cast<int>(members.size());
        if (abst * 2 > total) { ++st.refused_quorum; continue; }
        if (pos.empty() || neg.empty()) {
            // 同营带＝几何重叠簇：保留最高积分面，其余删
            const std::vector<int>& camp = pos.empty() ? neg : pos;
            int maxInteg = INT_MIN;
            for (int fi : camp)
                maxInteg = std::max(maxInteg,
                                    healthyDeg[static_cast<std::size_t>(fi)] -
                                        openDeg[static_cast<std::size_t>(fi)]);
            std::vector<int> losers;
            for (int fi : camp)
                if (healthyDeg[static_cast<std::size_t>(fi)] -
                        openDeg[static_cast<std::size_t>(fi)] < maxInteg)
                    losers.push_back(fi);
            if (losers.empty()) {
                if (camp.size() == 2) {
                    losers.push_back(camp[1]);
                } else {
                    ++st.refused_empty;
                    continue;
                }
            }
            if (static_cast<int>(losers.size()) > cap) {
                ++st.cap_hit;
                continue;
            }
            st.del += static_cast<int>(losers.size());
            ++st.removed_bands;
            if (!p.report_only)
                for (int fi : losers) dead[static_cast<std::size_t>(fi)] = 1;
            continue;
        }
        // 少数派出局；平票按积分决胜，同分拒绝
        const std::vector<int>* minority = nullptr;
        if (pos.size() == neg.size()) {
            long long sp = 0, sn = 0;
            for (int fi : pos)
                sp += static_cast<long long>(healthyDeg[static_cast<std::size_t>(fi)]) -
                      openDeg[static_cast<std::size_t>(fi)];
            for (int fi : neg)
                sn += static_cast<long long>(healthyDeg[static_cast<std::size_t>(fi)]) -
                      openDeg[static_cast<std::size_t>(fi)];
            if (sp == sn) { ++st.refused_tie; continue; }
            minority = (sp < sn) ? &pos : &neg;
        } else {
            minority = (pos.size() < neg.size()) ? &pos : &neg;
        }
        if (static_cast<int>(minority->size()) > cap) {
            ++st.cap_hit;
            continue;
        }
        st.del += static_cast<int>(minority->size());
        ++st.removed_bands;
        if (!p.report_only)
            for (int fi : *minority) dead[static_cast<std::size_t>(fi)] = 1;
    }

    // ---- 4.5) nm 重叠簇（cnt>=3，keep-2 契约＋薄片防护）----
    if (!nme2f.empty()) {
        std::vector<std::uint64_t> nmKeys;
        nmKeys.reserve(nme2f.size());
        for (const auto& kv : nme2f) nmKeys.push_back(kv.first);
        std::sort(nmKeys.begin(), nmKeys.end());
        struct NmCand { int fi; int integ; double area; };
        for (std::uint64_t k2 : nmKeys) {
            std::vector<NmCand> live;
            for (int fi : nme2f[k2])
                if (!dead[static_cast<std::size_t>(fi)]) {
                    const Eigen::Vector3i& t = tris[static_cast<std::size_t>(fi)];
                    Eigen::Vector3d cr =
                        (pts[static_cast<std::size_t>(t[1])] - pts[static_cast<std::size_t>(t[0])])
                            .cross(pts[static_cast<std::size_t>(t[2])] -
                                   pts[static_cast<std::size_t>(t[0])]);
                    live.push_back(NmCand{fi,
                                          healthyDeg[static_cast<std::size_t>(fi)] -
                                              openDeg[static_cast<std::size_t>(fi)],
                                          0.5 * cr.norm()});
                }
            if (live.size() <= 2) continue;
            const int losers = static_cast<int>(live.size()) - 2;
            if (losers > cap) { ++st.cap_hit; continue; }
            std::sort(live.begin(), live.end(), [](const NmCand& x, const NmCand& y) {
                if (x.integ != y.integ) return x.integ > y.integ;
                if (x.area != y.area) return x.area > y.area;
                return x.fi < y.fi;
            });
            int a0 = static_cast<int>(k2 >> 32), b0 = static_cast<int>(k2 & 0xffffffffu);
            for (std::size_t i = 2; i < live.size(); ++i) {
                const Eigen::Vector3i& t = tris[static_cast<std::size_t>(live[i].fi)];
                bool ok = true;
                for (int k = 0; k < 3; ++k) {
                    int a = t[k], b = t[(k + 1) % 3];
                    if (a == b) continue;
                    if (a == a0 && b == b0) continue;
                    if (a == b0 && b == a0) continue;
                    const detail::EdgeInfo& e2 = ec[detail::ekey(a, b)];
                    if (e2.cnt == 2) { ok = false; break; }
                }
                if (!ok) continue;
                ++st.nm_del;
                if (p.report_only) continue;
                dead[static_cast<std::size_t>(live[i].fi)] = 1;
                for (int k = 0; k < 3; ++k) {
                    int a = t[k], b = t[(k + 1) % 3];
                    if (a == b) continue;
                    detail::EdgeInfo& e3 = ec[detail::ekey(a, b)];
                    if (e3.cnt > 0) --e3.cnt;
                }
            }
        }
    }

    // ---- 5) 压实（保序；顶点不动）----
    const int totalDel = st.del + st.nm_del;
    if (!p.report_only && totalDel > 0) {
        std::vector<Eigen::Vector3i> kept;
        kept.reserve(F - static_cast<std::size_t>(totalDel));
        for (std::size_t fi = 0; fi < F; ++fi)
            if (!dead[fi]) kept.push_back(tris[fi]);
        tris.swap(kept);
    }
    return st;
}

} // namespace seamfold
} // namespace bp
} // namespace calib
