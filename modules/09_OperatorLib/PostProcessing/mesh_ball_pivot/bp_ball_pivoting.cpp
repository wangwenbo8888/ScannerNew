// ============================================================================
// bp_ball_pivoting.cpp — CPU 滚球内核实现（PCMP ball_pivoting.cpp 忠实移植）
// 移植口径：算法逐行同源（金标准复现）；剥离 KDTree 路径（恒 UniformGrid）、
// 诊断计时字段与 stderr 日志、env 开关（并行插入路径固定启用）。
// ============================================================================
#include "bp_ball_pivoting.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace calib {
namespace bp {

// ============================================================================
//                          FastBallPivoting 实现
// ============================================================================

FastBallPivoting::FastBallPivoting(const std::vector<Eigen::Vector3d>& points,
                                   const std::vector<Eigen::Vector3d>& normals)
    : points_(points), normals_(normals) {
    vertex_type_.assign(points_.size(), kOrphan);
    vertex_edges_.resize(points_.size());

    edges_.reserve(points_.size() * 6);
    triangles_.reserve(points_.size() * 2);
    out_triangles_.reserve(points_.size() * 2);
    out_normals_.reserve(points_.size() * 2);
    edge_map_.reserve(points_.size() * 6);
    for (auto& s : edge_map_shards_) s.reserve(points_.size() * 6 / EDGE_MAP_SHARDS + 1);
}

void FastBallPivoting::setSeedMask(const std::vector<char>& mask) {
    seed_mask_ = mask;
}

uint64_t FastBallPivoting::makeEdgeKey(int a, int b) {
    if (a > b) std::swap(a, b);
    return (static_cast<uint64_t>(a) << 32) | static_cast<uint32_t>(b);
}

int FastBallPivoting::getLinkingEdge(int v0, int v1) const {
    uint64_t key = makeEdgeKey(v0, v1);
    if (use_sharded_edge_map_) {
        int s = static_cast<int>(key & (EDGE_MAP_SHARDS - 1));
        auto it = edge_map_shards_[s].find(key);
        return (it != edge_map_shards_[s].end()) ? it->second : -1;
    }
    auto it = edge_map_.find(key);
    return (it != edge_map_.end()) ? it->second : -1;
}

int FastBallPivoting::getOrCreateEdge(int v0, int v1, bool& is_new) {
    uint64_t key = makeEdgeKey(v0, v1);
    if (use_sharded_edge_map_) {
        int s = static_cast<int>(key & (EDGE_MAP_SHARDS - 1));
        auto it = edge_map_shards_[s].find(key);
        if (it != edge_map_shards_[s].end()) {
            is_new = false;
            return it->second;
        }
        int idx = static_cast<int>(edges_.size());
        edges_.push_back(FastEdge{v0, v1, -1, -1, kFront});
        edge_map_shards_[s][key] = idx;
        is_new = true;
        return idx;
    }
    auto it = edge_map_.find(key);
    if (it != edge_map_.end()) {
        is_new = false;
        return it->second;
    }
    int idx = static_cast<int>(edges_.size());
    edges_.push_back(FastEdge{v0, v1, -1, -1, kFront});
    edge_map_[key] = idx;
    is_new = true;
    return idx;
}

void FastBallPivoting::addEdgeToVertex(int vidx, int eidx) {
    auto& ve = vertex_edges_[vidx];
    for (int e : ve)
        if (e == eidx) return;
    ve.push_back(eidx);
}

void FastBallPivoting::updateVertexType(int vidx) {
    const auto& ve = vertex_edges_[vidx];
    if (ve.empty()) {
        vertex_type_[vidx] = kOrphan;
        return;
    }
    for (int eidx : ve) {
        if (edges_[eidx].type != kInner) {
            vertex_type_[vidx] = kVFront;
            return;
        }
    }
    vertex_type_[vidx] = kVInner;
}

int FastBallPivoting::getOppositeVertex(int edge_idx) const {
    const FastEdge& edge = edges_[edge_idx];
    if (edge.triangle0 < 0) return -1;
    const FastTriangle& tri = triangles_[edge.triangle0];
    if (tri.vert0 != edge.source && tri.vert0 != edge.target) return tri.vert0;
    if (tri.vert1 != edge.source && tri.vert1 != edge.target) return tri.vert1;
    return tri.vert2;
}

void FastBallPivoting::addAdjacentTriangleToEdge(int edge_idx, int tri_idx) {
    FastEdge& edge = edges_[edge_idx];
    if (tri_idx != edge.triangle0 && tri_idx != edge.triangle1) {
        if (edge.triangle0 < 0) {
            edge.triangle0 = tri_idx;
            edge.type = kFront;
            int opp = getOppositeVertex(edge_idx);
            if (opp >= 0) {
                Eigen::Vector3d tr_norm =
                    (points_[edge.target] - points_[edge.source])
                        .cross(points_[opp] - points_[edge.source]);
                tr_norm /= tr_norm.norm();
                Eigen::Vector3d pt_norm =
                    normals_[edge.source] + normals_[edge.target] + normals_[opp];
                pt_norm /= pt_norm.norm();
                if (pt_norm.dot(tr_norm) < 0) {
                    std::swap(edge.target, edge.source);
                }
            }
        } else if (edge.triangle1 < 0) {
            edge.triangle1 = tri_idx;
            edge.type = kInner;
        }
    }
}

bool FastBallPivoting::computeBallCenter(int vidx1, int vidx2, int vidx3,
                                         double radius,
                                         Eigen::Vector3d& center) const {
    const Eigen::Vector3d& v1 = points_[vidx1];
    const Eigen::Vector3d& v2 = points_[vidx2];
    const Eigen::Vector3d& v3 = points_[vidx3];
    double c2 = (v2 - v1).squaredNorm();
    double b2 = (v1 - v3).squaredNorm();
    double a2 = (v3 - v2).squaredNorm();

    double alpha = a2 * (b2 + c2 - a2);
    double beta = b2 * (a2 + c2 - b2);
    double gamma = c2 * (a2 + b2 - c2);
    double abg = alpha + beta + gamma;

    if (abg < 1e-16) return false;

    double inv_abg = 1.0 / abg;
    alpha *= inv_abg;
    beta *= inv_abg;
    gamma *= inv_abg;

    Eigen::Vector3d circ_center = alpha * v1 + beta * v2 + gamma * v3;

    Eigen::Vector3d cr = (v2 - v1).cross(v3 - v1);
    double cross2 = cr.squaredNorm();
    if (cross2 < 1e-20) return false;
    double circ_radius2 = (a2 * b2 * c2) / (4.0 * cross2);

    double height2 = radius * radius - circ_radius2;
    if (height2 >= 0.0) {
        double inv_cross = 1.0 / std::sqrt(cross2);
        Eigen::Vector3d tr_norm = cr * inv_cross;
        Eigen::Vector3d pt_norm = normals_[vidx1] + normals_[vidx2] + normals_[vidx3];
        double pn2 = pt_norm.squaredNorm();
        if (pn2 < 1e-20) return false;
        pt_norm *= 1.0 / std::sqrt(pn2);
        if (tr_norm.dot(pt_norm) < 0) tr_norm *= -1;

        center = circ_center + std::sqrt(height2) * tr_norm;
        return true;
    }
    return false;
}

Eigen::Vector3d FastBallPivoting::computeFaceNormal(const Eigen::Vector3d& v0,
                                                    const Eigen::Vector3d& v1,
                                                    const Eigen::Vector3d& v2) const {
    Eigen::Vector3d normal = (v1 - v0).cross(v2 - v0);
    double norm = normal.norm();
    if (norm > 0) normal /= norm;
    return normal;
}

bool FastBallPivoting::isCompatible(int v0, int v1, int v2) const {
    Eigen::Vector3d normal = computeFaceNormal(points_[v0], points_[v1], points_[v2]);
    if (normal.dot(normals_[v0]) < -1e-16) normal *= -1;
    return normal.dot(normals_[v0]) > -1e-16 &&
           normal.dot(normals_[v1]) > -1e-16 &&
           normal.dot(normals_[v2]) > -1e-16;
}

bool FastBallPivoting::pointsCoplanar(const Eigen::Vector3d& p0,
                                      const Eigen::Vector3d& p1,
                                      const Eigen::Vector3d& p2,
                                      const Eigen::Vector3d& p3) {
    return (p1 - p0).dot((p2 - p0).cross(p3 - p0)) < 1e-12;
}

double FastBallPivoting::lineSegmentsMinimumDistance(const Eigen::Vector3d& p0,
                                                     const Eigen::Vector3d& p1,
                                                     const Eigen::Vector3d& q0,
                                                     const Eigen::Vector3d& q1) {
    const Eigen::Vector3d p10 = p1 - p0;
    const Eigen::Vector3d q10 = q1 - q0;
    const Eigen::Vector3d p0q0 = p0 - q0;
    double a = p10.dot(p10);
    double b = p10.dot(q10);
    double c = q10.dot(q10);
    double d = p10.dot(p0q0);
    double e = q10.dot(p0q0);
    double det = a * c - b * b;
    double s, t;
    if (det > 0) {
        double bte = b * e;
        double ctd = c * d;
        if (bte <= ctd) {
            if (e <= 0) {
                s = (-d >= a ? 1 : (-d > 0 ? -d / a : 0));
                t = 0;
            } else if (e < c) {
                s = 0;
                t = e / c;
            } else {
                s = (b - d >= a ? 1 : (b - d > 0 ? (b - d) / a : 0));
                t = 1;
            }
        } else {
            s = bte - ctd;
            if (s >= det) {
                if (b + e <= 0) {
                    s = (-d <= 0 ? 0 : (-d < a ? -d / a : 1));
                    t = 0;
                } else if (b + e < c) {
                    s = 1;
                    t = (b + e) / c;
                } else {
                    s = (b - d <= 0 ? 0 : (b - d < a ? (b - d) / a : 1));
                    t = 1;
                }
            } else {
                double ate = a * e;
                double btd = b * d;
                if (ate <= btd) {
                    s = (-d <= 0 ? 0 : (-d >= a ? 1 : -d / a));
                    t = 0;
                } else {
                    t = ate - btd;
                    if (t >= det) {
                        s = (b - d <= 0 ? 0 : (b - d >= a ? 1 : (b - d) / a));
                        t = 1;
                    } else {
                        s /= det;
                        t /= det;
                    }
                }
            }
        }
    } else {
        if (e <= 0) {
            s = (-d <= 0 ? 0 : (-d >= a ? 1 : -d / a));
            t = 0;
        } else if (e >= c) {
            s = (b - d <= 0 ? 0 : (b - d >= a ? 1 : (b - d) / a));
            t = 1;
        } else {
            s = 0;
            t = e / c;
        }
    }
    Eigen::Vector3d p = (1 - s) * p0 + s * p1;
    Eigen::Vector3d q = (1 - t) * q0 + t * q1;
    return (p - q).norm();
}

void FastBallPivoting::createTriangle(int v0, int v1, int v2,
                                      const Eigen::Vector3d& center) {
    int tri_idx = static_cast<int>(triangles_.size());
    triangles_.push_back(FastTriangle{v0, v1, v2, center});

    bool is_new = false;
    int e0 = getOrCreateEdge(v0, v1, is_new);
    addAdjacentTriangleToEdge(e0, tri_idx);
    if (is_new) { addEdgeToVertex(v0, e0); addEdgeToVertex(v1, e0); }

    int e1 = getOrCreateEdge(v1, v2, is_new);
    addAdjacentTriangleToEdge(e1, tri_idx);
    if (is_new) { addEdgeToVertex(v1, e1); addEdgeToVertex(v2, e1); }

    int e2 = getOrCreateEdge(v2, v0, is_new);
    addAdjacentTriangleToEdge(e2, tri_idx);
    if (is_new) { addEdgeToVertex(v2, e2); addEdgeToVertex(v0, e2); }

    updateVertexType(v0);
    updateVertexType(v1);
    updateVertexType(v2);

    Eigen::Vector3d face_normal =
        computeFaceNormal(points_[v0], points_[v1], points_[v2]);
    if (skipFlip_ || face_normal.dot(normals_[v0]) > -1e-16) {
        out_triangles_.emplace_back(v0, v1, v2);
    } else {
        out_triangles_.emplace_back(v0, v2, v1);
    }
    out_normals_.push_back(face_normal);
}

int FastBallPivoting::findCandidateVertex(int edge_idx, double radius,
                                          Eigen::Vector3d& candidate_center) {
    const FastEdge& edge = edges_[edge_idx];
    int src = edge.source;
    int tgt = edge.target;
    int opp = getOppositeVertex(edge_idx);
    if (opp < 0) return -1;

    Eigen::Vector3d mp = 0.5 * (points_[src] + points_[tgt]);
    const Eigen::Vector3d& center = triangles_[edge.triangle0].ball_center;

    Eigen::Vector3d v = points_[tgt] - points_[src];
    v /= v.norm();
    Eigen::Vector3d a = center - mp;
    a /= a.norm();

    std::vector<int> indices;
    std::vector<double> dists2;
    searchRadius(mp, 2 * radius, indices, dists2);

    double edgeLenHalf2 = (points_[src] - points_[tgt]).squaredNorm() * 0.25;
    double h2 = radius * radius - edgeLenHalf2;
    double preEffR2 = h2 > 0 ? radius * radius + 2.0 * radius * std::sqrt(h2) + h2 : radius * radius;

    int min_candidate = -1;
    double min_angle = 2 * M_PI;

    for (int nbidx : indices) {
        if (nbidx == src || nbidx == tgt || nbidx == opp) continue;
        if (vertex_type_[nbidx] == kVInner) continue;
        if (filterCandidates_ && !seed_mask_.empty() &&
            nbidx < (int)seed_mask_.size() && !seed_mask_[nbidx]) continue;

        bool coplanar = pointsCoplanar(points_[src], points_[tgt],
                                       points_[opp], points_[nbidx]);
        if (coplanar &&
            (lineSegmentsMinimumDistance(mp, points_[nbidx],
                                         points_[src], points_[opp]) < 1e-12 ||
             lineSegmentsMinimumDistance(mp, points_[nbidx],
                                         points_[tgt], points_[opp]) < 1e-12)) {
            continue;
        }

        Eigen::Vector3d new_center;
        if ((points_[nbidx] - mp).squaredNorm() > preEffR2) continue;
        if (!computeBallCenter(src, tgt, nbidx, radius, new_center)) continue;

        Eigen::Vector3d b = new_center - mp;
        b /= b.norm();

        double cosinus = std::min(1.0, std::max(-1.0, a.dot(b)));
        double angle = std::acos(cosinus);
        Eigen::Vector3d cvec = a.cross(b);
        if (cvec.dot(v) < 0) angle = 2 * M_PI - angle;

        if (angle >= min_angle) continue;

        bool empty_ball = true;
        double r_eps = radius - 1e-16;
        double r_eps2 = r_eps * r_eps;
        for (int nbidx2 : indices) {
            if (nbidx2 == src || nbidx2 == tgt || nbidx2 == nbidx) continue;
            if ((new_center - points_[nbidx2]).squaredNorm() < r_eps2) {
                empty_ball = false;
                break;
            }
        }
        if (empty_ball) {
            min_angle = angle;
            min_candidate = nbidx;
            candidate_center = new_center;
        }
    }
    return min_candidate;
}

void FastBallPivoting::expandTriangulation(double radius) {
    while (!edge_front_.empty()) {
        int edge_idx = edge_front_.front();
        edge_front_.pop_front();
        if (edges_[edge_idx].type != kFront) continue;

        Eigen::Vector3d center;
        int candidate = findCandidateVertex(edge_idx, radius, center);

        int edge_src = edges_[edge_idx].source;
        int edge_tgt = edges_[edge_idx].target;

        if (candidate < 0 ||
            vertex_type_[candidate] == kVInner ||
            !isCompatible(candidate, edge_src, edge_tgt)) {
            edges_[edge_idx].type = kBorder;
            border_edges_.push_back(edge_idx);
            continue;
        }

        int e0 = getLinkingEdge(candidate, edge_src);
        int e1 = getLinkingEdge(candidate, edge_tgt);
        if ((e0 >= 0 && edges_[e0].type != kFront) ||
            (e1 >= 0 && edges_[e1].type != kFront)) {
            edges_[edge_idx].type = kBorder;
            border_edges_.push_back(edge_idx);
            continue;
        }

        createTriangle(edge_src, edge_tgt, candidate, center);

        e0 = getLinkingEdge(candidate, edge_src);
        e1 = getLinkingEdge(candidate, edge_tgt);
        if (e0 >= 0 && edges_[e0].type == kFront) edge_front_.push_front(e0);
        if (e1 >= 0 && edges_[e1].type == kFront) edge_front_.push_front(e1);
    }
}

void FastBallPivoting::expandTriangulationParallel(double radius) {
    constexpr int MAX_BATCH_ROUNDS = 100;

    for (int batchRound = 0; batchRound < MAX_BATCH_ROUNDS; ++batchRound) {
        if (edge_front_.empty()) break;

        std::vector<int> batch(edge_front_.begin(), edge_front_.end());
        edge_front_.clear();
        int N = static_cast<int>(batch.size());

        struct ExpandResult {
            int edgeIdx;
            int candidate;
            Eigen::Vector3d center;
        };
        std::vector<ExpandResult> results(N);

#pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            int edge_idx = batch[i];
            results[i].edgeIdx = edge_idx;
            results[i].candidate = -1;

            if (edges_[edge_idx].type != kFront) continue;

            Eigen::Vector3d c;
            int cand = findCandidateVertex(edge_idx, radius, c);
            if (cand >= 0) {
                results[i].candidate = cand;
                results[i].center = c;
            }
        }

        for (int i = 0; i < N; ++i) {
            int edge_idx = results[i].edgeIdx;
            if (edges_[edge_idx].type != kFront) continue;

            int edge_src = edges_[edge_idx].source;
            int edge_tgt = edges_[edge_idx].target;
            int candidate = results[i].candidate;

            if (candidate < 0 ||
                vertex_type_[candidate] == kVInner ||
                !isCompatible(candidate, edge_src, edge_tgt)) {
                edges_[edge_idx].type = kBorder;
                border_edges_.push_back(edge_idx);
                continue;
            }

            int e0 = getLinkingEdge(candidate, edge_src);
            int e1 = getLinkingEdge(candidate, edge_tgt);
            if ((e0 >= 0 && edges_[e0].type != kFront) ||
                (e1 >= 0 && edges_[e1].type != kFront)) {
                edges_[edge_idx].type = kBorder;
                border_edges_.push_back(edge_idx);
                continue;
            }

            createTriangle(edge_src, edge_tgt, candidate, results[i].center);

            e0 = getLinkingEdge(candidate, edge_src);
            e1 = getLinkingEdge(candidate, edge_tgt);
            if (e0 >= 0 && edges_[e0].type == kFront) edge_front_.push_front(e0);
            if (e1 >= 0 && edges_[e1].type == kFront) edge_front_.push_front(e1);
        }
    }
}

bool FastBallPivoting::tryTriangleSeed(int v0, int v1, int v2,
                                       const std::vector<int>& nb_indices,
                                       double radius,
                                       Eigen::Vector3d& center) {
    if (!isCompatible(v0, v1, v2)) return false;

    double r_eps = radius - 1e-16;
    double r_eps2 = r_eps * r_eps;
    for (int nbidx : nb_indices) {
        if (nbidx == v0 || nbidx == v1 || nbidx == v2) continue;
        if ((center - points_[nbidx]).squaredNorm() < r_eps2) return false;
    }
    return true;
}

bool FastBallPivoting::trySeed(int v, double radius) {
    std::vector<int> indices;
    std::vector<double> dists2;
    searchRadius(points_[v], 2 * radius, indices, dists2);
    if (indices.size() < 3) return false;

    for (size_t nbidx0 = 0; nbidx0 < indices.size(); ++nbidx0) {
        int nb0 = indices[nbidx0];
        if (vertex_type_[nb0] != kOrphan || nb0 == v) continue;
        if (filterCandidates_ && !seed_mask_.empty() &&
            nb0 < (int)seed_mask_.size() && !seed_mask_[nb0]) continue;

        int candidate_vidx2 = -1;
        Eigen::Vector3d center;
        for (size_t nbidx1 = nbidx0 + 1; nbidx1 < indices.size(); ++nbidx1) {
            int nb1 = indices[nbidx1];
            if (vertex_type_[nb1] != kOrphan || nb1 == v) continue;
            if (filterCandidates_ && !seed_mask_.empty() &&
                nb1 < (int)seed_mask_.size() && !seed_mask_[nb1]) continue;
            if (tryTriangleSeed(v, nb0, nb1, indices, radius, center)) {
                candidate_vidx2 = nb1;
                break;
            }
        }

        if (candidate_vidx2 >= 0) {
            int e0 = getLinkingEdge(v, candidate_vidx2);
            if (e0 >= 0 && edges_[e0].type != kFront) continue;
            int e1 = getLinkingEdge(nb0, candidate_vidx2);
            if (e1 >= 0 && edges_[e1].type != kFront) continue;
            int e2 = getLinkingEdge(v, nb0);
            if (e2 >= 0 && edges_[e2].type != kFront) continue;

            createTriangle(v, nb0, candidate_vidx2, center);

            e0 = getLinkingEdge(v, candidate_vidx2);
            e1 = getLinkingEdge(nb0, candidate_vidx2);
            e2 = getLinkingEdge(v, nb0);
            if (e0 >= 0 && edges_[e0].type == kFront) edge_front_.push_front(e0);
            if (e1 >= 0 && edges_[e1].type == kFront) edge_front_.push_front(e1);
            if (e2 >= 0 && edges_[e2].type == kFront) edge_front_.push_front(e2);

            if (!edge_front_.empty()) return true;
        }
    }
    return false;
}

void FastBallPivoting::findSeedTriangle(double radius) {
    bool useMask = !seed_mask_.empty();
    for (size_t vidx = 0; vidx < points_.size(); ++vidx) {
        if (useMask && vidx < seed_mask_.size() && !seed_mask_[vidx]) continue;
        if (vertex_type_[vidx] == kOrphan) {
            if (trySeed(static_cast<int>(vidx), radius)) {
                expandTriangulation(radius);
            }
        }
    }
}

void FastBallPivoting::Run(const std::vector<double>& radii) {
    out_triangles_.clear();
    out_normals_.clear();

    if (useGrid_ && !radii.empty()) {
        double maxR = 0.0;
        for (double r : radii) maxR = std::max(maxR, r);
        grid_.build(points_, maxR);
    }

    for (double radius : radii) {
        if (radius <= 0) continue;

        for (auto it = border_edges_.begin(); it != border_edges_.end();) {
            int edge_idx = *it;
            const FastEdge& edge = edges_[edge_idx];
            const FastTriangle& tri = triangles_[edge.triangle0];

            Eigen::Vector3d center;
            if (computeBallCenter(tri.vert0, tri.vert1, tri.vert2, radius, center)) {
                std::vector<int> indices;
                std::vector<double> dists2;
                searchRadius(center, radius, indices, dists2);
                bool empty_ball = true;
                for (int idx : indices) {
                    if (idx != tri.vert0 && idx != tri.vert1 && idx != tri.vert2) {
                        empty_ball = false;
                        break;
                    }
                }
                if (empty_ball) {
                    edges_[edge_idx].type = kFront;
                    edge_front_.push_back(edge_idx);
                    it = border_edges_.erase(it);
                    continue;
                }
            }
            ++it;
        }

        if (edge_front_.empty()) {
            findSeedTriangle(radius);
        } else {
            expandTriangulation(radius);
        }
    }
}

void FastBallPivoting::buildGrid(double maxRadius) {
    useGrid_ = true;
    grid_.build(points_, maxRadius);
}

void FastBallPivoting::loadTriangles(const std::vector<Eigen::Vector3i>& tris, double radius) {
    std::fill(vertex_type_.begin(), vertex_type_.end(), kOrphan);
    for (auto& ve : vertex_edges_) ve.clear();
    edges_.clear();
    triangles_.clear();
    edge_map_.clear();
    edge_front_.clear();
    border_edges_.clear();
    out_triangles_.clear();
    out_normals_.clear();

    edges_.reserve(points_.size() * 6);
    triangles_.reserve(tris.size() + points_.size());
    out_triangles_.reserve(tris.size() + points_.size());
    out_normals_.reserve(tris.size() + points_.size());

    for (const auto& tri : tris) {
        int v0 = tri(0), v1 = tri(1), v2 = tri(2);

        Eigen::Vector3d center;
        if (!computeBallCenter(v0, v1, v2, radius, center)) {
            center = (points_[v0] + points_[v1] + points_[v2]) / 3.0;
        }

        int tri_idx = static_cast<int>(triangles_.size());
        triangles_.push_back(FastTriangle{v0, v1, v2, center});

        auto addEdge = [&](int a, int b) {
            uint64_t key = makeEdgeKey(a, b);
            auto it = edge_map_.find(key);
            int eidx;
            if (it != edge_map_.end()) {
                eidx = it->second;
            } else {
                eidx = static_cast<int>(edges_.size());
                edges_.push_back(FastEdge{a, b, -1, -1, kFront});
                edge_map_[key] = eidx;
                addEdgeToVertex(a, eidx);
                addEdgeToVertex(b, eidx);
            }
            FastEdge& edge = edges_[eidx];
            if (edge.triangle0 < 0) {
                edge.triangle0 = tri_idx;
                edge.type = kFront;
            } else if (edge.triangle1 < 0) {
                edge.triangle1 = tri_idx;
                edge.type = kInner;
            }
        };
        addEdge(v0, v1);
        addEdge(v1, v2);
        addEdge(v2, v0);

        updateVertexType(v0);
        updateVertexType(v1);
        updateVertexType(v2);

        out_triangles_.push_back(tri);
        out_normals_.push_back(computeFaceNormal(points_[v0], points_[v1], points_[v2]));
    }
}

void FastBallPivoting::loadTrianglesParallel(const std::vector<Eigen::Vector3i>& tris,
                                              double radius,
                                              const std::vector<Eigen::Vector3d>& centers) {
    (void)radius;
    size_t N = tris.size();
    if (N == 0) return;

    #pragma omp parallel for
    for (int64_t i = 0; i < (int64_t)vertex_type_.size(); ++i) vertex_type_[i] = kOrphan;
    for (auto& ve : vertex_edges_) ve.clear();
    edges_.clear();
    triangles_.clear();
    edge_map_.clear();
    for (auto& s : edge_map_shards_) s.clear();
    edge_front_.clear();
    border_edges_.clear();
    out_triangles_.clear();
    out_normals_.clear();

    triangles_.resize(N);
    out_triangles_.resize(N);
    out_normals_.resize(N);

    struct EdgeRec { uint64_t key; int triIdx; };
    std::vector<EdgeRec> edgeRecs(N * 3);
    #pragma omp parallel for
    for (int64_t i = 0; i < (int64_t)N; ++i) {
        int v0 = tris[i](0), v1 = tris[i](1), v2 = tris[i](2);
        triangles_[i] = FastTriangle{v0, v1, v2, centers[i]};
        out_triangles_[i] = tris[i];
        out_normals_[i] = Eigen::Vector3d(0, 0, 1);
        edgeRecs[i*3+0] = {makeEdgeKey(v0, v1), (int)i};
        edgeRecs[i*3+1] = {makeEdgeKey(v1, v2), (int)i};
        edgeRecs[i*3+2] = {makeEdgeKey(v2, v0), (int)i};
    }

    std::sort(edgeRecs.begin(), edgeRecs.end(),
              [](const EdgeRec& a, const EdgeRec& b) { return a.key < b.key; });

    int totalRecs = (int)edgeRecs.size();
    std::vector<int> isStart(totalRecs, 0);
    #pragma omp parallel for
    for (int i = 0; i < totalRecs; ++i) {
        if (i == 0 || edgeRecs[i].key != edgeRecs[i - 1].key) isStart[i] = 1;
    }
    std::vector<int> edgePrefix(totalRecs, 0);
    std::exclusive_scan(isStart.begin(), isStart.end(), edgePrefix.begin(), 0);
    int numUniqueEdges = edgePrefix.back() + isStart.back();

    edges_.resize(numUniqueEdges);
    #pragma omp parallel for
    for (int i = 0; i < totalRecs; ++i) {
        if (!isStart[i]) continue;
        int eidx = edgePrefix[i];
        uint64_t curKey = edgeRecs[i].key;
        int tri0 = edgeRecs[i].triIdx;
        const auto& t0 = triangles_[tri0];
        int src = -1, tgt = -1;
        auto tryEdge = [&](int a, int b) {
            if (makeEdgeKey(a, b) == curKey) { src = a; tgt = b; return true; }
            return false;
        };
        tryEdge(t0.vert0, t0.vert1) || tryEdge(t0.vert1, t0.vert2) || tryEdge(t0.vert2, t0.vert0);
        FastEdge fe;
        fe.source = src; fe.target = tgt;
        fe.triangle0 = tri0; fe.triangle1 = -1; fe.type = kFront;
        if (i + 1 < totalRecs && edgeRecs[i + 1].key == curKey) {
            fe.triangle1 = edgeRecs[i + 1].triIdx;
            fe.type = kInner;
        }
        edges_[eidx] = fe;
    }

    // 分片插入（shard-major 并行：每 shard 单写者；与沙盒 gates-passed 路径同构）
    for (auto& s : edge_map_shards_) { s.clear(); s.reserve(numUniqueEdges / EDGE_MAP_SHARDS * 2); }
    {
        std::vector<int> shardOf(numUniqueEdges);
        int cnt[EDGE_MAP_SHARDS + 1] = {0};
        for (int i = 0; i < numUniqueEdges; ++i) {
            const auto& e = edges_[i];
            int s = static_cast<int>(makeEdgeKey(e.source, e.target) & (EDGE_MAP_SHARDS - 1));
            shardOf[i] = s;
            ++cnt[s + 1];
        }
        for (int s = 0; s < EDGE_MAP_SHARDS; ++s) cnt[s + 1] += cnt[s];
        std::vector<int> grouped(numUniqueEdges);
        int cur[EDGE_MAP_SHARDS];
        for (int s = 0; s < EDGE_MAP_SHARDS; ++s) cur[s] = cnt[s];
        for (int i = 0; i < numUniqueEdges; ++i) grouped[cur[shardOf[i]]++] = i;
        auto& shards = edge_map_shards_;
        auto& edges = edges_;
        #pragma omp parallel for schedule(dynamic,1) default(none) shared(shards, edges, grouped, cnt)
        for (int s = 0; s < EDGE_MAP_SHARDS; ++s) {
            for (int p = cnt[s]; p < cnt[s + 1]; ++p) {
                int i = grouped[p];
                shards[s][makeEdgeKey(edges[i].source, edges[i].target)] = i;
            }
        }
    }
    use_sharded_edge_map_ = true;

    // 并行建 vertex_edges_ + updateVertexType
    std::vector<int> vEdgeCount(points_.size() + 1, 0);
    for (const auto& e : edges_) {
        vEdgeCount[e.source + 1]++;
        vEdgeCount[e.target + 1]++;
    }
    for (size_t i = 1; i < vEdgeCount.size(); ++i) vEdgeCount[i] += vEdgeCount[i - 1];
    for (size_t v = 0; v < vertex_edges_.size(); ++v) {
        vertex_edges_[v].clear();
        vertex_edges_[v].resize(vEdgeCount[v + 1] - vEdgeCount[v]);
    }
    std::vector<int> vFillPos(vertex_edges_.size(), 0);
    for (int eidx = 0; eidx < (int)edges_.size(); ++eidx) {
        const auto& e = edges_[eidx];
        if (e.source >= 0 && e.source < (int)vertex_edges_.size()) {
            int& pos = vFillPos[e.source];
            if (pos < (int)vertex_edges_[e.source].size()) vertex_edges_[e.source][pos++] = eidx;
        }
        if (e.target >= 0 && e.target < (int)vertex_edges_.size()) {
            int& pos = vFillPos[e.target];
            if (pos < (int)vertex_edges_[e.target].size()) vertex_edges_[e.target][pos++] = eidx;
        }
    }
    #pragma omp parallel for
    for (int64_t v = 0; v < (int64_t)points_.size(); ++v) {
        const auto& ve = vertex_edges_[v];
        if (ve.empty()) { vertex_type_[v] = kOrphan; continue; }
        bool allInner = true;
        for (int eidx : ve) { if (edges_[eidx].type != kInner) { allInner = false; break; } }
        vertex_type_[v] = allInner ? kVInner : kVFront;
    }
}

void FastBallPivoting::expandBorders(double radius) {
    for (size_t i = 0; i < edges_.size(); ++i)
        if (edges_[i].type == kFront)
            edge_front_.push_back(static_cast<int>(i));
    expandTriangulation(radius);
}

void FastBallPivoting::expandBordersMultiRadius(const std::vector<double>& radii) {
    for (size_t i = 0; i < edges_.size(); ++i)
        if (edges_[i].type == kFront)
            edge_front_.push_back(static_cast<int>(i));

    for (double radius : radii) {
        if (radius <= 0) continue;

        if (!border_edges_.empty()) {
            std::vector<int> borderVec(border_edges_.begin(), border_edges_.end());
            std::vector<char> reactivated(borderVec.size(), 0);

#pragma omp parallel for
            for (int bi = 0; bi < static_cast<int>(borderVec.size()); ++bi) {
                int edge_idx = borderVec[bi];
                const FastEdge& edge = edges_[edge_idx];
                const FastTriangle& tri = triangles_[edge.triangle0];

                Eigen::Vector3d center;
                if (computeBallCenter(tri.vert0, tri.vert1, tri.vert2, radius, center)) {
                    std::vector<int> indices;
                    std::vector<double> dists2;
                    searchRadius(center, radius, indices, dists2);
                    bool empty_ball = true;
                    for (int idx : indices) {
                        if (idx != tri.vert0 && idx != tri.vert1 && idx != tri.vert2) {
                            empty_ball = false;
                            break;
                        }
                    }
                    if (empty_ball) reactivated[bi] = 1;
                }
            }

            auto it = border_edges_.begin();
            size_t ri = 0;
            while (it != border_edges_.end() && ri < borderVec.size()) {
                int edge_idx = *it;
                if (reactivated[ri]) {
                    edges_[edge_idx].type = kFront;
                    edge_front_.push_back(edge_idx);
                    it = border_edges_.erase(it);
                } else {
                    ++it;
                }
                ++ri;
            }
        }

        if (!edge_front_.empty()) {
            expandTriangulationParallel(radius);
        }
    }
}

// ============================================================================
//                       ParallelBallPivoting 实现
// ============================================================================

ParallelBallPivoting::ParallelBallPivoting(
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<Eigen::Vector3d>& normals)
    : points_(points), normals_(normals) {}

uint64_t ParallelBallPivoting::triangleKey(int v0, int v1, int v2) {
    if (v0 > v1) std::swap(v0, v1);
    if (v1 > v2) std::swap(v1, v2);
    if (v0 > v1) std::swap(v0, v1);
    return (static_cast<uint64_t>(v0) << 42) |
           (static_cast<uint64_t>(v1) << 21) |
           static_cast<uint64_t>(v2);
}

ParallelBallPivoting::BlockResult ParallelBallPivoting::processBlock(
    const std::vector<int>& blockIndices,
    const std::vector<char>& seedMask,
    const std::vector<double>& radii) const {

    BlockResult result;
    size_t M = blockIndices.size();
    if (M < 3) return result;

    std::vector<Eigen::Vector3d> subPoints(M);
    std::vector<Eigen::Vector3d> subNormals(M);
    for (size_t i = 0; i < M; ++i) {
        subPoints[i] = points_[blockIndices[i]];
        subNormals[i] = normals_[blockIndices[i]];
    }

    FastBallPivoting fbp(subPoints, subNormals);
    fbp.setSeedMask(seedMask);
    fbp.setUseGrid(true);
    fbp.setCandidateFilter(true);
    fbp.Run(radii);

    const auto& localTris = fbp.triangles();
    const auto& localNrms = fbp.triangleNormals();
    result.triangles.reserve(localTris.size());
    result.normals.reserve(localNrms.size());
    for (const auto& tri : localTris) {
        result.triangles.emplace_back(
            blockIndices[tri(0)],
            blockIndices[tri(1)],
            blockIndices[tri(2)]);
    }
    result.normals = localNrms;
    return result;
}

void ParallelBallPivoting::mergeAndDedup(std::vector<BlockResult>& results) {
    struct Entry {
        int v0, v1, v2;
        int block;
        int tri;
    };

    int numThreads = static_cast<int>(std::thread::hardware_concurrency());
    if (numThreads <= 0) numThreads = 4;
    numThreads = std::min(numThreads, static_cast<int>(results.size()));

    const int NUM_BUCKETS = 64;
    const uint64_t BUCKET_MASK = NUM_BUCKETS - 1;

    auto hashKey = [](int a, int b, int c) -> uint64_t {
        if (a > b) std::swap(a, b);
        if (b > c) std::swap(b, c);
        if (a > b) std::swap(a, b);
        return ((uint64_t)a * 73856093) ^ ((uint64_t)b * 19349663) ^ ((uint64_t)c * 83492791);
    };

    std::vector<std::vector<std::vector<Entry>>> threadBuckets(
        numThreads, std::vector<std::vector<Entry>>(NUM_BUCKETS));

    std::atomic<int> nextBlock{0};
    std::vector<std::thread> workers;
    workers.reserve(numThreads);

    for (int t = 0; t < numThreads; ++t) {
        workers.emplace_back([&, t]() {
            for (auto& b : threadBuckets[t]) b.reserve(256);
            int b;
            while ((b = nextBlock.fetch_add(1)) < static_cast<int>(results.size())) {
                const auto& tris = results[b].triangles;
                for (int i = 0; i < static_cast<int>(tris.size()); ++i) {
                    int v0 = tris[i](0), v1 = tris[i](1), v2 = tris[i](2);
                    int bucket = static_cast<int>(hashKey(v0, v1, v2) & BUCKET_MASK);
                    threadBuckets[t][bucket].push_back({v0, v1, v2, b, i});
                }
            }
        });
    }
    for (auto& w : workers) w.join();

    std::vector<std::vector<Entry>> buckets(NUM_BUCKETS);
    for (int bucket = 0; bucket < NUM_BUCKETS; ++bucket) {
        size_t total = 0;
        for (int t = 0; t < numThreads; ++t)
            total += threadBuckets[t][bucket].size();
        buckets[bucket].reserve(total);
        for (int t = 0; t < numThreads; ++t) {
            auto& src = threadBuckets[t][bucket];
            buckets[bucket].insert(buckets[bucket].end(), src.begin(), src.end());
        }
    }
    threadBuckets.clear();
    threadBuckets.shrink_to_fit();

    std::vector<std::vector<Eigen::Vector3i>> bucketTris(NUM_BUCKETS);
    std::vector<std::vector<Eigen::Vector3d>> bucketNrms(NUM_BUCKETS);
    std::vector<size_t> bucketDups(NUM_BUCKETS, 0);

    std::atomic<int> nextBucket{0};
    workers.clear();

    for (int t = 0; t < numThreads; ++t) {
        workers.emplace_back([&]() {
            int bucket;
            while ((bucket = nextBucket.fetch_add(1)) < NUM_BUCKETS) {
                auto& entries = buckets[bucket];
                if (entries.empty()) continue;

                std::sort(entries.begin(), entries.end(),
                    [](const Entry& a, const Entry& b) {
                        return std::tie(a.v0, a.v1, a.v2) < std::tie(b.v0, b.v1, b.v2);
                    });

                auto& outT = bucketTris[bucket];
                auto& outN = bucketNrms[bucket];
                outT.reserve(entries.size());
                outN.reserve(entries.size());

                int lastV0 = -1, lastV1 = -1, lastV2 = -1;
                for (const auto& e : entries) {
                    if (e.v0 == lastV0 && e.v1 == lastV1 && e.v2 == lastV2) {
                        ++bucketDups[bucket];
                    } else {
                        lastV0 = e.v0; lastV1 = e.v1; lastV2 = e.v2;
                        outT.push_back(results[e.block].triangles[e.tri]);
                        if (e.tri < static_cast<int>(results[e.block].normals.size()))
                            outN.push_back(results[e.block].normals[e.tri]);
                    }
                }
            }
        });
    }
    for (auto& w : workers) w.join();

    out_triangles_.clear();
    out_normals_.clear();
    duplicates_removed_ = 0;

    size_t totalSurv = 0;
    for (int b = 0; b < NUM_BUCKETS; ++b) {
        totalSurv += bucketTris[b].size();
        duplicates_removed_ += bucketDups[b];
    }
    out_triangles_.reserve(totalSurv);
    out_normals_.reserve(totalSurv);
    for (int b = 0; b < NUM_BUCKETS; ++b) {
        out_triangles_.insert(out_triangles_.end(),
                              bucketTris[b].begin(), bucketTris[b].end());
        out_normals_.insert(out_normals_.end(),
                            bucketNrms[b].begin(), bucketNrms[b].end());
    }
}

void ParallelBallPivoting::Run(const std::vector<double>& radii) {
    Params p;
    Run(radii, p);
}

void ParallelBallPivoting::Run(const std::vector<double>& radii,
                                const Params& params) {
    out_triangles_.clear();
    out_normals_.clear();
    duplicates_removed_ = 0;

    size_t N = points_.size();
    if (N == 0 || radii.empty()) return;

    // ---- 确定分块数（4×cores 细粒度 + 动态分配；每块至少 500 点）----
    int numBlocks = params.numBlocks;
    if (numBlocks <= 0) {
        int hw = static_cast<int>(std::thread::hardware_concurrency());
        if (hw <= 0) hw = 4;
        numBlocks = hw * 4;
    }
    const size_t minPointsPerBlock = 500;
    while (static_cast<size_t>(numBlocks) > 1 &&
           N / numBlocks < minPointsPerBlock) {
        numBlocks /= 2;
    }
    if (numBlocks < 1) numBlocks = 1;
    num_blocks_used_ = numBlocks;

    if (numBlocks == 1) {
        num_threads_used_ = 1;
        std::vector<int> allIndices(N);
        for (size_t i = 0; i < N; ++i) allIndices[i] = static_cast<int>(i);
        std::vector<BlockResult> results(1);
        results[0] = processBlock(allIndices, {}, radii);
        mergeAndDedup(results);
        return;
    }

    double maxRadius = 0.0;
    for (double r : radii) maxRadius = std::max(maxRadius, r);
    double haloRadius = params.haloFactor * maxRadius;

    Eigen::Vector3d bboxMin = points_[0];
    Eigen::Vector3d bboxMax = points_[0];
    for (const auto& p : points_) {
        for (int d = 0; d < 3; ++d) {
            bboxMin(d) = std::min(bboxMin(d), p(d));
            bboxMax(d) = std::max(bboxMax(d), p(d));
        }
    }
    Eigen::Vector3d extent = bboxMax - bboxMin;
    int splitAxis = 0;
    if (extent(1) > extent(splitAxis)) splitAxis = 1;
    if (extent(2) > extent(splitAxis)) splitAxis = 2;

    std::vector<int> sortedIndices(N);
    for (size_t i = 0; i < N; ++i) sortedIndices[i] = static_cast<int>(i);
    std::sort(sortedIndices.begin(), sortedIndices.end(),
              [this, splitAxis](int a, int b) {
                  return points_[a](splitAxis) < points_[b](splitAxis);
              });

    struct BlockData {
        std::vector<int> indices;
        std::vector<char> seedMask;
    };
    std::vector<BlockData> blocks(numBlocks);

    std::vector<int> blockOfPoint(N, -1);
    for (int b = 0; b < numBlocks; ++b) {
        size_t startIdx = b * N / numBlocks;
        size_t endIdx = (b == numBlocks - 1) ? N : (b + 1) * N / numBlocks;
        for (size_t i = startIdx; i < endIdx; ++i)
            blockOfPoint[sortedIndices[i]] = b;
    }

    std::vector<double> blockOwnerLo(numBlocks, 1e30);
    std::vector<double> blockOwnerHi(numBlocks, -1e30);
    for (size_t i = 0; i < N; ++i) {
        int b = blockOfPoint[i];
        double coord = points_[i](splitAxis);
        if (coord < blockOwnerLo[b]) blockOwnerLo[b] = coord;
        if (coord > blockOwnerHi[b]) blockOwnerHi[b] = coord;
    }

    for (int b = 0; b < numBlocks; ++b) {
        double loExt = blockOwnerLo[b] - haloRadius;
        double hiExt = blockOwnerHi[b] + haloRadius;

        auto& blk = blocks[b];
        for (size_t i = 0; i < N; ++i) {
            double coord = points_[i](splitAxis);
            if (coord >= loExt && coord <= hiExt) {
                bool isOwner = (blockOfPoint[i] == b);
                blk.indices.push_back(static_cast<int>(i));
                blk.seedMask.push_back(isOwner ? 1 : 0);
            }
        }
    }

    int hwCores = static_cast<int>(std::thread::hardware_concurrency());
    int maxThreads = (params.numThreads > 0) ? std::min(params.numThreads, hwCores) : hwCores;
    num_threads_used_ = std::min(numBlocks, maxThreads);

    std::vector<BlockResult> results(numBlocks);
    std::atomic<int> nextBlock{0};
    std::vector<std::thread> workers;
    workers.reserve(num_threads_used_);

    for (int t = 0; t < num_threads_used_; ++t) {
        workers.emplace_back([this, &results, &blocks, &radii, &nextBlock]() {
            int b;
            while ((b = nextBlock.fetch_add(1)) < static_cast<int>(blocks.size())) {
                try {
                    results[b] = processBlock(blocks[b].indices,
                                              blocks[b].seedMask, radii);
                } catch (...) {
                    // 单块失败不影响其他块
                }
            }
        });
    }
    for (auto& t : workers) t.join();

    mergeAndDedup(results);

    // ---- 边界缝合（skipStitch=true 时由外层统一缝合）----
    if (numBlocks > 1 && !out_triangles_.empty() && !params.skipStitch) {
        double maxR = 0.0;
        for (double r : radii) maxR = std::max(maxR, r);

        FastBallPivoting globalBP(points_, normals_);
        globalBP.buildGrid(maxR);
        globalBP.loadTriangles(out_triangles_, maxR);
        globalBP.expandBordersMultiRadius(radii);

        out_triangles_ = globalBP.triangles();
        out_normals_ = globalBP.triangleNormals();
    }
}

} // namespace bp
} // namespace calib
