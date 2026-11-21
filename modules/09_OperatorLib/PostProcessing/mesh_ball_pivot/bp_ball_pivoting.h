#pragma once
// ============================================================================
// bp_ball_pivoting.h — CPU 滚球重建内核（PCMP ball_pivoting.h 移植）
//
// FastBallPivoting：单线程 BPA 内核（Open3D 语义重写：索引池＋O(1) 边查）
// ParallelBallPivoting：最长轴等分 N 块＋halo 重叠＋种子掩码＋三角哈希去重
// Open3D 剥离：kdtree_ 成员与 KDTree 路径删除——本移植恒用 UniformGrid
// （沙盒链 CPU 路径本就 useGrid=true，金标准同源）。
// ============================================================================
#include <Eigen/Dense>
#include <atomic>
#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "bp_uniform_grid.h"

namespace calib {
namespace bp {

class FastBallPivoting {
public:
    FastBallPivoting(const std::vector<Eigen::Vector3d>& points,
                     const std::vector<Eigen::Vector3d>& normals);

    void setSeedMask(const std::vector<char>& mask);
    void setUseGrid(bool use) { useGrid_ = use; }
    void setCandidateFilter(bool filter) { filterCandidates_ = filter; }
    void setSkipFlip(bool skip) { skipFlip_ = skip; }

    void Run(const std::vector<double>& radii);

    // ---- 边界缝合支持 ----
    void buildGrid(double maxRadius);
    void loadTriangles(const std::vector<Eigen::Vector3i>& tris, double radius);
    void loadTrianglesParallel(const std::vector<Eigen::Vector3i>& tris, double radius,
                               const std::vector<Eigen::Vector3d>& centers);
    void expandBorders(double radius);
    void expandBordersMultiRadius(const std::vector<double>& radii);

    const std::vector<Eigen::Vector3i>& triangles() const { return out_triangles_; }
    const std::vector<Eigen::Vector3d>& triangleNormals() const { return out_normals_; }

private:
    enum EdgeType { kBorder = 0, kFront = 1, kInner = 2 };
    enum VertexType { kOrphan = 0, kVFront = 1, kVInner = 2 };

    struct FastEdge {
        int source = -1;
        int target = -1;
        int triangle0 = -1;
        int triangle1 = -1;
        int type = kFront;
    };

    struct FastTriangle {
        int vert0 = -1, vert1 = -1, vert2 = -1;
        Eigen::Vector3d ball_center;
    };

    const std::vector<Eigen::Vector3d>& points_;
    const std::vector<Eigen::Vector3d>& normals_;
    UniformGrid grid_;
    bool useGrid_ = false;

    std::vector<FastEdge> edges_;
    std::vector<FastTriangle> triangles_;
    std::unordered_map<uint64_t, int> edge_map_;
    static constexpr int EDGE_MAP_SHARDS = 32;
    std::vector<std::unordered_map<uint64_t, int>> edge_map_shards_{EDGE_MAP_SHARDS};
    bool use_sharded_edge_map_ = false;
    std::vector<std::vector<int>> vertex_edges_;
    std::vector<int> vertex_type_;

    std::list<int> edge_front_;
    std::list<int> border_edges_;

    std::vector<Eigen::Vector3i> out_triangles_;
    std::vector<Eigen::Vector3d> out_normals_;

    std::vector<char> seed_mask_;
    bool filterCandidates_ = false;
    bool skipFlip_ = false;

    static uint64_t makeEdgeKey(int a, int b);
    int getLinkingEdge(int v0, int v1) const;
    int getOrCreateEdge(int v0, int v1, bool& is_new);
    void addEdgeToVertex(int vidx, int eidx);
    void updateVertexType(int vidx);
    int getOppositeVertex(int edge_idx) const;
    void addAdjacentTriangleToEdge(int edge_idx, int tri_idx);

    bool computeBallCenter(int vidx1, int vidx2, int vidx3,
                           double radius, Eigen::Vector3d& center) const;
    Eigen::Vector3d computeFaceNormal(const Eigen::Vector3d& v0,
                                      const Eigen::Vector3d& v1,
                                      const Eigen::Vector3d& v2) const;
    bool isCompatible(int v0, int v1, int v2) const;

    void createTriangle(int v0, int v1, int v2, const Eigen::Vector3d& center);
    int findCandidateVertex(int edge_idx, double radius,
                            Eigen::Vector3d& candidate_center);
    void expandTriangulation(double radius);
    void expandTriangulationParallel(double radius);
    bool tryTriangleSeed(int v0, int v1, int v2,
                         const std::vector<int>& nb_indices,
                         double radius, Eigen::Vector3d& center);
    bool trySeed(int v, double radius);
    void findSeedTriangle(double radius);

    inline int searchRadius(const Eigen::Vector3d& q, double r,
                            std::vector<int>& indices,
                            std::vector<double>& dists2) const {
        return grid_.searchRadius(q, r, indices, dists2);
    }

    static bool pointsCoplanar(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1,
                               const Eigen::Vector3d& p2, const Eigen::Vector3d& p3);
    static double lineSegmentsMinimumDistance(const Eigen::Vector3d& p0,
                                              const Eigen::Vector3d& p1,
                                              const Eigen::Vector3d& q0,
                                              const Eigen::Vector3d& q1);
};

class ParallelBallPivoting {
public:
    struct Params {
        int numBlocks = 0;
        int numThreads = 0;
        double haloFactor = 2.0;
        bool skipStitch = false;
    };

    ParallelBallPivoting(const std::vector<Eigen::Vector3d>& points,
                         const std::vector<Eigen::Vector3d>& normals);

    void Run(const std::vector<double>& radii);
    void Run(const std::vector<double>& radii, const Params& params);

    void setUseGrid(bool use) { useGrid_ = use; }

    const std::vector<Eigen::Vector3i>& triangles() const { return out_triangles_; }
    const std::vector<Eigen::Vector3d>& triangleNormals() const { return out_normals_; }

    int numBlocksUsed() const { return num_blocks_used_; }
    int numThreadsUsed() const { return num_threads_used_; }
    size_t duplicatesRemoved() const { return duplicates_removed_; }

private:
    const std::vector<Eigen::Vector3d>& points_;
    const std::vector<Eigen::Vector3d>& normals_;

    std::vector<Eigen::Vector3i> out_triangles_;
    std::vector<Eigen::Vector3d> out_normals_;

    int num_blocks_used_ = 0;
    int num_threads_used_ = 0;
    size_t duplicates_removed_ = 0;
    bool useGrid_ = false;

    struct BlockResult {
        std::vector<Eigen::Vector3i> triangles;
        std::vector<Eigen::Vector3d> normals;
    };
    BlockResult processBlock(const std::vector<int>& blockIndices,
                             const std::vector<char>& seedMask,
                             const std::vector<double>& radii) const;
    void mergeAndDedup(std::vector<BlockResult>& results);
    static uint64_t triangleKey(int v0, int v1, int v2);
};

} // namespace bp
} // namespace calib
