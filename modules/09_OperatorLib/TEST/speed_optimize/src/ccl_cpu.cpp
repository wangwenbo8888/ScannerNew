// ccl_cpu.cpp — CPU 连通域分析实现（RLE＋并查集；详见 ccl_cpu.h 头注释）
// 两个入口共享同一核心（并查集/过滤/输出序逐位一致）：
//   runCclCpu          —— 逐像素掩膜入口（自带 RLE 扫描）
//   runCclCpuFromRuns  —— 行程直入入口（消费 GPU-RLE 产物，【150FPS】）
#include "ccl_cpu.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <numeric>

namespace scr {

namespace {

// 并查集（按最小标签合并：根标签＝组件光栅首遇顺序号）
inline int ufFind(std::vector<int>& parent, int x) {
    int root = x;
    while (parent[root] != root) root = parent[root];
    while (parent[x] != root) {           // 路径压缩
        const int next = parent[x];
        parent[x] = root;
        x = next;
    }
    return root;
}

struct RunEntry { int x0, x1, label; };   // x1 含端（[x0,x1]）

// 【N2·2026-10-23】runsByRow 池化：thread_local 跨调用复用——每帧免 H≈2048 个
// vector 头重建＋逐 run 内层分配（第三轮审核 N2 项；两入口互斥不复用冲突，
// 多 marker lane 各自线程独立池）。保容量逐行 clear，冷启动首帧 assign。
std::vector<std::vector<RunEntry>>& pooledRunsByRow(int H) {
    static thread_local std::vector<std::vector<RunEntry>> pool;
    if (static_cast<int>(pool.size()) != H)
        pool.assign(static_cast<size_t>(H), {});
    else
        for (auto& r : pool) r.clear();
    return pool;
}

// —— 核心：逐行行程（每行升序）→ 8 连通并查集 → 面积/边框统计 → 过滤输出 ——
calib::RegionAnalysisResult cclFromRunsByRow(std::vector<std::vector<RunEntry>>& runsByRow,
                                             int W, int H, int minArea, int maxArea,
                                             int topXCount) {
    calib::RegionAnalysisResult result;
    std::vector<int> parent;              // 并查集（parent[l]=l 为根）
    parent.reserve(4096);
    const std::vector<RunEntry>* prevRuns = nullptr;
    int nextLabel = 0;
    for (int y = 0; y < H; ++y) {
        auto& curRuns = runsByRow[y];
        // 与上一行行程做 8 连通合并（重叠区 [x0-1, x1+1]）
        size_t pi = 0;
        for (auto& r : curRuns) {
            int root = -1;
            if (prevRuns) {
                while (pi < prevRuns->size() && (*prevRuns)[pi].x1 + 1 < r.x0) ++pi;
                size_t pj = pi;
                while (pj < prevRuns->size() && (*prevRuns)[pj].x0 <= r.x1 + 1) {
                    const int pl = ufFind(parent, (*prevRuns)[pj].label);
                    if (root < 0) root = pl;
                    else if (pl != root) {
                        if (pl < root) { parent[root] = pl; root = pl; }
                        else parent[pl] = root;
                        root = ufFind(parent, root);
                    }
                    ++pj;
                }
            }
            if (root < 0) {
                parent.push_back(static_cast<int>(parent.size()));
                root = nextLabel++;
            }
            r.label = root;
        }
        prevRuns = &runsByRow[y];
    }

    // —— 根统计：面积＋包围盒（按根标签聚合）——
    struct Stat { int area = 0, minX = 0, minY = 0, maxX = -1, maxY = -1; };
    std::vector<Stat> stats(parent.size());
    for (int y = 0; y < H; ++y) {
        for (const auto& r : runsByRow[y]) {
            const int root = ufFind(parent, r.label);
            Stat& s = stats[root];
            s.area += r.x1 - r.x0 + 1;
            if (s.maxY < 0) {                       // 首遇（行序遍历→minY 即当前行）
                s.minX = r.x0; s.maxX = r.x1; s.minY = y; s.maxY = y;
            } else {
                if (r.x0 < s.minX) s.minX = r.x0;
                if (r.x1 > s.maxX) s.maxX = r.x1;
                s.maxY = y;
            }
        }
    }

    // —— 面积过滤［minArea, maxArea］＋（可选）topX ——
    int effMinArea = minArea;
    if (topXCount > 0) {
        std::vector<int> areas;
        areas.reserve(parent.size());
        for (int l = 0; l < static_cast<int>(parent.size()); ++l)
            if (parent[l] == l && stats[l].area >= minArea && stats[l].area <= maxArea)
                areas.push_back(stats[l].area);
        if (static_cast<int>(areas.size()) > topXCount) {
            std::sort(areas.begin(), areas.end(), std::greater<int>());
            effMinArea = std::max(minArea, areas[static_cast<size_t>(topXCount) - 1]);
        }
    }
    // 输出顺序＝根标签升序＝光栅首遇序（对齐 GPU buildRemap 升序遍历）
    int newId = 0;
    for (int l = 0; l < static_cast<int>(parent.size()); ++l) {
        if (parent[l] != l) continue;
        const Stat& s = stats[l];
        if (s.area < effMinArea || s.area > maxArea) continue;
        calib::ComponentStats cs;
        cs.label = ++newId;
        cs.boundingBoxX = s.minX;
        cs.boundingBoxY = s.minY;
        cs.boundingBoxWidth = s.maxX - s.minX + 1;
        cs.boundingBoxHeight = s.maxY - s.minY + 1;
        cs.area = s.area;
        result.components.push_back(cs);
    }
    result.componentCount = static_cast<int>(result.components.size());
    result.qualityFlag = result.componentCount == 0
                             ? calib::QualityFlag::Warning
                             : (result.componentCount > 200
                                    ? calib::QualityFlag::Degraded
                                    : calib::QualityFlag::Normal);
    result.success = true;
    if (result.componentCount == 0)
        result.message = "No connected components found after filtering";
    return result;
}

} // namespace

calib::RegionAnalysisResult runCclCpu(const cv::Mat& mask, int minArea, int maxArea,
                                      int topXCount) {
    calib::RegionAnalysisResult result;
    if (mask.empty() || mask.type() != CV_8UC1) {
        result.message = "ccl_cpu: 掩膜空或类型非 CV_8UC1";
        return result;
    }
    const int W = mask.cols, H = mask.rows;
    auto& runsByRow = pooledRunsByRow(H);   // 【N2】池化复用
    for (int y = 0; y < H; ++y) {
        const uchar* row = mask.ptr<uchar>(y);
        auto& curRuns = runsByRow[static_cast<size_t>(y)];
        int x = 0;
        while (x < W) {
            // 8 字节整块跳全黑段（掩膜稀疏，绝大多数块全零）
            while (x + 8 <= W) {
                uint64_t v;
                std::memcpy(&v, row + x, 8);
                if (v != 0) break;
                x += 8;
            }
            if (x >= W) break;
            while (x < W && row[x] == 0) ++x;
            if (x >= W) break;
            const int x0 = x;
            while (x < W && row[x] != 0) ++x;
            curRuns.push_back({x0, x - 1, -1});
        }
    }
    return cclFromRunsByRow(runsByRow, W, H, minArea, maxArea, topXCount);
}

calib::RegionAnalysisResult runCclCpuFromRuns(const std::vector<int>& runsTriple,
                                              int nRuns, int W, int H,
                                              int minArea, int maxArea, int topXCount) {
    auto& runsByRow = pooledRunsByRow(H);   // 【N2】池化复用
    for (int i = 0; i < nRuns; ++i) {
        const int row = runsTriple[static_cast<size_t>(i) * 3];
        const int x0 = runsTriple[static_cast<size_t>(i) * 3 + 1];
        const int x1 = runsTriple[static_cast<size_t>(i) * 3 + 2];
        if (row < 0 || row >= H) continue;    // 防御（GPU 产物不应越界）
        runsByRow[static_cast<size_t>(row)].push_back({x0, x1, -1});
    }
    // GPU 行程按线程槽位写出（非行内 x 升序）——并查集要求行内升序：起点行内
    // 唯一，按 x0 排序后与像素 RLE 路径完全同序（确定性等价）
    for (auto& runs : runsByRow)
        std::sort(runs.begin(), runs.end(),
                  [](const RunEntry& a, const RunEntry& b) { return a.x0 < b.x0; });
    return cclFromRunsByRow(runsByRow, W, H, minArea, maxArea, topXCount);
}

} // namespace scr
