#include "tsdf_shell_fuse.h"
#include "tsdf_mc_tables.h"
#include "common/calib_logging.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>
#include <vector>

namespace calib {

CALIB_DEFINE_LOG_TAG(02T, TsdfShellFuseCpu)

// ============================================================
// pack20 体素键（ZG EdgeFusion/InstantMesh 同族: 每轴 20bit 符号-数值,
// 负数 = |v|∨0x80000; 复现工程 edf_core.h PackKey 同款）
// ============================================================
namespace {

inline uint64_t encAxis(int32_t v) {
    return v < 0 ? (uint64_t(-v) | 0x80000u) : (uint64_t(v) & 0x7FFFFu);
}
inline uint64_t packKey(int32_t x, int32_t y, int32_t z) {
    return encAxis(z) | (encAxis(y) << 20) | (encAxis(x) << 40);
}
constexpr int32_t kMaxAxis = 0x7FFFE;  // 键域保护

// 棱键混合（MC 顶点去重用: 同一条被切棱 → 同一 64bit 键）
inline uint64_t mixEdgeKey(uint64_t a, uint64_t b) {
    uint64_t h = a * 0x9E3779B97F4A7C15ULL;
    h ^= b + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    return h;
}

// 3×3 对称阵 Jacobi 特征分解（PCA 法向用）
void jacobiEig3(double A[9], double vec[9], double val[3]) {
    double V[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    for (int sweep = 0; sweep < 50; ++sweep) {
        double off = 0;
        for (int p = 0; p < 2; ++p)
            for (int q = p + 1; q < 3; ++q) off += std::fabs(A[p * 3 + q]);
        if (off < 1e-18) break;
        for (int p = 0; p < 2; ++p)
            for (int q = p + 1; q < 3; ++q) {
                if (std::fabs(A[p * 3 + q]) < 1e-20) continue;
                const double th = (A[q * 3 + q] - A[p * 3 + p]) / (2.0 * A[p * 3 + q]);
                const double tt = (th >= 0 ? 1.0 : -1.0) / (std::fabs(th) + std::sqrt(th * th + 1.0));
                const double c = 1.0 / std::sqrt(tt * tt + 1.0);
                const double s = tt * c;
                for (int k = 0; k < 3; ++k) {
                    const double akp = A[k * 3 + p], akq = A[k * 3 + q];
                    A[k * 3 + p] = c * akp - s * akq;
                    A[k * 3 + q] = s * akp + c * akq;
                }
                for (int k = 0; k < 3; ++k) {
                    const double apk = A[p * 3 + k], aqk = A[q * 3 + k];
                    A[p * 3 + k] = c * apk - s * aqk;
                    A[q * 3 + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < 3; ++k) {
                    const double vkp = V[k * 3 + p], vkq = V[k * 3 + q];
                    V[k * 3 + p] = c * vkp - s * vkq;
                    V[k * 3 + q] = s * vkp + c * vkq;
                }
            }
    }
    for (int i = 0; i < 3; ++i) val[i] = A[i * 3 + i];
    for (int i = 0; i < 9; ++i) vec[i] = V[i];
}

// 带奇偶并查集（法向定向传播: 最大生成树 Kruskal）。
// 语义: 每点带翻转位 f_i, 有效法向 = n_i×(-1)^f_i; 树上要求有效点积 ≥ 0。
class FlipUf {
public:
    explicit FlipUf(int n) : parent_(n), size_(n, 1), par_(n, 0) {
        for (int i = 0; i < n; ++i) parent_[i] = i;
    }
    int find(int x, uint8_t& px) {
        // 两趟: 先沿路累计奇偶, 再路径压缩
        int root = x;
        uint8_t acc = 0;
        while (parent_[root] != root) {
            acc ^= par_[root];
            root = parent_[root];
        }
        px = acc;
        int cur = x;
        uint8_t c = acc;
        while (parent_[cur] != root) {
            const int nxt = parent_[cur];
            const uint8_t pnext = (uint8_t)(c ^ par_[cur]);
            parent_[cur] = root;
            par_[cur] = c;
            c = pnext;
            cur = nxt;
        }
        return root;
    }
    // 合并 i/j 所在分量, 传"原始法线点积"; 保证合并后有效 n_i·n_j ≥ 0
    void unite(int i, int j, double rawDot) {
        uint8_t fi = 0, fj = 0;
        int a = find(i, fi), b = find(j, fj);
        if (a == b) return;
        const double s = (((fi ^ fj) != 0) ? -rawDot : rawDot);
        const uint8_t toggle = (s < 0.0) ? 1 : 0;  // a 分量整体是否翻转
        if (size_[a] < size_[b]) {
            parent_[a] = b;
            par_[a] = toggle;
            size_[b] += size_[a];
        } else {
            parent_[b] = a;
            par_[b] = toggle;
            size_[a] += size_[b];
        }
    }

private:
    std::vector<int> parent_, size_;
    std::vector<uint8_t> par_;
};

}  // namespace

// ============================================================
// Params / Stats 序列化
// ============================================================

void TsdfShellFuseParams::validate() const {
    if (!(voxelSize > 0.0f)) throw std::invalid_argument("voxelSize must be > 0");
    if (!(truncFactor >= 1.0f)) throw std::invalid_argument("truncFactor must be >= 1");
    if (!(lateralFactor > 0.0f)) throw std::invalid_argument("lateralFactor must be > 0");
    if (!(keyframeDispDiv > 0.0f)) throw std::invalid_argument("keyframeDispDiv must be > 0");
    if (!(keyframeRotDeg >= 0.0f)) throw std::invalid_argument("keyframeRotDeg must be >= 0");
    if (mcCornerStep < 1) throw std::invalid_argument("mcCornerStep must be >= 1");
    if (normalKnn < 4) throw std::invalid_argument("normalKnn must be >= 4");
}

nlohmann::json TsdfShellFuseParams::toJson() const {
    nlohmann::json j;
    j["voxelSize"] = voxelSize;
    j["truncFactor"] = truncFactor;
    j["lateralFactor"] = lateralFactor;
    j["useKeyframeGate"] = useKeyframeGate;
    j["keyframeDispDiv"] = keyframeDispDiv;
    j["keyframeRotDeg"] = keyframeRotDeg;
    j["mcCornerStep"] = mcCornerStep;
    j["isoLevel"] = isoLevel;
    j["useCornerRayGate"] = useCornerRayGate;
    j["cornerRayGateDot"] = cornerRayGateDot;
    j["normalKnn"] = normalKnn;
    return j;
}

TsdfShellFuseParams TsdfShellFuseParams::fromJson(const nlohmann::json& j) {
    TsdfShellFuseParams p;
    if (j.contains("voxelSize")) p.voxelSize = j.at("voxelSize").get<float>();
    if (j.contains("truncFactor")) p.truncFactor = j.at("truncFactor").get<float>();
    if (j.contains("lateralFactor")) p.lateralFactor = j.at("lateralFactor").get<float>();
    if (j.contains("useKeyframeGate")) p.useKeyframeGate = j.at("useKeyframeGate").get<bool>();
    if (j.contains("keyframeDispDiv")) p.keyframeDispDiv = j.at("keyframeDispDiv").get<float>();
    if (j.contains("keyframeRotDeg")) p.keyframeRotDeg = j.at("keyframeRotDeg").get<float>();
    if (j.contains("mcCornerStep")) p.mcCornerStep = j.at("mcCornerStep").get<int>();
    if (j.contains("isoLevel")) p.isoLevel = j.at("isoLevel").get<float>();
    if (j.contains("useCornerRayGate")) p.useCornerRayGate = j.at("useCornerRayGate").get<bool>();
    if (j.contains("cornerRayGateDot")) p.cornerRayGateDot = j.at("cornerRayGateDot").get<float>();
    if (j.contains("normalKnn")) p.normalKnn = j.at("normalKnn").get<int>();
    return p;
}

nlohmann::json TsdfShellStats::toJson() const {
    nlohmann::json j;
    j["integratedPoints"] = integratedPoints;
    j["fusedFrames"] = fusedFrames;
    j["skippedFrames"] = skippedFrames;
    j["droppedPoints"] = droppedPoints;
    j["voxelCount"] = voxelCount;
    j["cubesCut"] = cubesCut;
    j["triangleCount"] = triangleCount;
    j["vertexCount"] = vertexCount;
    j["meanPointToMesh"] = meanPointToMesh;
    j["p95PointToMesh"] = p95PointToMesh;
    j["maxPointToMesh"] = maxPointToMesh;
    return j;
}

// ============================================================
// Impl
// ============================================================

struct TsdfShellFuseCpu::Impl {
    // 每体素: 截断带符号距离(归一化) + 观测计数 + 胜出写入时的单位观察方向 + 帧号
    struct Vox {
        float tsdf = 0.0f;
        float weight = 0.0f;
        float rx = 0.0f, ry = 0.0f, rz = 0.0f;
        float frame = 0.0f;
    };
    std::unordered_map<uint64_t, Vox> hash;

    bool hasPrevPose = false;
    cv::Matx33d prevR = cv::Matx33d::eye();
    cv::Vec3d prevT = cv::Vec3d(0, 0, 0);
    float frameCounter = 0.0f;

    void clear() {
        hash.clear();
        hasPrevPose = false;
        frameCounter = 0.0f;
    }

    // 泼溅核心: 样本 p 世界坐标 + 单位观察方向 u（传感器→点）。
    // 球窗 tau + 横向窗; sdf=along/tau 夹 [-1,1]; 更新=运行加权平均(偏差 D-T3)。
    void splat(const TsdfShellFuseParams& prm, TsdfShellStats& st,
               double px, double py, double pz,
               double ux, double uy, double uz, float frameIdx) {
        const double grid = prm.voxelSize;
        const double tau = prm.truncFactor * grid;
        const int cx = (int)std::floor(px / grid);
        const int cy = (int)std::floor(py / grid);
        const int cz = (int)std::floor(pz / grid);
        if (std::abs(cx) > kMaxAxis || std::abs(cy) > kMaxAxis ||
            std::abs(cz) > kMaxAxis) {
            ++st.droppedPoints;
            return;
        }
        const int r = (int)std::ceil(tau / grid) + 1;
        const double latHalf = double(prm.lateralFactor) * grid;
        const double lat2Max = latHalf * latHalf;
        // 【偏差 D-T2, 显式登记】球窗半径² = tau² + grid²（ZG 原版恰为 tau²）:
        // 立方体角(MC 角步长 2 格 → 深至 ±tau)在采样横向间距处恰好落在球面上,
        // 紧邻样本够不着 → 角缺失 → 整立方体跳过 → 壳出洞。补一格余量后角
        // 覆盖稳定; 超出 tau 的 sdf 仍被夹断成 ±1, 场语义不变。
        const double win2 = tau * tau + grid * grid;
        for (int dz = -r; dz <= r; ++dz)
            for (int dy = -r; dy <= r; ++dy)
                for (int dx = -r; dx <= r; ++dx) {
                    if ((double)(dx * dx + dy * dy + dz * dz) * grid * grid >
                        win2)
                        continue;  // 球形窗（立方体对角会远超 tau）
                    const double vx = (cx + dx + 0.5) * grid;
                    const double vy = (cy + dy + 0.5) * grid;
                    const double vz = (cz + dz + 0.5) * grid;
                    const double rxv = vx - px, ryv = vy - py, rzv = vz - pz;
                    const double along = rxv * ux + ryv * uy + rzv * uz;
                    const double lat2 = rxv * rxv + ryv * ryv + rzv * rzv -
                                        along * along;
                    if (lat2 > lat2Max) continue;  // 只刷视线两旁一条
                    double sdf = along / tau;
                    if (sdf > 1.0) sdf = 1.0;
                    if (sdf < -1.0) sdf = -1.0;
                    // 记录"本格此次写入的观察方向" = 传感器→样本射线 u。
                    // 【修正】ZG 语义是 unit(v − t)（传感器→格子）; standoff(≈300mm)
                    // >> tau(≈1mm) 时对所有近旁格子都 ≈ u。若误存 unit(格子−样本)
                    // 则面两侧的格子方向正好相反（dot≈−1）, MC 角门会把跨面立方体
                    // 的对侧角全部误杀 → 壳出洞（rayGate 诊断暴露的正是此病灶）。
                    const float grx = (float)ux, gry = (float)uy, grz = (float)uz;
                const uint64_t key = packKey(cx + dx, cy + dy, cz + dz);
                auto it = hash.find(key);
                if (it == hash.end()) {
                    Vox v;
                    v.tsdf = (float)sdf;
                    v.weight = 1.0f;
                    v.rx = grx; v.ry = gry; v.rz = grz;
                    v.frame = frameIdx;
                    hash.emplace(key, v);
                } else {
                    Vox& v = it->second;
                    // 【偏差 D-T3, 显式登记】运行加权平均（KinectFusion 经典式）,
                    // 替代 ZG 的 min|sdf| "就近者胜"。ZG 的 min 规则依托"逐帧稠密
                    // 积分"（每格由同一帧连续深度场写入, 跨帧取最近视角的平滑场）;
                    // 本算子是逐点稀疏泼溅——min 会逐格挑选单点噪声, 表面处符号
                    // 随噪声逐格翻转, MC 切出海绵状碎片（实测球壳三角数×5、欧拉
                    // 数转负）。平均使噪声按 √N 递减; "正反两面互相抵消"的 ZG
                    // 顾虑由 MC 角观察方向门承担（两侧观察方向 dot<−0.5 即拒绝
                    // 缝合）, 双面薄板的彻底解法留两槽扩展位。
                    const float wSum = v.weight + 1.0f;
                    v.tsdf = (v.weight * v.tsdf + (float)sdf) / wSum;
                    float rx = v.weight * v.rx + grx;
                    float ry = v.weight * v.ry + gry;
                    float rz = v.weight * v.rz + grz;
                    const float rl = std::sqrt(rx * rx + ry * ry + rz * rz);
                    if (rl > 1e-12f) {
                        rx /= rl; ry /= rl; rz /= rl;
                    } else {
                        rx = grx; ry = gry; rz = grz;
                    }
                    v.rx = rx; v.ry = ry; v.rz = rz;
                    v.weight = wSum;
                    v.frame = frameIdx;
                }
                }
    }
};

TsdfShellFuseCpu::TsdfShellFuseCpu(const TsdfShellFuseParams& params)
    : pImpl_(std::make_unique<Impl>()), params_(params) {
    params_.validate();
    CALIB_LOG_INFO("TsdfShellFuseCpu init (voxel={}, trunc={}, knn={})",
                   params_.voxelSize, params_.truncFactor, params_.normalKnn);
}

TsdfShellFuseCpu::~TsdfShellFuseCpu() = default;

void TsdfShellFuseCpu::Destroy() { pImpl_.reset(); }

void TsdfShellFuseCpu::Clear() noexcept {
    pImpl_->clear();
    stats_ = TsdfShellStats{};
}

void TsdfShellFuseCpu::SetParams(const TsdfShellFuseParams& params) {
    params.validate();
    params_ = params;
}

bool TsdfShellFuseCpu::addFrame(const std::vector<cv::Point3f>& localPts,
                                const cv::Matx33d& R, const cv::Vec3d& T) {
    if (!pImpl_) return false;
    // 关键帧门（ZG 常数: 位移²>格宽/15 或传感器 x 轴夹角>0.05°）
    if (params_.useKeyframeGate && pImpl_->hasPrevPose) {
        const double dx = T(0) - pImpl_->prevT(0);
        const double dy = T(1) - pImpl_->prevT(1);
        const double dz = T(2) - pImpl_->prevT(2);
        const double dsq = dx * dx + dy * dy + dz * dz;
        const float thr = params_.voxelSize / params_.keyframeDispDiv;
        bool fuse = (float)dsq > thr;
        if (!fuse) {
            // 传感器 x 轴夹角 = angle(R·e_x, Rp·e_x)（首列点积; 夹 clamp 防 NaN）
            double coldot = R(0, 0) * pImpl_->prevR(0, 0) +
                            R(1, 0) * pImpl_->prevR(1, 0) +
                            R(2, 0) * pImpl_->prevR(2, 0);
            if (coldot > 1.0) coldot = 1.0;
            if (coldot < -1.0) coldot = -1.0;
            const float deg =
                (float)(std::acos(coldot) * 180.0 / 3.14159265358979323846);
            fuse = deg > params_.keyframeRotDeg;
        }
        if (!fuse) {
            ++stats_.skippedFrames;
            return true;
        }
    }
    const float frameIdx = pImpl_->frameCounter++;
    for (const auto& p : localPts) {
        const double lx = p.x, ly = p.y, lz = p.z;
        const double wx = R(0, 0) * lx + R(0, 1) * ly + R(0, 2) * lz + T(0);
        const double wy = R(1, 0) * lx + R(1, 1) * ly + R(1, 2) * lz + T(1);
        const double wz = R(2, 0) * lx + R(2, 1) * ly + R(2, 2) * lz + T(2);
        double ux = wx - T(0), uy = wy - T(1), uz = wz - T(2);
        const double ul = std::sqrt(ux * ux + uy * uy + uz * uz);
        if (ul < 1e-12) { ++stats_.droppedPoints; continue; }
        ux /= ul; uy /= ul; uz /= ul;
        pImpl_->splat(params_, stats_, wx, wy, wz, ux, uy, uz, frameIdx);
        ++stats_.integratedPoints;
    }
    pImpl_->hasPrevPose = true;
    pImpl_->prevR = R;
    pImpl_->prevT = T;
    ++stats_.fusedFrames;
    return true;
}

bool TsdfShellFuseCpu::addCloudWithNormals(const std::vector<cv::Point3f>& pts,
                                           const std::vector<cv::Point3f>& normals) {
    if (!pImpl_) return false;
    if (pts.size() != normals.size()) {
        CALIB_LOG_ERROR("addCloudWithNormals: pts({}) != normals({})",
                        pts.size(), normals.size());
        return false;
    }
    const float frameIdx = pImpl_->frameCounter++;
    for (size_t i = 0; i < pts.size(); ++i) {
        const double nx = normals[i].x, ny = normals[i].y, nz = normals[i].z;
        const double nl = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (nl < 1e-9) { ++stats_.droppedPoints; continue; }
        // 法线朝外(朝传感器侧) → 观察方向 = -n
        pImpl_->splat(params_, stats_, pts[i].x, pts[i].y, pts[i].z,
                      -nx / nl, -ny / nl, -nz / nl, frameIdx);
        ++stats_.integratedPoints;
    }
    ++stats_.fusedFrames;
    return true;
}

bool TsdfShellFuseCpu::addCloudAutoNormals(const std::vector<cv::Point3f>& pts) {
    if (!pImpl_) return false;
    const int n = (int)pts.size();
    if (n < 8) {
        CALIB_LOG_ERROR("addCloudAutoNormals: too few points ({})", n);
        return false;
    }
    // ---- 1) 网格分桶 ----
    double minx = pts[0].x, maxx = pts[0].x;
    double miny = pts[0].y, maxy = pts[0].y;
    double minz = pts[0].z, maxz = pts[0].z;
    for (const auto& p : pts) {
        if (p.x < minx) minx = p.x; if (p.x > maxx) maxx = p.x;
        if (p.y < miny) miny = p.y; if (p.y > maxy) maxy = p.y;
        if (p.z < minz) minz = p.z; if (p.z > maxz) maxz = p.z;
    }
    const double ext = std::max(std::max(maxx - minx, maxy - miny), maxz - minz);
    const double bin = std::max(4.0 * params_.voxelSize, ext / 1024.0);
    const double invBin = 1.0 / bin;
    std::unordered_map<uint64_t, std::vector<int>> bins;
    bins.reserve(pts.size() * 2);
    auto bkey = [](int bx, int by, int bz) { return packKey(bx, by, bz); };
    std::vector<int> cellBx(n), cellBy(n), cellBz(n);
    for (int i = 0; i < n; ++i) {
        cellBx[i] = (int)std::floor((pts[i].x - minx) * invBin);
        cellBy[i] = (int)std::floor((pts[i].y - miny) * invBin);
        cellBz[i] = (int)std::floor((pts[i].z - minz) * invBin);
        bins[bkey(cellBx[i], cellBy[i], cellBz[i])].push_back(i);
    }
    const int knn = std::min(params_.normalKnn, n - 1);
    std::vector<cv::Point3f> normals(n, cv::Point3f(0, 0, 0));
    std::vector<std::vector<int>> nbrs(n);
    for (int i = 0; i < n; ++i) {
        // 逐"外壳环"扩搜（只扫 max-norm==ring 的格子, 不重复累加）
        std::vector<std::pair<double, int>> cand;
        cand.reserve(knn * 2 + 8);
        for (int ring = 0; ring <= 3; ++ring) {
            for (int dz = -ring; dz <= ring; ++dz)
                for (int dy = -ring; dy <= ring; ++dy)
                    for (int dx = -ring; dx <= ring; ++dx) {
                        const int m = std::max(std::abs(dx), std::max(std::abs(dy), std::abs(dz)));
                        if (m != ring) continue;
                        auto it = bins.find(bkey(cellBx[i] + dx, cellBy[i] + dy, cellBz[i] + dz));
                        if (it == bins.end()) continue;
                        for (int j : it->second) {
                            if (j == i) continue;
                            const double ddx = pts[j].x - pts[i].x;
                            const double ddy = pts[j].y - pts[i].y;
                            const double ddz = pts[j].z - pts[i].z;
                            cand.emplace_back(ddx * ddx + ddy * ddy + ddz * ddz, j);
                        }
                    }
            if ((int)cand.size() >= knn + 1) break;
        }
        if ((int)cand.size() < 4) { ++stats_.droppedPoints; continue; }  // 孤立点
        if ((int)cand.size() > knn + 1) {
            std::partial_sort(cand.begin(), cand.begin() + knn + 1, cand.end());
            cand.resize(knn + 1);
        }
        // ---- 2) PCA: 协方差最小|特征值|向量 = 法向 ----
        double mx = 0, my = 0, mz = 0;
        for (auto& c : cand) {
            mx += pts[c.second].x; my += pts[c.second].y; mz += pts[c.second].z;
        }
        mx /= (double)cand.size();
        my /= (double)cand.size();
        mz /= (double)cand.size();
        double A[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (auto& c : cand) {
            const double ddx = pts[c.second].x - mx;
            const double ddy = pts[c.second].y - my;
            const double ddz = pts[c.second].z - mz;
            A[0] += ddx * ddx; A[1] += ddx * ddy; A[2] += ddx * ddz;
            A[4] += ddy * ddy; A[5] += ddy * ddz;
            A[8] += ddz * ddz;
        }
        A[3] = A[1]; A[6] = A[2]; A[7] = A[5];
        double vec[9], val[3];
        jacobiEig3(A, vec, val);
        int small = 0;
        if (std::fabs(val[1]) < std::fabs(val[small])) small = 1;
        if (std::fabs(val[2]) < std::fabs(val[small])) small = 2;
        const double nx = vec[small], ny = vec[3 + small], nz = vec[6 + small];
        const double nl = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (nl < 1e-12) { ++stats_.droppedPoints; continue; }
        normals[i] = cv::Point3f((float)(nx / nl), (float)(ny / nl), (float)(nz / nl));
        nbrs[i].reserve(cand.size());
        for (auto& c : cand) nbrs[i].push_back(c.second);
    }
    // ---- 3) 定向传播: |n_i·n_j| 最大生成树（Kruskal + 带奇偶并查集） ----
    struct Edge { double w; int a, b; };
    std::vector<Edge> edges;
    edges.reserve((size_t)n * 4);
    auto isZero = [](const cv::Point3f& v) {
        return v.x == 0.0f && v.y == 0.0f && v.z == 0.0f;
    };
    for (int i = 0; i < n; ++i) {
        if (isZero(normals[i])) continue;
        for (int j : nbrs[i]) {
            if (j <= i || isZero(normals[j])) continue;
            const double dot01 = (double)normals[i].x * normals[j].x +
                                 (double)normals[i].y * normals[j].y +
                                 (double)normals[i].z * normals[j].z;
            edges.push_back({std::fabs(dot01), i, j});
        }
    }
    std::sort(edges.begin(), edges.end(),
              [](const Edge& a, const Edge& b) { return a.w > b.w; });
    FlipUf uf(n);
    for (const auto& e : edges) {
        const double dot01 = (double)normals[e.a].x * normals[e.b].x +
                             (double)normals[e.a].y * normals[e.b].y +
                             (double)normals[e.a].z * normals[e.b].z;
        uf.unite(e.a, e.b, dot01);
    }
    for (int i = 0; i < n; ++i) {
        if (isZero(normals[i])) continue;
        uint8_t p = 0;
        uf.find(i, p);
        if (p) normals[i] = cv::Point3f(-normals[i].x, -normals[i].y, -normals[i].z);
    }
    // ---- 4) 全局符号: 多数法线背离质心 → 朝外 ----
    double gx = 0, gy = 0, gz = 0;
    for (const auto& p : pts) { gx += p.x; gy += p.y; gz += p.z; }
    gx /= n; gy /= n; gz /= n;
    double acc = 0;
    for (int i = 0; i < n; ++i)
        if (!isZero(normals[i]))
            acc += (double)normals[i].x * (pts[i].x - gx) +
                   (double)normals[i].y * (pts[i].y - gy) +
                   (double)normals[i].z * (pts[i].z - gz);
    if (acc < 0)
        for (int i = 0; i < n; ++i)
            if (!isZero(normals[i]))
                normals[i] = cv::Point3f(-normals[i].x, -normals[i].y, -normals[i].z);

    // ---- 5) 过滤零法线点后入表 ----
    std::vector<cv::Point3f> keepPts, keepNrm;
    keepPts.reserve(n);
    keepNrm.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (!isZero(normals[i])) {
            keepPts.push_back(pts[i]);
            keepNrm.push_back(normals[i]);
        }
    }
    CALIB_LOG_INFO("autoNormals: {} pts -> {} kept (knn={}, dropped={})",
                   n, keepPts.size(), knn, n - (int)keepPts.size());
    return addCloudWithNormals(keepPts, keepNrm);
}

// ------------------------------------------------------------
// Marching Cubes 抽壳（ZG vf_core.cpp 同构; 顶点=世界 mm）
// ------------------------------------------------------------
TsdfShellMesh TsdfShellFuseCpu::extractMesh() const { return extractMeshDiag(nullptr); }

TsdfShellMesh TsdfShellFuseCpu::extractMeshDiag(TsdfShellFuseCpu::McDiag* diag) const {
    TsdfShellMesh mesh;
    if (!pImpl_) return mesh;
    const float grid = params_.voxelSize;
    const int step = params_.mcCornerStep;
    const float iso = params_.isoLevel;
    // 诊断计数（diag 为空时照常累计, 结尾不回填）
    int nMiss = 0, nGate = 0, nSkipInc = 0, nGateCube = 0, nL1 = 0;

    std::vector<uint64_t> keys;
    keys.reserve(pImpl_->hash.size());
    for (const auto& kv : pImpl_->hash) keys.push_back(kv.first);
    mesh.voxelCount = (int)keys.size();

    // 顶点登记簿: 棱键 → 顶点号（同一条被切棱只登记一次——防裂缝）。
    // 【偏差 D-MC1, 显式登记】ZG 原版用"锚角"去重（顶点记在 |sdf| 较小的角上,
    // 三角形三锚撞角即整块丢弃）——在对称场（|sdf| 并列常见）下高频撞锚丢面,
    // 与本算子"水密薄壳"目标冲突; 故改用经典棱键去重（每条棱一个顶点,
    // 相邻立方体共享棱键 → 无裂缝且不丢面）。代价仅是顶点数略多。
    std::unordered_map<uint64_t, uint32_t> vid;

    auto unpack = [](uint64_t key, int32_t& x, int32_t& y, int32_t& z) {
        auto dec = [](uint64_t v) -> int32_t {
            return (v > 0x80000u) ? -(int32_t)(v & 0x7FFFFu) : (int32_t)v;
        };
        z = dec(key & 0xFFFFF);
        y = dec((key >> 20) & 0xFFFFF);
        x = dec((key >> 40) & 0xFFFFF);
    };
    // 角 sdf 读取 + 观察方向门
    auto sdfAt = [&](uint64_t ckey, const double uc[3], bool neighbor,
                     bool& found) -> double {
        auto it = pImpl_->hash.find(ckey);
        if (it == pImpl_->hash.end()) {
            ++nMiss;
            found = false;
            return 1.0;  // 未观测格读 +1（全观测立方体规则下不参与切面）
        }
        if (neighbor && params_.useCornerRayGate &&
            (uc[0] != 0.0 || uc[1] != 0.0 || uc[2] != 0.0)) {
            const double un[3] = {it->second.rx, it->second.ry, it->second.rz};
            const double ln = std::sqrt(un[0] * un[0] + un[1] * un[1] + un[2] * un[2]);
            if (!(ln <= 1e-12)) {  // 零方向跳过门（同 ZG: dot=0 恒过）
                const double dot = (un[0] * uc[0] + un[1] * uc[1] + un[2] * uc[2]) / ln;
                if (!(dot > (double)params_.cornerRayGateDot)) {
                    ++nGate;
                    found = false;
                    return 1.0;  // 观察方向差太远: 不可作角
                }
            }
        }
        found = true;
        return it->second.tsdf;
    };

    struct Vtx {
        uint64_t ekey;  // 棱键（两角键混合; 偏差 D-MC1）
        uint32_t idx;
        float x, y, z;
    };

    for (uint64_t key : keys) {
        int32_t kx, ky, kz;
        unpack(key, kx, ky, kz);
        // 锚点格点化: 只允许各轴坐标均为 step 整数倍的体素当立方体锚。
        // 否则每个体素都当锚 → 步长 step 的锚立方体互相重叠 2³=8 层,
        // 同一表面被八套错位网格各切一遍（三角数×8、壳成夹层）。
        // 无信息损失: 锚格即 0 号角, 无条目的锚立方体本就被"八角全有"规则跳过。
        {
            auto md = [](int32_t v, int s) {
                int32_t m = v % s;
                return m < 0 ? m + s : m;
            };
            if (md(kx, step) || md(ky, step) || md(kz, step)) continue;
        }
        const auto& cen = pImpl_->hash.at(key);
        double uc[3] = {cen.rx, cen.ry, cen.rz};
        const double lc = std::sqrt(uc[0] * uc[0] + uc[1] * uc[1] + uc[2] * uc[2]);
        if (lc > 1e-12) {
            uc[0] /= lc; uc[1] /= lc; uc[2] /= lc;
        } else {
            uc[0] = uc[1] = uc[2] = 0.0;
        }

        int32_t cs[8][3];
        double sdf[8];
        bool found[8], inside[8];
        int anyFound = 0;
        for (int i = 0; i < 8; ++i) {
            // kMcCornerOff 为 key 空间 {0,2} 基步长 → 按 mcCornerStep 缩放
            cs[i][0] = kx + (tsdfshell::kMcCornerOff[i][0] ? step : 0);
            cs[i][1] = ky + (tsdfshell::kMcCornerOff[i][1] ? step : 0);
            cs[i][2] = kz + (tsdfshell::kMcCornerOff[i][2] ? step : 0);
            sdf[i] = sdfAt(packKey(cs[i][0], cs[i][1], cs[i][2]), uc, i != 0, found[i]);
            anyFound += found[i] ? 1 : 0;
            inside[i] = sdf[i] < iso;
        }
        if (anyFound != 8) {
            ++nSkipInc;  // 只切八角全看过的立方体
            continue;
        }

        unsigned int ci = 0;
        for (int i = 0; i < 8; ++i)
            if (inside[i]) ci |= (1u << i);
        const unsigned int et = tsdfshell::kMcEdgeTable[ci];
        if (et == 0 || et == 0xFF || tsdfshell::kMcTriTable[ci][0] == 0xFF) {
            ++nGateCube;  // 全同号/表空——正常跳过
            continue;
        }

        auto cornerMm = [&](int i, double& x, double& y, double& z) {
            x = (cs[i][0] + 0.5) * grid;
            y = (cs[i][1] + 0.5) * grid;
            z = (cs[i][2] + 0.5) * grid;
        };
        auto vertexOnEdge = [&](int e, Vtx& v) {
            const int c0 = tsdfshell::kMcEdgeCorners[e][0];
            const int c1 = tsdfshell::kMcEdgeCorners[e][1];
            const double a = sdf[c0], b = sdf[c1];
            double t = 0.5;
            if (std::fabs(b - a) > 1e-12) t = (iso - a) / (b - a);
            t = std::max(0.0, std::min(1.0, t));
            double x0, y0, z0, x1, y1, z1;
            cornerMm(c0, x0, y0, z0);
            cornerMm(c1, x1, y1, z1);
            v.x = (float)(x0 + t * (x1 - x0));
            v.y = (float)(y0 + t * (y1 - y0));
            v.z = (float)(z0 + t * (z1 - z0));
            // 棱键 = 两角键混合（同一几何棱在相邻立方体中得同一键——防裂缝）。
            // ⚠ 次序规范化: 相邻立方体枚举同一条棱时端点次序可能相反,
            // mix 对 (a,b)/(b,a) 不对称 → 必须先按大小排序再混合。
            const uint64_t ka = packKey(cs[c0][0], cs[c0][1], cs[c0][2]);
            const uint64_t kb = packKey(cs[c1][0], cs[c1][1], cs[c1][2]);
            v.ekey = ka < kb ? mixEdgeKey(ka, kb) : mixEdgeKey(kb, ka);
            v.idx = 0;
        };
        auto registerVertex = [&](Vtx& v) {
            auto it = vid.find(v.ekey);
            if (it != vid.end()) {
                v.idx = it->second;
                // 同一棱由多个立方体重复插值, 位置相同; 对半平均为无害兜底
                auto& V = mesh.vertices;
                V[3 * v.idx] = (V[3 * v.idx] + v.x) * 0.5f;
                V[3 * v.idx + 1] = (V[3 * v.idx + 1] + v.y) * 0.5f;
                V[3 * v.idx + 2] = (V[3 * v.idx + 2] + v.z) * 0.5f;
            } else {
                v.idx = (uint32_t)(mesh.vertices.size() / 3);
                mesh.vertices.push_back(v.x);
                mesh.vertices.push_back(v.y);
                mesh.vertices.push_back(v.z);
                vid.emplace(v.ekey, v.idx);
            }
        };
        // 立方体几何朝向: winding 朝"外"(外侧角心质心 − 内侧角心质心)
        double io[3] = {0, 0, 0}, oo[3] = {0, 0, 0};
        int ni = 0, no = 0;
        for (int i = 0; i < 8; ++i) {
            double* acc = inside[i] ? io : oo;
            (inside[i] ? ni : no) += 1;
            acc[0] += cs[i][0]; acc[1] += cs[i][1]; acc[2] += cs[i][2];
        }
        const double dir[3] = {oo[0] / no - io[0] / ni, oo[1] / no - io[1] / ni,
                               oo[2] / no - io[2] / ni};

        bool cubeEmitted = false;
        for (int k = 0; k < 16; k += 3) {
            const int e0 = tsdfshell::kMcTriTable[ci][k];
            if (e0 == 0xFF) break;
            const int e1 = tsdfshell::kMcTriTable[ci][k + 1];
            const int e2 = tsdfshell::kMcTriTable[ci][k + 2];
            Vtx v0, v1, v2;
            vertexOnEdge(e0, v0);
            vertexOnEdge(e1, v1);
            vertexOnEdge(e2, v2);
            // （棱键天然互异, 无撞锚丢面——偏差 D-MC1）
            const double l1_01 = std::fabs((double)v1.x - v0.x) +
                                 std::fabs((double)v1.y - v0.y) +
                                 std::fabs((double)v1.z - v0.z);
            const double l1_02 = std::fabs((double)v2.x - v0.x) +
                                 std::fabs((double)v2.y - v0.y) +
                                 std::fabs((double)v2.z - v0.z);
            const double l1_12 = std::fabs((double)v2.x - v1.x) +
                                 std::fabs((double)v2.y - v1.y) +
                                 std::fabs((double)v2.z - v1.z);
            // 退化过滤（ZG 阈 1e-6 在格点分辨率下会误杀"顶点几乎贴角"的合法
            // 薄片三角 → 壳留 4 边界棱小洞（实测球壳 l1Drop=2 ⇔ boundary=4）。
            // 降到 1e-9：仅拦数值上完全重合的三角; 近退化薄片对法向贡献≈0 无害。
            if (l1_01 < 1e-9 || l1_02 < 1e-9 || l1_12 < 1e-9) {
                ++nL1;  // 完全退化
                continue;
            }
            registerVertex(v0);
            registerVertex(v1);
            registerVertex(v2);
            uint32_t a = v0.idx, b = v1.idx, c = v2.idx;
            const float* p0 = &mesh.vertices[3 * a];
            const float* p1 = &mesh.vertices[3 * b];
            const float* p2 = &mesh.vertices[3 * c];
            const double ux = (double)p1[0] - p0[0], uy = (double)p1[1] - p0[1],
                         uz = (double)p1[2] - p0[2];
            const double wx = (double)p2[0] - p0[0], wy = (double)p2[1] - p0[1],
                         wz = (double)p2[2] - p0[2];
            const double nx = uy * wz - uz * wy, ny = uz * wx - ux * wz,
                         nz = ux * wy - uy * wx;
            if (nx * dir[0] + ny * dir[1] + nz * dir[2] < 0) std::swap(b, c);
            mesh.indices.push_back(a);
            mesh.indices.push_back(b);
            mesh.indices.push_back(c);
            cubeEmitted = true;
        }
        if (cubeEmitted) ++mesh.cubesCut;
    }
    mesh.vertexCount = (int)(mesh.vertices.size() / 3);
    mesh.triangleCount = (int)(mesh.indices.size() / 3);
    if (diag) {
        diag->cubesVisited = (int)keys.size();
        diag->cornerMissing = nMiss;
        diag->cornerRayGate = nGate;
        diag->cubesSkippedIncomplete = nSkipInc;
        diag->cubesGate = nGateCube;
        diag->trianglesL1Dropped = nL1;
    }

    // 面法向累积归一（退化 → (0,0,1)）
    const size_t nt = mesh.indices.size() / 3;
    mesh.normals.assign(mesh.vertices.size(), 0.0f);
    for (size_t i = 0; i < nt; ++i) {
        const uint32_t a = mesh.indices[3 * i], b = mesh.indices[3 * i + 1],
                       c = mesh.indices[3 * i + 2];
        const float ax = mesh.vertices[3 * b] - mesh.vertices[3 * a];
        const float ay = mesh.vertices[3 * b + 1] - mesh.vertices[3 * a + 1];
        const float az = mesh.vertices[3 * b + 2] - mesh.vertices[3 * a + 2];
        const float bx = mesh.vertices[3 * c] - mesh.vertices[3 * a];
        const float by = mesh.vertices[3 * c + 1] - mesh.vertices[3 * a + 1];
        const float bz = mesh.vertices[3 * c + 2] - mesh.vertices[3 * a + 2];
        const float nx = ay * bz - az * by;
        const float ny = az * bx - ax * bz;
        const float nz = ax * by - ay * bx;
        for (uint32_t v : {a, b, c}) {
            mesh.normals[3 * v] += nx;
            mesh.normals[3 * v + 1] += ny;
            mesh.normals[3 * v + 2] += nz;
        }
    }
    for (size_t v = 0; v < mesh.normals.size() / 3; ++v) {
        const float nl = std::sqrt(mesh.normals[3 * v] * mesh.normals[3 * v] +
                                   mesh.normals[3 * v + 1] * mesh.normals[3 * v + 1] +
                                   mesh.normals[3 * v + 2] * mesh.normals[3 * v + 2]);
        if (nl < 1e-20f) {
            mesh.normals[3 * v] = 0;
            mesh.normals[3 * v + 1] = 0;
            mesh.normals[3 * v + 2] = 1;
        } else {
            mesh.normals[3 * v] /= nl;
            mesh.normals[3 * v + 1] /= nl;
            mesh.normals[3 * v + 2] /= nl;
        }
    }
    return mesh;
}

// ------------------------------------------------------------
// 调试探针（见头文件）
// ------------------------------------------------------------
bool TsdfShellFuseCpu::probeVoxel(double x, double y, double z, float& tsdf,
                                  float& weight, float ray[3]) const {
    if (!pImpl_) return false;
    const double grid = params_.voxelSize;
    const int cx = (int)std::floor(x / grid);
    const int cy = (int)std::floor(y / grid);
    const int cz = (int)std::floor(z / grid);
    auto it = pImpl_->hash.find(packKey(cx, cy, cz));
    if (it == pImpl_->hash.end()) return false;
    tsdf = it->second.tsdf;
    weight = it->second.weight;
    ray[0] = it->second.rx;
    ray[1] = it->second.ry;
    ray[2] = it->second.rz;
    return true;
}

// ------------------------------------------------------------
// 云→网格最近顶点距离统计（网格分桶近似; 量化薄壳化效果）
// ------------------------------------------------------------
TsdfShellStats computeCloudToMeshStats(const TsdfShellMesh& mesh,
                                       const std::vector<cv::Point3f>& pts,
                                       const TsdfShellStats& stats, float voxel) {
    TsdfShellStats out = stats;
    const size_t nv = mesh.vertices.size() / 3;
    if (nv == 0 || pts.empty()) return out;
    const float w = std::max(voxel, 1e-3f);
    std::unordered_map<uint64_t, std::vector<uint32_t>> bins;
    bins.reserve(nv * 2);
    for (uint32_t v = 0; v < nv; ++v) {
        const int bx = (int)std::floor(mesh.vertices[3 * v] / w);
        const int by = (int)std::floor(mesh.vertices[3 * v + 1] / w);
        const int bz = (int)std::floor(mesh.vertices[3 * v + 2] / w);
        bins[packKey(bx, by, bz)].push_back(v);
    }
    std::vector<double> dists;
    dists.reserve(pts.size());
    for (const auto& p : pts) {
        const int bx = (int)std::floor(p.x / w);
        const int by = (int)std::floor(p.y / w);
        const int bz = (int)std::floor(p.z / w);
        double best = -1.0;
        for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    auto it = bins.find(packKey(bx + dx, by + dy, bz + dz));
                    if (it == bins.end()) continue;
                    for (uint32_t v : it->second) {
                        const double ddx = (double)mesh.vertices[3 * v] - p.x;
                        const double ddy = (double)mesh.vertices[3 * v + 1] - p.y;
                        const double ddz = (double)mesh.vertices[3 * v + 2] - p.z;
                        const double d2 = ddx * ddx + ddy * ddy + ddz * ddz;
                        if (best < 0 || d2 < best) best = d2;
                    }
                }
        if (best >= 0) dists.push_back(std::sqrt(best));
    }
    if (dists.empty()) return out;
    double sum = 0;
    for (double d : dists) sum += d;
    out.meanPointToMesh = sum / dists.size();
    out.maxPointToMesh = *std::max_element(dists.begin(), dists.end());
    const size_t k95 = (size_t)std::min<double>((double)(dists.size() - 1),
                                                std::ceil(0.95 * (double)dists.size()));
    std::nth_element(dists.begin(), dists.begin() + k95, dists.end());
    out.p95PointToMesh = dists[k95];
    return out;
}

OperatorInfo getTsdfShellFuseCpuInfo() {
    return OperatorInfo{"TsdfShellFuseCpu", SCANNER_VERSION_MAJOR, SCANNER_VERSION_MINOR,
                        OperatorType::CPU};
}

}  // namespace calib
