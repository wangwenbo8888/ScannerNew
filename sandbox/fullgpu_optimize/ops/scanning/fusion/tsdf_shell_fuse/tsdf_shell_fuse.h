#pragma once
// ============================================================================
// tsdf_shell_fuse.h — TSDF 距离场融合 + Marching Cubes 抽薄壳（备用算子, CPU）
// ============================================================================
// 【定位】对标软件（ZG RigelScan）融合思路的自研重构备用算子：
//   ① 体素不再存"代表点"（laser_cloud_fuse_cuda 现行制），改存"离最近表面多远"
//      （TSDF：截断带符号距离场, 归一化 [-1,1], 截断=truncFactor×格宽）；
//   ② 写入规则＝沿观察方向"泼溅"（球窗 tau + 横向 1.5 格）＋**运行加权平均**
//      （偏差 D-T3：ZG 原版 min|sdf|"就近者胜"依托逐帧稠密积分, 本算子逐点
//      稀疏泼溅下 min 逐格挑单点噪声→表面符号逐格翻转→海绵碎片; 平均使噪声
//      √N 递减, "双面抵消"顾虑由 MC 角观察方向门拦截——详见 cpp 注释）；
//   ③ 关键帧门（逐帧模式）：位移² > 格宽/15 或传感器 x 轴夹角 > 0.05° 才融合
//      （ZG 常数, 复现工程指令级修正：0.05 是"度"不是弧度）；
//   ④ Marching Cubes 抽壳：经典 Lorensen-Cline 查表（表值取自 ZG DLL 逐字节
//      提取版）；只切"八角全看过"的立方体（防覆盖边界假面）；角间观察方向门
//      （dot > -0.5, 防正反两面缝成假面）；顶点按"棱键"去重（同一条被切棱
//      只登记一个顶点, 相邻立方体共享 → 无裂缝; 偏差 D-MC1——ZG 原版"锚角"
//      规则在对称场下撞锚丢面, 与水密目标冲突, 详见 cpp 注释）；退化三角过滤；
//      面法向累积归一。
// 目的：把"有厚度的点云"（激光双面观测/噪声/抖动叠加）抽出一张薄壳面。
//
// 【证据源】本仓 ZG 复现工程（数学等价, ctest 全绿）：
//   prodline_benchmark/ZG/operator_porting/voxel_fusion_meshing/src/{vf_impl,vf_core}.cpp
//   prodline_benchmark/ZG/operator_porting/edge_fusion/src/core/edf_core.h（pack20 键）
//
// 【与现行主路的关系】备用算子——不接 ScanPipeline 主链, 由 tools/tsdf_shell_extract
//   与单测直接消费；laser_cloud_fuse_cuda（代表点制）行为零改动。
// ============================================================================
#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common/calib_types.h"
#include "common/scanner_api.h"
#include "common/version.h"

namespace calib {

// ------------------------------------------------------------
// 参数（全部有 ZG 证据或显式缺省, JSON 可往返）
// ------------------------------------------------------------
struct TsdfShellFuseParams {
    float voxelSize     = 0.5f;   // mm, 体素格宽
    float truncFactor   = 2.0f;   // 截断距离 = truncFactor×格宽（ZG 设计 D-7: 2.0）
    float lateralFactor = 1.5f;   // 泼溅横向半窗 = lateralFactor×格宽（ZG: 1.5）

    // 关键帧门（仅 addFrame 逐帧模式生效; ZG 常数）
    bool  useKeyframeGate = true;
    float keyframeDispDiv = 15.0f;  // 位移² > 格宽/15 → 融合（1.6.0 版 Options 缺省; 旧版 1/16）
    float keyframeRotDeg  = 0.05f;  // 传感器 x 轴行夹角 > 0.05° → 融合（度! 非 弧度）

    // Marching Cubes
    // 【偏差 D-T4】锚步长缺省 1（ZG 实证为 2＝立方体棱长·τ）。ZG 的角步 2 依托
    // 逐帧稠密积分（±τ 带内全格覆盖, 远角必有数据）; 本算子逐点稀疏泼溅的覆盖
    // 带随深度横向收窄, 角步 2 的立方体远角(离锚 2 格)越过覆盖带 → 只有锚落在
    // ~0.1mm 窗内的立方体完整 → 壳大量开洞。步长 1 时立方体棱长=格宽, 远角
    // 必在带内, 壳闭合; 网格分辨率同步提高到格宽级。
    int   mcCornerStep     = 1;
    float isoLevel         = 0.0f;   // 等值面（sdf=0 即表面）
    bool  useCornerRayGate = true;   // 角间观察方向门（dot > cornerRayGateDot 才可用作角）
    float cornerRayGateDot = -0.5f;  // ≈120° 容限（ZG 常数）

    // addCloudAutoNormals 的法向估计
    int normalKnn = 16;              // k 近邻 PCA + 最大生成树定向传播

    void validate() const;
    nlohmann::json toJson() const;
    static TsdfShellFuseParams fromJson(const nlohmann::json& j);
};

// ------------------------------------------------------------
// 产物：薄壳网格（世界系 mm）＋ 计数
// ------------------------------------------------------------
struct TsdfShellMesh {
    std::vector<float>    vertices;   // xyz×N（mm, 世界系）
    std::vector<float>    normals;    // xyz×N（面法向累积归一; 退化→(0,0,1)）
    std::vector<uint32_t> indices;    // 三角形顶点号×3

    int   voxelCount    = 0;   // 有效体素数
    int   cubesCut      = 0;   // 实际切了三角形的立方体数
    int   vertexCount   = 0;
    int   triangleCount = 0;
};

// 运行统计（toJson 可落盘）
struct TsdfShellStats {
    int    integratedPoints = 0;   // 实际进入泼溅的样本点数
    int    fusedFrames      = 0;   // 实际融合的帧数（关键帧门后）
    int    skippedFrames    = 0;   // 被关键帧门跳过的帧数
    int    droppedPoints    = 0;   // 坐标超 20bit 键域等被丢弃的点
    size_t voxelCount       = 0;
    int    cubesCut         = 0;
    int    triangleCount    = 0;
    int    vertexCount      = 0;
    // 云→网格距离（mm; 由 computeCloudToMeshStats 填, 量化"抽壳后残厚"）
    double meanPointToMesh  = -1.0;
    double p95PointToMesh   = -1.0;
    double maxPointToMesh   = -1.0;

    nlohmann::json toJson() const;
};

// ------------------------------------------------------------
// 算子本体
// ------------------------------------------------------------
// ===== 算子规范 §4 状态模型 =====
// 状态类别: 自持久化
// 说明: 实例内部维护 TSDF 体素哈希累积器, 跨 Add*/add* 调用累积并影响 extractMesh
//       结果, 通过 Clear() 重置。
// 重置接口: Clear()
// 并发策略: 每实例非线程安全, 多实例并行各自独占
// ==============================
class SCANNER_API TsdfShellFuseCpu {
public:
    static constexpr const char* kLogTag = "02T-TsdfShellFuseCpu";

    explicit TsdfShellFuseCpu(const TsdfShellFuseParams& params = {});
    ~TsdfShellFuseCpu();

    TsdfShellFuseCpu(const TsdfShellFuseCpu&) = delete;
    TsdfShellFuseCpu& operator=(const TsdfShellFuseCpu&) = delete;

    // —— 输入模式 A：逐帧点云＋位姿（ZG PointFusion 同形态）——
    // localPts: 本机系 3D 点; R/T: 本机系→全局系位姿（世界 = R·p + T, 传感器位于 T）。
    // 观察方向 = unit(R·p)（传感器→点）。首帧必融; 之后按关键帧门跳帧。
    bool addFrame(const std::vector<cv::Point3f>& localPts,
                  const cv::Matx33d& R, const cv::Vec3d& T);

    // —— 输入模式 B：单朵全局云＋法线 ——
    // normals 须朝外（朝传感器侧）; 观察方向取 -normal（传感器在法线一侧照射）。
    bool addCloudWithNormals(const std::vector<cv::Point3f>& pts,
                             const std::vector<cv::Point3f>& normals);

    // —— 输入模式 C：单朵全局云无法线 ——
    // 内部 kNN-PCA 估法向（normalKnn）→ 最大生成树定向传播（保一致性）→
    // 全局符号取"多数法线背离质心"（闭壳朝外）。数据量自适应桶宽。
    bool addCloudAutoNormals(const std::vector<cv::Point3f>& pts);

    // 抽壳（幂等: 不改累积器, 可重复调用）; diag 非空时回填诊断直方图
    TsdfShellMesh extractMesh() const;
    struct McDiag {
        int cubesVisited = 0;      // 迭代的体素数
        int cornerMissing = 0;     // 角格子无条目次数
        int cornerRayGate = 0;     // 角被观察方向门拒绝次数
        int cubesSkippedIncomplete = 0;  // 八角不全被跳过的立方体数
        int cubesGate = 0;         // 边表/三角表门槛跳过（全同号等, 正常）
        int trianglesL1Dropped = 0;  // L1 退化过滤丢弃的三角形数
    };
    TsdfShellMesh extractMeshDiag(McDiag* diag) const;

    const TsdfShellStats& stats() const noexcept { return stats_; }

    // 调试探针: 查询世界坐标 (x,y,z) 处体素的 tsdf/weight/观察方向。
    // 无条目返回 false（诊断/测试用; 非生产接口）。
    bool probeVoxel(double x, double y, double z, float& tsdf, float& weight,
                    float ray[3]) const;

    void Clear() noexcept;
    void SetParams(const TsdfShellFuseParams& params);
    const TsdfShellFuseParams& params() const noexcept { return params_; }
    void Destroy();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
    TsdfShellFuseParams params_;
    TsdfShellStats stats_;
};

// 云→网格最近顶点距离统计（网格分桶, 近似最近顶点而非最近面; 量化"薄壳化"效果）。
// 返回填好三个距离字段的 stats 副本; mesh 为空时原样返回。
TsdfShellStats computeCloudToMeshStats(const TsdfShellMesh& mesh,
                                       const std::vector<cv::Point3f>& pts,
                                       const TsdfShellStats& stats,
                                       float voxel);

OperatorInfo getTsdfShellFuseCpuInfo();

}  // namespace calib
