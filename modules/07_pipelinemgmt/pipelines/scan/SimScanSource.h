#pragma once
// ============================================================================
// SimScanSource.h — 调试件：中段模拟提取源（260917 架构对齐版）
//
// 用途：真机前端＋生产链主干的中段数据替换——流水线 gpu/p 图像链照常运行，
// 每帧的「标志点/激光 3D 提取结果」由本源产出模拟观测（SimFrameObs，设备系）：
//   - 场景（全局系真值）：标志点集（D:/markers_30.ply 或 builtin）＋激光点云
//     （app「导入点云」stash 或 builtin）；
//   - 轨迹 G_k = T(k·t)·R(k·θ)：设备位姿逐帧小步递增（标志点集合整体移动）；
//   - 观测 = G_k⁻¹·真值＋噪声：标志点逐点高斯 σ（≈检测重建误差）；激光 =
//     视场内子集 + 逐点高斯抖动 σ（≈逐帧部分表面观测）；
//   - **不配准**：R/T 由生产链 ScanChains::runRegistration 对全局 prevState
//     锚估计（与真机完全同路径）；真值位姿仅供日志/测试对照（lastTruePose）。
//
// 线程模型：next() 自带互斥（runtime 多 lane 的 gpuChain 可能并发调用——
// 与旧注入线程单线程独占不同）。其余方法非线程安全。
// 落位：编入 mod_pipelinemgmt；调试件，默认不参与生产链（app 侧 UI 开关注入）。
// ============================================================================
#include <cstdint>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "base/types.h"
#include "core/marker/marker_cloud_fuse_cpu/marker_cloud_fuse_cpu.h"  // calib::MarkerCloudPoint
#include "pipelines/scan/ScanTypes.h"                                  // SimFrameObs

namespace Scanner::pipeline {

// —— 场景（全局系真值）——
struct SimScene {
    std::vector<calib::MarkerCloudPoint> markers;  // normal/whiteRadius 缺省填充
    std::vector<cv::Point3f> laser;                // 可为空（builtin fallback 或无点云）

    /// 内置场景（固定 seed 可复现；测试用——不依赖外部文件）：
    /// 30 标志点（6×5 网格 @30mm，z=-400 平面，贴近 markers_30.ply 量纲）
    /// ＋ 6000 激光点（200×200mm 浅抛物面片，模拟表面采样）
    static SimScene builtin(uint32_t seed = 42);

    /// 标志点从 PLY 读入（06 fileio::importPLY；仅 xyz，normal 缺省 (0,0,1)、
    /// whiteRadius=1.5、globalId=下标）。laser 字段不触碰（调用方另行填充）。
    static Scanner::Result loadMarkersFromPly(const std::string& path, SimScene& out);
};

// —— 轨迹与抖动参数 ——
struct SimTrajParams {
    double rotDegPerFrame = 0.3;      // 整体移动：绕 axis 每帧转角（deg）
    double transMmPerFrame = 0.5;     // 整体移动：每帧平移（mm，沿 x）
    cv::Vec3d rotAxis{0.0, 1.0, 0.0}; // 旋转轴（内部归一化）
    double markerNoiseSigmaMm = 0.05; // 标志点检测噪声 σ（≈图像链重建误差；0=无噪声
                                      //   对照组——精确断言用例须置 0）
    double jitterSigmaMm = 0.05;      // 激光逐点抖动 σ（< marker voxel 0.5mm 的 1/5）
    size_t laserPerFrame = 100000;    // 每帧激光子采样上限（大点云防逐帧全量上传；
                                      //   M 小于该值则全量。逐帧重抽≈部分表面观测）
    uint64_t maxFrames = 600;         // 帧数上限（默认 600 帧＝20s@30fps）

    // —— 视场观测窗（部分观测——设备只看得见视场内的点，复刻真实扫描
    //    「扫到哪看到哪」；全 0=不限（全场景每帧可见——精确断言/恒等对照组用）。
    //    设备系深度=−z；FOV 判定 |x|,|y| ≤ 深度·tan(半角) ——
    double fovHalfHDeg = 0.0;         // 水平半视场（0=不限）
    double fovHalfVDeg = 0.0;         // 垂直半视场（0=不限）
    double depthMinMm = 0.0;          // 观测距离下限（0=不限）
    double depthMaxMm = 0.0;          // 观测距离上限（0=不限）
    // 纵向扫描行程：视场窗中心沿 y 从场景包围盒 yMin 匀速扫到 yMax（须配合
    // fovHalfVDeg）——渐进覆盖：标志点/激光逐帧进入视场，融合云逐渐增长
    bool sweepY = false;

    // —— 递增观测调度（260919 用户定版「模拟真实扫描过程」）：标志点按 y 排序
    //    后滑动窗口——帧0 观测 markerWindowStart 个；每 markerStepFrames 帧为一步，
    //    窗口前移 markerAdvancePerStep（与上一步重合 W-advance 个标志点——重叠
    //    驱动配准的受控版：8 个→9 个其中 ~7 个重合→逐步 8→9→10…→N 逐帧增加）；
    //    激光观测＝当帧窗口覆盖区域子集（区域门控）。本模式下 FOV 锥不再门控
    //    标志点/激光（观测判据=窗口/区域+深度窗），sweepY 应关（sweep 速度超
    //    光流匹配阈 2mm 曾致配准逐帧全败）。
    //    markerStepFrames=0（缺省）＝旧 FOV 视场窗观测（既有测试口径不变）
    uint64_t markerStepFrames = 0;     // 每步停留帧数（0=关闭递增调度）
    size_t markerWindowStart = 8;      // 起始窗口大小（帧0 观测标志点数）
    size_t markerGrowEvery = 2;        // 每 N 步窗口 +1（观测数逐帧增加的节奏）
    size_t markerAdvancePerStep = 1;   // 每步窗口前移数（与上一步重合 W-advance）
};

// —— 逐帧源（设备系原始观测；无配准——R/T 归生产链）——
class SimScanSource {
public:
    SimScanSource(SimScene scene, SimTrajParams params = {});

    void reset();                     // 帧计数/随机态归零（场景不变）
    /// 产下一帧观测；false＝已到 maxFrames 或场景全空（out 置空）
    bool next(SimFrameObs& out);
    uint64_t generated() const { return frame_; }
    /// 最近一帧轨迹真值位姿（行主序 R/3 平移；精度对照——测试/日志用）
    void lastTruePose(double R[9], double T[3]) const;

private:
    SimScene scene_;
    SimTrajParams params_;
    uint64_t frame_ = 0;              // 已产帧数（G_k 的 k）
    std::vector<uint32_t> laserIdx_;  // 激光洗牌池（legacy FOV 模式游标环扫用）
    size_t laserCursor_ = 0;          // 池游标（每帧前进 scanBudget，跨帧覆盖全池）
    // 行分桶索引（260919 提速：subset 模式取行——30M 池随机游标探测 ~150ms/帧
    // ×4 lane 互斥串行=600ms/帧=7fps 根因；按全局 y 2.5mm 行分桶后取行只读
    // 选中桶（~3.5 万顺序读）<1ms。桶内下标→scene_.laser；构造期建（场景不变）
    std::vector<std::vector<uint32_t>> rowBuckets_;
    long rowBucketBase_ = 0;          // 桶 0 对应的全局行号（floor(y/2.5)）
    size_t bucketCursor_ = 0;         // 桶内取数起点（帧间轮转——配额摊到全部
                                      // 选中行，防首桶填满=覆盖塌缩）
    std::vector<size_t> markerOrder_; // 标志点按 y 升序下标（递增调度滑动窗用）
    std::mt19937 rng_;
    mutable std::mutex mtx_;          // next() 多 lane 并发互斥（gpuChain 调用方）

    double trueR_[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    double trueT_[3] = {0, 0, 0};

    // —— 场景包围盒（sweepY 行程拟合用；ctor 一次算好）——
    double sceneYMin_ = 0.0;
    double sceneYMax_ = 0.0;
    double sceneMeanDepth_ = 400.0;              // −z 均值（FOV 窗宽估算基准）
};

} // namespace Scanner::pipeline
