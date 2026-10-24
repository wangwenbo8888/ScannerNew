#pragma once
// ============================================================================
// harness.h — 链装配＋执行＋计时＋dump（scan_chain_replay 核心）
// ============================================================================
// 设计：docs/plans/2026-10-04-scan-chain-replay-test-design.md §5/§6
// 接线依据：docs/plans/2026-10-04-scan-chain-replay-impl.md 算子接线速查表
// （全部签名经算子头文件核验）。单线程顺序逐 cycle；单 cv::cuda::Stream。
// 计时：算子内置 timings 优先（mask_separation），外部 steady_clock＋stream
// 同步兜底；warmupCycles 个预热 cycle 不计时。
#include <chrono>
#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>

#include "calib_loader.h"
#include "frame_source.h"
#include "scr_config.h"
#include "triangle_index.h"

#include <memory>
#include <thread>

// —— 09 算子（mod_operatorlib PUBLIC include 可达）——
#include "scanning/preprocess/mask_separation/laser_markingpoint_mask_separation_cuda.h"
#include "core/vision/ccl/region_analyze_cuda.h"
#include "core/marker/image_split/image_split_cpu.h"
#include "core/marker/zernike_edge/zernike_edge_cpu.h"
#include "core/marker/image_merge/image_merge_cpu.h"
#include "core/marker/undistort_cpu/undistort_points_cpu.h"
#include "core/marker/ellipse_fit/ellipse_fit_cpu.h"
#include "core/marker/marker_match/marker_match_cpu.h"
#include "core/marker/conic_reconstruct/conic_reconstruct_cpu.h"
#include "core/marker/optical_flow_fuse/marker_optical_flow_fuse_cpu.h"
#include "core/marker/frame_fuse/frame_fuse_cpu.h"
#include "core/common/steger_types.h"
#include "core/laser/steger_fast/steger_fast.h"
#include "core/laser/flat_center/flat_center.h"
#include "core/laser/undistort_cuda/undistort_points_cuda.h"
#include "core/laser/epipolar_interp_dual/epipolar_interp_dual_cuda.h"
#include "scanning/laser/laser_match_scan_v3/laser_match_scan_v3_cuda.h"
#include "core/laser/laser_reconstruct/laser_reconstruct_cuda.h"
#include "scanning/fusion/laser_cloud_fuse_cuda/laser_cloud_fuse_cuda.h"

namespace scr {
namespace fc { class FiveChainExecutor; }   // 【五链 P2】前置声明（friend 用）

/// 规则网格板检测（2026-10-15 裁决：网格板平移对称＝原理性不可分辨——错一格时
/// 每点完美匹配邻格点、全防线穿透且永不报警，宁拒勿错）。
/// 判据：≥8 点且各点最近邻距离变异系数 < cvTol。校准依据（ALL 399 帧全量实测）：
/// 真实随机板 CV 分布 min=0.0179 / P50=0.138（可见子集偶发整齐）；真网格
/// （±0.5mm 噪声/100mm 格距）CV≈0.005——默认阈值 0.012 两侧各留 ≥1.4 倍余量，
/// 且 <8 点不判（小样本子集易偶发整齐，实测唯一 CV<0.02 帧恰为 6 点子集）。
/// 返回 true＝检测到规则阵列。
bool detectRegularGrid(const std::vector<cv::Point3d>& pts, double cvTol = 0.012);

/// 单 cycle（＝单帧，同帧模式）产物（内存持有；GpuMat 经 shared_ptr 生命周期安全）
struct CycleProducts {
    // —— 本帧输入（同帧：激光链与标记点链共用）——
    cv::Mat frameGrayL, frameGrayR;

    // —— 同帧掩膜分离（一次执行，双链分叉消费）——
    // 2026-10-04 算子细激光形态重设计后：原生出激光/标记双掩膜，harness 直接
    // 消费 d_laserMask / d_markingPointMask（历史组合掩膜补丁已退役）
    calib::LaserMarkingSeparationResult sepL, sepR;

    // —— 激光链（吃同帧 d_laserMask）——
    calib::StegerResult stegerL, stegerR;
    calib::UndistortPointsResult undistCudaL, undistCudaR;
    calib::EpipolarInterpResult interpL, interpR;
    calib::LaserMatchScanResultV3 matchV3;
    calib::LaserReconstructResult recon;
    // —— 标记点链（吃同帧 d_markingPointMask）——
    calib::RegionAnalysisResult cclL, cclR;
    calib::ImageSplitCPUResult splitL, splitR;
    std::vector<calib::ZernikeEdgeCPUResult> zernL, zernR;
    calib::ImageMergeCPUResult mergeL, mergeR;
    calib::StereoUndistortResult undistCpu;
    std::vector<calib::EllipseFitCPUResult> ellL, ellR;
    calib::MarkerMatchCPUResult mm;
    calib::ConicReconstructCPUResult conic;
    // —— 配准 ——
    calib::MarkerOpticalFlowFuseCPUResult regFlow;
    bool usedFrameFuse = false;
    bool fuseSlowPath = false;   // 兜底经步5 三角索引慢路径（诊断：预测失效频率）
    calib::FrameFuseCPUResult regFuse;
    /// flow|fuse|hold|first（hold=全失败沿用上帧。relock 已按 2026-10-15 人工裁决删除）
    std::string regSource;
    /// 【E3】fivechain 语义：1b/chain1 流已排空（含 evGate 传递）→ RLE 环 D2H 宿主
    /// 可见，拷出免事件同步。serial 置真无害（无消费者）；pipe 置真但语义不可用
    /// （lane 流≠stream_）——pipe 侧严禁消费（R2-B 裁定，注释钉死）
    bool gpuDrained = false;
    double R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    double T[3] = {0, 0, 0};
    // —— 配准诊断（summary.json flowDiag 节；断链可诊断）——
    int flowMatchedDiag = -1;
    double flowRmseDiag = -1.0;
    std::string flowMessageDiag;

    // —— 计时与步骤状态 ——
    std::vector<std::pair<std::string, double>> stepMs;   // name → ms（顺序）
    std::map<std::string, std::string> stepSource;        // internal|external
    std::map<std::string, std::string> stepError;         // 失败步骤 → 消息
    std::map<std::string, bool> stepSuccess;

    // —— 统计（report.csv / 断言用）——
    int markersDetected = 0;     // ellipse 拟合成功数
    int markersValid = 0;        // conic_reconstruct 有效标记点（空间圆）
    int regMatched = 0;
    int laserMatched = 0;
    int laserPoints3D = 0;
    double rowResidMedPx = -1.0; // 激光配对行对齐残差中位数（矫正系极线自洽）
    double dispMedPx = -1.0;     // 视差中位数
};

class Harness {
public:
    Harness(const ScrConfig& cfg, const CalibData& calib, const FrameSource& src,
            std::string& err);
    ~Harness();   // 全算子 Destroy()

    // 【150FPS·方案 D】标记点链算子组：09 CPU 算子非线程安全——串行路径一组，
    // 流水线每 lane 一组（多 lane 并行处理不同帧的完整标记点链）
    struct MarkerOps {
        std::unique_ptr<calib::ImageSplitCPU> splitOp_;
        std::unique_ptr<calib::ZernikeEdgeCPU> zernOp_;
        std::unique_ptr<calib::ImageMergeCPU> mergeOp_;
        std::unique_ptr<calib::MarkerUndistortCPU> undistCpuOp_;
        std::unique_ptr<calib::EllipseFitCPU> ellOp_;
        std::unique_ptr<calib::MarkerMatchCPU> mmOp_;
        std::unique_ptr<calib::ConicReconstructCPU> conicOp_;
    };    /// 执行第 cycleIdx 个 cycle（0 基）。计时随 recordTiming。
    bool runCycle(int cycleIdx, CycleProducts& out, bool recordTiming, std::string& err);
    /// 双链矫正系一致性检查（首个标记点 cycle 后调用一次）：
    /// undistCpu 输出（应回显注入的温表档）vs CalibData 档值 max|Δ|
    double checkConsistency(const CycleProducts& p, std::string& detail) const;

    /// 产物落盘：full=true 全量；false 仅 summary.json
    void dumpCycle(int cycleIdx, const CycleProducts& p, bool full) const;
    /// 步骤筛选版（--step）：onlySteps 非空时仅落所选步骤的产物（summary 恒写；
    /// 前置仍内存执行——见设计 §6 单步模式 v1 口径）
    void dumpCycleFiltered(int cycleIdx, const CycleProducts& p, bool full,
                           const std::vector<std::string>* onlySteps) const;

    /// 首帧（配准全局系）是否已初始化
    bool registrationInitialized() const { return !prevState_.empty(); }

    const std::string& outDir() const { return outDir_; }
    /// prev R/T（供平滑性断言；未初始化返回 false）
    bool lastRT(double R[9], double T[3]) const;
    /// 全局标记点地图（配准后的完整三维点——全局坐标系；供链外导出 PLY）
    const calib::GlobalMarkerSet& globalMarkerSet() const { return globalSet_; }
    /// 端到端 P0：累积全局激光点云（每帧激光 3D × R/T → 全局系，含 hold 帧沿用值）
    void accumulateGlobalLaser(const CycleProducts& p);
    /// 端到端 P0：导出全局激光点云 PLY（返回点数；空返 0）
    size_t saveGlobalLaserPly(const std::string& path) const;
    /// 端到端 P3：库健康度（库大小 / 候选池大小 / 写库版本号）
    size_t librarySize() const { return globalSet_.positions.size(); }
    size_t candidatePoolSize() const { return candidates_.size(); }
    size_t libraryVersion() const { return libVersion_; }
    /// 【五链 P1】体素融合统计（C5 定容/C1 耗时对账源）
    size_t fuseVoxelCount() const { return voxelFuseOp_ ? voxelFuseOp_->GetVoxelCount() : 0; }
    size_t fuseFusedPointCount() const { return voxelFuseOp_ ? voxelFuseOp_->GetFusedPointCount() : 0; }
    size_t fuseInputTotal() const { return fuseInputTotal_; }
    size_t fuseFrameCount() const { return fuseFrameCount_; }
    size_t fuseRehashCount() const { return fuseRehashCount_; }
    size_t fuseOverloadCount() const { return fuseOverloadCount_; }
    /// 【五链 P2】通道对账计数（0=读 1=上传 2=A 3=B 4=C 5=D 6=E 7=F；无执行器返 -1）
    int fcChCount(int i) const;
    int fcTombCount() const;

private:
    // —— 【利用率专项】激光链算子组化：双收尾 lane 需双实例（09 算子非线程
    //    安全）。组内含 steger/flat/undist/interp/v3/recon 十算子；d_gray/stream_
    //    仍共享（单流 FIFO＋laserFed 闸门保覆写序）
    struct LaserOps {
        std::unique_ptr<calib::StegerExtractorFast> stegerL_, stegerR_;
        std::unique_ptr<calib::FlatCenterExtractor> flatL_, flatR_;
        std::unique_ptr<calib::UndistortPointsCuda> undistCudaL_, undistCudaR_;
        std::unique_ptr<calib::EpipolarInterpDualCuda> interpL_, interpR_;
        std::unique_ptr<calib::LaserMatchScanCudaV3> matchOp_;
        std::unique_ptr<calib::LaserReconstructCuda> reconOp_;
        void destroy();
    };
    // 【利用率专项·lane 解耦】d_gray 每收尾 lane 独立（闸门 fed≥N-kLaserN＋
    // 单流 FIFO 保证本 lane 前帧 steger 先执行——单缓冲即可）；串行用 [0]
    std::vector<cv::cuda::GpuMat> d_grayLanesL_, d_grayLanesR_;
    void ensureGrayLanes(int lanes);           // 惰性建（Harness 构造建 [0]）
    /// 同帧分离（一次执行，双链分叉消费）；laneIdx＝上传目标缓冲（串行传 0）
    void runSeparation(const cv::Mat& grayL, const cv::Mat& grayR, CycleProducts& p,
                       bool record, int laneIdx = 0);
    void runLaserChain(CycleProducts& p, bool record, bool tubeOdd);
    /// 【利用率专项·收尾线程】激光链两段拆分：steger 段（读 d_gray＋分离内部
    /// 激光掩膜——跨线程覆写敏感区，入队后须放行 laserFed 闸门）／收尾段
    /// （undist→interp→v3→recon→行残差统计——算子仅收尾线程触碰，天然串行）。
    /// 串行路径仍走 runLaserChain（＝两段顺序连调，行为逐字不变）。
    /// L＝本 lane 算子组（双 lane 双实例；串行走 laserOps_）；laneIdx＝d_gray 环；
    /// ls＝本 lane 专用流（多流解耦：lane 内全流同步只排空自己——串行走 stream_）
    void runLaserSteger(LaserOps& L, CycleProducts& p, bool record, bool tubeOdd,
                        int laneIdx, cv::cuda::Stream& ls);
    void runLaserTail(LaserOps& L, CycleProducts& p, bool record, bool tubeOdd,
                      cv::cuda::Stream& ls);
    /// 【E4/E5·五链专用】激光前段提交（零宿主同步）：flat ExecuteDeferred（容量宽
    /// 视图＋stats D2H 在飞）→ undistort（容量网格——1:1 映射，尾区无害多算 ~μs）
    /// → interp ExecuteDeferred（设备计数界＋键尾戳＋计数 D2H→pinned）。
    /// 返回：0=未提交（掩膜空/异常/提交失败——失败已记账，无在飞 GPU 工作）；
    /// 1=已提交·延迟完成（**须排空流后调 finishLaserFront**）；
    /// 2=已提交·legacy（steger/dedup-cpu 路径——内部已同步，finish 段轻量直通）
    int submitLaserFront(LaserOps& L, CycleProducts& p, bool record, bool tubeOdd,
                         int laneIdx, cv::cuda::Stream& ls);
    /// 【E4/E5】排水后完成段（kind＝submitLaserFront 返回值）：FinishDeferred×2＋
    /// FinishInterp×2（延迟时）＋零点检查（消息口径逐字保留）＋步骤记账
    void finishLaserFront(LaserOps& L, CycleProducts& p, bool record, int kind);
    /// 标记点链（掩膜经参数注入：串行传 sep 原视图＋evSepDone_；流水线传 D2D
    /// 快照＋per-job 就绪事件——09 算子非线程安全，ops 逐 lane 独立实例组）
    void runMarkerChain(CycleProducts& p, bool record, MarkerOps& ops,
                        const std::shared_ptr<cv::cuda::GpuMat>& maskL,
                        const std::shared_ptr<cv::cuda::GpuMat>& maskR,
                        cudaEvent_t evMaskReady, cv::cuda::Stream& dlStream);
    /// 【150FPS】行程直入版（GPU-RLE 产物；标记点 lane 零 CUDA）
    void runMarkerChainRuns(CycleProducts& p, bool record, MarkerOps& ops,
                            const std::vector<int>& runsL, int nL,
                            const std::vector<int>& runsR, int nR, int W, int H);
    /// ccl 之后的共享链体（split→…→conic）
    void runMarkerChainCore(CycleProducts& p, bool record, MarkerOps& ops);
    void runRegistration(CycleProducts& p, bool record);
    /// 收编一次成功的光流/重锁结果：置 R/T、重建 prevState（含新点发号）、
    /// 维护全局地图（新号追加、旧号最新覆盖）。2026-10-14 配准优化专项。
    void adoptRegistration(calib::MarkerOpticalFlowFuseCPUResult& res, CycleProducts& p);
    // —— frame_fuse 兜底增强（2026-10-15 步2/步5）——
    /// 视野捞点：库点经上帧 R/T 逆变换到当前帧系，按到当前标记质心的距离取近邻
    /// 子集（半径随 hold 连击自适应放大；半径内不足时退取全库最近）
    std::vector<int> cullByView(const calib::MarkerPointSet& cur, int maxCount) const;
    /// 子集喂算子（契约：绝不把全库直接传算子——超 triangleMaxPoints 会静默回退 PFH）
    calib::FrameFuseCPUResult fuseExecute(const calib::MarkerPointSet& cur,
                                          const std::vector<int>& libIdx) const;
    /// 链内验收门（config runtime.fuseMinMatched/fuseMaxRmseMm——数据定标值）
    bool acceptFuse(const calib::FrameFuseCPUResult& r) const;
    /// 批2 ①：帧级法线合理性——双峰法线（正反混合/大面积翻转）时拒走兜底
    /// （治已知边界：孪生布点＋法线翻转→折中混合解）。返回反对票占比。
    double checkNormalPlausibility(const std::vector<cv::Vec3d>& normals) const;
    /// 批2 ②：光流路径归并时检查法线同向（对齐兜底路径批1 F5 做法）
    bool mergeNormalCompatible(const cv::Vec3d& a, const cv::Vec3d& b) const;
    void timeBegin();
    double timeEnd(const char* step, CycleProducts& p, bool record, bool gpu);
    void fail(const char* step, CycleProducts& p, const std::string& msg) const;
    /// 【利用率专项】stepSuccess/stepMs 跨线程写读（内部锁 stepRecMtx_——
    /// 流水线下标记 lane/激光收尾/配准线程并发触碰同一 p 的容器）
    void setStepOk(CycleProducts& p, const char* step, bool ok);
    bool stepOk(const CycleProducts& p, const char* step) const;

    ScrConfig cfg_;
    CalibData calib_;
    std::string outDir_;
    // 去重路径：operator=算子内 GPU 去重（默认）｜cpu=外挂 CPU 位移式（对照 oracle）
    std::string dedupPath_;
    // 【150FPS·方案 A】CCL 路径：cpu=RLE 并查集（默认）｜gpu=原 RegionAnalyzerCUDA
    std::string cclPath_;
    // 【产线口径 2026-10-19·用户裁定】JMW_PROD=1：跳过回放测试专用记账（行残差
    // 统计等）——产线不消费，省收尾车道每帧 2×N 点下载＋同步＋中位统计
    bool prodProfile_ = false;
    int cclMinArea_ = 800, cclMaxArea_ = 30000, cclTopX_ = 0;   // CPU 路径参数（同 config ccl 节）
    // 【150FPS】分离完成事件：CPU CCL 下载掩膜前只拦到 sep 完成（不等激光链）
    cudaEvent_t evSepDone_ = nullptr;

    // —— 算子池（每算子实例；⚠ per-eye GPU 算子 L/R 各一实例——结果为内部
    //    持久缓冲视图，同实例第二次 Execute 会使首次结果视图失效）——
    std::unique_ptr<calib::LaserMarkingSeparationCUDA> sepLaserL_, sepLaserR_;
    std::unique_ptr<calib::RegionAnalyzerCUDA> cclL_, cclR_;   // GPU ccl 路径（cclPath=gpu）

    MarkerOps serialOps_;                    // 串行/JMW_PARALLEL 路径用
    std::vector<MarkerOps> markerLanes_;     // 流水线 lane 用（惰性建）
    /// 标记点链算子组装配（串行组与流水线 lane 组同参各建一份）
    MarkerOps makeMarkerOps();

    // 【利用率专项】LaserOps 定义已前移至本类私有区首部（声明先于使用）；
    // laserOps_＝串行/单 lane 组；laserLanes_＝流水线收尾 lane 组
    LaserOps laserOps_;
    std::vector<LaserOps> laserLanes_;       // 流水线激光收尾 lane 用（惰性建）
    /// 激光链算子组装配（laneIdx=0 打印注入日志；>0 静默——stdout 口径不变）
    LaserOps makeLaserOps(int laneIdx, std::string& err);

    std::unique_ptr<calib::MarkerOpticalFlowFuseCPU> flowOp_;
    std::unique_ptr<calib::FrameFuseCPU> fuseOp_;
    // 【flat_center】平顶专用备用算子开关（runtime.extractor=flat 启用；步骤槽位
    // 键恒为 "steger_fast"——下游/报表/导出零改动；flat 实例在各 LaserOps 组内）
    std::string extractor_;
    // 【利用率专项】原十算子散成员收编入 LaserOps（见上），串行＝laserOps_
    // d_gray 每 lane 独立缓冲与 ensureGrayLanes 声明见本类前部（原 d_grayL_/R_ 位）

    cv::cuda::Stream stream_;              // 主流：激光链共用（分步同步计时）
    cv::cuda::Stream streamSepR_;          // 分离右目专用（L/R 双流并行；备用）
    cv::cuda::Stream streamMarker_;        // 标记点链专用（ccl 与激光链 GPU 并行）
    cv::cuda::Stream streamLaserR_;        // 【速度优化·三轮】steger 右目专用（双线程真重叠）
    // —— 双链并行（全链速度优化 2026-10-16）：标记点链（CPU）工作线程与激光链
    //    （GPU 主线程）并行。计时基点线程感知；产物先落独立 CycleProducts 再合并。
    std::thread::id mainTid_{};            // runCycle 所在线程（构造时钉定）
    // 【利用率专项】跨线程产物字段写护栏：流水线多线程并发写同一 CycleProducts
    // 的 stepMs/stepSource（timeEnd）与 stepSuccess/stepError（fail）——容器并发
    // 变更＝UB/堆损坏（RLE 回填提前后标记 lane 与激光收尾线程首次真并发）
    mutable std::mutex stepRecMtx_;
    std::unique_ptr<FramePrefetcher> prefetch_;   // PNG 解码 IO 预取（非流水线计算）
    const FrameSource* src_ = nullptr;     // 帧读取（构造注入，生命周期由调用方保证）

    // —— 双激光管温表（奇偶＝管号；不同管不同标定参数。缺省两管共表）——
    std::shared_ptr<const calib::LaserPlaneMapTempTable> tableOdd_, tableEven_;

    // —— 配准链状态（单线程直接持有）——
    calib::PrevFrameState prevState_;
    calib::GlobalMarkerSet globalSet_;      // 已 transform 累积（globalId 去重）
    std::map<int, size_t> globalIndexById_;
    int nextGid_ = 0;                       // 会话级新点发号计数器（光流/兜底共用）
    bool gridBoardRejected_ = false;        // 规则网格板检测锁存（2026-10-15 裁决硬拒）
    std::vector<std::array<double, 3>> trajT_;   // 逐帧 T 轨迹（M11_trajectory.png 用）
    // 误差趋势门基线（2026-10-15 加固）：前 50 个光流成功帧误差中位，冻结后作全会话参照
    // ——专抓温漂/斜视角等系统性慢变（帧间差分防线结构性看不见）
    std::vector<double> flowRmseBaseline_;
    double flowRmseBaselineMed_ = -1.0;

    // —— frame_fuse 兜底增强（2026-10-15 步2/步5）——
    calib::FrameFuseCPUParams fuseParams_;  // config 装配时的兜底参数快照
    int holdStreak_ = 0;                    // 连续 hold 帧计数（捞点半径自适应＋停手门）
    size_t libVersion_ = 1;                 // 库写计数（索引脏标记）
    std::unique_ptr<TriangleLibraryIndex> libIndex_;   // 步5 全库索引（惰性重建）
    size_t libIndexVersion_ = 0;            // 索引对应库版本
    // 批3：索引重建节流——写库即脏但不必每次都重建（N=500 重建 ~1.7ms，
    // 连续扫描时每帧都脏→每帧重建＝浪费；节流：距上次重建 ≥5 帧才重建）
    int cycleCount_ = 0;                    // 总帧计数（节流计时用）
    int lastIndexRebuildCycle_ = -1;        // 上次索引重建的帧号
    // 批2 ①：法线合理性检查结果（per-frame，诊断落 summary）
    double lastNormalOpposeRatio_ = 0.0;
    // 批2 ②：光流归并法线同向门（与兜底共用 fuseParams_.triangleNormalMaxAngleDeg）
    // 批2 ③：新点候选期——{位置, 法线, 计数}，连续 N 次重观测一致才转正入图
    struct CandidateEntry {
        cv::Point3d position;
        cv::Vec3d normal;
        int count = 1;
    };
    std::vector<CandidateEntry> candidates_;   // 候选池（与库分离， prevState 可引用）
    static constexpr int kPromoteThreshold = 3; // 转正门槛：连续 3 次
    static constexpr double kCandidateMergeMm = 2.0; // 候选匹配半径（同合并半径）

    // 端到端 P0→【五链 P1】：全局激光点云＝产线体素融合算子（0.25mm 去重语义；
    // 自持久累积器，仅主线程/CUDA链2 触碰；A/B 重跑经 Clear 保险）
    std::unique_ptr<calib::LaserCloudFuseCuda> voxelFuseOp_;
    size_t fuseInputTotal_ = 0;     // Σ 逐帧 inputCount（对账 Σ 非hold帧 laserPoints3D）
    size_t fuseFrameCount_ = 0;     // 已融合帧数
    size_t fuseRehashCount_ = 0;    // 扩容事件计数
    size_t fuseOverloadCount_ = 0;  // 负载断言越限帧数（应恒 0）

    // ============================================================================
    // 【150FPS·方案 D】跨帧流水线（GPU lane ×1 ∥ 标记点 lane ×3 ∥ 配准线程×1）
    // 帧间无 GPU 依赖（激光链跨帧独立；配准跨帧依赖在 CPU 侧严格保序）。
    // 流水线消费 Harness 私有方法与成员——友元访问；持有期独占（串行头帧之外
    // 主线程只在 wait() 消费产物）。声明于末位＝成员逆序析构中最先析构（先停
    // 线程再 Destroy 算子）。
    // ============================================================================
    friend class PipelineExecutor;
    std::unique_ptr<class PipelineExecutor> pipe_;   // 惰性建（首个流水线帧）
    bool pipeHeadDone_ = false;                      // 串行头（dump 帧）已过

    // ============================================================================
    // 【五链 P2】fivechain 执行器（声明于 pipe_ 之后＝逆序析构中最先析构——
    // 先停线程再 Destroy 算子）。激活：config fivechain.executor / env JMW_FIVECHAIN。
    // ============================================================================
    friend class fc::FiveChainExecutor;
    std::unique_ptr<class fc::FiveChainExecutor> fc_;
    bool fcActive_ = false;              // 融合分流：CUDA链2 已融合（main 免重复）
    cv::cuda::Stream* fcFuseStreamPtr_ = nullptr;   // PLY 导出前同步的正确流

    // 计时
    std::chrono::steady_clock::time_point t0_;
    // 【全链速度优化】事件计时制：纯 GPU 步骤免逐步同步——timeBegin/timeEnd(gpu)
    // 只在流上记事件，帧尾一次同步后统一算时长（真实 GPU 段时长，精度更高）。
    struct StepEvent {
        std::string step;
        cudaEvent_t evStart = nullptr, evEnd = nullptr;
    };
    std::vector<StepEvent> stepEvents_;
    cudaEvent_t evPendingStart_ = nullptr;    // 当前 timeBegin 的起始事件（复用池）
    void flushStepEvents(CycleProducts& p, bool record);   // 帧尾：同步＋算时长＋记账
    static cudaEvent_t acquireEvent();
    std::map<std::string, double> hostStepMs_;  // gpu=false 步骤的 host 时长暂存
};

} // namespace scr
