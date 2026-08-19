#pragma once
// ============================================================================
// PostProcessPipeline.h — E 后处理流水线（04 后处理工作流的编排引擎）
//
// 离线批处理五阶段（docs/模块功能/04-后处理.md，Q5 定案边界）：
//   0 法线重算 → 1 封装(网格化) → 2 补洞 → 3 光顺(HC) → 4 边界优化，出 STL。
// 不走底座、不碰设备；记账（进度/取消/产物）归 04，本对象只编排。
//
// 网格四族算子（封装/补洞/光顺/边界）09 待建——以 IMeshStageOp 桩接口占位：
// 默认内置桩返回 degraded("operator pending")（09 落地后由 04/09 经
// setStageOp 替换实现，编排不变；法线阶段 T27 实接 laser_cloud_normal 适配）。
//
// STL 导出：06 file_io 有真能力（file_io.h exportSTL），但 file_io.cpp 现编入
// app 的 scan_demo 且依赖 OSG（osg::Vec3）——07 库不链 OSG 不可直调。故以
// StlExportFn 注入点占位（默认空=产物占位+Degraded 1902）；T27/接入期由
// app 侧（同拥 file_io.cpp 与 OSG）适配接线。
//
// 事件码（07 E 后处理族，19xx）：
//   1900 完成（reportCompletion：Normal→Info 也发）
//   1901 阶段失败（Fault，fail-fast 中止）
//   1902 STL 导出未接线（Degraded，产物占位）
//   1903 STL 导出失败（Fault，不中止完成上报）
//
// 取消语义：阶段前检查点检测取消 → 安全退出返回 degraded（B 链惯例
// cancel=degraded——用户意图中止非数据问题，不记 Fault）；不导出、不发完成事件。
//
// 生命周期：ctor → setStageOp（可选，装配期替换桩）→ configure(deps) →
//   run(阻塞批算，04 触发) 或 attachCloud + start()/stop()（IPipelineObject
//   适配，内部即后台 run / cancel+join，语义同 B/D）。
// ============================================================================
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "base/types.h"
#include "pipelines/IPipelineObject.h"
#include "pipelines/PipelineDeps.h"
#include "pipelines/PipelineEventSink.h"
#include "pipelines/calibcompute/CalibComputeTypes.h"   // ProgressCb/CancelToken

namespace Scanner::pipeline {

// —— 五阶段流转的轻量数据句柄（真形态随 09 网格算子定，编排只透传）——
struct MeshData {
    std::vector<float> xyz;            // 点（host）
    std::vector<float> normals;        // 法线
    std::vector<uint32_t> triangles;   // 网格阶段后：三角片索引（桩阶段不动）
    size_t pointCount() const { return xyz.size() / 3; }
};

// 网格阶段算子桩接口（09 网格四族落地后替换实现，编排不变）
struct IMeshStageOp {
    virtual ~IMeshStageOp() = default;
    virtual std::string name() const = 0;
    // 返回 ok=已执行 / degraded="operator pending" 桩 / fail
    virtual Result run(MeshData& io, const CancelToken& cancel) = 0;
};

class PostProcessPipeline : public IPipelineObject {
public:
    static constexpr int kStageCount = 5;   // 法线/封装/补洞/光顺/边界

    struct Config {
        uint32_t skipStages = 0;             // 位掩码：bit0 法线 bit1 封装 bit2 补洞 bit3 光顺 bit4 边界
        std::string outputPath = "output.stl";
    };

    /// STL 导出函数（T27/接入期接 06 file_io::exportSTL 适配；空=产物占位+Degraded）
    using StlExportFn = std::function<bool(const std::string& path, const MeshData& mesh)>;

    explicit PostProcessPipeline(Config cfg = {});
    ~PostProcessPipeline() override;                    // 安全网：stop()（join 后台线程）
    PostProcessPipeline(const PostProcessPipeline&) = delete;
    PostProcessPipeline& operator=(const PostProcessPipeline&) = delete;

    // —— 阶段算子注入（默认内置桩：封装/补洞/光顺/边界四族 09 落地后由 04/09
    //    替换；法线 T27 实接。stageIdx 有效域 0..4，域外忽略）——
    void setStageOp(int stageIdx, std::unique_ptr<IMeshStageOp> op);

    /// STL 导出注入（configure 前后均可；见 StlExportFn 注）
    void setStlExporter(StlExportFn fn);

    // —— IPipelineObject ——
    Result configure(const PipelineDeps& deps) override;   // sink（事件）；STL 经 06 file_io（接线 T27/接入期）

    /// 阻塞批算（04 触发）：五阶段顺序执行（skipStages 位跳过）→ STL 导出。
    /// 进度 5 段均分 0..100（跳过段占份额报 skipped）；桩 degraded → 整体
    /// Degraded 继续（显式跳过不算降级）；fail → 立即 fail 返回
    Result run(MeshData&& cloud, ProgressCb cb, CancelToken& cancel);
    const MeshData& output() const;                    // 上一次 run 产物（未 run 过为空）

    // —— IPipelineObject 适配（内部即后台 run/取消）——
    void attachCloud(MeshData cloud);                  // start() 异步 run 的数据源
    Result start() override;                           // 后台线程对 attachCloud 数据 run
    void stop() override;                              // cancel + join
    bool isRunning() const override;

private:
    Result runInternal(MeshData cloud, const ProgressCb& cb, CancelToken& cancel);

    Config cfg_;
    std::unique_ptr<IMeshStageOp> ops_[kStageCount];   // ctor 填内置 pending 桩
    StlExportFn stlExport_;
    std::unique_ptr<PipelineEventSink> sink_;          // 事件出口（configure 内建）
    bool configured_ = false;

    MeshData out_;                                     // 最近一次 run 产物
    MeshData attachedCloud_;                           // start() 路径数据源
    std::mutex runMutex_;                              // run/start 不可重入
    std::thread worker_;
    std::atomic<bool> running_{false};
    std::shared_ptr<CancelToken> cancelToken_;         // start/stop 路径令牌（start 重建）
};

} // namespace Scanner::pipeline
