#pragma once
// ============================================================================
// PostProcessWorkflow.h — 后处理工作流（编排/记账壳；五阶段编排移交 07 E 对象）
//
// P6-T29c（07 文档 §3 修改 3）：旧七阶段 sleep 空壳退役——GBA/重融合已在列
// 但 Q5 定案归 02（GlobalOptimStage），本工作流不再涉及。帧/批处理改为装配
// 07 PostProcessPipeline（07 §1.1-E，离线批处理五阶段：法线→封装→补洞→
// 光顺→边界→出 STL）；本类只留 IWorkflow 生命周期 + 进度回调透传 + 产物记账。
//
// 数据流：WorkflowContext 点云快照（PointCloudBuffer.getSnapshot）→
// MeshData（xyz）→ pipeline run（skipStages 位掩码透传）→ STL 经
// StlExportFn 接 06 file_io exportSTL——file_io.cpp 编入 app 且 07 库不链
// OSG，故导出适配在本文件（app 编译单元）内接线。
//
// TODO(接入期接线)：
//   - 输入改 06 点云仓库（app 存活件）内存句柄（现 PointCloudBuffer 快照，
//     02 GBA 修正点云写入后此处即为修正结果）。
//   - 网格四族算子（封装/补洞/光顺/边界）09 落地后经 setStageOp 注入
//     （现内置桩恒 degraded "operator pending"，法线阶段已实接）。
// ============================================================================

#include "IWorkflow.h"
#include "WorkflowContext.h"
#include "pipelines/postprocess/PostProcessPipeline.h"   // MeshData/CancelToken
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>

namespace Scanner::workflow {

struct PostProcessResult {
    bool success = false;
    int meshTriangles = 0;
    double smoothTimeMs = 0.0;
    std::string outputPath;
    std::string message;
};

class PostProcessWorkflow : public IWorkflow {
public:
    explicit PostProcessWorkflow(WorkflowContext* ctx);
    ~PostProcessWorkflow() override;

    void setOutputPath(const std::string& path) { outputPath_ = path; }
    /// 位掩码透传 07 E：bit0 法线 bit1 封装 bit2 补洞 bit3 光顺 bit4 边界
    void setSkipStages(uint32_t mask) { skipStages_ = mask; }

    // —— 完成回报钩子（P5-T16 门禁接线，同 01-T14/02-T15 模式）——
    /// 会话终态回报（postThread_ 批算线程尾调用，bool=ok）。
    /// app 侧注入 lambda 调 CommandGate::notifyCompleted 合账切 S2——
    /// 依赖方向 app→04/10，04 不 include 10 头（经回调反向解耦）。
    void setOnFinished(std::function<void(bool)> cb) { onFinished_ = std::move(cb); }

    // IWorkflow
    std::string getName() const override { return "PostProcessWorkflow"; }
    Result initialize() override;
    Result start() override;
    Result pause() override { return Result::ok(); }
    Result resume() override { return Result::ok(); }
    Result stop() override;
    WorkflowState getState() const override { return state_.load(); }
    Result setProgressCallback(WorkflowCallback cb) override;

    PostProcessResult getResult() const { return result_; }

private:
    WorkflowContext* ctx_;
    std::string outputPath_ = "output.stl";
    uint32_t skipStages_ = 0;

    std::atomic<WorkflowState> state_{WorkflowState::Idle};
    WorkflowCallback callback_;
    PostProcessResult result_;

    // —— 07 E 编排引擎（会话私有件：start 建、stop 收）——
    std::unique_ptr<Scanner::pipeline::PostProcessPipeline> pipeline_;
    std::unique_ptr<Scanner::pipeline::CancelToken> cancelToken_;   // stop 取消令牌

    std::thread postThread_;
    std::atomic<bool> running_{false};
    std::function<void(bool)> onFinished_;       // 完成回报（app 注入；可空）

    /// 点云快照 → MeshData（xyz；法线由阶段 0 重算）
    Scanner::Result makeCloudData(Scanner::pipeline::MeshData& out) const;
    /// 阻塞批算线程体（进度透传 + 产物记账）
    void postProcessLoop(Scanner::pipeline::MeshData cloud);
};

} // namespace Scanner::workflow
