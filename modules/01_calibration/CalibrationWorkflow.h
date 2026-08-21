#pragma once
// ============================================================================
// CalibrationWorkflow.h — 标定工作流（编排/记账壳；帧处理/批算移交 07 A/B）
//
// P6-T29b（07 文档 §3 修改 2）：旧棋盘格简化 6 步链（采集→角点检测→内参→
// 外参→立体矫正→温度表）退役——改为装配 07 两流水线对象：
//   A 姿态判断（PosturePipeline，01-⑤）：逐周期认姿态、连续命中确认、
//     集齐 25 自动收口（completionHook 异步移交本类）
//   B 标定计算（CalibComputePipeline，01-⑥）：相机链‖激光链（PJC）批算 +
//     质量门禁；run 尾经 ICalibRepoWriter 自动写
// 本类只留 IWorkflow 生命周期编排 + 进度回调透传 UI + 结果记账。
//
// 结果落盘：本类内定义小适配器 RepositoryWriter 实现 ICalibRepoWriter
// （.cpp）——write 直写 06 标定仓库（载荷原文 + imageSize，内存+原子落盘）。
//
// 参数来源：initialize 装载 06 会话档 calib_session.json 三键（initialParams
// 原文/targets/boardPoints）；缺档保持空参防言语义（paramsReady 拦截）。
//
// TODO(接入期接线)：
//   - 08 采集侧：周期帧组 CycleUnit 写入姿态环（Backpressure 反压；
//     当前无真帧源——A 空转等周期，07 防御路径保留）。
//   - 08 采集控制门面：PipelineDeps.acquisition（集齐收口自动停采；现空）。
//   - 03 渲染：PipelineDeps.sceneFeed（实时姿态/标志点检出推送；现空）。
// ============================================================================

#include "IWorkflow.h"
#include "WorkflowContext.h"
#include "CalibSessionData.h"
#include "pipelines/posture/PosturePipeline.h"   // PostureInitialParams/PostureSessionData
#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <thread>
#include <vector>
#include <opencv2/core.hpp>

// 前向声明（07 流水线对象）
namespace Scanner::pipeline { class CalibComputePipeline; class ICalibRepoWriter; class CancelToken; }

namespace Scanner::workflow {

struct CalibrationResult {
    bool success = false;
    double reprojErrorLeft = 0.0;
    double reprojErrorRight = 0.0;
    double stereoError = 0.0;
    std::string message;
};

class CalibrationWorkflow : public IWorkflow {
public:
    explicit CalibrationWorkflow(WorkflowContext* ctx);
    ~CalibrationWorkflow() override;

    // —— 装配参数注入（A/B 同源；真实来源 TODO 接入期，缺参 fail 不崩）——
    /// 初始参数组（2-6/3-1 严格同组：A 标记点链与 B 相机链单源注入）
    void setInitialParams(const Scanner::pipeline::PostureInitialParams& p) { initialParams_ = p; }
    /// 25×4×4 row-major 目标姿态表
    void setTargets(std::vector<std::array<double, 16>> targets) { targets_ = std::move(targets); }
    /// 温度补偿后板点（B 相机链 2-6 消费）
    void setBoardPoints(std::vector<cv::Point3f> pts) { boardPoints_ = std::move(pts); }

    // —— 完成回报钩子（P5-T14 门禁接线）——
    /// 会话终态回报（B 批算线程尾/B 线程创建失败处调用，bool=ok）。
    /// app 侧注入 lambda 调 CommandGate::notifyCompleted 合账切 S2——
    /// 依赖方向 app→01/10，01 不 include 10 头（经回调反向解耦）。
    void setOnFinished(std::function<void(bool)> cb) { onFinished_ = std::move(cb); }

    // IWorkflow
    std::string getName() const override { return "CalibrationWorkflow"; }
    Result initialize() override;
    Result start() override;
    Result pause() override { return Result::ok(); }
    Result resume() override { return Result::ok(); }
    Result stop() override;
    WorkflowState getState() const override { return state_.load(); }
    Result setProgressCallback(WorkflowCallback cb) override;

    CalibrationResult getResult() const { return result_; }

private:
    WorkflowContext* ctx_;
    std::atomic<WorkflowState> state_{WorkflowState::Idle};
    WorkflowCallback callback_;
    CalibrationResult result_;

    // —— 装配参数（缺省空：initialize 防御 fail）——
    Scanner::pipeline::PostureInitialParams initialParams_;
    std::vector<std::array<double, 16>> targets_;
    std::vector<cv::Point3f> boardPoints_;

    // —— 07 A/B 对象（会话私有件：start 建、stop 收）——
    std::unique_ptr<Scanner::pipeline::PosturePipeline> posture_;
    std::unique_ptr<Scanner::pipeline::CalibComputePipeline> compute_;
    std::unique_ptr<Scanner::pipeline::ICalibRepoWriter> calibRepo_;  // 06 标定仓库适配（.cpp 定义）

    // —— 06 会话件（会话档装载 + 姿态环 8 槽 Backpressure；08 采集侧写入——TODO 接入期）——
    Scanner::data::CalibSessionData calibSession_;

    std::unique_ptr<Scanner::pipeline::CancelToken> cancelToken_;   // B 批算取消令牌（stop 触发）

    std::thread calibThread_;                      // completionHook 异步移交 → B 批算线程
    std::atomic<bool> running_{false};
    std::function<void(bool)> onFinished_;         // 完成回报（app 注入；可空）

    /// 装配参数是否齐备（初始参数/目标表/板点）
    bool paramsReady() const;
    /// A 装配+启动（start 内调用；失败自清理）
    Result startPosture();
    /// B 批算+记账（completionHook 异步线程内；阻塞 run）
    void runCompute(Scanner::pipeline::PostureSessionData session);

    void notifyProgress(int current, int total, const std::string& stage);
    void notifyError(const std::string& msg);
};

} // namespace Scanner::workflow
