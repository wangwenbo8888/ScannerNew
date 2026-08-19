#pragma once
// ============================================================================
// CalibComputePipeline.h — B 标定计算流水线对象（01-⑥；两线程总装骨架）
// ============================================================================
// 组合根：相机链（CameraChain，0..50）∥ 激光链（LaserChain，50..100）经
// promise/future 衔接（3-4 兑现 → 4-5 起跟随），join 后合成输出与质量。
//
// 用法（02 装配后 01/自动段调用）：
//   attachBoardPoints(温度补偿后板点) + attachInitialParams(与 2-6 同组初始参数)
//   → configure(deps)（记录 calibRepo；落盘 T23 接线）→ run（阻塞批算）。
//
// start/stop = IPipelineObject 适配（内部即 run/置停，语义注明）：
//   start()：后台线程对 attachSession() 注入的会话执行 run（异步，进度回调空）；
//   stop() ：cancel + join 后台线程；
//   isRunning()：run/start 执行中标志。
//
// TestHooks：两条链的 run 函数可注入（假链验证并行 join/输出合并/进度合并）。
// 进度合并：相机链与激光链各自本就映射在 0..50 / 50..100，总装层再做单调门
// （仅转发 > 已报告值的进度），保证用户侧回调 0..100 单调不回退。
#include <atomic>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

#include "base/types.h"
#include "pipelines/IPipelineObject.h"
#include "pipelines/PipelineDeps.h"
#include "pipelines/calibcompute/CalibComputeTypes.h"
#include "pipelines/posture/PostureTypes.h"

namespace Scanner::pipeline {

class CalibComputePipeline : public IPipelineObject {
public:
    struct Config {
        cv::Vec3d pjcInitialT{80.0, 3.0, 3.0};   // PJC 初值（→ LaserChain::Deps）
    };

    // 两条链的 run 签名（= CameraChain::run / LaserChain::run）
    using CameraRunFn = std::function<Scanner::Result(
        const PostureSessionData&, const InitialCalibParams&,
        const std::vector<cv::Point3f>&, StereoParams&,
        std::promise<StereoParams>&, CalibComputeOutput&,
        const ProgressCb&, const CancelToken&)>;
    using LaserRunFn = std::function<Scanner::Result(
        const PostureSessionData&, std::future<StereoParams>,
        CalibComputeOutput&, const ProgressCb&, const CancelToken&)>;

    explicit CalibComputePipeline(const Config& cfg = {});
    ~CalibComputePipeline() override;

    CalibComputePipeline(const CalibComputePipeline&) = delete;
    CalibComputePipeline& operator=(const CalibComputePipeline&) = delete;

    // —— 装配期注入 ——
    void attachBoardPoints(std::vector<cv::Point3f> board3D);   // 板点（调用方温度补偿后）
    void attachInitialParams(InitialCalibParams init);          // 与 2-6 严格同组
    void attachSession(PostureSessionData session);             // start() 异步 run 的输入

    // IPipelineObject：注入依赖（只接线不拥有）——本版仅记录 calibRepo（落盘 T23 接）
    Scanner::Result configure(const PipelineDeps& deps) override;

    // 阻塞批算（双线程：cam/las 并行 join；cancel 贯穿两链）
    Scanner::Result run(const PostureSessionData& in,
                        const ProgressCb& cb,
                        CancelToken& cancel);

    // 上一次 run 的输出（未 run 过为空壳）
    const CalibComputeOutput& output() const;

    // —— IPipelineObject 适配（内部即 run/置停，见文件头语义注明）——
    Scanner::Result start() override;   // 后台线程跑 attachSession 会话
    void stop() override;               // 取消并 join
    bool isRunning() const override;

    // 测试注入（假链；空的钩子 → 真链）
    struct TestHooks {
        CameraRunFn cameraRun;
        LaserRunFn laserRun;
    };
    void setTestHooks(TestHooks hooks);

private:
    Scanner::Result runLocked(const PostureSessionData& in,
                              const ProgressCb& cb,
                              CancelToken& cancel);

    Config cfg_;
    InitialCalibParams init_;
    bool hasInit_ = false;
    std::vector<cv::Point3f> board_;
    PostureSessionData session_;
    bool hasSession_ = false;
    TestHooks hooks_;                       // 空 → 真链
    ICalibRepoWriter* calibRepo_ = nullptr; // T23 落盘接线（configure 记录）

    CalibComputeOutput out_;
    Scanner::Result lastResult_ = Scanner::Result::fail("not run");
    std::mutex runMutex_;                   // run/start 不可重入
    std::thread worker_;                    // start() 后台线程
    std::atomic<bool> running_{false};
    std::shared_ptr<CancelToken> cancelToken_;  // start/stop 路径令牌（start 重建；worker 持共享保活）
};

} // namespace Scanner::pipeline
