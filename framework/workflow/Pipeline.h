#pragma once
// ============================================================================
// Pipeline.h — 多阶段流水线（Workflow 层）
//
// Stage 基类 + Pipeline 编排器。
// Stage 间通过 RingBuffer 通信，支持背压。
// ============================================================================

#include "common/types.h"
#include "data/RingBuffer.h"
#include <string>
#include <vector>
#include <thread>
#include <functional>

namespace Scanner::workflow {

// ============================================================================
// Stage — 流水线阶段基类
// ============================================================================
class Stage {
public:
    explicit Stage(std::string name) : name_(std::move(name)) {}
    virtual ~Stage() = default;

    const std::string& getName() const { return name_; }

    /// 处理一帧数据（子类实现）
    virtual Result process() = 0;

    /// 停止阶段
    virtual void stop() { stopped_ = true; }

    bool isStopped() const { return stopped_; }

protected:
    std::string name_;
    std::atomic<bool> stopped_{false};
};

// ============================================================================
// Pipeline — 多阶段流水线编排器
// ============================================================================
class Pipeline {
public:
    Pipeline() = default;
    ~Pipeline();

    /// 添加阶段（按顺序）
    void addStage(Stage* stage);

    /// 启动流水线（每个阶段一个线程）
    Result start();

    /// 停止流水线
    void stop();

    /// 等待完成
    void waitComplete();

    /// 获取阶段数
    size_t getStageCount() const { return stages_.size(); }

private:
    void stageLoop(Stage* stage);

    std::vector<Stage*> stages_;
    std::vector<std::thread> threads_;
    std::atomic<bool> running_{false};
};

} // namespace Scanner::workflow
