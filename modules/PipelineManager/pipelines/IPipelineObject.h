#pragma once
// ============================================================================
// IPipelineObject.h — 五对象公共轻量生命周期（组合优先，仅此一抽象）
// ============================================================================
#include "base/types.h"
#include "PipelineDeps.h"

namespace Scanner::pipeline {

class IPipelineObject {
public:
    virtual ~IPipelineObject() = default;
    virtual Result configure(const PipelineDeps& deps) = 0;  // 注入依赖（只接线不拥有）
    virtual Result start() = 0;
    virtual void stop() = 0;      // 请求停止并等待收尾
    virtual bool isRunning() const = 0;
};

} // namespace Scanner::pipeline
