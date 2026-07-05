#pragma once
#include "../common/types.h"
namespace Scanner::data {
class IFrameSink {  // 依赖倒置：HAL 回填帧
public:
    virtual ~IFrameSink() = default;
    virtual void onFrame(const Frame&) = 0;
};
class IDeviceStateSink { public: virtual ~IDeviceStateSink() = default; };
class IDataStore { public: virtual ~IDataStore() = default; };
class DataContext {};
class WorkflowArtifactStore {};  // ADR 7.10
// ADR 7.6 预留：FusionStateHandle（有状态融合接缝的不透明句柄）待实现
}
