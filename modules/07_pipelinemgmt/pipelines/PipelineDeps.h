#pragma once
// ============================================================================
// PipelineDeps.h — 装配期依赖注入接口集（07 定义窄接口，工作流侧实现）
// ============================================================================
#include <string>

#include "base/EventBus.h"

namespace Scanner::pipeline {

// 采集启停窄接口（A 姿态流水线收口用；01 工作流以 08 门面适配实现）
class IAcquisitionControl {
public:
    virtual ~IAcquisitionControl() = default;
    virtual void stopAcquisition() = 0;
};

// 标定结果落盘窄接口（B 标定计算流水线 run 尾自动写；01 以 06 标定结果仓库适配）
class ICalibRepoWriter {
public:
    virtual ~ICalibRepoWriter() = default;
    virtual bool write(const std::string& json) = 0;   // 序列化格式实施期定
};

// 点云入库窄接口（D 全局优化 run 尾自动写；02 以 06 点云仓库适配）
class ICloudRepoWriter {
public:
    virtual ~ICloudRepoWriter() = default;
    virtual bool write(const std::string& tag) = 0;    // 点云句柄交接实施期定
};

struct PipelineDeps {
    Scanner::infra::EventBus* eventBus{nullptr};
    class ISceneFeed* sceneFeed{nullptr};
    IAcquisitionControl* acquisition{nullptr};
    ICalibRepoWriter* calibRepo{nullptr};
    ICloudRepoWriter* cloudRepo{nullptr};
};

} // namespace Scanner::pipeline
