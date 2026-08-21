#pragma once
// ============================================================================
// WorkflowContext.h — 工作流统一依赖入口（Workflow 层）
//
// 聚合 Service(Data)/Data(DataPlane+DataStore)/EventBus，窄接口注入 Workflow。
// ADR 7.7: Workflow 只依赖 WorkflowContext，不直接持有各层指针。
// ============================================================================

#include "base/types.h"
#include <string>
#include <functional>

namespace Scanner::data   { class FrameBuffer; class PointCloudBuffer; class DeviceStateCache; class CalibrationRepository; }
namespace Scanner::service{ class StateMachine; class ParameterManager; }
namespace Scanner::infra  { class EventBus; }

namespace Scanner::workflow {

class WorkflowContext {
public:
    WorkflowContext();
    ~WorkflowContext();

    // === Data 层 ===
    void setFrameBuffer(data::FrameBuffer* fb) { frameBuffer_ = fb; }
    void setPointCloudBuffer(data::PointCloudBuffer* pcb) { pointCloudBuffer_ = pcb; }
    void setDeviceStateCache(data::DeviceStateCache* dsc) { deviceStateCache_ = dsc; }
    void setCalibRepo(data::CalibrationRepository* cr) { calibRepo_ = cr; }   // 06 标定仓库

    data::FrameBuffer* frameBuffer() { return frameBuffer_; }
    data::PointCloudBuffer* pointCloudBuffer() { return pointCloudBuffer_; }
    data::DeviceStateCache* deviceStateCache() { return deviceStateCache_; }
    data::CalibrationRepository* calibRepo() { return calibRepo_; }

    // === Service 层 ===
    void setStateMachine(service::StateMachine* sm) { stateMachine_ = sm; }
    void setParameterManager(service::ParameterManager* pm) { paramManager_ = pm; }

    service::StateMachine* stateMachine() { return stateMachine_; }
    service::ParameterManager* params() { return paramManager_; }

    // === HAL 层 ===（A-T17 撤口：camera/mcu 收进 DeviceManager 门面——铁规不漏
    // 零件指针，工作流经 app 组合根取门面薄转发；全库核实无消费者）

    // === Infra ===
    void setEventBus(infra::EventBus* bus) { eventBus_ = bus; }
    infra::EventBus* eventBus() { return eventBus_; }

    // === EventBus 发布快捷方法 ===
    void publishProgress(int currentStage, int totalStages, const std::string& stageName, float progress);
    void publishEvent(EventType type, int64_t param1 = 0, int64_t param2 = 0);

private:
    data::FrameBuffer*       frameBuffer_ = nullptr;
    data::PointCloudBuffer*  pointCloudBuffer_ = nullptr;
    data::DeviceStateCache*  deviceStateCache_ = nullptr;
    data::CalibrationRepository* calibRepo_ = nullptr;

    service::StateMachine*    stateMachine_ = nullptr;
    service::ParameterManager* paramManager_ = nullptr;

    infra::EventBus* eventBus_ = nullptr;
};

} // namespace Scanner::workflow
