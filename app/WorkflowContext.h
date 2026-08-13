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

namespace Scanner::data   { class FrameBuffer; class PointCloudBuffer; class DeviceStateCache; class CalibStore; }
namespace Scanner::service{ class StateMachine; class ParameterManager; class SessionService; }
namespace Scanner::infra  { class EventBus; }
namespace Scanner::hal    { class IScannerCamera; class IMCU; }

namespace Scanner::workflow {

class WorkflowContext {
public:
    WorkflowContext();
    ~WorkflowContext();

    // === Data 层 ===
    void setFrameBuffer(data::FrameBuffer* fb) { frameBuffer_ = fb; }
    void setPointCloudBuffer(data::PointCloudBuffer* pcb) { pointCloudBuffer_ = pcb; }
    void setDeviceStateCache(data::DeviceStateCache* dsc) { deviceStateCache_ = dsc; }
    void setCalibStore(data::CalibStore* cs) { calibStore_ = cs; }

    data::FrameBuffer* frameBuffer() { return frameBuffer_; }
    data::PointCloudBuffer* pointCloudBuffer() { return pointCloudBuffer_; }
    data::DeviceStateCache* deviceStateCache() { return deviceStateCache_; }
    data::CalibStore* calibStore() { return calibStore_; }

    // === Service 层 ===
    void setStateMachine(service::StateMachine* sm) { stateMachine_ = sm; }
    void setParameterManager(service::ParameterManager* pm) { paramManager_ = pm; }
    void setSessionService(service::SessionService* ss) { sessionService_ = ss; }

    service::StateMachine* stateMachine() { return stateMachine_; }
    service::ParameterManager* params() { return paramManager_; }
    service::SessionService* session() { return sessionService_; }

    // === HAL 层 ===
    void setCamera(hal::IScannerCamera* cam) { camera_ = cam; }
    void setMCU(hal::IMCU* mcu) { mcu_ = mcu; }

    hal::IScannerCamera* camera() { return camera_; }
    hal::IMCU* mcu() { return mcu_; }

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
    data::CalibStore*        calibStore_ = nullptr;

    service::StateMachine*    stateMachine_ = nullptr;
    service::ParameterManager* paramManager_ = nullptr;
    service::SessionService*  sessionService_ = nullptr;

    hal::IScannerCamera* camera_ = nullptr;
    hal::IMCU*           mcu_ = nullptr;

    infra::EventBus* eventBus_ = nullptr;
};

} // namespace Scanner::workflow
