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
namespace Scanner::pipeline { class ISceneFeed; }

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

    // === 渲染推送（P2 渲染加固：app SceneFeedAdapter 实现 07 ISceneFeed）===
    void setSceneFeed(class Scanner::pipeline::ISceneFeed* feed) { sceneFeed_ = feed; }
    Scanner::pipeline::ISceneFeed* sceneFeed() { return sceneFeed_; }

    // === P-4 按键域档位快照（261002：02 会话启动读——体素密度档值 mm〔0=默认
    //     0.25 口径〕＋景深档〔0 近/1 远〕；app 组合根重写取 08 门面快照）===
    virtual double voxelDensityMm() const { return 0.0; }
    virtual int depthOfField() const { return 0; }
    // P-4 景深区间（camera.json depthOfField 节；02 会话启动注入 07 ScanChains）
    virtual void dofRange(double& nearMin, double& nearMax,
                          double& farMin, double& farMax) const {
        nearMin = 150.0; nearMax = 500.0; farMin = 400.0; farMax = 700.0;
    }

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
    Scanner::pipeline::ISceneFeed* sceneFeed_ = nullptr;   // P2 渲染加固（app 注入，可空）
};

} // namespace Scanner::workflow
