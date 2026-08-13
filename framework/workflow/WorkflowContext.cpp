#include "WorkflowContext.h"
#include "FrameBuffer.h"
#include "PointCloudBuffer.h"
#include "DeviceStateCache.h"
#include "service/StateMachine.h"
#include "ParameterManager.h"
#include "service/SessionService.h"
#include "infra/EventBus.h"
#include "hal/IScannerCamera.h"
#include "hal/IMCU.h"
#include <chrono>

namespace Scanner::workflow {

WorkflowContext::WorkflowContext() {}
WorkflowContext::~WorkflowContext() {}

void WorkflowContext::publishProgress(int currentStage, int totalStages,
                                       const std::string& stageName, float progress) {
    if (!eventBus_) return;
    Event evt;
    evt.type = EventType::ScanFrameReady;
    evt.param1 = currentStage;
    evt.param2 = totalStages;
    evt.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    eventBus_->publish(evt);
}

void WorkflowContext::publishEvent(EventType type, int64_t param1, int64_t param2) {
    if (!eventBus_) return;
    Event evt;
    evt.type = type;
    evt.param1 = param1;
    evt.param2 = param2;
    evt.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    eventBus_->publish(evt);
}

} // namespace Scanner::workflow
