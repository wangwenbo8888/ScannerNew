#pragma once
// ============================================================================
// IState.h — 状态机接口（Service 层）
//
// 扫描仪全局状态机：管理设备就绪→标定→扫描→后处理的状态流转。
// ============================================================================

#include "common/types.h"
#include <string>

namespace Scanner::service {

enum class ScannerState : uint8_t {
    Init,           // 初始化中
    DeviceReady,    // 设备就绪
    Calibrating,    // 标定中
    Calibrated,     // 已标定
    Scanning,       // 扫描中
    Paused,         // 暂停
    PostProcessing, // 后处理中
    Error,          // 故障
    EmergencyStop   // 急停
};

class IState {
public:
    virtual ~IState() = default;

    virtual ScannerState getCurrentState() const = 0;
    virtual std::string getStateName() const = 0;

    /// 状态转移（由事件驱动）
    virtual Result transition(EventType event, int64_t param = 0) = 0;

    /// 查询当前状态是否允许某操作
    virtual bool canOperate(const std::string& operation) const = 0;
};

} // namespace Scanner::service
