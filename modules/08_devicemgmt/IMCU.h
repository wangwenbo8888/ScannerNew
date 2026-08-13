#pragma once
// ============================================================================
// IMCU.h — 下位机 MCU 接口（HAL 层）
//
// 控制节点：同步触发、按键、温度、急停硬线。
// ============================================================================

#include "base/types.h"
#include <functional>

namespace Scanner::hal {

// ============================================================================
// MCU 事件回调
// ============================================================================
enum class McuEventType : uint8_t {
    KeyEvent,       // 按键
    Temperature,    // 温度上报
    EmergencyStop,  // 急停
    Heartbeat       // 心跳
};

struct McuEvent {
    McuEventType type;
    int64_t param1 = 0;
    int64_t param2 = 0;
    TimestampMs timestamp = 0;
};

using McuEventCallback = std::function<void(const McuEvent&)>;

// ============================================================================
// IMCU — 下位机接口
// ============================================================================
class IMCU {
public:
    virtual ~IMCU() = default;

    virtual Result open(const std::string& portOrDevice) = 0;
    virtual Result close() = 0;
    virtual bool isOpen() const = 0;

    // 触发
    virtual Result sendSoftwareTrigger() = 0;
    virtual Result setHardwareTriggerMode(bool enabled) = 0;

    // 激光/光源控制
    virtual Result setLaserOn(bool on) = 0;
    virtual Result setLaserPower(int level) = 0;
    virtual Result setLedOn(bool on) = 0;

    // 急停
    virtual Result emergencyStop() = 0;
    virtual bool isEmergencyStop() const = 0;

    // 传感器
    virtual double getTemperature() const = 0;

    // 事件
    virtual Result registerCallback(McuEventCallback cb) = 0;
    virtual Result unregisterCallback() = 0;
};

} // namespace Scanner::hal
