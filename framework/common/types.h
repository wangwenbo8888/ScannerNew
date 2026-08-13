#pragma once
// ============================================================================
// types.h — 框架公共类型定义
//
// 全层通用：Result、帧号、位姿、Sink 接口前向声明等。
// 无业务逻辑，无第三方依赖（除 std）。
// ============================================================================

#include <cstdint>
#include <string>
#include <memory>
#include <vector>
#include <functional>

namespace Scanner {

// ============================================================================
// 质量标记
// ============================================================================
enum class QualityFlag : uint8_t {
    Normal,     // 正常
    Degraded,   // 性能降级
    Warning,    // 警告（可继续）
    Fault       // 故障（需干预）
};

// ============================================================================
// 版本契约分级
// ============================================================================
enum class ContractLevel : uint8_t {
    Stable,      // SDK / 设备接口（语义化版本）
    Internal,    // 层间契约（同 major 可变）
    Experimental // 算子
};

// ============================================================================
// Result — 算子/操作返回值（不抛异常）
// ============================================================================
struct Result {
    bool success = true;
    int32_t errorCode = 0;
    std::string message;
    QualityFlag qualityFlag = QualityFlag::Normal;

    static Result ok(const std::string& msg = "") {
        return {true, 0, msg, QualityFlag::Normal};
    }
    static Result fail(int32_t code, const std::string& msg) {
        return {false, code, msg, QualityFlag::Fault};
    }
    static Result fail(const std::string& msg) {
        return {false, -1, msg, QualityFlag::Fault};
    }
    static Result degraded(const std::string& msg = "") {
        return {true, 1, msg, QualityFlag::Degraded};
    }
    static Result warning(const std::string& msg = "") {
        return {true, 2, msg, QualityFlag::Warning};
    }

    bool isDegraded() const { return qualityFlag == QualityFlag::Degraded; }
    bool hasWarning() const { return qualityFlag == QualityFlag::Warning; }
    bool isFault() const { return qualityFlag == QualityFlag::Fault; }
};

// ============================================================================
// 帧号 / 时间戳
// ============================================================================
using FrameId = uint64_t;
using TimestampMs = uint64_t;

// ============================================================================
// 位姿（R/t 或 4x4 矩阵）
// ============================================================================
struct Pose {
    double R[9] = {1,0,0, 0,1,0, 0,0,1};  // 3x3 旋转（行主序）
    double t[3] = {0, 0, 0};                // 平移
    FrameId frameId = 0;
    TimestampMs timestamp = 0;

    void identity() {
        R[0]=1; R[1]=0; R[2]=0;
        R[3]=0; R[4]=1; R[5]=0;
        R[6]=0; R[7]=0; R[8]=1;
        t[0]=t[1]=t[2]=0;
    }
};

// ============================================================================
// 设备状态
// ============================================================================
enum class DeviceState : uint8_t {
    Offline,
    Connected,
    Streaming,
    Error
};

// ============================================================================
// 故障严重级别
// ============================================================================
enum class FaultSeverity : uint8_t {
    Info,
    Warning,
    Error,
    Critical
};

// ============================================================================
// 扫描模式
// ============================================================================
enum class ScanMode : uint8_t {
    MarkerOnly = 0,       // 纯标记点
    MarkerPlusLaser = 1   // 标记点 + 激光
};

// ============================================================================
// 事件类型（EventBus 用）
// ============================================================================
enum class EventType : uint16_t {
    // 设备
    DeviceConnected = 0x0100,
    DeviceDisconnected = 0x0101,
    DeviceError = 0x0102,

    // 扫描
    ScanStarted = 0x0200,
    ScanStopped = 0x0201,
    ScanPaused = 0x0202,
    ScanFrameReady = 0x0203,

    // 安全
    EmergencyStop = 0x0300,
    TemperatureUpdate = 0x0301,

    // 故障
    FaultOccurred = 0x0400,
    FaultCleared = 0x0401,

    // 会话
    SessionStarted = 0x0500,
    SessionStopped = 0x0501,
    SessionSaved = 0x0502,

    // 状态
    StateChanged = 0x0600,

    // 用户自定义起点
    UserDefined = 0x1000
};

// ============================================================================
// 事件载荷（轻量，不持有大对象）
// ============================================================================
struct Event {
    EventType type = EventType::UserDefined;
    uint32_t sourceId = 0;
    TimestampMs timestamp = 0;
    int64_t param1 = 0;
    int64_t param2 = 0;
};

// ============================================================================
// 智能指针别名
// ============================================================================
template<typename T>
using UPtr = std::unique_ptr<T>;

template<typename T>
using SPtr = std::shared_ptr<T>;

} // namespace Scanner
