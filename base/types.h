#pragma once
// ============================================================================
// base/types.h — 公共类型（base 共享内核）
//
// 全层通用：Result、帧号、位姿、事件类型等。无业务逻辑，无第三方依赖（除 std）。
// Phase 1：自 framework/common/types.h 整体迁入（EventType/DeviceState/ScanMode
//   将在后续 Phase 拆出到 07/06/02，本阶段保持合一以维持双轨兼容）。
// ============================================================================

#include <cstdint>
#include <string>
#include <memory>
#include <vector>
#include <functional>

namespace Scanner {

enum class QualityFlag : uint8_t {
    Normal,
    Degraded,
    Warning,
    Fault
};

enum class ContractLevel : uint8_t {
    Stable,
    Internal,
    Experimental
};

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

using FrameId = uint64_t;
using TimestampMs = uint64_t;

struct Pose {
    double R[9] = {1,0,0, 0,1,0, 0,0,1};
    double t[3] = {0, 0, 0};
    FrameId frameId = 0;
    TimestampMs timestamp = 0;

    void identity() {
        R[0]=1; R[1]=0; R[2]=0;
        R[3]=0; R[4]=1; R[5]=0;
        R[6]=0; R[7]=0; R[8]=1;
        t[0]=t[1]=t[2]=0;
    }
};

enum class DeviceState : uint8_t {
    Offline,
    Connected,
    Streaming,
    Error
};

enum class FaultSeverity : uint8_t {
    Info,
    Warning,
    Error,
    Critical
};

enum class ScanMode : uint8_t {
    MarkerOnly = 0,
    MarkerPlusLaser = 1
};

// 健康指标快照（08 侧采集 / 10 侧消费；字段对齐 08 文档差距清单）
struct HealthMetrics {
    double cpuTempC{-1.0};      // CPU 温度 ℃
    double cpuPercent{-1.0};    // CPU 占用 %（仅记录）
    double gpuMemPercent{-1.0}; // GPU 显存 %
    double gpuTempC{-1.0};      // GPU 温度 ℃
    double memPercent{-1.0};    // 内存水位 %
    double diskFreeGB{-1.0};    // 磁盘剩余 GB
    double captureFps{-1.0};    // 采集帧率
    double processFps{-1.0};    // 处理帧率
    double dropRate{-1.0};      // 丢帧率 %
    int64_t timestampMs{0};     // -1=未知/未采
};

enum class EventType : uint16_t {
    DeviceConnected = 0x0100,
    DeviceDisconnected = 0x0101,
    DeviceError = 0x0102,

    ScanStarted = 0x0200,
    ScanStopped = 0x0201,
    ScanPaused = 0x0202,
    ScanFrameReady = 0x0203,

    EmergencyStop = 0x0300,
    TemperatureUpdate = 0x0301,

    FaultOccurred = 0x0400,
    FaultCleared = 0x0401,

    SessionStarted = 0x0500,
    SessionStopped = 0x0501,
    SessionSaved = 0x0502,

    StateChanged = 0x0600,

    // —— mod10 可观测性扩展（2026-08-20 设计方案 §9）——
    CalibStarted        = 0x0204,
    CalibFinished       = 0x0205,
    PostProcessStarted  = 0x0206,
    PostProcessFinished = 0x0207,
    SystemReady         = 0x0601,
    CommandRejected     = 0x0602,
    SelfCheckPassed     = 0x0402,
    LedControl          = 0x0302,   // param1: 0=灭 1=红 2=绿
    HealthReport        = 0x0303,   // param1: 0=正常 1=降级摘要码

    UserDefined = 0x1000
};

struct Event {
    EventType type = EventType::UserDefined;
    uint32_t sourceId = 0;
    TimestampMs timestamp = 0;
    int64_t param1 = 0;
    int64_t param2 = 0;
};

template<typename T>
using UPtr = std::unique_ptr<T>;

template<typename T>
using SPtr = std::shared_ptr<T>;

} // namespace Scanner
