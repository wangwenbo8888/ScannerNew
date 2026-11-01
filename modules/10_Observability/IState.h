#pragma once
// IState.h — 全局门禁状态机接口（7 态模型 S1–S7，2026-08-20 设计方案 §2）
#include "base/types.h"
#include <string>

namespace Scanner::service {

enum class SystemState : uint8_t {
    Init = 1,           // S1 初始化（自检：通讯/加密狗/授权）
    Standby = 2,        // S2 待机（门禁主战场）
    Calibrating = 3,    // S3 标定
    ScanMarker = 4,     // S4 扫描标记点
    ScanMarkerLaser = 5,// S5 扫描标记点+激光
    PostProcessing = 6, // S6 后处理（免疫故障转态）
    FaultSelfCheck = 7  // S7 设备故障+自检中
};

class IState {
public:
    virtual ~IState() = default;
    virtual SystemState getCurrentState() const = 0;
    virtual std::string getStateName() const = 0;
    virtual Result transition(EventType event, int64_t param = 0) = 0;
    virtual bool canOperate(const std::string& operation) const = 0;
};

} // namespace Scanner::service
