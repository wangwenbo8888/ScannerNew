#pragma once
// ============================================================================
// ModeController.h — 模式黑板（2026-08-18 设计 §2.1；§4-1 命令组成功回调才落板）
//
// 黑板三词：待机/标定中/扫描中（原子读写）；另记采集子态 开/关（S4/S5 子态真相源
// ——10 设计 §2.5：子态不占全局态、10 不记账）。原 3 号灯职责：设备在干嘛以此为准。
// 改板规矩：request 先问门禁（拒→fail 带原因；过→ok 但不落板——落板由
// commit() 在命令组成功回调调，防「板已切命令没到」）。
// same-mode commit 口径：不广播（防重复命令组重复刷 UI；D-T9 钉死）。
// ============================================================================
#include "base/types.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

namespace Scanner::device {

enum class DeviceMode : uint8_t { Idle, Calibrating, Scanning };

class ModeController {
public:
    using GateQuery = std::function<Result(const std::string& op)>;  // op 如 "enter_scan"
    explicit ModeController(GateQuery gate);

    // 请求改板：门禁拒→fail（带原因）；过→ok（黑板不变）——调用方（DeviceManager）
    // 随后发命令组，成功回调里调 commit
    Result request(DeviceMode newMode, const std::string& op);
    // 命令组成功后落板：广播回调 onChange(旧,新)；same-mode 不广播
    void commit(DeviceMode newMode);

    DeviceMode mode() const;                    // 黑板（原子）
    bool isCapturing() const;                   // 采集子态（原子）
    void setCapturing(bool on);                 // N11 H1/H0 命令成功后置（幂等）

    std::function<void(DeviceMode, DeviceMode)> onChange;   // 落板广播（可空）

private:
    GateQuery gate_;
    std::atomic<DeviceMode> mode_{DeviceMode::Idle};
    std::atomic<bool> capturing_{false};
};

} // namespace Scanner::device
