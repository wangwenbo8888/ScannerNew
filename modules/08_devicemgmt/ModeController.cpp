// ============================================================================
// ModeController.cpp — 模式黑板实现（D-T9；测试 test_mode_controller.cpp 钉死）
// ============================================================================
#include "ModeController.h"

namespace Scanner::device {

ModeController::ModeController(GateQuery gate) : gate_(std::move(gate)) {}

Result ModeController::request(DeviceMode /*newMode*/, const std::string& op) {
    // 黑板不拦（same-mode 照问防死锁——重复进入由门禁判）
    if (!gate_) return Result::ok();            // 无门禁场景/测试便利：直接过（仍不落板）
    const Result r = gate_(op);                 // 先问门禁
    if (!r.success) return r;                   // 拒→fail 原因透传，板不动
    return Result::ok();                        // 过→ok 但不落板（commit 才落）
}

void ModeController::commit(DeviceMode newMode) {
    const DeviceMode old = mode_.exchange(newMode);   // 原子落板
    if (old == newMode) return;                       // same-mode 不广播（口径钉死，防重复刷 UI）
    if (onChange) onChange(old, newMode);             // 落板广播（回调可空）
}

DeviceMode ModeController::mode() const { return mode_.load(); }
bool ModeController::isCapturing() const { return capturing_.load(); }
void ModeController::setCapturing(bool on) { capturing_.store(on); }

} // namespace Scanner::device
