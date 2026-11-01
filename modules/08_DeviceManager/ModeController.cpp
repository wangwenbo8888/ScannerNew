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
void ModeController::setCapturing(bool on) {
    // 261002 暂停编辑门禁：翻转沿广播（同值幂等不广播）——app 层据此把扫描
    // 工作流同步 Paused/Resume（M 停采即可编辑，无需点模式键二次关会话）
    const bool was = capturing_.exchange(on);
    if (was != on && onCapturingChange) onCapturingChange(on);
}

} // namespace Scanner::device
