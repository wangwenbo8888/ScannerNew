// ============================================================================
// CommandGate.cpp — 统一命令通道实现
// 锁策略：mtx_ 仅护 cmds_ 注册表——Spec 拷贝到局部立即解锁，handler/pre/转态
//   全在锁外执行（handler 可能回调 gate/注册新命令，持锁即死锁）。
// 并发仲裁：不靠 gate 锁防重入，靠状态机 CAS——多线程同时过三关时，
//   transition 唯一成功，败者要么门禁关（态已非 S2）要么切态关（无边可走）。
// ============================================================================
#include "CommandGate.h"
#include "base/EventBus.h"
#include <spdlog/spdlog.h>

namespace Scanner::service {

CommandGate::CommandGate(StateMachine* sm, infra::EventBus* bus) : sm_(sm), bus_(bus) {}

Result CommandGate::registerCommand(Spec spec) {
    if (spec.name.empty()) {
        return Result::fail("命令名为空");
    }
    std::string name = spec.name; // emplace 败时 spec 已被 move 进临时节点，先留名
    std::lock_guard lock(mtx_);
    if (!cmds_.emplace(spec.name, std::move(spec)).second) {
        return Result::fail("命令已注册: " + name);
    }
    return Result::ok();
}

void CommandGate::publishRejected(const std::string& name, const std::string& reason) {
    spdlog::warn("[CommandGate] 拒绝命令: {} ({})", name, reason);
    if (bus_) {
        Event evt;
        evt.type = EventType::CommandRejected;
        bus_->publish(evt);
    }
}

Result CommandGate::submit(const std::string& name, int64_t payload) {
    // ① 查注册：拷贝 Spec 即解锁（handler/pre 执行不许持锁）
    Spec spec;
    {
        std::lock_guard lock(mtx_);
        auto it = cmds_.find(name);
        if (it == cmds_.end()) {
            publishRejected(name, "未注册");
            return Result::fail("命令未注册: " + name);
        }
        spec = it->second;
    }

    // ② 门禁：gateOp 非空才查（内部命令/触发型免检）
    if (!spec.gateOp.empty() && !sm_->canOperate(spec.gateOp)) {
        publishRejected(name, "门禁拒绝（当前态 " + sm_->getStateName() + "）");
        return Result::fail("门禁拒绝: " + name);
    }

    // ③ 业务前置谓词（如 02 参数就绪）；fail message 透传
    if (spec.pre) {
        Result pr = spec.pre();
        if (!pr.success) {
            publishRejected(name, pr.message);
            return Result::fail(pr.message);
        }
    }

    // ④ 即切态（startedEvent≠0）：败 = 并发抢先（他线程已 CAS 切走）
    bool switched = false;
    SystemState from = SystemState::Init;
    if (spec.startedEvent != kNoEvent) {
        from = sm_->getCurrentState();
        Result tr = sm_->transition(spec.startedEvent, payload);
        if (!tr.success) {
            publishRejected(name, "并发抢先切态");
            return Result::fail("命令被并发抢先: " + name);
        }
        switched = true;
    }

    // ⑤ handler 点火（锁外，毫秒级返回）；同步失败且切过态 → ⑥ 回滚收尾态
    if (spec.handler) {
        Result hr = spec.handler();
        if (!hr.success) {
            if (switched && spec.finishedEvent != kNoEvent) {
                sm_->transition(spec.finishedEvent); // 回滚 S2（§3.3）
            }
            spdlog::error("[CommandGate] handler 同步失败{}: {} - {}",
                          switched ? "（已回滚）" : "", name, hr.message);
            publishRejected(name, hr.message);
            return Result::fail(hr.message);
        }
    }

    // ⑦ 成功：publish startedEvent（param1=from param2=to）；StateChanged 由状态机自发，不重复
    if (switched && bus_) {
        Event evt;
        evt.type = spec.startedEvent;
        evt.param1 = static_cast<int64_t>(from);
        evt.param2 = static_cast<int64_t>(sm_->getCurrentState());
        bus_->publish(evt);
    }
    return Result::ok();
}

Result CommandGate::notifyCompleted(const std::string& name, bool ok) {
    Spec spec;
    {
        std::lock_guard lock(mtx_);
        auto it = cmds_.find(name);
        if (it == cmds_.end()) {
            return Result::fail("命令未注册: " + name);
        }
        spec = it->second;
    }
    if (spec.finishedEvent == kNoEvent) {
        return Result::fail("命令无完成转态: " + name);
    }
    if (!ok) {
        // 非设备类异步失败：仍切回 S2（不许悬死 S3–S6，§3.3），但记 ERROR
        spdlog::error("[CommandGate] 命令以失败收尾，切回待机: {}", name);
    }
    return sm_->transition(spec.finishedEvent);
}

} // namespace Scanner::service
