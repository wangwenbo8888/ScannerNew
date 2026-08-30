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
#include "jmw_logging.h"
#include <exception>

namespace Scanner::service {

CommandGate::CommandGate(StateMachine* sm, infra::EventBus* bus) : sm_(sm), bus_(bus) {}

Result CommandGate::registerCommand(Spec spec) {
    if (spec.name.empty()) {
        return Result::fail("命令名为空");
    }
    // 组合校验（P1-T6 放宽后语义，两类约束）：
    //  a) 纯触发型（startedEvent==kNoEvent && finishedEvent!=kNoEvent，如 finish_scan）
    //     submit 不切态即点火，若无 gateOp 则任意态可提交——必配 gateOp 声明合法态
    //  b) 切态型（startedEvent!=kNoEvent）无 finishedEvent → 收尾禁用型，允许注册
    //     （P1-T6 实施期补正：system_ready（S1→S2 自检全过）不是工作流命令，无「收尾」
    //      概念。安全性已有保障：notifyCompleted 对其 fail（无转态可切，T5 已测
    //      NotifyCompletedNoFinishEventFails）；handler 同步失败不回滚——§3.3 回滚
    //      依赖 finishedEvent，此处无态可回，命令语义即「切过去就成」）
    if (spec.startedEvent == kNoEvent && spec.finishedEvent != kNoEvent && spec.gateOp.empty()) {
        return Result::fail("触发型命令必须配 gateOp: " + spec.name);
    }
    std::string name = spec.name; // emplace 败时 spec 已被 move 进临时节点，先留名
    std::lock_guard lock(mtx_);
    if (!cmds_.emplace(spec.name, std::move(spec)).second) {
        return Result::fail("命令已注册: " + name);
    }
    return Result::ok();
}

void CommandGate::publishRejected(const std::string& name, const std::string& reason) {
    JMW_LOG_WARN("10-CommandGate", "[CommandGate] 拒绝命令: {} ({})", name, reason);
    if (bus_) {
        Event evt;
        evt.type = EventType::CommandRejected;
        bus_->publish(evt);
    }
}

Result CommandGate::submit(const std::string& name, int64_t payload) {
    // ① 查注册：拷贝 Spec 即解锁（handler/pre 执行不许持锁；publish 一律锁外，
    //   锁内只收集拒绝原因）
    Spec spec;
    std::string reject;
    {
        std::lock_guard lock(mtx_);
        auto it = cmds_.find(name);
        if (it == cmds_.end()) {
            reject = "未注册";
        } else {
            spec = it->second;
        }
    }
    if (!reject.empty()) {
        publishRejected(name, reject);          // publishRejected 内统一打日志（不走点）
        return Result::fail("命令未注册: " + name);
    }

    // ② 门禁：gateOp 非空才查（内部命令/触发型免检）
    if (!spec.gateOp.empty() && !sm_->canOperate(spec.gateOp)) {
        publishRejected(name, "门禁拒绝（当前态 " + sm_->getStateName() + "）");
        return Result::fail("门禁拒绝: " + name);
    }

    // ③ 业务前置谓词（如 02 参数就绪）；fail message 透传；异常视同 fail 走拒绝路径，
    //   不得外泄调用方线程
    if (spec.pre) {
        Result pr;
        try {
            pr = spec.pre();
        } catch (const std::exception& e) {
            pr = Result::fail(std::string("precondition 异常: ") + e.what());
        } catch (...) {
            pr = Result::fail("precondition 异常: unknown");
        }
        if (!pr.success) {
            publishRejected(name, "前置失败: " + pr.message);   // 原因并入（不走点）
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

    // ⑤ handler 点火（锁外，毫秒级返回）；异常视同同步失败（不得外泄调用方线程）；
    //   同步失败且切过态 → ⑥ 回滚收尾态
    if (spec.handler) {
        Result hr;
        try {
            hr = spec.handler();
        } catch (const std::exception& e) {
            hr = Result::fail(std::string("handler 异常: ") + e.what());
        } catch (...) {
            hr = Result::fail("handler 异常: unknown");
        }
        if (!hr.success) {
            if (switched && spec.finishedEvent != kNoEvent) {
                const Result rb = sm_->transition(spec.finishedEvent); // 回滚 S2（§3.3）
                if (rb.success) {
                    JMW_LOG_INFO("10-CommandGate", "[CommandGate] 已回滚至 {}", sm_->getStateName());
                } else {
                    JMW_LOG_WARN("10-CommandGate", "[CommandGate] 回滚失败（态已被并发迁走，落点合法）");
                }
            }
            JMW_LOG_ERROR("10-CommandGate", "[CommandGate] handler 同步失败{}: {} - {}",
                          switched ? "（已尝试回滚）" : "", name, hr.message);
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
        JMW_LOG_ERROR("10-CommandGate", "[CommandGate] 命令以失败收尾，切回待机: {}", name);
    }
    return sm_->transition(spec.finishedEvent);
}

} // namespace Scanner::service
