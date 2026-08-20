#pragma once
// ============================================================================
// CommandGate.h — 统一命令通道（submit 触发 / notifyCompleted 收尾切态）
// 设计基准：docs/plans/2026-08-20-可观测性模块10设计方案.md §3
//   · 双口（§3.1）：submit 走三关（注册/门禁/pre）→ 即切态 → handler 仅点火；
//     notifyCompleted 是唯一切态收尾口（start→finish 成对，非第三扇门）
//   · 触发型命令（finish_scan 型）startedEvent=kNoEvent：submit 只点火不切态
//   · handler 同步失败：回滚 finishedEvent + ERROR 日志 + CommandRejected（§3.3）
//   · gate 不开线程（§3.4）：三关+切态在调用方线程，查表/CAS 微秒级
// ============================================================================

#include "StateMachine.h"
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace Scanner::infra { class EventBus; }

namespace Scanner::service {

class CommandGate {
public:
    using Handler = std::function<Result()>;        // 仅点火，毫秒级返回（§3.4）
    using Precondition = std::function<Result()>;   // 业务前置谓词（如 02 参数就绪）

    // EventType 枚举无 0 值（最小 0x0100），0 作「无切态」哨兵
    static constexpr EventType kNoEvent{0};

    struct Spec {
        std::string name;
        std::string gateOp;        // canOperate 操作名；""=不查门禁（内部命令如 system_ready）
        EventType startedEvent{kNoEvent};  // kNoEvent=触发型不切态（finish_scan 型）；非 0=切态事件
        SystemState startedTo{};   // 调试/文档用（实际目标由状态机矩阵定）
        EventType finishedEvent{kNoEvent}; // notifyCompleted 切态事件；kNoEvent=无切态（system_ready 型）
        Precondition pre;
        Handler handler;
    };

    explicit CommandGate(StateMachine* sm, infra::EventBus* bus = nullptr);

    Result registerCommand(Spec spec);                  // 重名注册 fail
    Result submit(const std::string& name, int64_t payload = 0);
    Result notifyCompleted(const std::string& name, bool ok);

private:
    void publishRejected(const std::string& name, const std::string& reason);

    StateMachine* sm_;
    infra::EventBus* bus_;
    std::mutex mtx_;   // 仅护 cmds_ 注册表；handler/pre/转态执行一律锁外
    std::unordered_map<std::string, Spec> cmds_;
};

} // namespace Scanner::service
