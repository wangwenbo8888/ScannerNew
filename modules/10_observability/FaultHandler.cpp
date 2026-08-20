#include "FaultHandler.h"
#include "StateMachine.h"
#include "base/EventBus.h"
#include <algorithm>
#include <chrono>
#include <spdlog/spdlog.h>

namespace Scanner::service {

namespace {
// 同源同级聚合窗口（ms）：窗口内重复上报只计数，不发事件不转态（防风暴，§4.1）
constexpr int64_t kAggregateWindowMs = 1000;

// EventBus 同步分发：reportFault 末尾 publish 的 FaultOccurred 会在总线锁内
// 同步回调本 handler 的订阅 lambda——thread_local 自发布守卫防止同线程重复
// 处理（档案计数翻倍）。按实例指针匹配，多实例互不误伤。
thread_local const void* t_selfPublish = nullptr;
} // namespace

FaultHandler::FaultHandler(infra::EventBus* bus) : eventBus_(bus) {}

FaultHandler::~FaultHandler() { stop(); }

int FaultHandler::registerSource(const std::string& name) {
    std::lock_guard lock(faultMtx_);
    auto it = sources_.find(name);
    if (it != sources_.end()) return it->second;  // 重名同 id（稳定标识）
    const int id = nextId_++;
    sources_.emplace(name, id);
    sourceNames_.emplace(id, name);
    return id;
}

void FaultHandler::reportFault(int sourceId, FaultSeverity severity,
                               const std::string& message) {
    handleFault(sourceId, severity, message, /*viaBus=*/false);
}

void FaultHandler::clearFault(const std::string& faultId) {
    {
        std::lock_guard lock(faultMtx_);
        active_.erase(std::remove_if(active_.begin(), active_.end(),
                          [&faultId](const FaultRecord& r) { return r.id == faultId; }),
                      active_.end());
    }
    spdlog::info("[FaultHandler] 故障已清除: {}", faultId);
    if (eventBus_) {
        Event evt;
        evt.type = EventType::FaultCleared;
        eventBus_->publish(evt);
    }
    // 状态机矩阵无 FaultCleared 边（S7 出口唯 SelfCheckPassed，§4.4）——
    // 此处不转态、不因转态缺失报错；恢复语义走 selfCheckPassed()
}

void FaultHandler::selfCheckPassed() {
    if (!stateMachine_) return;
    if (stateMachine_->getCurrentState() != SystemState::FaultSelfCheck) return;
    if (!stateMachine_->transition(EventType::SelfCheckPassed).success) return;  // 竞态下他者已离开 S7
    spdlog::info("[FaultHandler] 自检通过，恢复 lastHealthy 态");
    if (eventBus_) {
        Event evt;
        evt.type = EventType::LedControl;
        evt.param1 = 2;  // 绿
        eventBus_->publish(evt);
    }
}

std::vector<FaultHandler::FaultRecord> FaultHandler::activeFaults() const {
    std::lock_guard lock(faultMtx_);
    return active_;
}

void FaultHandler::start() {
    if (started_ || !eventBus_) return;
    subscriberId_ = eventBus_->subscribe(EventType::FaultOccurred,
        [this](const Event& evt) {
            if (t_selfPublish == this) return;  // 本线程自发布（reportFault 已处理）
            handleFault(static_cast<int>(evt.param2),
                        static_cast<FaultSeverity>(evt.param1),
                        "external fault, source=" + std::to_string(evt.param2),
                        /*viaBus=*/true);
        });
    started_ = true;
    spdlog::info("[FaultHandler] 已启动");
}

void FaultHandler::stop() {
    if (!started_ || !eventBus_) return;
    eventBus_->unsubscribe(subscriberId_);
    started_ = false;
}

std::string FaultHandler::sourceName(int id) const {
    std::lock_guard lock(faultMtx_);
    auto it = sourceNames_.find(id);
    return it != sourceNames_.end() ? it->second : std::to_string(id);
}

void FaultHandler::handleFault(int sourceId, FaultSeverity severity,
                               const std::string& message, bool viaBus) {
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    // ── 档案层（锁内，直调/订阅两路并发安全）：同 source+severity 1s 窗口聚合 ──
    {
        std::lock_guard lock(faultMtx_);
        FaultRecord* hit = nullptr;
        for (auto& r : active_) {
            if (r.sourceId == sourceId && r.severity == severity) { hit = &r; break; }
        }
        if (hit) {
            if (nowMs - hit->firstTimeMs < kAggregateWindowMs) {
                ++hit->count;  // 防风暴：只计数——不发事件不转态
                if (hit->count == 2) {  // 窗口内首次抑制打一条汇总（后续抑制不刷屏）
                    auto nit = sourceNames_.find(sourceId);  // 锁内直查（sourceName 会重入锁）
                    spdlog::info("[FaultHandler] 同源同类故障 1s 窗口内聚合：source={}({}) severity={} count={}",
                                 nit != sourceNames_.end() ? nit->second : std::to_string(sourceId),
                                 sourceId, static_cast<int>(severity), hit->count);
                }
                return;
            }
            hit->firstTimeMs = nowMs;  // 窗口已过：新一次故障，重置沿用档位
            hit->count = 1;
            hit->message = message;
        } else {
            FaultRecord rec;
            rec.id = std::to_string(sourceId) + ":" + std::to_string(static_cast<int>(severity));
            rec.sourceId = sourceId;
            rec.severity = severity;
            rec.message = message;
            rec.firstTimeMs = nowMs;
            rec.count = 1;
            active_.push_back(std::move(rec));
        }
    }

    // ── 日志层（锁外）：关键路径强制日志——Error 级以上 error，其余 info
    const std::string name = sourceName(sourceId);
    if (severity >= FaultSeverity::Error) {
        spdlog::error("[FaultHandler] 故障 source={}({}) severity={} : {}",
                      name, sourceId, static_cast<int>(severity), message);
    } else {
        spdlog::info("[FaultHandler] 故障 source={}({}) severity={} : {}",
                     name, sourceId, static_cast<int>(severity), message);
    }

    // ── 故障链（仅直调口，§4.4）：Error 级以上且非 S6（免疫）/S7（不重转）→ S7 ──
    // 订阅口不做故障链动作：EventBus 同步分发持总线锁，锁内 transition 会经
    // StateChanged publish 重入死锁——外部源按 §4.6 接 reportFault 注入口
    if (viaBus && severity >= FaultSeverity::Error) {
        spdlog::warn("[FaultHandler] 外部 Error 级故障经总线通道仅记档——应接 reportFault 注入口走完整故障链");
    }
    if (!viaBus && severity >= FaultSeverity::Error && stateMachine_) {
        const SystemState cur = stateMachine_->getCurrentState();
        if (cur != SystemState::PostProcessing && cur != SystemState::FaultSelfCheck) {
            if (stateMachine_->transition(EventType::FaultOccurred).success) {
                if (safeStopCb_) safeStopCb_();  // app 注入的安全停机（10 不链 08）
                if (eventBus_) {
                    Event led;
                    led.type = EventType::LedControl;
                    led.param1 = 1;  // 红
                    eventBus_->publish(led);
                }
            }
        }
    }

    // ── 事件层（仅直调口；订阅口事件已在总线上）：修「丢 source」缺陷（§4.2）──
    if (!viaBus && eventBus_) {
        struct Guard {  // 自发布守卫 RAII（防 handler 回跳本实例重复处理）
            const void* prev;
            explicit Guard(const void* self) : prev(t_selfPublish) { t_selfPublish = self; }
            ~Guard() { t_selfPublish = prev; }
        } guard(this);
        Event evt;
        evt.type = EventType::FaultOccurred;
        evt.param1 = static_cast<int64_t>(severity);
        evt.param2 = sourceId;
        eventBus_->publish(evt);
    }
}

} // namespace Scanner::service
