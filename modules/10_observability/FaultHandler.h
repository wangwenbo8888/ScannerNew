#pragma once
// ============================================================================
// FaultHandler.h — 故障处理器（Service 层）
//
// 故障档案表 + 来源标识 + 1s 聚合防风暴 + S7 故障链（设计方案 §4）：
//   · reportFault 直调口 = 完整故障链（记档 → S6/S7 判定 → 转态 →
//     safeStop 回调 → LedControl 红 → FaultOccurred 事件）
//   · EventBus 订阅口（start 后）= 仅记档：外部源直接 publish
//     FaultOccurred(param1=severity, param2=sourceId) 也进档案；
//     EventBus 同步分发持总线锁，锁内转态/发灯事件会重入死锁——
//     故障链动作归直调口（§4.6：08 落地接 reportFault 注入口）
//   · 红灯走 LedControl(param1: 1=红 2=绿)，不复用 EmergencyStop
//     通道（§4.3 语义解耦）
// ============================================================================

#include "base/types.h"
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace Scanner::infra { class EventBus; }
namespace Scanner::service { class StateMachine; }

namespace Scanner::service {

class FaultHandler {
public:
    struct FaultRecord {
        std::string id;         // "<sourceId>:<severity>"（同源同类聚合一档）
        int sourceId;
        FaultSeverity severity;
        std::string message;
        int64_t firstTimeMs;
        uint32_t count;         // 1s 窗口聚合计数
    };

    explicit FaultHandler(infra::EventBus* bus = nullptr);
    ~FaultHandler();

    FaultHandler(const FaultHandler&) = delete;
    FaultHandler& operator=(const FaultHandler&) = delete;

    void setStateMachine(StateMachine* sm) { stateMachine_ = sm; }
    void setSafeStopCallback(std::function<void()> cb) { safeStopCb_ = std::move(cb); }  // app 注入（10 不链 08）

    int registerSource(const std::string& name);          // 稳定 id（重名同 id）
    void reportFault(int sourceId, FaultSeverity severity, const std::string& message);
    void clearFault(const std::string& faultId);          // 仅清档+FaultCleared 事件（恢复走 selfCheckPassed）
    void selfCheckPassed();                               // S7→lastHealthy + LedControl 绿

    std::vector<FaultRecord> activeFaults() const;        // 档案快照（测试/查询）

    void start();   // 订阅 EventBus FaultOccurred（param1=severity param2=sourceId）
    void stop();

private:
    void handleFault(int sourceId, FaultSeverity severity, const std::string& message,
                     bool viaBus);                        // 直调/订阅两路共用的内部处理
    std::string sourceName(int id) const;

    mutable std::mutex faultMtx_;                         // 档案+来源表并发防护（审核 F）
    std::vector<FaultRecord> active_;
    std::map<std::string, int> sources_;
    std::map<int, std::string> sourceNames_;
    int nextId_ = 1;

    infra::EventBus* eventBus_ = nullptr;
    StateMachine* stateMachine_ = nullptr;
    std::function<void()> safeStopCb_;
    uint32_t subscriberId_ = 0;
    bool started_ = false;
};

} // namespace Scanner::service
