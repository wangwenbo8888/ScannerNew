#pragma once
// ============================================================================
// CommandChannel.h — 下行可靠通道（设计方案 §2.3；非阻塞发送+tick 驱动对账）
// 非阻塞铁律：send 立即返回——ACK 消费(onAck)与超时判定(tick)都在逻辑线程
// 自己的分流/tick 里做；send 内等待=自锁（A1 审查修正）。
// 单线程属主：所有公开方法仅逻辑线程调用（无锁设计）。
// v2 降级（reliable=false）：send 发不等/不挂表/不判败（恒 ok「未确认」）。
// 命令组步链（R2-A2）：前一条 ACK 完成回调里发下一条；任一步 3 败整组短路
// （组失败回调一次，不发后续）；组全 ACK 组成功回调一次。已发命令幂等无回滚。
// 口径：write 失败=消耗一次尝试（不立即补发，tick 推进）；判败与最后一发同
// tick 收口（总尝试 1+maxRetries 发用尽即败）；表满逐出最旧=判超时
// （onDone(false)+onFault(payload, 当次尝试数)）。
// ============================================================================
#include "serial/FrameCodec.h"
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace Scanner::device::serial {

class CommandChannel {
public:
    using DoneCb = std::function<void(bool ok, const std::string& payload)>;
    // onFault(payload, attempts) —— Fault 事件出口（DeviceManager 转 EventBus）
    using FaultCb = std::function<void(const std::string&, int)>;

    struct Deps {
        FrameCodec* codec = nullptr;                       // 组帧（encode）
        std::function<bool(const std::string& frame)> write; // 写帧（返回是否成功）
        std::function<int64_t()> nowMs;                    // 时钟（测试注入假钟）
        bool reliable = true;                              // false=v2 降级
        int ackTimeoutMs = 100;
        int maxRetries = 3;                                // 总尝试=1+maxRetries
    };

    explicit CommandChannel(Deps d);

    // 关键命令：挂待确认表+组帧+写——立即返回（挂表序号；v2 恒触发 onDone(true,"未确认"))
    void send(const std::string& payload, DoneCb onDone);
    // 查询类（N15）：发不等不挂表
    void sendFireAndForget(const std::string& payload);
    // 命令组步链：payloads 依序，前条 ACK→发下条；任一步 3 败→短路+onGroupDone(false)
    void sendGroup(std::vector<std::string> payloads, DoneCb onGroupDone);
    // ACK 回填（逻辑线程分流 A 帧时）：命中待确认表→销项→触发完成回调
    void onAck(uint16_t ackedSeq);
    // 逻辑线程 10ms tick：到期未销项→重传（间隔=ackTimeoutMs）；3 败→完成回调(fail)+onFault
    void tick();

    FaultCb onFault;                                       // 故障出口（可空）
    uint16_t nextSeq();                                    // 0~255 循环（组帧用）

private:
    static constexpr size_t kTableCap = 8;                 // 待确认表容量（满则最旧先判超时）

    struct PendingCmd {
        uint16_t seq = 0;
        std::string payload;
        DoneCb onDone;
        int attempts = 1;                                  // 已用尝试数（首发=1）
        int64_t dueMs = 0;                                 // 下次对账到期时刻
    };

    Deps deps_;
    uint16_t seq_ = 0;
    std::vector<PendingCmd> table_;                        // 挂表序（front=最旧）

    bool writeFrame(uint16_t seq, const std::string& payload);  // 组帧+写（失败=尝试已消耗）
    void failEntry(PendingCmd e);                          // onDone(false)+onFault（先销项后调用）
};

} // namespace Scanner::device::serial
