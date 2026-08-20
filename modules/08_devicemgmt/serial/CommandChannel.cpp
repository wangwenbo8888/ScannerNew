// ============================================================================
// CommandChannel.cpp — 实现见头文件契约（设计方案 §2.3）
// ============================================================================
#include "serial/CommandChannel.h"

namespace Scanner::device::serial {

CommandChannel::CommandChannel(Deps d) : deps_(std::move(d)) {}

uint16_t CommandChannel::nextSeq() {
    const uint16_t s = seq_;
    seq_ = static_cast<uint16_t>((seq_ + 1) % 256);
    return s;
}

bool CommandChannel::writeFrame(uint16_t seq, const std::string& payload) {
    const std::string frame = deps_.codec ? deps_.codec->encode(payload, seq) : payload;
    return deps_.write ? deps_.write(frame) : false;
}

void CommandChannel::send(const std::string& payload, DoneCb onDone) {
    if (!deps_.reliable) {  // v2 降级：发不等/不挂表/不判败
        writeFrame(nextSeq(), payload);
        if (onDone) onDone(true, "未确认");
        return;
    }
    bool evicted = false;
    PendingCmd victim;
    if (table_.size() >= kTableCap) {  // 满：最旧先判超时
        victim = std::move(table_.front());
        table_.erase(table_.begin());
        evicted = true;
    }
    PendingCmd e;
    e.seq = nextSeq();
    e.payload = payload;
    e.onDone = std::move(onDone);
    e.attempts = 1;
    e.dueMs = deps_.nowMs() + deps_.ackTimeoutMs;
    writeFrame(e.seq, payload);
    table_.push_back(std::move(e));
    if (evicted) failEntry(std::move(victim));  // 先挂新再补发被逐回调（回调内可再 send）
}

void CommandChannel::sendFireAndForget(const std::string& payload) {
    writeFrame(nextSeq(), payload);
}

void CommandChannel::sendGroup(std::vector<std::string> payloads, DoneCb onGroupDone) {
    if (payloads.empty()) {
        if (onGroupDone) onGroupDone(true, "");
        return;
    }
    std::string first = std::move(payloads.front());
    payloads.erase(payloads.begin());
    send(first, [this, rest = std::move(payloads), onGroupDone = std::move(onGroupDone)](
                    bool ok, const std::string& p) {
        if (!ok) {  // 任一步 3 败 → 整组短路（后续不发）
            if (onGroupDone) onGroupDone(false, p);
            return;
        }
        if (rest.empty()) {  // 组全 ACK → 组成功回调一次
            if (onGroupDone) onGroupDone(true, p);
            return;
        }
        sendGroup(rest, onGroupDone);  // 前条 ACK 完成回调里发下一条
    });
}

void CommandChannel::onAck(uint16_t ackedSeq) {
    for (size_t i = 0; i < table_.size(); ++i) {
        if (table_[i].seq == ackedSeq) {
            PendingCmd e = std::move(table_[i]);
            table_.erase(table_.begin() + i);  // 先销项再回调（回调内可再 send）
            if (e.onDone) e.onDone(true, e.payload);
            return;
        }
    }
}

void CommandChannel::tick() {
    if (!deps_.reliable) return;
    for (;;) {
        const int64_t now = deps_.nowMs();
        size_t idx = table_.size();
        for (size_t i = 0; i < table_.size(); ++i) {  // 表序即旧序 → 首个到期=最旧到期
            if (now >= table_[i].dueMs) {
                idx = i;
                break;
            }
        }
        if (idx == table_.size()) break;
        if (table_[idx].attempts >= 1 + deps_.maxRetries) {  // 已用尽仍挂表（防御）→ 直判
            PendingCmd dead = std::move(table_[idx]);
            table_.erase(table_.begin() + idx);
            failEntry(std::move(dead));
            continue;
        }
        PendingCmd& e = table_[idx];
        writeFrame(e.seq, e.payload);  // 重传（write 失败同计数——尝试已消耗）
        ++e.attempts;
        e.dueMs = now + deps_.ackTimeoutMs;
        if (e.attempts >= 1 + deps_.maxRetries) {  // 最后一发与判败同 tick 收口（3 败）
            PendingCmd dead = std::move(e);
            table_.erase(table_.begin() + idx);
            failEntry(std::move(dead));
        }
    }
}

void CommandChannel::failEntry(PendingCmd e) {
    if (e.onDone) e.onDone(false, e.payload);
    if (onFault) onFault(e.payload, e.attempts);
}

} // namespace Scanner::device::serial
