#pragma once
// ============================================================================
// FrameCodec.h — v2/v3 帧成拆器（设计方案 §2.1；开关收敛此处）
// v2: "载荷;"            ';' 分帧直通，无校验（兼容现状固件）
// v3: "$载荷<seq><crc>;"  seq=2位hex(0~255循环)，crc=4位hex CRC16-CCITT 查表（覆盖 $…seq）
// 约束：载荷内禁 '$' ';'；帧长上限 32B；半帧超时 50ms（advanceTimeout 推进）
// ============================================================================
#include <cstdint>
#include <string>
#include <vector>

namespace Scanner::device::serial {

class FrameCodec {
public:
    enum class Version { V2, V3 };
    explicit FrameCodec(Version v);

    struct Frame { std::string payload; uint16_t seq = 0; };  // 拆帧产物（CRC 已验）

    // 喂字节流（串口rx线程调）：内部累积+分帧，完整帧追加到 out
    void feed(const std::string& bytes, std::vector<Frame>& out);
    // 逻辑时钟推进（半帧超时 A3）：超时则丢弃挂起缓冲并复位（返回是否发生丢弃）
    // 线程契约：须与 feed 同线程调用（挂起缓冲无锁——S-T5 前置清理注明）
    bool advanceTimeout(int64_t ms);
    // 组帧：V3 校验载荷禁 '$' ';'（违规返回空串）；V2 = payload + ';'
    std::string encode(const std::string& payload, uint16_t seq) const;
    // CRC16-CCITT 查表（多项式 0x1021、初值 0xFFFF）；已知向量 "123456789" → 0x29B1
    static uint16_t crc16ccitt(const std::string& bytes, uint16_t init);

    Version version() const { return v_; }

private:
    Version v_;
    std::string pending_;     // v3 挂起缓冲（收到 '$' 未收到 ';'）
    int64_t pendingAgeMs_ = 0;
    void parsePending(std::vector<Frame>& out);
};

} // namespace Scanner::device::serial
