// ============================================================================
// FrameCodec.cpp — 裸 ';' 分帧实现（契约见 FrameCodec.h）
// ============================================================================
#include "FrameCodec.h"

namespace Scanner::device::serial {

namespace {
constexpr int64_t kHalfTimeoutMs = 50;
}

void FrameCodec::feed(const std::string& bytes, std::vector<Frame>& out) {
    for (char ch : bytes) {
        if (ch == ';') {
            if (!pending_.empty()) out.push_back(Frame{std::move(pending_)});
            pending_.clear();
            pendingAgeMs_ = 0;
        } else {
            if (pending_.empty()) pendingAgeMs_ = 0;
            pending_.push_back(ch);
        }
    }
}

bool FrameCodec::advanceTimeout(int64_t ms) {
    if (pending_.empty()) return false;
    pendingAgeMs_ += ms;
    if (pendingAgeMs_ < kHalfTimeoutMs) return false;
    pending_.clear();
    pendingAgeMs_ = 0;
    return true;
}

std::string FrameCodec::encode(const std::string& payload) const { return payload + ";"; }

} // namespace Scanner::device::serial
