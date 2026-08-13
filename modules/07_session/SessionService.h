#pragma once
// ============================================================================
// SessionService.h — 扫描会话管理（Service 层）
//
// 启/停/暂停/恢复扫描会话、覆盖率统计、断电续扫元数据。
// ============================================================================

#include "base/types.h"
#include <mutex>
#include <string>
#include <atomic>
#include <chrono>

namespace Scanner::service {

struct SessionMetadata {
    std::string sessionId;
    std::string projectName;
    TimestampMs startTime = 0;
    TimestampMs endTime = 0;
    uint64_t totalFrames = 0;
    uint64_t fusedFrames = 0;
    double coveragePercent = 0.0;
    bool saved = false;
};

class SessionService {
public:
    SessionService();
    ~SessionService();

    Result startSession(const std::string& projectName = "");
    Result stopSession();
    Result pauseSession();
    Result resumeSession();

    void onFrameProcessed();
    void onFrameFused();
    void setCoverage(double percent);

    SessionMetadata getMetadata() const;
    bool isActive() const { return active_.load(); }
    bool isPaused() const { return paused_.load(); }

    // 断电续扫
    Result saveCheckpoint(const std::string& path);
    Result loadCheckpoint(const std::string& path);

private:
    std::atomic<bool> active_{false};
    std::atomic<bool> paused_{false};
    mutable std::mutex mutex_;
    SessionMetadata metadata_;
};

} // namespace Scanner::service
