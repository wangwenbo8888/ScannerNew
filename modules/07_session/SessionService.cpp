#include "SessionService.h"
#include <spdlog/spdlog.h>
#include <fstream>
#include <iomanip>

namespace Scanner::service {

SessionService::SessionService() {}
SessionService::~SessionService() { stopSession(); }

Result SessionService::startSession(const std::string& projectName) {
    if (active_.load()) return Result::fail("会话已在运行");

    std::lock_guard lock(mutex_);
    metadata_ = {};
    metadata_.sessionId = "session_" + std::to_string(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    metadata_.projectName = projectName.empty() ? "未命名工程" : projectName;
    metadata_.startTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    active_.store(true);
    paused_.store(false);

    spdlog::info("[SessionService] 会话启动: {} ({})", metadata_.sessionId, metadata_.projectName);
    return Result::ok();
}

Result SessionService::stopSession() {
    if (!active_.load()) return Result::ok();

    active_.store(false);
    paused_.store(false);

    {
        std::lock_guard lock(mutex_);
        metadata_.endTime = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }

    spdlog::info("[SessionService] 会话停止: {} (帧: {})",
                 metadata_.sessionId, metadata_.totalFrames);
    return Result::ok();
}

Result SessionService::pauseSession() {
    if (!active_.load()) return Result::fail("无活跃会话");
    paused_.store(true);
    spdlog::info("[SessionService] 会话暂停");
    return Result::ok();
}

Result SessionService::resumeSession() {
    if (!active_.load()) return Result::fail("无活跃会话");
    paused_.store(false);
    spdlog::info("[SessionService] 会话恢复");
    return Result::ok();
}

void SessionService::onFrameProcessed() {
    if (!active_.load() || paused_.load()) return;
    std::lock_guard lock(mutex_);
    ++metadata_.totalFrames;
}

void SessionService::onFrameFused() {
    if (!active_.load() || paused_.load()) return;
    std::lock_guard lock(mutex_);
    ++metadata_.fusedFrames;
}

void SessionService::setCoverage(double percent) {
    std::lock_guard lock(mutex_);
    metadata_.coveragePercent = percent;
}

SessionMetadata SessionService::getMetadata() const {
    std::lock_guard lock(mutex_);
    return metadata_;
}

Result SessionService::saveCheckpoint(const std::string& path) {
    std::lock_guard lock(mutex_);
    std::ofstream ofs(path);
    if (!ofs) return Result::fail("无法写入: " + path);

    ofs << "{\n";
    ofs << "  \"sessionId\": \"" << metadata_.sessionId << "\",\n";
    ofs << "  \"projectName\": \"" << metadata_.projectName << "\",\n";
    ofs << "  \"startTime\": " << metadata_.startTime << ",\n";
    ofs << "  \"totalFrames\": " << metadata_.totalFrames << ",\n";
    ofs << "  \"fusedFrames\": " << metadata_.fusedFrames << ",\n";
    ofs << "  \"coveragePercent\": " << std::fixed << std::setprecision(2) << metadata_.coveragePercent << "\n";
    ofs << "}\n";

    metadata_.saved = true;
    spdlog::info("[SessionService] checkpoint 已保存: {}", path);
    return Result::ok();
}

Result SessionService::loadCheckpoint(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) return Result::fail("无法读取: " + path);

    // 简单解析（后续可用 nlohmann_json）
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    spdlog::info("[SessionService] checkpoint 已加载: {} ({} bytes)", path, content.size());

    // TODO: 解析 JSON 恢复 metadata_
    return Result::ok();
}

} // namespace Scanner::service
