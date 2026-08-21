// ============================================================================
// CalibrationRepository.cpp — 标定结果仓库实现（T6: write/load/clear/lastPath；
// typed getters 与 readyForScan 由 T7/T8 补齐）
// ============================================================================
#include "CalibrationRepository.h"

#include <ctime>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>

namespace Scanner::data {

namespace {

// 07 serializeCalib 载荷五顶层键（缺任一拒写）
constexpr const char* kPayloadKeys[] = {"stereo", "tempTables", "pjc", "planeMap", "quality"};

std::tm utcNow(std::time_t& t) {
    t = std::time(nullptr);
    std::tm tmBuf{};
    gmtime_s(&tmBuf, &t);
    return tmBuf;
}

std::string isoNow() {
    std::time_t t;
    const std::tm tmBuf = utcNow(t);
    char buf[24];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tmBuf);
    return buf;
}

std::string makeCalibId() {
    std::time_t t;
    const std::tm tmBuf = utcNow(t);
    char stamp[20];
    std::strftime(stamp, sizeof(stamp), "%Y%m%dT%H%M%S", &tmBuf);
    std::ostringstream oss;
    oss << stamp << '-' << std::hex << (std::random_device{}() & 0xffff);
    return oss.str();
}

} // namespace

Scanner::Result CalibrationRepository::write(const std::string& payloadJson, cv::Size imageSize,
                                             const std::string& path) {
    nlohmann::json payload;
    try {
        payload = nlohmann::json::parse(payloadJson);
    } catch (const std::exception&) {
        return Scanner::Result::fail("标定载荷 JSON 解析失败");
    }
    if (!payload.is_object()) {
        return Scanner::Result::fail("标定载荷不是 JSON 对象");
    }
    for (const char* key : kPayloadKeys) {
        if (!payload.contains(key)) {
            return Scanner::Result::fail(std::string("标定载荷缺少顶层键 ") + key);
        }
    }

    nlohmann::json doc;
    doc["meta"] = {{"version", 1},
                   {"createdAt", isoNow()},
                   {"calibId", makeCalibId()},
                   {"imageSize", {{"width", imageSize.width}, {"height", imageSize.height}}}};
    for (const char* key : kPayloadKeys) {
        doc[key] = payload[key];
    }

    // 原子落盘：tmp 写全 → remove 目标（Windows rename 不覆盖既有）→ rename
    const std::string tmpPath = path + ".tmp";
    try {
        std::ofstream ofs(tmpPath, std::ios::binary | std::ios::trunc);
        if (!ofs.is_open()) {
            return Scanner::Result::fail("临时文件创建失败: " + tmpPath);
        }
        ofs << doc.dump(2);
        ofs.close();
        if (!ofs) {
            std::error_code ecCleanup;
            std::filesystem::remove(tmpPath, ecCleanup);
            return Scanner::Result::fail("临时文件写入失败: " + tmpPath);
        }
    } catch (const std::exception& e) {
        return Scanner::Result::fail(std::string("临时文件异常: ") + e.what());
    }
    std::error_code ec;
    std::filesystem::remove(path, ec);           // 目标不存在时忽略
    std::filesystem::rename(tmpPath, path, ec);
    if (ec) {
        return Scanner::Result::fail("原子改名失败: " + ec.message());
    }

    std::lock_guard<std::mutex> lock(mtx_);
    doc_ = std::move(doc);
    hasData_ = true;
    lastPath_ = path;
    return Scanner::Result::ok("标定仓库已落盘: " + path);
}

Scanner::Result CalibrationRepository::load(const std::string& path) {
    nlohmann::json doc;
    try {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.is_open()) {
            return Scanner::Result::fail("仓库文件打开失败: " + path);
        }
        doc = nlohmann::json::parse(ifs);
    } catch (const std::exception&) {
        return Scanner::Result::fail("仓库文件 JSON 解析失败(坏档不崩): " + path);
    }
    if (!doc.is_object()) {
        return Scanner::Result::fail("仓库档不是 JSON 对象: " + path);
    }

    std::lock_guard<std::mutex> lock(mtx_);
    doc_ = std::move(doc);
    hasData_ = true;
    lastPath_ = path;
    return Scanner::Result::ok();
}

void CalibrationRepository::clear() {
    std::lock_guard<std::mutex> lock(mtx_);
    doc_ = nlohmann::json::object();
    hasData_ = false;
}

} // namespace Scanner::data
