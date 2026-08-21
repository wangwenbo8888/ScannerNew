// ============================================================================
// CalibrationRepository.cpp — 标定结果仓库实现（T6: write/load/clear/lastPath；
// typed getters 与 readyForScan 由 T7/T8 补齐）
// ============================================================================
#include "CalibrationRepository.h"

#include <algorithm>
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

const nlohmann::json kNullJson;

// 09 jsonToMatAuto 等价（06 不链 09）：二维嵌套数组 → CV_64F 行主序
cv::Mat matFromJson(const nlohmann::json& j) {
    if (!j.is_array() || j.empty() || !j[0].is_array()) return {};
    const int rows = static_cast<int>(j.size());
    const int cols = static_cast<int>(j[0].size());
    cv::Mat mat(rows, cols, CV_64F);
    for (int r = 0; r < rows; ++r) {
        if (!j[r].is_array() || static_cast<int>(j[r].size()) != cols) return {};
        for (int c = 0; c < cols; ++c) {
            mat.at<double>(r, c) = j[r][c].get<double>();
        }
    }
    return mat;
}

// 二维数组 → Matx 逐元素（形状不符返全零默认并置错误）
template <typename MatxT>
MatxT matxFromJson(const nlohmann::json& j, bool& ok) {
    constexpr int kRows = MatxT::rows;
    constexpr int kCols = MatxT::cols;
    MatxT m = MatxT::zeros();
    if (!j.is_array() || static_cast<int>(j.size()) != kRows) {
        ok = false;
        return m;
    }
    for (int r = 0; r < kRows; ++r) {
        if (!j[r].is_array() || static_cast<int>(j[r].size()) != kCols) {
            ok = false;
            return m;
        }
        for (int c = 0; c < kCols; ++c) {
            m(r, c) = j[r][c].get<double>();
        }
    }
    return m;
}

// 逐层下钻取子树（任一层缺失返 nullptr）
const nlohmann::json* findNested(const nlohmann::json& root, std::initializer_list<const char*> keys) {
    const nlohmann::json* cur = &root;
    for (const char* key : keys) {
        if (!cur->is_object()) return nullptr;
        const auto it = cur->find(key);
        if (it == cur->end()) return nullptr;
        cur = &*it;
    }
    return cur;
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
        return Scanner::Result::fail("原子改名失败(数据保留于 " + tmpPath + "): " + ec.message());
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

std::string CalibrationRepository::lastPath() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return lastPath_;
}

StereoData CalibrationRepository::stereo() const {
    std::lock_guard<std::mutex> lock(mtx_);
    StereoData data;
    const nlohmann::json* st = findNested(doc_, {"stereo"});
    if (st) {
        auto fill = [st](const char* key, cv::Mat& dst) {
            const auto it = st->find(key);
            if (it != st->end()) dst = matFromJson(*it);
        };
        fill("cameraMatrixL", data.cameraMatrixL);
        fill("cameraMatrixR", data.cameraMatrixR);
        fill("distCoeffsL", data.distCoeffsL);
        fill("distCoeffsR", data.distCoeffsR);
        fill("R", data.R);
        fill("T", data.T);
        fill("R1", data.R1);
        fill("R2", data.R2);
        fill("P1", data.P1);
        fill("P2", data.P2);
        fill("Q", data.Q);
    }
    if (const nlohmann::json* size = findNested(doc_, {"meta", "imageSize"}); size && size->is_object()) {
        data.imageSize = cv::Size(size->value("width", 0), size->value("height", 0));
    }
    return data;
}

StereoTempTable CalibrationRepository::stereoTempTable() const {
    std::lock_guard<std::mutex> lock(mtx_);
    StereoTempTable table;
    const nlohmann::json* tab = findNested(doc_, {"tempTables", "rectify", "table"});
    if (!tab || !tab->is_array()) return table;
    for (const auto& entry : *tab) {
        if (!entry.is_object() || !entry.contains("temperature") || !entry["temperature"].is_number()) {
            continue;
        }
        StereoTempTier tier;
        tier.tempC = entry["temperature"].get<double>();
        bool ok = true;
        auto field = [&entry](const char* key) -> const nlohmann::json& {
            const auto it = entry.find(key);
            return it == entry.end() ? kNullJson : *it;
        };
        tier.R1 = matxFromJson<cv::Matx33d>(field("R1"), ok);
        tier.R2 = matxFromJson<cv::Matx33d>(field("R2"), ok);
        tier.P1 = matxFromJson<cv::Matx34d>(field("P1"), ok);
        tier.P2 = matxFromJson<cv::Matx34d>(field("P2"), ok);
        tier.Q = matxFromJson<cv::Matx44d>(field("Q"), ok);
        if (!ok) continue;  // 档内形状不符整档弃用（防御）
        table.tiers.push_back(tier);
    }
    std::sort(table.tiers.begin(), table.tiers.end(),
              [](const StereoTempTier& a, const StereoTempTier& b) { return a.tempC < b.tempC; });
    return table;
}

PlaneMapTempTableRef CalibrationRepository::planeMapTiers() const {
    std::lock_guard<std::mutex> lock(mtx_);
    PlaneMapTempTableRef ref;
    const nlohmann::json* tab = findNested(doc_, {"planeMap", "tempTable", "table"});
    if (!tab || !tab->is_array()) return ref;
    for (const auto& entry : *tab) {
        if (entry.is_object() && entry.contains("temperature") && entry["temperature"].is_number()) {
            ref.tiers.push_back(PlaneMapTempTierRef{entry["temperature"].get<double>()});
        }
    }
    std::sort(ref.tiers.begin(), ref.tiers.end(),
              [](const PlaneMapTempTierRef& a, const PlaneMapTempTierRef& b) { return a.tempC < b.tempC; });
    return ref;
}

nlohmann::json CalibrationRepository::planeMapTempTableRaw() const {
    std::lock_guard<std::mutex> lock(mtx_);
    const nlohmann::json* sub = findNested(doc_, {"planeMap", "tempTable"});
    return sub ? *sub : nlohmann::json();
}

nlohmann::json CalibrationRepository::tempTablesRaw() const {
    std::lock_guard<std::mutex> lock(mtx_);
    const nlohmann::json* sub = findNested(doc_, {"tempTables"});
    return sub ? *sub : nlohmann::json();
}

Scanner::Result CalibrationRepository::readyForScan(ReadyReport& out) const {
    std::lock_guard<std::mutex> lock(mtx_);
    out.ready = false;
    out.missing.clear();

    auto hasMat = [](const nlohmann::json* obj, const char* key) {
        if (!obj || !obj->is_object()) return false;
        const auto it = obj->find(key);
        return it != obj->end() && it->is_array() && !it->empty();
    };
    const nlohmann::json* st = findNested(doc_, {"stereo"});
    if (!hasMat(st, "cameraMatrixL")) out.missing.push_back("相机内参 L");
    if (!hasMat(st, "cameraMatrixR")) out.missing.push_back("相机内参 R");
    if (!hasMat(st, "R") || !hasMat(st, "T")) out.missing.push_back("外参 R/T");

    const nlohmann::json* rectTab = findNested(doc_, {"tempTables", "rectify", "table"});
    if (!rectTab || !rectTab->is_array() || rectTab->empty()) out.missing.push_back("立体温度表(rectify)");

    const nlohmann::json* pmTab = findNested(doc_, {"planeMap", "tempTable", "table"});
    if (!pmTab || !pmTab->is_array() || pmTab->empty()) out.missing.push_back("激光档表(planeMap)");

    const nlohmann::json* size = findNested(doc_, {"meta", "imageSize"});
    if (!size || !size->is_object() || size->value("width", 0) <= 0 || size->value("height", 0) <= 0) {
        out.missing.push_back("meta.imageSize");
    }

    out.ready = out.missing.empty();
    return Scanner::Result::ok();
}

} // namespace Scanner::data
