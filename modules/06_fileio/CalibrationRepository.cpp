// ============================================================================
// CalibrationRepository.cpp — 标定结果仓库实现（T6: write/load/clear/lastPath；
// typed getters 与 readyForScan 由 T7/T8 补齐）
// ============================================================================
#include "CalibrationRepository.h"

#include <spdlog/spdlog.h>
#include "jmw_logging.h"
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

// ============================================================================
// 工厂标定档适配（方案 A：load 识别 factory_calib schema → 主工程内部档结构；
// 数据零换算纯搬运——工厂 stereoRectifyTempTable 已按温度逐条产出矫正五件套）
// ============================================================================
nlohmann::json adaptFactoryCalib(const nlohmann::json& src) {
    nlohmann::json doc;
    // —— stereo：内参/畸变（intrinsic.left/right）＋外参（extrinsic.R/T）
    //            ＋矫正五件套（rectify）——键路径映射，Mat 数组原样 ——
    const auto* intr = findNested(src, {"intrinsic"});
    const auto* extr = findNested(src, {"extrinsic"});
    const auto* rec  = findNested(src, {"rectify"});
    nlohmann::json stereo = nlohmann::json::object();
    if (intr && intr->is_object()) {
        if (const auto* L = findNested(*intr, {"left", "camera_matrix"}))  stereo["cameraMatrixL"] = *L;
        if (const auto* L = findNested(*intr, {"left", "dist_coeffs"}))   stereo["distCoeffsL"] = *L;
        if (const auto* R = findNested(*intr, {"right", "camera_matrix"})) stereo["cameraMatrixR"] = *R;
        if (const auto* R = findNested(*intr, {"right", "dist_coeffs"}))  stereo["distCoeffsR"] = *R;
    }
    if (extr && extr->is_object()) {
        if (const auto* m = findNested(*extr, {"R"})) stereo["R"] = *m;
        if (const auto* m = findNested(*extr, {"T"})) stereo["T"] = *m;
    }
    if (rec && rec->is_object()) {
        for (const char* k : {"R1", "R2", "P1", "P2", "Q"})
            if (const auto* m = findNested(*rec, {k})) stereo[k] = *m;
    }
    doc["stereo"] = std::move(stereo);

    // —— meta.imageSize：工厂档 imageSize=[W,H]（数组）→ 主工程 {width,height} ——
    if (const auto* sz = findNested(src, {"imageSize"}); sz && sz->is_array() && sz->size() == 2) {
        doc["meta"] = {{"imageSize", {{"width", (*sz)[0]},
                                      {"height", (*sz)[1]}}}};
    }

    // —— tempTables.rectify.table：工厂 stereoRectifyTempTable.table 原样搬运
    //    （条目已含 temperature+R1/R2/P1/P2/Q+validRoi——stereoTempTable() 按主工程
    //    键取所需字段，多余键天然忽略；两表其余原文档透传 raw 口）——
    if (const auto* tab = findNested(src, {"stereoRectifyTempTable"})) {
        nlohmann::json tt = nlohmann::json::object();
        tt["table"] = tab->is_object() && tab->contains("table") ? (*tab)["table"] : nlohmann::json::array();
        doc["tempTables"] = {{"rectify", std::move(tt)}};
    }
    // laser_calib.json（工厂第二档）由 load 调用方另行合并：见 load 内 fcMerge 逻辑
    return doc;
}

// 工厂激光档并入主工程档（planeMap.tempTable ← laser_calib.json 顶层 tempTable 子树；
// 工厂 schema 现状无该子树时置空数组——readyForScan 报缺"激光档表"如实反映）
void mergeFactoryLaser(nlohmann::json& doc, const nlohmann::json& laser) {
    nlohmann::json pm = nlohmann::json::object();
    pm["tempTable"] = nlohmann::json::object();
    pm["tempTable"]["table"] = laser.contains("tempTable") && laser["tempTable"].contains("table")
                                   ? laser["tempTable"]["table"]
                                   : nlohmann::json::array();
    doc["planeMap"] = std::move(pm);
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

    {
        std::lock_guard<std::mutex> lock(mtx_);
        doc_ = std::move(doc);
        hasData_ = true;
        lastPath_ = path;
    }
    // 落盘快照（一次性）＋门禁自检：写完即验 readyForScan，缺项当场 warn——
    // 02 门禁失败时日志里早有根因，不用等扫描期才发现（锁外调——readyForScan 自取锁）
    {
        ReadyReport rr;
        const Scanner::Result rdy = readyForScan(rr);
        std::string missing;
        for (size_t i = 0; i < rr.missing.size(); ++i) {
            if (i) missing += ",";
            missing += rr.missing[i];
        }
        JMW_LOG_INFO("06-CalibRepo",
            "[CalibRepo] 落盘: {} ({}x{}) readyForScan={} 缺项={}",
            path, imageSize.width, imageSize.height,
            rdy.success ? "ok" : "FAIL", missing.empty() ? "无" : missing);
        if (!missing.empty())
            JMW_LOG_WARN("06-CalibRepo", "[CalibRepo] 落盘后门禁缺项: {}", missing);
    }
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

    // 工厂档识别与适配（方案 A）：factory_calib.camera_calib.v1 → 主工程内部档
    // 结构（纯键路径映射，数据零换算）；同目录 laser_calib.json 存在则合并激光档
    if (doc.contains("schema") && doc["schema"].is_string() &&
        doc["schema"].get<std::string>().rfind("factory_calib.", 0) == 0) {
        nlohmann::json adapted = adaptFactoryCalib(doc);
        // 激光第二档：与相机档同目录的 laser_calib.json（工厂产线约定）；
        // 不存在/坏档＝激光档表缺（readyForScan 如实报缺，不阻塞相机侧）
        std::filesystem::path p(path);
        std::filesystem::path laserPath = p.parent_path() / "laser_calib.json";
        std::error_code ec;
        if (std::filesystem::exists(laserPath, ec)) {
            try {
                std::ifstream lfs(laserPath);
                nlohmann::json laser = nlohmann::json::parse(lfs);
                if (laser.is_object()) mergeFactoryLaser(adapted, laser);
            } catch (const std::exception&) {
                JMW_LOG_WARN("06-CalibRepo", "工厂激光档解析失败(忽略，激光档表将报缺): {}", laserPath.string());
            }
        }
        doc = std::move(adapted);
    }

    {
        std::lock_guard<std::mutex> lock(mtx_);
        doc_ = std::move(doc);
        hasData_ = true;
        lastPath_ = path;
    }
    // 装载快照（一次性）——表档数量即后续查表/门禁的运行依据
    JMW_LOG_INFO("06-CalibRepo", "[CalibRepo] 装载: {} 立体档={}档 激光档={}档",
                 path, stereoTempTable().tiers.size(), planeMapTiers().tiers.size());
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
