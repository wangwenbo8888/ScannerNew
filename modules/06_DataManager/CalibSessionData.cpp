// CalibSessionData.cpp — 01 标定会话件实现（会话档三键装载；姿态环归头文件）
#include "CalibSessionData.h"

#include <fstream>
#include <stdexcept>

namespace Scanner::data {

Scanner::Result CalibSessionData::load(const std::string& path,
                                       CalibSessionConfig& out) const {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return Result::fail("标定会话档打开失败: " + path);

    nlohmann::json doc;
    try {
        doc = nlohmann::json::parse(ifs);
    } catch (const std::exception& e) {
        return Result::fail(std::string("标定会话档解析失败: ") + e.what());
    }
    if (!doc.is_object()) return Result::fail("标定会话档顶层非 JSON 对象");

    // targets：必须存在、非空数组，每条恰 16 数（4×4 齐次）
    if (!doc.contains("targets") || !doc["targets"].is_array()
        || doc["targets"].empty())
        return Result::fail("标定会话档 targets 缺失或为空");
    std::vector<std::array<double, 16>> targets;
    targets.reserve(doc["targets"].size());
    for (const auto& row : doc["targets"]) {
        if (!row.is_array() || row.size() != 16)
            return Result::fail("标定会话档 targets 条目非 16 数数组");
        std::array<double, 16> a{};
        for (size_t i = 0; i < 16; ++i) {
            if (!row[i].is_number())                       // 逐元素类型校验（防坏档抛异常）
                return Result::fail("标定会话档 targets 条目含非数值元素");
            a[i] = row[i].get<double>();
        }
        targets.push_back(a);
    }

    // boardPoints：必须存在、非空数组，每点恰 [x,y,z] 三数
    if (!doc.contains("boardPoints") || !doc["boardPoints"].is_array()
        || doc["boardPoints"].empty())
        return Result::fail("标定会话档 boardPoints 缺失或为空");
    std::vector<cv::Point3f> board;
    board.reserve(doc["boardPoints"].size());
    for (const auto& p : doc["boardPoints"]) {
        if (!p.is_array() || p.size() != 3)
            return Result::fail("标定会话档 boardPoints 条目非 [x,y,z]");
        for (const auto& v : p)
            if (!v.is_number())
                return Result::fail("标定会话档 boardPoints 条目含非数值元素");
        board.emplace_back(p[0].get<float>(), p[1].get<float>(), p[2].get<float>());
    }

    // initialParams：缺省容空对象；存在则须对象，原文透传不解析（07 类型归 01 换算）
    nlohmann::json initial = nlohmann::json::object();
    if (doc.contains("initialParams")) {
        if (!doc["initialParams"].is_object())
            return Result::fail("标定会话档 initialParams 非 JSON 对象");
        initial = doc["initialParams"];
    }

    out.initialParams = std::move(initial);
    out.targets = std::move(targets);
    out.boardPoints = std::move(board);
    return Result::ok();
}

} // namespace Scanner::data
