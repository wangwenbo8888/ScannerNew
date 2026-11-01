#pragma once
// CalibSessionData.h — 01 标定会话件（设计 §4.3）
// initialParams 以 JSON 原文透传（07 类型归 01 边界换算，06 不链 07）。
// 会话档格式（§8-2 定案，JSON 三键）：initialParams（07 PostureInitialParams 字段
// 原文）/ targets（每条 16 数，4×4 齐次）/ boardPoints（每点 [x,y,z]）。
// 装载规则：targets 缺键或空数组 fail、boardPoints 缺键或空数组 fail；
//           initialParams 缺省容空对象（01 侧 paramsReady 防言语义兜底）。
#include "CycleUnit.h"
#include "SlotRing.h"
#include "base/types.h"
#include <array>
#include <cstddef>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace Scanner::data {

struct CalibSessionConfig {
    nlohmann::json initialParams;                   // 07 PostureInitialParams 原文
    std::vector<std::array<double, 16>> targets;    // 25 目标姿态（4×4 齐次）
    std::vector<cv::Point3f> boardPoints;           // 温补后板点
};

class CalibSessionData {
public:
    Scanner::Result load(const std::string& path, CalibSessionConfig& out) const;
    // 姿态环（Backpressure；07 PosturePipeline attachRing 注入）
    SlotRing<CycleUnit>& cycleRing() { return ring_; }
private:
    static constexpr size_t kSlots = 8;             // 沿用 01 现状槽位
    SlotRing<CycleUnit> ring_{kSlots, SlotRing<CycleUnit>::WriterMode::Backpressure};
};

} // namespace Scanner::data
