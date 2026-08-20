#pragma once
// ============================================================================
// jmw_logging.h — 全模块日志宏（设计方案 §5/§8.2）
//
// 与 09 的 CALIB_LOG_*（core/common/calib_logging.h）同构同底（spdlog+fmt，
// 纯宏直透——fmt::format(__VA_ARGS__) 单层展开在本工程使用面无逗号陷阱，
// 保持与 09 完全一致优先）：tag 显式传参，输出 "[模块-对象] msg" 前缀
// （§8.2 格式纪律）。
// 09 现有 CALIB_LOG_* 不动；新代码一律 JMW_LOG。
// 依赖：先经 ObsLogger.h 的 obsLoggerInit 装配 default logger（否则走 spdlog
// 内建 stdout logger，仅控制台无文件）。
// ============================================================================

#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

#define JMW_LOG_INFO(tag, ...)  SPDLOG_INFO("[{}] {}", tag, fmt::format(__VA_ARGS__))
#define JMW_LOG_WARN(tag, ...)  SPDLOG_WARN("[{}] {}", tag, fmt::format(__VA_ARGS__))
#define JMW_LOG_ERROR(tag, ...) SPDLOG_ERROR("[{}] {}", tag, fmt::format(__VA_ARGS__))
