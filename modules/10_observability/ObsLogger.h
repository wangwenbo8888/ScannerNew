#pragma once
// ============================================================================
// ObsLogger.h — 统一日志装配（控制台 + 文件轮转 10MB×5）+ 排障包（设计方案 §5）
//
//   · obsLoggerInit：stdout_color + rotating_file 双 sink 组 spdlog default
//     logger（pattern "[%H:%M:%S.%e][%l][%n] %v"，logger 名 jmw，level info）；
//     main 最先装配（§7.3：ObsLogger → CrashHandler → 其余件）。幂等——重复
//     调用先拆旧装配再重建（旧 sink 文件句柄随之释放）
//   · obsLoggerShutdown：flush 余量 + drop 全部 logger/sink——Windows 下删除
//     日志目录前必须先调（rotating sink 持有 jmw.log 句柄）
//   · obsExportDiagnosticsPackage：logs 副本 + dump（若有）+ version.txt 收进
//     logs 目录旁 diagnostics_<YYYYmmdd_HHMMSS>/；每步容错（缺则跳过不报错）；
//     不引 zip 依赖（YAGNI）——目录形态即交付物，zip 留待后续
//
// 日志书写用 jmw_logging.h 的 JMW_LOG_* 宏（未装配时走 spdlog 内建 stdout）。
// ============================================================================

#include <cstddef>
#include <string>

namespace Scanner::service {

struct ObsLoggerConfig {
    std::string logDir = "logs";            // 相对 exe 工作目录
    size_t maxBytes = 10 * 1024 * 1024;     // 单文件 10MB
    size_t maxFiles = 5;                    // 轮转保留 5 份（含当前 jmw.log）
    std::string baseName = "jmw";
};

void obsLoggerInit(const ObsLoggerConfig& cfg);
void obsLoggerShutdown();
std::string obsExportDiagnosticsPackage(const std::string& dumpPath = "");

} // namespace Scanner::service
