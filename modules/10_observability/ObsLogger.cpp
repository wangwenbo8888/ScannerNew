// ============================================================================
// ObsLogger.cpp — 统一日志装配实现（设计方案 §5）
//
//   · 双 sink（stdout_color + daily_file）组 spdlog default logger；
//     幂等 = init 内先 spdlog::shutdown() 拆旧装配（registry 无永久关闭
//     标志，drop_all 后可重建；旧 sink 析构释放当日文件句柄并落盘余量）
//   · daily_file_sink(base, 0, 0, false, maxFiles)：文件名 jmw_YYYY-MM-DD.log
//     （日期插在扩展名前）；跨零点首次写入自动换新文件；maxFiles=保留份数
//     （sink 构造与每次轮转时删最旧——ISO 日期字典序即时间序）
//   · 排障包每步 std::error_code 容错（缺则跳过）；spdlog 以共享模式打开
//     日志文件，sink 未关闭也能读走已 flush 部分
// ============================================================================
#include "ObsLogger.h"

#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <vector>

#ifdef _WIN32
#include <windows.h>   // SetConsoleOutputCP——控制台按 UTF-8 解码（见 obsLoggerInit）
#endif

// 工程版本（09 core/common/version.h 为头文件内联，取宏不引链接依赖；
// 不可含则写 unknown——__has_include 守卫保证构建永不因缺它而断）
#if defined(__has_include)
#if __has_include("../../09_operatorlib/core/common/version.h")
#include "../../09_operatorlib/core/common/version.h"
#define JMW_HAVE_VERSION_H 1
#endif
#endif
#ifndef JMW_HAVE_VERSION_H
#define JMW_HAVE_VERSION_H 0
#endif

namespace fs = std::filesystem;

namespace Scanner::service {

namespace {

std::mutex g_obsMutex;          // init/shutdown/export 串行（default logger 是进程级全局）
ObsLoggerConfig g_cfg;          // 排障包导出所需的当前装配
bool g_inited = false;

std::string timestampName() {
    const std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
    localtime_s(&tm, &t);
    char buf[24];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
    return buf;
}

// 按日保留清理：扫 logDir 内 <baseName>_YYYY-MM-DD.log（ISO 日期字典序=时间序），
// 超过 keepCount 份删最旧。sink 的 max_files 只在跨零点轮转时清——历史积压由
// init 时主动清一次（确定性、不依赖 spdlog 版本行为）。
void pruneOldDailyLogs(const fs::path& logDir, const std::string& baseName,
                       size_t keepCount) {
    std::error_code ec;
    std::vector<fs::path> files;
    for (fs::directory_iterator it(logDir, ec), end; it != end; it.increment(ec)) {
        const fs::path& p = it->path();
        if (it->is_regular_file(ec) && p.extension() == ".log" &&
            p.stem().string().rfind(baseName + "_", 0) == 0)
            files.push_back(p);
    }
    if (files.size() <= keepCount) return;
    std::sort(files.begin(), files.end());          // 文件名字典序 = 日期时间序
    const size_t removeCount = files.size() - keepCount;
    for (size_t i = 0; i < removeCount; ++i)
        fs::remove(files[i], ec);                   // 单个失败不中断（尽力而为）
}

std::string todayDateStr() {
    const std::time_t t = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());
    std::tm tm{};
    localtime_s(&tm, &t);
    char buf[16];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d", &tm);
    return buf;
}

} // namespace

void obsLoggerInit(const ObsLoggerConfig& cfg) {
    std::lock_guard<std::mutex> lock(g_obsMutex);
    spdlog::shutdown();   // 幂等：先拆旧装配（flush 由 sink 析构保证）

#ifdef _WIN32
    // 控制台解码对齐：spdlog 输出 UTF-8 字节，Windows 控制台默认 GBK(936)——
    // 不设则中文全乱码（等价 chcp 65001；仅本进程控制台会话生效）
    SetConsoleOutputCP(CP_UTF8);
#endif

    std::error_code ec;
    fs::create_directories(cfg.logDir, ec);
    // 历史积压先清（sink 的 max_files 只管跨零点轮转时）：当日文件已存在（重启
    // 场景）占一个槽位；尚未存在则多让出一格给即将新建的当日文件
    const size_t keep = cfg.maxFiles > 0 ? cfg.maxFiles : 1;
    const fs::path todayFile =
        fs::path(cfg.logDir) / (cfg.baseName + "_" + todayDateStr() + ".log");
    const size_t keepExisting =
        fs::exists(todayFile, ec) ? keep : (keep > 0 ? keep - 1 : 0);
    pruneOldDailyLogs(cfg.logDir, cfg.baseName, keepExisting);

    auto console = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto file = std::make_shared<spdlog::sinks::daily_file_sink_mt>(
        (fs::path(cfg.logDir) / (cfg.baseName + ".log")).string(),
        /*rotation_hour=*/0, /*rotation_minute=*/0,
        /*truncate=*/false,
        static_cast<uint16_t>(cfg.maxFiles > 0xFFFF ? 0xFFFF : cfg.maxFiles));

    auto logger = std::make_shared<spdlog::logger>(
        "jmw", spdlog::sinks_init_list{console, file});
    logger->set_level(spdlog::level::info);
    logger->flush_on(spdlog::level::warn);          // warn+ 即时落盘（§7 崩溃留痕）
    // %s/%# = 源文件名(仅 basename)/行号——由 JMW_LOG_*（SPDLOG 宏族）在调用点
    // 自动捕获；散装 spdlog::info 等函数调用无 source_loc，显示为空段 [:0]
    logger->set_pattern("[%H:%M:%S.%e][%l][%n][%s:%#] %v");
    spdlog::set_default_logger(logger);

    g_cfg = cfg;
    g_inited = true;
}

void obsLoggerShutdown() {
    std::lock_guard<std::mutex> lock(g_obsMutex);
    if (g_inited) {
        if (auto lg = spdlog::default_logger()) lg->flush();   // 落盘余量
    }
    spdlog::shutdown();   // drop_all → 全部 sink 析构，释放当日日志文件句柄
    g_inited = false;
}

std::string obsExportDiagnosticsPackage(const std::string& dumpPath) {
    std::lock_guard<std::mutex> lock(g_obsMutex);
    std::error_code ec;

    const fs::path logDir(g_cfg.logDir);
    const fs::path baseDir = logDir.parent_path().empty() ? fs::path(".") : logDir.parent_path();
    const fs::path dst = baseDir / ("diagnostics_" + timestampName());
    fs::create_directories(dst, ec);

    // ① 日志目录副本（不存在则跳过）
    if (fs::exists(logDir, ec)) {
        fs::copy(logDir, dst / "logs",
                 fs::copy_options::recursive | fs::copy_options::overwrite_existing, ec);
    }

    // ② dump 副本（路径空或不存在 → 跳过，容错不报错）
    if (!dumpPath.empty()) {
        const fs::path dump(dumpPath);
        if (fs::exists(dump, ec)) {
            fs::create_directories(dst / "dumps", ec);
            fs::copy(dump, dst / "dumps" / dump.filename(),
                     fs::copy_options::overwrite_existing, ec);
        }
    }

    // ③ 版本信息文件
    {
        std::ofstream out(dst / "version.txt");
#if JMW_HAVE_VERSION_H
        out << "scanner_version=" << SCANNER_VERSION_MAJOR << '.' << SCANNER_VERSION_MINOR
            << '.' << SCANNER_VERSION_PATCH << '\n';
#else
        out << "scanner_version=unknown\n";
#endif
    }

    // zip 打包暂缓（YAGNI）：目录形态即交付物
    return dst.string();
}

} // namespace Scanner::service
