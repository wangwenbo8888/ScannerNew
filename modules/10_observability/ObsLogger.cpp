// ============================================================================
// ObsLogger.cpp — 统一日志装配实现（设计方案 §5）
//
//   · 双 sink（stdout_color + rotating_file）组 spdlog default logger；
//     幂等 = init 内先 spdlog::shutdown() 拆旧装配（registry 无永久关闭
//     标志，drop_all 后可重建；旧 sink 析构释放 jmw.log 句柄并落盘余量）
//   · maxFiles 语义 = 含当前文件共保留 N 份：spdlog 侧传 maxFiles-1
//     （spdlog max_files 指轮转副本数，N+1 才是实际文件数）
//   · 排障包每步 std::error_code 容错（缺则跳过）；spdlog 以共享模式打开
//     日志文件，sink 未关闭也能读走已 flush 部分
// ============================================================================
#include "ObsLogger.h"

#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>

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

} // namespace

void obsLoggerInit(const ObsLoggerConfig& cfg) {
    std::lock_guard<std::mutex> lock(g_obsMutex);
    spdlog::shutdown();   // 幂等：先拆旧装配（flush 由 sink 析构保证）

    std::error_code ec;
    fs::create_directories(cfg.logDir, ec);

    auto console = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto file = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        (fs::path(cfg.logDir) / (cfg.baseName + ".log")).string(),
        cfg.maxBytes,
        cfg.maxFiles > 0 ? cfg.maxFiles - 1 : 0);   // spdlog max_files=轮转副本数

    auto logger = std::make_shared<spdlog::logger>(
        "jmw", spdlog::sinks_init_list{console, file});
    logger->set_level(spdlog::level::info);
    logger->flush_on(spdlog::level::warn);          // warn+ 即时落盘（§7 崩溃留痕）
    logger->set_pattern("[%H:%M:%S.%e][%l][%n] %v");
    spdlog::set_default_logger(logger);

    g_cfg = cfg;
    g_inited = true;
}

void obsLoggerShutdown() {
    std::lock_guard<std::mutex> lock(g_obsMutex);
    if (g_inited) {
        if (auto lg = spdlog::default_logger()) lg->flush();   // 落盘余量
    }
    spdlog::shutdown();   // drop_all → 全部 sink 析构，释放 jmw.log 句柄
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
